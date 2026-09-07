"""
Design C — Stage 53B-v2
Reconstruct the complete 2015–2025 observation-level ISS groundwater panel,
including the separately acquired authoritative 2024 and 2025 ARPA extensions,
then link the already-frozen Sentinel-1 exposure.

CRITICAL FIREWALL
-----------------
This stage DOES NOT fit or reveal any groundwater–Sentinel-1 association model.

Why v2?
-------
The historical observation table groundwater_clean.csv ends in 2023, but the
project had already acquired and frozen authoritative ARPA groundwater updates
for 2024 and 2025 in Stage 14 of the post-2021 workflow. This script reuses the
audited Stage-14 raw loaders so that the new Design-C Stage-52 August-MEDIAN
outcome can be reconstructed from observation-level records through 2025.

Primary groundwater population remains the frozen shallow ISS population:
    68 total metadata stations (context only)
    37 historical ISS wells
    minus 2 pre-existing S1-era exclusions
    = 35 candidate ISS wells

Frozen exclusions:
    PO0180930U0006
    PO0181870U0001

Groundwater temporal rules from Stage 52
----------------------------------------
Outcome:
    gw_aug_m = median of all valid August groundwater-depth observations.

Antecedent:
    gw_pre_last_janfeb_m = last valid Jan–Feb groundwater-depth observation.

Frozen S1 spatial linkage
-------------------------
Primary:
    unweighted median flooding_like_s1_primary within 10 km.

Sensitivities:
    5 km
    20 km
    complete-4-track exposure within 10 km.

No feature/radius/station selection from groundwater association performance.

Outputs
-------
outputs/diagnostics/design_c/c2zg_groundwater_panel/
    c2zg_groundwater_observation_source_provenance.csv
    c2zg_stage14_extension_reproduction_qa.csv
    c2zg_groundwater_temporal_panel_2015_2025.csv
    c2zg_well_s1_spatial_linkage_qa.csv
    c2zg_groundwater_s1_analysis_panel_2015_2025.csv
    c2zg_groundwater_s1_panel_coverage.csv
    c2zg_groundwater_s1_identifying_variation_qa.csv
    c2zg_groundwater_s1_panel_qa.json
    c2zg_groundwater_s1_panel_summary.txt
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

HIST_CLEAN = (
    ROOT / "data" / "processed" / "publication_groundwater"
    / "groundwater_clean.csv"
)
META = (
    ROOT / "data" / "processed" / "publication_groundwater"
    / "groundwater_station_metadata.csv"
)
ANNUAL_2025 = (
    ROOT / "data" / "processed" / "post2021"
    / "groundwater_annual_measures_2008_2025.csv"
)
STAGE14 = (
    ROOT / "scripts" / "05_post2021_flooding"
    / "14_extend_groundwater_annual_measures.py"
)

P52 = (
    D / "c2ze_groundwater_protocol"
    / "c2ze_groundwater_association_protocol.json"
)
FREEZE51 = (
    D / "c2zd_exposure_freeze"
    / "c2zd_frozen_s1_exposure_manifest.json"
)
S1 = (
    D / "c2zc_cross_track_exposure"
    / "c2zc_support_year_flooding_like_exposure.csv.gz"
)

OUT = D / "c2zg_groundwater_panel"
OUT.mkdir(parents=True, exist_ok=True)

PROV_OUT = OUT / "c2zg_groundwater_observation_source_provenance.csv"
REPRO_OUT = OUT / "c2zg_stage14_extension_reproduction_qa.csv"
GW_TEMP_OUT = OUT / "c2zg_groundwater_temporal_panel_2015_2025.csv"
LINK_OUT = OUT / "c2zg_well_s1_spatial_linkage_qa.csv"
PANEL_OUT = OUT / "c2zg_groundwater_s1_analysis_panel_2015_2025.csv"
COVERAGE_OUT = OUT / "c2zg_groundwater_s1_panel_coverage.csv"
VAR_OUT = OUT / "c2zg_groundwater_s1_identifying_variation_qa.csv"
QA_OUT = OUT / "c2zg_groundwater_s1_panel_qa.json"
SUMMARY_OUT = OUT / "c2zg_groundwater_s1_panel_summary.txt"

for f in [HIST_CLEAN, META, ANNUAL_2025, STAGE14, P52, FREEZE51, S1]:
    if not f.exists():
        raise FileNotFoundError(f)

p52 = json.loads(P52.read_text(encoding="utf-8"))
freeze51 = json.loads(FREEZE51.read_text(encoding="utf-8"))

if p52.get("status") != "PROTOCOL_FROZEN_GROUNDWATER_VALUES_UNREAD":
    raise RuntimeError(
        f"Unexpected Stage-52 protocol status: {p52.get('status')!r}"
    )

if freeze51.get("status") != \
        "PASS_FINAL_S1_EXPOSURE_FROZEN_FOR_HYDROLOGICAL_ANALYSIS":
    raise RuntimeError(
        f"Unexpected Stage-51 freeze status: {freeze51.get('status')!r}"
    )

PRIMARY_SUPPORT_COL = \
    p52["upstream_s1_freeze"]["primary_support_year_exposure_column"]
COMPLETE4_SUPPORT_COL = \
    p52["upstream_s1_freeze"]["complete4_sensitivity_column"]

EXCLUDED = {
    x["station"]
    for x in p52["population"]["outcome_blind_preexisting_exclusions"]
}

if EXCLUDED != {"PO0180930U0006", "PO0181870U0001"}:
    raise RuntimeError(f"Unexpected frozen exclusion set: {sorted(EXCLUDED)}")

# -------------------------------------------------------------------------
# Load Stage-14 module without executing its command-line main block.
# -------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("stage14_groundwater", STAGE14)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Stage-14 groundwater extension module.")

stage14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage14)

meta = pd.read_csv(META, low_memory=False)
meta["station"] = meta["station"].astype(str)
meta["aquifer_group"] = (
    meta["aquifer_group"].astype(str).str.upper().str.strip()
)

full_station_n = int(meta["station"].nunique())
class_counts = (
    meta.drop_duplicates("station")["aquifer_group"]
    .value_counts(dropna=False)
    .to_dict()
)
iss_meta = (
    meta.loc[meta["aquifer_group"].eq("ISS")]
    .drop_duplicates("station")
    .copy()
)

hard_flags = []

if full_station_n != 68:
    hard_flags.append(f"FULL_METADATA_STATION_UNIVERSE:{full_station_n}!=68")
if len(iss_meta) != 37:
    hard_flags.append(f"ISS_METADATA_UNIVERSE:{len(iss_meta)}!=37")

# -------------------------------------------------------------------------
# Historical observation-level source (through 2023)
# -------------------------------------------------------------------------
hist = pd.read_csv(HIST_CLEAN, low_memory=False)

required = {
    "station", "province", "commune", "gw_depth_m", "date",
    "utm_e", "utm_n", "gwb", "aquifer_group"
}
missing = sorted(required - set(hist.columns))
if missing:
    raise AssertionError(f"Historical groundwater source missing: {missing}")

hist["station"] = hist["station"].astype(str)
hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
hist["gw_depth_m"] = pd.to_numeric(hist["gw_depth_m"], errors="coerce")
hist["aquifer_group"] = (
    hist["aquifer_group"].astype(str).str.upper().str.strip()
)
hist = hist[hist["date"].notna()].copy()
hist["year"] = hist["date"].dt.year.astype(int)
hist["month"] = hist["date"].dt.month.astype(int)
hist["doy"] = hist["date"].dt.dayofyear.astype(int)

# Historical production file must stop before the separately acquired updates.
if int(hist["year"].max()) > 2023:
    hard_flags.append(
        f"HISTORICAL_SOURCE_UNEXPECTEDLY_EXTENDS_TO:{int(hist['year'].max())}"
    )

# -------------------------------------------------------------------------
# Reuse the already-audited Stage-14 authoritative raw loaders.
# Only the first tuple element is the harmonized observation-level dataframe.
# -------------------------------------------------------------------------
res24 = stage14.load_authoritative_2024(meta)
res25 = stage14.load_authoritative_2025(meta)

obs24 = res24[0].copy()
obs25 = res25[0].copy()

for d, yr in [(obs24, 2024), (obs25, 2025)]:
    d["station"] = d["station"].astype(str)
    d["date"] = pd.to_datetime(d["date"], errors="raise")
    d["gw_depth_m"] = pd.to_numeric(d["gw_depth_m"], errors="coerce")
    d["aquifer_group"] = (
        d["aquifer_group"].astype(str).str.upper().str.strip()
    )
    d["year"] = d["date"].dt.year.astype(int)
    d["month"] = d["date"].dt.month.astype(int)
    d["doy"] = d["date"].dt.dayofyear.astype(int)
    if not d["year"].eq(yr).all():
        hard_flags.append(f"AUTHORITATIVE_{yr}_DATE_YEAR_MISMATCH")

# -------------------------------------------------------------------------
# Stage-14 reproduction check for 2024/2025 ISS annual quantities.
# This independently verifies that the raw loaders still reproduce the frozen
# annual extension for key fields. We do NOT use gw_aug_mean_m as Design-C Y.
# -------------------------------------------------------------------------
frozen_annual = pd.read_csv(ANNUAL_2025, low_memory=False)
frozen_annual["station"] = frozen_annual["station"].astype(str)
frozen_annual["year"] = pd.to_numeric(
    frozen_annual["year"], errors="raise"
).astype(int)

repro_rows = []
check_cols = [
    "gw_obs_n",
    "gw_janfeb_n",
    "gw_aug_n",
    "gw_pre_last_janfeb_m",
    "gw_pre_last_janfeb_date",
    "gw_aug_mean_m",
    "gw_aug_first_m",
    "gw_aug_last_m",
    "gw_aug_nearest_aug23_m",
]

for yr, raw_obs in [(2024, obs24), (2025, obs25)]:
    r = raw_obs[
        raw_obs["aquifer_group"].eq("ISS")
        & raw_obs["station"].isin(set(iss_meta["station"]))
    ].copy()

    generated = (
        r.groupby("station", sort=True, group_keys=False)
        .apply(stage14.yearly_record, include_groups=False)
        .reset_index()
    )
    generated["year"] = yr

    frozen = frozen_annual[
        frozen_annual["year"].eq(yr)
        & frozen_annual["station"].isin(set(iss_meta["station"]))
    ].copy()

    # Only compare station-years with at least one raw observation.
    frozen = frozen[frozen["station"].isin(set(generated["station"]))].copy()

    m = generated.merge(
        frozen[["station", "year"] + check_cols],
        on=["station", "year"],
        how="outer",
        suffixes=("_generated", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )

    if not m["_merge"].eq("both").all():
        hard_flags.append(f"STAGE14_{yr}_STATION_KEY_REPRODUCTION_FAILURE")

    for col in check_cols:
        a = m[f"{col}_generated"]
        b = m[f"{col}_frozen"]

        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            aa = pd.to_numeric(a, errors="coerce").to_numpy(float)
            bb = pd.to_numeric(b, errors="coerce").to_numpy(float)
            equal = np.isclose(
                aa, bb, rtol=0, atol=1e-12, equal_nan=True
            )
        else:
            aa = a.astype("string").fillna("<NA>")
            bb = b.astype("string").fillna("<NA>")
            equal = aa.to_numpy() == bb.to_numpy()

        mismatch_n = int((~equal).sum())
        repro_rows.append({
            "year": yr,
            "column": col,
            "rows_compared": int(len(equal)),
            "mismatch_n": mismatch_n,
            "exact_reproduction": mismatch_n == 0,
        })

        if mismatch_n:
            hard_flags.append(
                f"STAGE14_{yr}_REPRO_MISMATCH:{col}:{mismatch_n}"
            )

repro_df = pd.DataFrame(repro_rows)
repro_df.to_csv(REPRO_OUT, index=False)

# -------------------------------------------------------------------------
# Combine true observation-level sources.
# Historical through 2023 + authoritative 2024 + authoritative 2025.
# -------------------------------------------------------------------------
common_cols = sorted(
    set(hist.columns) & set(obs24.columns) & set(obs25.columns)
)
required_common = {
    "station", "gw_depth_m", "date", "year", "month", "doy",
    "utm_e", "utm_n", "aquifer_group"
}
if not required_common.issubset(common_cols):
    raise AssertionError(
        "Combined observation sources lack required harmonized columns."
    )

h = hist[hist["year"] <= 2023][common_cols].copy()
o24 = obs24[common_cols].copy()
o25 = obs25[common_cols].copy()

combined = pd.concat([h, o24, o25], ignore_index=True)
combined["station"] = combined["station"].astype(str)
combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
combined["gw_depth_m"] = pd.to_numeric(
    combined["gw_depth_m"], errors="coerce"
)
combined["year"] = combined["date"].dt.year.astype("Int64")
combined["month"] = combined["date"].dt.month.astype("Int64")
combined["aquifer_group"] = (
    combined["aquifer_group"].astype(str).str.upper().str.strip()
)

prov = pd.DataFrame([
    {
        "source_role": "historical_observation_level",
        "path": str(HIST_CLEAN.relative_to(ROOT)),
        "years_used": "2015-2023",
        "authoritative_for": "2015-2023",
    },
    {
        "source_role": "authoritative_2024_observation_level",
        "path": str(stage14.RAW_2024.relative_to(ROOT)),
        "years_used": "2024",
        "authoritative_for": "2024",
    },
    {
        "source_role": "authoritative_2025_observation_level",
        "path": str(stage14.RAW_2025.relative_to(ROOT)),
        "years_used": "2025",
        "authoritative_for": "2025",
    },
])
prov.to_csv(PROV_OUT, index=False)

# Frozen ISS analysis population.
iss = combined[
    combined["aquifer_group"].eq("ISS")
    & combined["station"].isin(set(iss_meta["station"]))
    & ~combined["station"].isin(EXCLUDED)
    & combined["year"].between(2015, 2025)
    & combined["date"].notna()
    & np.isfinite(combined["gw_depth_m"])
].copy()

# No duplicate station-date values may remain in the harmonized ISS path.
dup = iss.duplicated(["station", "date"], keep=False)
if dup.any():
    hard_flags.append(
        f"DUPLICATE_STATION_DATE_AFTER_HARMONIZATION:{int(dup.sum())}"
    )

# Coordinate stability against frozen station metadata.
meta_xy = iss_meta.set_index("station")[["utm_e", "utm_n"]]
coord_bad = 0
for st, s in iss.groupby("station"):
    if st not in meta_xy.index:
        coord_bad += 1
        continue
    e0 = float(meta_xy.loc[st, "utm_e"])
    n0 = float(meta_xy.loc[st, "utm_n"])
    ee = pd.to_numeric(s["utm_e"], errors="coerce").to_numpy(float)
    nn = pd.to_numeric(s["utm_n"], errors="coerce").to_numpy(float)
    # 2025 source was frozen with 0.01-m precision tolerance.
    if not np.all(np.isclose(ee, e0, rtol=0, atol=0.01, equal_nan=False)):
        coord_bad += 1
    if not np.all(np.isclose(nn, n0, rtol=0, atol=0.01, equal_nan=False)):
        coord_bad += 1

if coord_bad:
    hard_flags.append(f"ISS_COORDINATE_METADATA_MISMATCHES:{coord_bad}")

# -------------------------------------------------------------------------
# Construct exact frozen Design-C temporal variables from observations.
# -------------------------------------------------------------------------
gw_rows = []

for (station, year), s in iss.groupby(["station", "year"], sort=True):
    s = s.sort_values("date")
    aug = s[s["month"].eq(8)].copy()
    jf = s[s["month"].isin([1, 2])].copy()

    if len(aug) == 0 or len(jf) == 0:
        continue

    last_jf = jf.iloc[-1]

    gw_rows.append({
        "station": station,
        "year": int(year),
        "utm_e": float(meta_xy.loc[station, "utm_e"]),
        "utm_n": float(meta_xy.loc[station, "utm_n"]),
        "gw_aug_m": float(aug["gw_depth_m"].median()),
        "gw_aug_obs_n": int(len(aug)),
        "gw_aug_mean_m_descriptive": float(aug["gw_depth_m"].mean()),
        "gw_aug_first_date": str(aug["date"].min().date()),
        "gw_aug_last_date": str(aug["date"].max().date()),
        "gw_pre_last_janfeb_m": float(last_jf["gw_depth_m"]),
        "gw_pre_last_janfeb_date": str(last_jf["date"].date()),
        "gw_janfeb_obs_n": int(len(jf)),
    })

gw_panel = pd.DataFrame(gw_rows)

if gw_panel.empty:
    hard_flags.append("NO_TEMPORALLY_ELIGIBLE_GROUNDWATER_ROWS")

if not gw_panel.empty and \
        gw_panel.duplicated(["station", "year"], keep=False).any():
    hard_flags.append("DUPLICATE_GROUNDWATER_STATION_YEAR_KEYS")

gw_panel.to_csv(GW_TEMP_OUT, index=False)

# Frozen post-2021 complete availability expectations from Stage 14.
expected_complete = {2022: 17, 2023: 13, 2024: 15, 2025: 17}
observed_complete = (
    gw_panel.groupby("year")["station"].nunique().to_dict()
    if not gw_panel.empty else {}
)

for yr, expected in expected_complete.items():
    observed = int(observed_complete.get(yr, 0))
    if observed != expected:
        hard_flags.append(
            f"COMPLETE_ISS_COUNT_{yr}:{observed}!=frozen_{expected}"
        )

# -------------------------------------------------------------------------
# Load frozen S1 support-year exposure and spatially link to wells.
# -------------------------------------------------------------------------
s1 = pd.read_csv(S1, compression="gzip", low_memory=False)

required_s1 = {
    "support_id", "year", "lon", "lat",
    PRIMARY_SUPPORT_COL, COMPLETE4_SUPPORT_COL,
}
missing = sorted(required_s1 - set(s1.columns))
if missing:
    raise AssertionError(f"Frozen S1 table missing columns: {missing}")

s1["support_id"] = s1["support_id"].astype(str)
s1["year"] = pd.to_numeric(s1["year"], errors="raise").astype(int)
s1["lon"] = pd.to_numeric(s1["lon"], errors="coerce")
s1["lat"] = pd.to_numeric(s1["lat"], errors="coerce")
s1[PRIMARY_SUPPORT_COL] = pd.to_numeric(
    s1[PRIMARY_SUPPORT_COL], errors="coerce"
)
s1[COMPLETE4_SUPPORT_COL] = pd.to_numeric(
    s1[COMPLETE4_SUPPORT_COL], errors="coerce"
)

support_coords = (
    s1[["support_id", "lon", "lat"]]
    .drop_duplicates()
    .sort_values("support_id")
    .reset_index(drop=True)
)

if support_coords["support_id"].nunique() != len(support_coords):
    hard_flags.append("S1_SUPPORT_COORDINATE_DUPLICATION")

transformer = Transformer.from_crs(
    "EPSG:4326", "EPSG:32632", always_xy=True
)
e, n = transformer.transform(
    support_coords["lon"].to_numpy(float),
    support_coords["lat"].to_numpy(float),
)
support_coords["s1_utm_e"] = np.asarray(e, float)
support_coords["s1_utm_n"] = np.asarray(n, float)

stations = (
    gw_panel[["station", "utm_e", "utm_n"]]
    .drop_duplicates()
    .sort_values("station")
    .reset_index(drop=True)
)

membership = {}
link_rows = []

for _, wr in stations.iterrows():
    st = str(wr["station"])
    we = float(wr["utm_e"])
    wn = float(wr["utm_n"])

    dist = np.sqrt(
        (support_coords["s1_utm_e"].to_numpy() - we) ** 2
        + (support_coords["s1_utm_n"].to_numpy() - wn) ** 2
    )

    for radius in [5000, 10000, 20000]:
        mask = dist <= radius
        ids = support_coords.loc[mask, "support_id"].astype(str).tolist()
        membership[(st, radius)] = ids

        link_rows.append({
            "station": st,
            "radius_m": radius,
            "support_points_n": int(len(ids)),
            "nearest_support_distance_m": float(np.min(dist)),
            "farthest_included_support_distance_m":
                float(np.max(dist[mask])) if mask.any() else np.nan,
        })

link_df = pd.DataFrame(link_rows)
link_df.to_csv(LINK_OUT, index=False)

if (link_df.query("radius_m == 10000")["support_points_n"] == 0).any():
    hard_flags.append("ONE_OR_MORE_WELLS_HAVE_ZERO_10KM_S1_SUPPORT")

s1_by_year = {
    int(y): d.set_index("support_id")
    for y, d in s1.groupby("year", sort=False)
}

panel_rows = []

for _, r in gw_panel.iterrows():
    st = str(r["station"])
    yr = int(r["year"])

    if yr not in s1_by_year:
        continue

    sy = s1_by_year[yr]

    def aggregate(radius: int, col: str):
        ids = membership[(st, radius)]
        present = [sid for sid in ids if sid in sy.index]
        if not present:
            return np.nan, 0, 0
        vals = pd.to_numeric(sy.loc[present, col], errors="coerce")
        finite = vals[np.isfinite(vals)]
        return (
            float(finite.median()) if len(finite) else np.nan,
            int(len(present)),
            int(len(finite)),
        )

    x5, n5, v5 = aggregate(5000, PRIMARY_SUPPORT_COL)
    x10, n10, v10 = aggregate(10000, PRIMARY_SUPPORT_COL)
    x20, n20, v20 = aggregate(20000, PRIMARY_SUPPORT_COL)
    xc4, nc4, vc4 = aggregate(10000, COMPLETE4_SUPPORT_COL)

    out = dict(r)
    out.update({
        "s1_flooding_like_5km": x5,
        "s1_support_points_5km_n": n5,
        "s1_valid_support_points_5km_n": v5,
        "s1_flooding_like_10km": x10,
        "s1_support_points_10km_n": n10,
        "s1_valid_support_points_10km_n": v10,
        "s1_flooding_like_20km": x20,
        "s1_support_points_20km_n": n20,
        "s1_valid_support_points_20km_n": v20,
        "s1_flooding_like_complete4_10km": xc4,
        "s1_complete4_support_points_10km_n": nc4,
        "s1_complete4_valid_support_points_10km_n": vc4,
    })
    panel_rows.append(out)

panel = pd.DataFrame(panel_rows)

panel["primary_complete"] = (
    np.isfinite(panel["gw_aug_m"])
    & np.isfinite(panel["gw_pre_last_janfeb_m"])
    & np.isfinite(panel["s1_flooding_like_10km"])
)

panel.to_csv(PANEL_OUT, index=False)

coverage_rows = []
for yr in range(2015, 2026):
    y = panel[panel["year"].eq(yr)]
    coverage_rows.append({
        "year": yr,
        "temporally_eligible_well_years_n": int(len(y)),
        "primary_complete_well_years_n": int(y["primary_complete"].sum()),
        "stations_n": int(y["station"].nunique()),
        "s1_5km_available_n":
            int(np.isfinite(y["s1_flooding_like_5km"]).sum()),
        "s1_10km_available_n":
            int(np.isfinite(y["s1_flooding_like_10km"]).sum()),
        "s1_20km_available_n":
            int(np.isfinite(y["s1_flooding_like_20km"]).sum()),
        "s1_complete4_10km_available_n":
            int(np.isfinite(y["s1_flooding_like_complete4_10km"]).sum()),
    })

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(COVERAGE_OUT, index=False)

primary = panel[panel["primary_complete"]].copy()

var_rows = []
for st, s in primary.groupby("station", sort=True):
    x = pd.to_numeric(s["s1_flooding_like_10km"], errors="coerce")
    var_rows.append({
        "station": st,
        "primary_years_n": int(len(s)),
        "first_primary_year": int(s["year"].min()),
        "last_primary_year": int(s["year"].max()),
        "s1_10km_mean": float(x.mean()),
        "s1_10km_sd_within_station":
            float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "s1_10km_range_within_station":
            float(x.max() - x.min()) if len(x) else np.nan,
    })

var_df = pd.DataFrame(var_rows)
var_df.to_csv(VAR_OUT, index=False)

if len(primary) == 0:
    hard_flags.append("NO_PRIMARY_COMPLETE_WELL_YEARS")

primary_wells = int(primary["station"].nunique())
wells_ge2 = int((primary.groupby("station").size() >= 2).sum())
years_present = sorted(primary["year"].astype(int).unique().tolist())

status = (
    "PASS_COMPLETE_2015_2025_GROUNDWATER_S1_PANEL_BUILT"
    if not hard_flags else
    "FAIL_REVIEW_REQUIRED"
)

qa = {
    "stage": "DESIGN_C_STAGE53B_V2_COMPLETE_GROUNDWATER_S1_PANEL",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,

    "station_universe_context": {
        "all_metadata_stations_n": full_station_n,
        "aquifer_group_counts": {
            str(k): int(v) for k, v in class_counts.items()
        },
        "historical_iss_wells_n": int(len(iss_meta)),
        "frozen_exclusions": sorted(EXCLUDED),
        "candidate_iss_after_exclusions_n":
            int(len(set(iss_meta["station"]) - EXCLUDED)),
    },

    "groundwater_temporally_eligible_well_years_n": int(len(gw_panel)),
    "primary_complete_well_years_n": int(len(primary)),
    "primary_wells_n": primary_wells,
    "primary_wells_with_at_least_2_years_n": wells_ge2,
    "primary_years": years_present,

    "frozen_post2021_complete_counts_expected": expected_complete,
    "frozen_post2021_complete_counts_observed": {
        str(k): int(v) for k, v in observed_complete.items()
        if k in expected_complete
    },

    "hard_flags": hard_flags,

    "groundwater_values_read": True,
    "sentinel1_frozen_exposure_read": True,
    "irrigation_flow_values_read": False,
    "hydrological_association_model_fitted": False,
    "groundwater_driven_station_exclusion": False,
    "groundwater_driven_radius_selection": False,
    "groundwater_driven_s1_feature_selection": False,

    "sha256": {
        "historical_groundwater_clean":
            hashlib.sha256(HIST_CLEAN.read_bytes()).hexdigest(),
        "raw_2024":
            hashlib.sha256(stage14.RAW_2024.read_bytes()).hexdigest(),
        "raw_2025":
            hashlib.sha256(stage14.RAW_2025.read_bytes()).hexdigest(),
        "stage52_protocol":
            hashlib.sha256(P52.read_bytes()).hexdigest(),
        "stage51_s1_exposure":
            hashlib.sha256(S1.read_bytes()).hexdigest(),
        "analysis_panel":
            hashlib.sha256(PANEL_OUT.read_bytes()).hexdigest(),
    },
}

QA_OUT.write_text(
    json.dumps(qa, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 53B-v2 COMPLETE 2015-2025 GROUNDWATER-S1 PANEL",
    "=" * 82,
    "",
    "GROUNDWATER UNIVERSE",
    "--------------------",
    f"All metadata stations: {full_station_n}",
    f"Historical ISS wells: {len(iss_meta)}",
    f"Frozen S1-era exclusions: {len(EXCLUDED)}",
    f"Candidate ISS wells after exclusions: "
    f"{len(set(iss_meta['station']) - EXCLUDED)}",
    "",
    "AUTHORITATIVE OBSERVATION SOURCES",
    "---------------------------------",
    "2015-2023: publication_groundwater/groundwater_clean.csv",
    f"2024: {stage14.RAW_2024.relative_to(ROOT)}",
    f"2025: {stage14.RAW_2025.relative_to(ROOT)}",
    "",
    "STAGE-14 EXTENSION REPRODUCTION",
    "-------------------------------",
    f"Reproduction checks: {len(repro_df)}",
    f"Reproduction failures: "
    f"{int((~repro_df['exact_reproduction']).sum())}",
    "",
    "FROZEN POST-2021 TEMPORAL ELIGIBILITY",
    "-------------------------------------",
]
for yr in [2022, 2023, 2024, 2025]:
    lines.append(
        f"{yr}: observed {int(observed_complete.get(yr, 0))} "
        f"/ frozen expected {expected_complete[yr]}"
    )

lines += [
    "",
    "FINAL PANEL",
    "-----------",
    f"Temporally eligible groundwater well-years: {len(gw_panel)}",
    f"Primary complete groundwater-S1 well-years: {len(primary)}",
    f"Primary wells: {primary_wells}",
    f"Primary wells with >=2 years: {wells_ge2}",
    f"Primary years represented: {years_present}",
    "",
    "YEAR COVERAGE",
    "-------------",
    coverage.to_string(index=False),
    "",
    "HARD QA FLAGS",
    "-------------",
]
lines += hard_flags if hard_flags else ["None"]
lines += [
    "",
    "FIREWALL",
    "--------",
    "Groundwater values read: True",
    "Frozen Sentinel-1 exposure read: True",
    "Irrigation-flow values read: False",
    "Hydrological association model fitted: False",
    "Groundwater-driven station exclusion: False",
    "Groundwater-driven radius selection: False",
    "Groundwater-driven S1 feature selection: False",
    "",
    f"STAGE 53B-v2 STATUS: {status}",
]

SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

if hard_flags:
    raise RuntimeError(
        "Stage 53B-v2 failed hard QA; inspect generated diagnostics."
    )
