"""
Design C — Stage 54
Independent audit and freeze-readiness check for the Stage-53C frozen
groundwater–Sentinel-1 primary panel.

Purpose
-------
Verify the frozen 193-row primary panel before any groundwater–Sentinel-1
association coefficient is fitted or inspected.

This stage:
- verifies exact row/well/year counts;
- verifies no duplicate station-year keys;
- verifies the two documented 10-km geometry-zero wells are absent;
- verifies all retained rows have valid primary Y, antecedent GW, and 10-km S1;
- verifies post-2021 retention counts;
- recomputes the frozen 10-km well exposure independently from the Stage-50
  support-year table for a deterministic sample of station-years;
- audits fixed-effects design rank;
- quantifies identifying variation in the frozen S1 exposure after:
    * station FE,
    * year FE,
    * station + year FE,
    * station + year FE + antecedent groundwater;
- DOES NOT fit or report the groundwater association coefficient.

Outputs
-------
outputs/diagnostics/design_c/c2zi_primary_panel_audit/
    c2zi_primary_panel_recompute_checks.csv
    c2zi_identifying_variation_audit.csv
    c2zi_primary_panel_audit.json
    c2zi_primary_panel_audit_summary.txt
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

P52 = (
    D / "c2ze_groundwater_protocol"
    / "c2ze_groundwater_association_protocol.json"
)
M53 = (
    D / "c2zh_groundwater_panel_freeze"
    / "c2zh_frozen_primary_panel_manifest.json"
)
PANEL = (
    D / "c2zh_groundwater_panel_freeze"
    / "c2zh_frozen_primary_groundwater_s1_panel.csv"
)
S1 = (
    D / "c2zc_cross_track_exposure"
    / "c2zc_support_year_flooding_like_exposure.csv.gz"
)

OUT = D / "c2zi_primary_panel_audit"
OUT.mkdir(parents=True, exist_ok=True)

RECOMP_OUT = OUT / "c2zi_primary_panel_recompute_checks.csv"
VAR_OUT = OUT / "c2zi_identifying_variation_audit.csv"
QA_OUT = OUT / "c2zi_primary_panel_audit.json"
SUMMARY_OUT = OUT / "c2zi_primary_panel_audit_summary.txt"

for f in [P52, M53, PANEL, S1]:
    if not f.exists():
        raise FileNotFoundError(f)

p52 = json.loads(P52.read_text(encoding="utf-8"))
m53 = json.loads(M53.read_text(encoding="utf-8"))

if p52.get("status") != "PROTOCOL_FROZEN_GROUNDWATER_VALUES_UNREAD":
    raise RuntimeError(f"Unexpected Stage-52 status: {p52.get('status')!r}")

if m53.get("status") != \
        "PASS_PRIMARY_PANEL_FROZEN_WITH_DOCUMENTED_STRUCTURAL_NONOVERLAP":
    raise RuntimeError(f"Unexpected Stage-53C status: {m53.get('status')!r}")

panel = pd.read_csv(PANEL, low_memory=False)
s1 = pd.read_csv(S1, compression="gzip", low_memory=False)

required_panel = {
    "station", "year", "utm_e", "utm_n",
    "gw_aug_m", "gw_pre_last_janfeb_m",
    "s1_flooding_like_10km",
}
missing = sorted(required_panel - set(panel.columns))
if missing:
    raise AssertionError(f"Frozen panel missing columns: {missing}")

panel["station"] = panel["station"].astype(str)
panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)

for col in [
    "utm_e", "utm_n", "gw_aug_m",
    "gw_pre_last_janfeb_m", "s1_flooding_like_10km",
]:
    panel[col] = pd.to_numeric(panel[col], errors="coerce")

required_s1 = {
    "support_id", "year", "lon", "lat", "flooding_like_s1_primary"
}
missing = sorted(required_s1 - set(s1.columns))
if missing:
    raise AssertionError(f"Frozen S1 table missing columns: {missing}")

s1["support_id"] = s1["support_id"].astype(str)
s1["year"] = pd.to_numeric(s1["year"], errors="raise").astype(int)
s1["lon"] = pd.to_numeric(s1["lon"], errors="coerce")
s1["lat"] = pd.to_numeric(s1["lat"], errors="coerce")
s1["flooding_like_s1_primary"] = pd.to_numeric(
    s1["flooding_like_s1_primary"], errors="coerce"
)

hard_flags = []

# ---------------------------------------------------------------------
# Exact panel structure
# ---------------------------------------------------------------------
if len(panel) != 193:
    hard_flags.append(f"PRIMARY_ROWS:{len(panel)}!=193")

if panel["station"].nunique() != 32:
    hard_flags.append(f"PRIMARY_WELLS:{panel['station'].nunique()}!=32")

expected_years = list(range(2015, 2026))
years = sorted(panel["year"].unique().tolist())
if years != expected_years:
    hard_flags.append(f"PRIMARY_YEARS:{years}")

if panel.duplicated(["station", "year"]).any():
    hard_flags.append("DUPLICATE_STATION_YEAR_KEYS")

if not np.isfinite(panel["gw_aug_m"]).all():
    hard_flags.append("MISSING_PRIMARY_OUTCOME")
if not np.isfinite(panel["gw_pre_last_janfeb_m"]).all():
    hard_flags.append("MISSING_ANTECEDENT_GW")
if not np.isfinite(panel["s1_flooding_like_10km"]).all():
    hard_flags.append("MISSING_PRIMARY_S1")

geometry_zero = {"PO0181220U0001", "PO0181550U0001"}
if set(panel["station"]) & geometry_zero:
    hard_flags.append("GEOMETRY_ZERO_WELL_PRESENT_IN_FROZEN_PRIMARY_PANEL")

expected_post = {2022: 17, 2023: 13, 2024: 15, 2025: 17}
for yr, expected in expected_post.items():
    observed = int((panel["year"] == yr).sum())
    if observed != expected:
        hard_flags.append(f"POST_COUNT_{yr}:{observed}!={expected}")

# ---------------------------------------------------------------------
# Independent S1 spatial recomputation on deterministic station-year sample
# ---------------------------------------------------------------------
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
support_coords["utm_e"] = np.asarray(e, float)
support_coords["utm_n"] = np.asarray(n, float)

s1_by_year = {
    int(y): d.set_index("support_id")
    for y, d in s1.groupby("year", sort=False)
}

# Deterministic audit sample:
# first, middle, last station alphabetically × selected years spanning archive,
# plus every post-2021 year for one central station.
stations_sorted = sorted(panel["station"].unique().tolist())
audit_stations = [
    stations_sorted[0],
    stations_sorted[len(stations_sorted)//2],
    stations_sorted[-1],
]
audit_pairs = set()

for st in audit_stations:
    st_years = sorted(panel.loc[panel["station"].eq(st), "year"].tolist())
    if st_years:
        for y in {st_years[0], st_years[len(st_years)//2], st_years[-1]}:
            audit_pairs.add((st, int(y)))

central_st = stations_sorted[len(stations_sorted)//2]
for y in [2022, 2023, 2024, 2025]:
    if ((panel["station"] == central_st) & (panel["year"] == y)).any():
        audit_pairs.add((central_st, y))

recomp_rows = []

for st, yr in sorted(audit_pairs):
    r = panel[(panel["station"] == st) & (panel["year"] == yr)].iloc[0]
    dist = np.sqrt(
        (support_coords["utm_e"].to_numpy() - float(r["utm_e"]))**2
        + (support_coords["utm_n"].to_numpy() - float(r["utm_n"]))**2
    )
    ids = support_coords.loc[dist <= 10000, "support_id"].astype(str).tolist()

    sy = s1_by_year[yr]
    present = [sid for sid in ids if sid in sy.index]
    vals = pd.to_numeric(
        sy.loc[present, "flooding_like_s1_primary"],
        errors="coerce",
    )
    vals = vals[np.isfinite(vals)]

    recomputed = float(vals.median()) if len(vals) else np.nan
    frozen = float(r["s1_flooding_like_10km"])
    abs_diff = abs(recomputed - frozen) if np.isfinite(recomputed) else np.nan
    ok = bool(np.isfinite(recomputed) and abs_diff <= 1e-12)

    recomp_rows.append({
        "station": st,
        "year": yr,
        "support_points_10km_n": len(ids),
        "valid_support_points_n": int(len(vals)),
        "frozen_s1_flooding_like_10km": frozen,
        "recomputed_s1_flooding_like_10km": recomputed,
        "absolute_difference": abs_diff,
        "exact_within_1e_12": ok,
    })

    if not ok:
        hard_flags.append(f"S1_RECOMPUTE_FAILURE:{st}:{yr}")

recomp = pd.DataFrame(recomp_rows)
recomp.to_csv(RECOMP_OUT, index=False)

# ---------------------------------------------------------------------
# Design-rank and identifying-variation audit WITHOUT reporting beta
# ---------------------------------------------------------------------
# Build dummy matrix for station/year FE.
station_d = pd.get_dummies(
    panel["station"], prefix="station", drop_first=True, dtype=float
)
year_d = pd.get_dummies(
    panel["year"], prefix="year", drop_first=True, dtype=float
)

ones = np.ones((len(panel), 1), dtype=float)
A = panel["gw_pre_last_janfeb_m"].to_numpy(float)[:, None]
X_exposure = panel["s1_flooding_like_10km"].to_numpy(float)

# Full design includes intercept, exposure, antecedent, station FE, year FE.
X_full = np.column_stack([
    ones,
    X_exposure[:, None],
    A,
    station_d.to_numpy(float),
    year_d.to_numpy(float),
])

rank = int(np.linalg.matrix_rank(X_full))
cols_n = int(X_full.shape[1])

if rank != cols_n:
    hard_flags.append(f"FULL_DESIGN_RANK_DEFICIENT:{rank}/{cols_n}")

def residualize(x: np.ndarray, Z: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(Z, x, rcond=None)
    return x - Z @ coef

Z_station = np.column_stack([ones, station_d.to_numpy(float)])
Z_year = np.column_stack([ones, year_d.to_numpy(float)])
Z_twfe = np.column_stack([
    ones,
    station_d.to_numpy(float),
    year_d.to_numpy(float),
])
Z_full_adjust = np.column_stack([
    ones,
    A,
    station_d.to_numpy(float),
    year_d.to_numpy(float),
])

r_station = residualize(X_exposure, Z_station)
r_year = residualize(X_exposure, Z_year)
r_twfe = residualize(X_exposure, Z_twfe)
r_full = residualize(X_exposure, Z_full_adjust)

def desc(label: str, x: np.ndarray) -> dict:
    return {
        "stage": label,
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "variance": float(np.var(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
    }

raw = desc("raw_s1_exposure", X_exposure)
rows = [
    raw,
    desc("after_station_fe", r_station),
    desc("after_year_fe", r_year),
    desc("after_station_plus_year_fe", r_twfe),
    desc("after_station_year_fe_plus_antecedent_gw", r_full),
]

raw_var = raw["variance"]
for rr in rows:
    rr["variance_fraction_of_raw"] = (
        rr["variance"] / raw_var if raw_var > 0 else np.nan
    )

var_df = pd.DataFrame(rows)
var_df.to_csv(VAR_OUT, index=False)

if not np.isfinite(r_full).all():
    hard_flags.append("NONFINITE_FULLY_RESIDUALIZED_EXPOSURE")
if float(np.std(r_full, ddof=1)) <= 0:
    hard_flags.append("ZERO_IDENTIFYING_VARIATION_AFTER_FULL_ADJUSTMENT")

status = (
    "PASS_PRIMARY_PANEL_AUDITED_AND_READY_FOR_FIRST_ASSOCIATION_FIT"
    if not hard_flags
    else "FAIL_REVIEW_REQUIRED"
)

qa = {
    "stage": "DESIGN_C_STAGE54_INDEPENDENT_PRIMARY_PANEL_AUDIT",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "panel_rows_n": int(len(panel)),
    "panel_wells_n": int(panel["station"].nunique()),
    "panel_years": years,
    "post2021_counts": {
        str(y): int((panel["year"] == y).sum())
        for y in [2022, 2023, 2024, 2025]
    },
    "independent_s1_recompute_checks_n": int(len(recomp)),
    "independent_s1_recompute_failures_n":
        int((~recomp["exact_within_1e_12"]).sum()),
    "design_matrix": {
        "columns_n": cols_n,
        "rank": rank,
        "full_rank": rank == cols_n,
    },
    "identifying_variation": {
        "raw_s1_sd": raw["sd"],
        "twfe_residual_s1_sd":
            float(np.std(r_twfe, ddof=1)),
        "fully_adjusted_residual_s1_sd":
            float(np.std(r_full, ddof=1)),
        "fully_adjusted_variance_fraction_of_raw":
            float(np.var(r_full, ddof=1) / raw_var) if raw_var > 0 else None,
    },
    "hard_flags": hard_flags,
    "firewall": {
        "groundwater_values_read": True,
        "frozen_s1_exposure_read": True,
        "association_coefficient_fitted": False,
        "association_coefficient_reported": False,
        "irrigation_flow_values_read": False,
        "model_search_performed": False,
    },
    "sha256": {
        "stage52_protocol": hashlib.sha256(P52.read_bytes()).hexdigest(),
        "stage53c_manifest": hashlib.sha256(M53.read_bytes()).hexdigest(),
        "frozen_primary_panel": hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        "stage50_s1_support_year": hashlib.sha256(S1.read_bytes()).hexdigest(),
    },
}

QA_OUT.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

lines = [
    "DESIGN C - STAGE 54 INDEPENDENT PRIMARY PANEL AUDIT",
    "=" * 74,
    "",
    "PANEL STRUCTURE",
    "---------------",
    f"Rows: {len(panel)}",
    f"Wells: {panel['station'].nunique()}",
    f"Years: {years}",
    f"Duplicate station-years: {int(panel.duplicated(['station','year']).sum())}",
    "",
    "POST-2021 COUNTS",
    "----------------",
]
for y in [2022, 2023, 2024, 2025]:
    lines.append(f"{y}: {int((panel['year'] == y).sum())}")

lines += [
    "",
    "INDEPENDENT 10-KM S1 RECOMPUTATION",
    "----------------------------------",
    f"Checks: {len(recomp)}",
    f"Failures: {int((~recomp['exact_within_1e_12']).sum())}",
    "",
    "DESIGN RANK",
    "-----------",
    f"Columns: {cols_n}",
    f"Rank: {rank}",
    f"Full rank: {rank == cols_n}",
    "",
    "IDENTIFYING VARIATION IN FROZEN S1 EXPOSURE",
    "-------------------------------------------",
    f"Raw SD: {np.std(X_exposure, ddof=1):.12f}",
    f"After station FE SD: {np.std(r_station, ddof=1):.12f}",
    f"After year FE SD: {np.std(r_year, ddof=1):.12f}",
    f"After station + year FE SD: {np.std(r_twfe, ddof=1):.12f}",
    f"After station + year FE + antecedent GW SD: "
    f"{np.std(r_full, ddof=1):.12f}",
    f"Fully adjusted variance / raw variance: "
    f"{(np.var(r_full, ddof=1)/raw_var):.12f}",
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
    "Association coefficient fitted: False",
    "Association coefficient reported: False",
    "Irrigation-flow values read: False",
    "Model search performed: False",
    "",
    f"STAGE 54 STATUS: {status}",
]

SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

if hard_flags:
    raise RuntimeError("Stage 54 failed hard QA; do not fit the association model.")
