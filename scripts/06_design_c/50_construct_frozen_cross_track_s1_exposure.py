"""
Design C — Stage 50 execute the frozen cross-track Sentinel-1 representation.

Purpose
-------
Apply the Stage-49 protocol to the frozen Stage-44/45 Sentinel-1 phenology
archive and construct one support-year flooding-like Sentinel-1 exposure for
2015–2025.

This stage DOES NOT read:
- groundwater values
- irrigation-flow values
- raw RiceFloodIT flooding-frequency values

It uses only the already-frozen Stage-49 orientation rule, which itself records
the construct direction established in Stage 47/48.

Frozen representation
---------------------
1. Source feature: construct-valid feature from Stage 49 (currently vv_p10_db).
2. Robust-standardize separately by Sentinel-1 track over the full 2015–2025
   support-year archive:
       robust_z = (x - track_median) / (1.4826 * track_MAD)
3. Apply the frozen construct-orientation multiplier.
4. For each support_id × year, aggregate available oriented track values using
   the unweighted median.
5. Primary eligibility requires >=3 of 4 available tracks.
6. Exactly-4-track rows are retained as a prespecified sensitivity subset.
7. No imputation and no performance weighting.

Inputs
------
outputs/diagnostics/design_c/c2x_phenology/
    c2x_s1_track_year_phenology_features.csv.gz

outputs/diagnostics/design_c/c2zb_representation_protocol/
    c2zb_cross_track_representation_protocol.json

Outputs
-------
outputs/diagnostics/design_c/c2zc_cross_track_exposure/
    c2zc_track_standardization_parameters.csv
    c2zc_support_year_flooding_like_exposure.csv.gz
    c2zc_support_year_coverage_qa.csv
    c2zc_cross_track_exposure_qa.json
    c2zc_cross_track_exposure_summary.txt

Run
---
python -u scripts/06_design_c/50_construct_frozen_cross_track_s1_exposure.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

PHENO = (
    D / "c2x_phenology"
    / "c2x_s1_track_year_phenology_features.csv.gz"
)
PROTOCOL = (
    D / "c2zb_representation_protocol"
    / "c2zb_cross_track_representation_protocol.json"
)

OUT = D / "c2zc_cross_track_exposure"
OUT.mkdir(parents=True, exist_ok=True)

PARAM_OUT = OUT / "c2zc_track_standardization_parameters.csv"
EXPOSURE_OUT = OUT / "c2zc_support_year_flooding_like_exposure.csv.gz"
COVERAGE_OUT = OUT / "c2zc_support_year_coverage_qa.csv"
QA_OUT = OUT / "c2zc_cross_track_exposure_qa.json"
SUMMARY_OUT = OUT / "c2zc_cross_track_exposure_summary.txt"

EXPECTED_SUPPORT_N = 4331
EXPECTED_YEARS = list(range(2015, 2026))
EXPECTED_TRACKS = {
    ("ascending", 15),
    ("ascending", 88),
    ("descending", 66),
    ("descending", 168),
}
EXPECTED_TRACK_YEAR_ROWS = EXPECTED_SUPPORT_N * len(EXPECTED_YEARS) * len(EXPECTED_TRACKS)
EXPECTED_SUPPORT_YEAR_ROWS = EXPECTED_SUPPORT_N * len(EXPECTED_YEARS)

for f in [PHENO, PROTOCOL]:
    if not f.exists():
        raise FileNotFoundError(f)

protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

if protocol.get("status") != "PROTOCOL_FROZEN_NO_HYDROLOGICAL_OUTCOME_READ":
    raise RuntimeError(
        "Stage-49 protocol is not in the expected frozen pre-outcome state: "
        f"{protocol.get('status')!r}"
    )

feature = str(protocol["source_feature"])
orientation = int(
    protocol["construct_orientation"]["orientation_multiplier"]
)
if orientation not in {-1, 1}:
    raise RuntimeError(f"Invalid orientation multiplier: {orientation}")

aggregation_method = str(
    protocol["cross_track_aggregation"]["method"]
).lower()
if aggregation_method != "unweighted median":
    raise RuntimeError(
        f"Unexpected frozen aggregation method: {aggregation_method!r}"
    )

min_tracks = int(
    protocol["cross_track_aggregation"]["primary_minimum_available_tracks"]
)
max_tracks = int(
    protocol["cross_track_aggregation"]["maximum_tracks"]
)

if min_tracks != 3 or max_tracks != 4:
    raise RuntimeError(
        f"Unexpected frozen eligibility rule: min={min_tracks}, max={max_tracks}"
    )

print("DESIGN C - STAGE 50 FROZEN CROSS-TRACK SENTINEL-1 EXPOSURE")
print("=" * 80)
print(f"Source feature: {feature}")
print(f"Orientation multiplier: {orientation:+d}")
print("Aggregation: unweighted median")
print("Primary eligibility: >=3 of 4 tracks")
print()
print("FIREWALL")
print("--------")
print("Groundwater read: False")
print("Irrigation-flow read: False")
print("Raw RiceFloodIT values read: False")
print("Hydrological association model fitted: False")
print()

x = pd.read_csv(PHENO, compression="gzip", low_memory=False)

required = {
    "support_id", "year", "orbit_state", "relative_orbit",
    "lon", "lat", feature
}
missing = sorted(required - set(x.columns))
if missing:
    raise AssertionError(f"Phenology table missing columns: {missing}")

x["support_id"] = x["support_id"].astype(str)
x["year"] = pd.to_numeric(x["year"], errors="raise").astype(int)
x["orbit_state"] = x["orbit_state"].astype(str).str.lower()
x["relative_orbit"] = pd.to_numeric(
    x["relative_orbit"], errors="raise"
).astype(int)
x[feature] = pd.to_numeric(x[feature], errors="coerce")
x["lon"] = pd.to_numeric(x["lon"], errors="coerce")
x["lat"] = pd.to_numeric(x["lat"], errors="coerce")

hard_flags = []

def hard(code, condition):
    if bool(condition):
        hard_flags.append(code)

hard(
    f"TRACK_YEAR_ROW_COUNT_MISMATCH:{len(x)}!=expected_{EXPECTED_TRACK_YEAR_ROWS}",
    len(x) != EXPECTED_TRACK_YEAR_ROWS,
)

hard(
    "DUPLICATE_SUPPORT_YEAR_TRACK_KEYS",
    x.duplicated(
        ["support_id", "year", "orbit_state", "relative_orbit"],
        keep=False,
    ).any(),
)

supports = sorted(x["support_id"].unique())
hard(
    f"SUPPORT_UNIVERSE_MISMATCH:{len(supports)}!=expected_{EXPECTED_SUPPORT_N}",
    len(supports) != EXPECTED_SUPPORT_N,
)

years = sorted(x["year"].unique().tolist())
hard(
    f"YEAR_UNIVERSE_MISMATCH:{years}",
    years != EXPECTED_YEARS,
)

tracks = set(
    x[["orbit_state", "relative_orbit"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
)
hard(
    f"TRACK_UNIVERSE_MISMATCH:{sorted(tracks)}",
    tracks != EXPECTED_TRACKS,
)

# Fixed support coordinates must remain invariant across all years/tracks.
coord_n = (
    x.groupby("support_id")[["lon", "lat"]]
    .nunique(dropna=False)
)
hard(
    "SUPPORT_COORDINATE_INSTABILITY",
    ((coord_n["lon"] != 1) | (coord_n["lat"] != 1)).any(),
)

# -------------------------------------------------------------------------
# Track-specific full-archive robust standardization
# -------------------------------------------------------------------------
param_rows = []

for (state, orbit), g in x.groupby(
    ["orbit_state", "relative_orbit"], sort=True
):
    v = pd.to_numeric(g[feature], errors="coerce")
    vf = v[np.isfinite(v)]

    if len(vf) == 0:
        hard_flags.append(f"NO_FINITE_VALUES_FOR_TRACK:{state}_{orbit}")
        center = scale = mad = np.nan
    else:
        center = float(vf.median())
        mad = float((vf - center).abs().median())
        scale = float(1.4826 * mad)

        if not np.isfinite(scale) or scale <= 0:
            hard_flags.append(
                f"INVALID_ROBUST_SCALE:{state}_{orbit}:mad={mad}:scale={scale}"
            )

    param_rows.append({
        "orbit_state": state,
        "relative_orbit": int(orbit),
        "source_feature": feature,
        "finite_values_n": int(len(vf)),
        "track_median": center,
        "track_mad": mad,
        "robust_scale_1p4826_mad": scale,
        "orientation_multiplier": orientation,
    })

params = pd.DataFrame(param_rows)
params.to_csv(PARAM_OUT, index=False)

x = x.merge(
    params[
        [
            "orbit_state",
            "relative_orbit",
            "track_median",
            "robust_scale_1p4826_mad",
        ]
    ],
    on=["orbit_state", "relative_orbit"],
    how="left",
    validate="many_to_one",
)

x["robust_z"] = (
    (x[feature] - x["track_median"])
    / x["robust_scale_1p4826_mad"]
)
x.loc[~np.isfinite(x["robust_z"]), "robust_z"] = np.nan

x["flooding_like_z"] = orientation * x["robust_z"]

# -------------------------------------------------------------------------
# Support-year composite
# -------------------------------------------------------------------------
track_labels = {
    ("ascending", 15): "asc15",
    ("ascending", 88): "asc88",
    ("descending", 66): "desc66",
    ("descending", 168): "desc168",
}
x["track_label"] = [
    track_labels[(s, int(o))]
    for s, o in zip(x["orbit_state"], x["relative_orbit"])
]

# One coordinate per support.
coords = (
    x[["support_id", "lon", "lat"]]
    .drop_duplicates()
    .sort_values("support_id")
)

wide = x.pivot(
    index=["support_id", "year"],
    columns="track_label",
    values="flooding_like_z",
).reset_index()

for col in ["asc15", "asc88", "desc66", "desc168"]:
    if col not in wide.columns:
        wide[col] = np.nan

wide = wide.merge(
    coords,
    on="support_id",
    how="left",
    validate="many_to_one",
)

track_cols = ["asc15", "asc88", "desc66", "desc168"]

wide["available_tracks_n"] = (
    wide[track_cols].notna().sum(axis=1).astype(int)
)
wide["primary_eligible_ge3_tracks"] = (
    wide["available_tracks_n"] >= min_tracks
)
wide["complete_4track_subset"] = (
    wide["available_tracks_n"] == max_tracks
)

# Median across available tracks, but set primary exposure missing when <3.
wide["flooding_like_s1_median_all_available"] = (
    wide[track_cols].median(axis=1, skipna=True)
)
wide["flooding_like_s1_primary"] = (
    wide["flooding_like_s1_median_all_available"]
    .where(wide["primary_eligible_ge3_tracks"])
)

wide["flooding_like_s1_complete4"] = (
    wide["flooding_like_s1_median_all_available"]
    .where(wide["complete_4track_subset"])
)

wide["source_feature"] = feature
wide["orientation_multiplier"] = orientation
wide["aggregation_rule"] = "unweighted_median_of_oriented_track_robust_z"

wide = wide[
    [
        "support_id", "year", "lon", "lat",
        "asc15", "asc88", "desc66", "desc168",
        "available_tracks_n",
        "primary_eligible_ge3_tracks",
        "complete_4track_subset",
        "flooding_like_s1_median_all_available",
        "flooding_like_s1_primary",
        "flooding_like_s1_complete4",
        "source_feature",
        "orientation_multiplier",
        "aggregation_rule",
    ]
].sort_values(["year", "support_id"]).reset_index(drop=True)

hard(
    f"SUPPORT_YEAR_ROW_COUNT_MISMATCH:{len(wide)}!=expected_{EXPECTED_SUPPORT_YEAR_ROWS}",
    len(wide) != EXPECTED_SUPPORT_YEAR_ROWS,
)

hard(
    "DUPLICATE_SUPPORT_YEAR_ROWS",
    wide.duplicated(["support_id", "year"], keep=False).any(),
)

per_support_years = wide.groupby("support_id")["year"].nunique()
hard(
    "SUPPORT_MISSING_ONE_OR_MORE_YEARS",
    (per_support_years != len(EXPECTED_YEARS)).any(),
)

# Exposure invariants.
hard(
    "PRIMARY_EXPOSURE_PRESENT_WITH_LT3_TRACKS",
    (
        (wide["available_tracks_n"] < 3)
        & wide["flooding_like_s1_primary"].notna()
    ).any(),
)
hard(
    "PRIMARY_EXPOSURE_MISSING_WITH_GE3_TRACKS",
    (
        (wide["available_tracks_n"] >= 3)
        & wide["flooding_like_s1_primary"].isna()
    ).any(),
)
hard(
    "COMPLETE4_EXPOSURE_PRESENT_WITH_LT4_TRACKS",
    (
        (wide["available_tracks_n"] < 4)
        & wide["flooding_like_s1_complete4"].notna()
    ).any(),
)
hard(
    "COMPLETE4_EXPOSURE_MISSING_WITH_4_TRACKS",
    (
        (wide["available_tracks_n"] == 4)
        & wide["flooding_like_s1_complete4"].isna()
    ).any(),
)

wide.to_csv(
    EXPOSURE_OUT,
    index=False,
    compression="gzip",
)

# -------------------------------------------------------------------------
# Coverage QA by year
# -------------------------------------------------------------------------
coverage = (
    wide.groupby("year", as_index=False)
    .agg(
        support_rows_n=("support_id", "size"),
        zero_track_rows_n=(
            "available_tracks_n",
            lambda s: int((s == 0).sum()),
        ),
        one_track_rows_n=(
            "available_tracks_n",
            lambda s: int((s == 1).sum()),
        ),
        two_track_rows_n=(
            "available_tracks_n",
            lambda s: int((s == 2).sum()),
        ),
        three_track_rows_n=(
            "available_tracks_n",
            lambda s: int((s == 3).sum()),
        ),
        four_track_rows_n=(
            "available_tracks_n",
            lambda s: int((s == 4).sum()),
        ),
        primary_eligible_rows_n=(
            "primary_eligible_ge3_tracks",
            lambda s: int(pd.Series(s).astype(bool).sum()),
        ),
        complete4_rows_n=(
            "complete_4track_subset",
            lambda s: int(pd.Series(s).astype(bool).sum()),
        ),
        primary_exposure_median=(
            "flooding_like_s1_primary", "median"
        ),
        primary_exposure_p10=(
            "flooding_like_s1_primary",
            lambda s: float(np.nanquantile(
                pd.to_numeric(s, errors="coerce"), 0.10
            )),
        ),
        primary_exposure_p90=(
            "flooding_like_s1_primary",
            lambda s: float(np.nanquantile(
                pd.to_numeric(s, errors="coerce"), 0.90
            )),
        ),
    )
    .sort_values("year")
)

coverage["primary_eligible_fraction"] = (
    coverage["primary_eligible_rows_n"]
    / coverage["support_rows_n"]
)
coverage["complete4_fraction"] = (
    coverage["complete4_rows_n"]
    / coverage["support_rows_n"]
)

coverage.to_csv(COVERAGE_OUT, index=False)

# -------------------------------------------------------------------------
# Freeze fingerprints
# -------------------------------------------------------------------------
pheno_sha256 = hashlib.sha256(PHENO.read_bytes()).hexdigest()
protocol_sha256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
exposure_sha256 = hashlib.sha256(EXPOSURE_OUT.read_bytes()).hexdigest()

status = (
    "PASS_FROZEN_CROSS_TRACK_EXPOSURE_CONSTRUCTED"
    if not hard_flags
    else "FAIL_REVIEW_REQUIRED"
)

qa = {
    "stage": "DESIGN_C_STAGE50_CONSTRUCT_FROZEN_CROSS_TRACK_S1_EXPOSURE",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,

    "source_feature": feature,
    "orientation_multiplier": orientation,
    "aggregation_method": "unweighted median",
    "primary_minimum_available_tracks": min_tracks,

    "input_phenology_sha256": pheno_sha256,
    "input_stage49_protocol_sha256": protocol_sha256,
    "output_exposure_sha256": exposure_sha256,

    "track_year_rows_n": int(len(x)),
    "expected_track_year_rows_n": EXPECTED_TRACK_YEAR_ROWS,
    "support_year_rows_n": int(len(wide)),
    "expected_support_year_rows_n": EXPECTED_SUPPORT_YEAR_ROWS,

    "primary_eligible_support_year_rows_n":
        int(wide["primary_eligible_ge3_tracks"].sum()),
    "complete4_support_year_rows_n":
        int(wide["complete_4track_subset"].sum()),

    "hard_flags": hard_flags,

    "raw_ricefloodit_values_read": False,
    "ricefloodit_information_used":
        "only frozen orientation multiplier from Stage-49 protocol",
    "groundwater_values_read": False,
    "irrigation_flow_values_read": False,
    "groundwater_association_models_fitted": 0,
    "irrigation_association_models_fitted": 0,
    "representation_tuned_to_hydrological_outcomes": False,
    "missing_track_values_imputed": False,
    "performance_weighted_tracks": False,
}

QA_OUT.write_text(
    json.dumps(qa, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 50 FROZEN CROSS-TRACK SENTINEL-1 EXPOSURE",
    "=" * 80,
    "",
    f"Source feature: {feature}",
    f"Orientation multiplier: {orientation:+d}",
    "Track standardization: full-archive track-specific median / 1.4826×MAD",
    "Cross-track aggregation: unweighted median",
    "",
    f"Track-year rows: {len(x)} / expected {EXPECTED_TRACK_YEAR_ROWS}",
    f"Support-year rows: {len(wide)} / expected {EXPECTED_SUPPORT_YEAR_ROWS}",
    f"Primary eligible (>=3 tracks): {int(wide['primary_eligible_ge3_tracks'].sum())}",
    f"Complete 4-track subset: {int(wide['complete_4track_subset'].sum())}",
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
    "Raw RiceFloodIT values read: False",
    "Groundwater read: False",
    "Irrigation-flow read: False",
    "Hydrological association model fitted: False",
    "Representation tuned to hydrological outcomes: False",
    "",
    f"STAGE 50 STATUS: {status}",
]

summary = "\n".join(lines) + "\n"
SUMMARY_OUT.write_text(summary, encoding="utf-8")
print(summary, end="")

if hard_flags:
    raise RuntimeError(
        "Stage 50 failed hard QA; inspect the QA and coverage outputs."
    )
