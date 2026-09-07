"""
Design C — Stage 47 execute the frozen RiceFloodIT construct validation.

This script implements the Stage-46 protocol against the already frozen
Stage-44/45 Sentinel-1 phenology representation.

Important firewall
------------------
RiceFloodIT flooding-frequency values are now permitted for EXTERNAL CONSTRUCT
VALIDATION ONLY.

Still forbidden:
- groundwater values
- irrigation-flow values
- threshold tuning
- classifier fitting
- feature redesign based on RiceFloodIT performance
- groundwater/irrigation association modelling

Observed RiceFloodIT source
---------------------------
data/processed/publication_groundwater/ricefloodit_georef.csv

Required columns:
year, ff, lon, lat

The raw observed source is data/raw/RiceFloodIT/ffavg_2021.csv; the processed
georeference adds lon/lat without changing the observed FF construct.

Frozen S1 input
---------------
outputs/diagnostics/design_c/c2x_phenology/
    c2x_s1_track_year_phenology_features.csv.gz

Frozen protocol
---------------
outputs/diagnostics/design_c/c2y_validation_protocol/
    c2y_ricefloodit_construct_validation_protocol.json

Outputs
-------
outputs/diagnostics/design_c/c2z_construct_validation/
    c2z_ricefloodit_support_linkage_qa.csv
    c2z_primary_feature_year_track_correlations.csv
    c2z_primary_feature_track_summary.csv
    c2z_primary_feature_cross_track_summary.csv
    c2z_primary_feature_bootstrap_ci.csv
    c2z_secondary_feature_sensitivity.csv
    c2z_construct_validation_qa.json
    c2z_construct_validation_summary.txt

Run
---
python -u scripts/06_design_c/47_execute_ricefloodit_construct_validation.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

PHENO = D / "c2x_phenology" / "c2x_s1_track_year_phenology_features.csv.gz"
PROTOCOL = D / "c2y_validation_protocol" / "c2y_ricefloodit_construct_validation_protocol.json"
RFIT = ROOT / "data" / "processed" / "publication_groundwater" / "ricefloodit_georef.csv"

OUT = D / "c2z_construct_validation"
OUT.mkdir(parents=True, exist_ok=True)

LINK_QA_OUT = OUT / "c2z_ricefloodit_support_linkage_qa.csv"
CELL_OUT = OUT / "c2z_primary_feature_year_track_correlations.csv"
TRACK_OUT = OUT / "c2z_primary_feature_track_summary.csv"
CROSS_OUT = OUT / "c2z_primary_feature_cross_track_summary.csv"
BOOT_OUT = OUT / "c2z_primary_feature_bootstrap_ci.csv"
SECONDARY_OUT = OUT / "c2z_secondary_feature_sensitivity.csv"
QA_OUT = OUT / "c2z_construct_validation_qa.json"
SUMMARY_OUT = OUT / "c2z_construct_validation_summary.txt"

for f in [PHENO, PROTOCOL, RFIT]:
    if not f.exists():
        raise FileNotFoundError(f)

protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

if protocol.get("status") != "PROTOCOL_FROZEN_NO_VALIDATION_EXECUTED":
    raise RuntimeError(
        "Stage-46 protocol is not in the expected frozen pre-validation state."
    )

years = [int(y) for y in protocol["validation_target"]["years"]]
primary_features = list(protocol["primary_features"])
secondary_features = list(protocol["secondary_sensitivity_features"])

min_cell_n = int(
    protocol["unit_of_validation"]["minimum_valid_supports_per_year_track_feature_cell"]
)
screen = protocol["construct_consistency_screen"]
min_years = int(screen["minimum_eligible_years_per_track"])
min_sign_agreement = float(screen["minimum_within_track_sign_agreement_share"])
min_abs_rho = float(screen["minimum_within_track_median_absolute_spearman"])
min_tracks = int(screen["minimum_tracks_with_consistent_majority_direction"])

boot = protocol["spatial_bootstrap"]
B = int(boot["replicates"])
SEED = int(boot["seed"])
BLOCK_M = float(boot["block_size_m"])
CONF = float(boot["confidence_level"])
alpha = 1.0 - CONF

print("DESIGN C - STAGE 47 FROZEN RICEFLOODIT CONSTRUCT VALIDATION")
print("=" * 78)
print("RiceFloodIT flood values read: True")
print("Groundwater read: False")
print("Irrigation-flow read: False")
print("Threshold selected: False")
print("Classifier fitted: False")
print("Groundwater/irrigation association model: False")
print()

s1 = pd.read_csv(PHENO, compression="gzip", low_memory=False)
rf = pd.read_csv(RFIT, low_memory=False)

req_s1 = {
    "support_id", "year", "orbit_state", "relative_orbit", "lon", "lat",
    *primary_features, *secondary_features,
}
missing = sorted(req_s1 - set(s1.columns))
if missing:
    raise AssertionError(f"Frozen S1 table missing columns: {missing}")

req_rf = {"year", "ff", "lon", "lat"}
missing = sorted(req_rf - set(rf.columns))
if missing:
    raise AssertionError(f"RiceFloodIT georef missing columns: {missing}")

s1["year"] = pd.to_numeric(s1["year"], errors="coerce").astype("Int64")
rf["year"] = pd.to_numeric(rf["year"], errors="coerce").astype("Int64")
rf["ff"] = pd.to_numeric(rf["ff"], errors="coerce")

s1 = s1[s1["year"].isin(years)].copy()
rf = rf[rf["year"].isin(years)].copy()

# -------------------------------------------------------------------------
# Exact coordinate linkage audit
# -------------------------------------------------------------------------
# The frozen S1 support coordinates were constructed from the RiceFloodIT-
# compatible support grid. We therefore require coordinate identity after a
# conservative 6-decimal normalization (~0.1 m latitude scale), not nearest-
# neighbor substitution.
for df in [s1, rf]:
    df["lon6"] = pd.to_numeric(df["lon"], errors="coerce").round(6)
    df["lat6"] = pd.to_numeric(df["lat"], errors="coerce").round(6)

# Guard against ambiguous RiceFloodIT coordinates within a year.
rf_key_counts = (
    rf.groupby(["year", "lon6", "lat6"], dropna=False)
      .size()
      .reset_index(name="n")
)
ambig = rf_key_counts[rf_key_counts["n"] > 1]
if len(ambig):
    raise RuntimeError(
        f"Ambiguous RiceFloodIT year-coordinate keys after 6-decimal "
        f"normalization: {len(ambig)}"
    )

rf_small = rf[["year", "lon6", "lat6", "ff"]].copy()

joined = s1.merge(
    rf_small,
    on=["year", "lon6", "lat6"],
    how="left",
    validate="many_to_one",
)

link_qa = (
    joined.groupby(["year", "orbit_state", "relative_orbit"], as_index=False)
    .agg(
        s1_rows_n=("support_id", "size"),
        unique_supports_n=("support_id", "nunique"),
        ff_linked_n=("ff", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
    )
)
link_qa["ff_link_fraction"] = (
    link_qa["ff_linked_n"] / link_qa["s1_rows_n"]
)
link_qa.to_csv(LINK_QA_OUT, index=False)

print("LINKAGE")
print("-------")
print(link_qa.to_string(index=False))
print()

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

def eligible_pair(g: pd.DataFrame, feature: str):
    z = g[["ff", feature, "lon", "lat"]].copy()
    z["ff"] = pd.to_numeric(z["ff"], errors="coerce")
    z[feature] = pd.to_numeric(z[feature], errors="coerce")
    z = z[np.isfinite(z["ff"]) & np.isfinite(z[feature])].copy()
    return z

def rho_of(z: pd.DataFrame, feature: str):
    if len(z) < 2:
        return np.nan
    if z["ff"].nunique() < 2 or z[feature].nunique() < 2:
        return np.nan
    r = spearmanr(z["ff"], z[feature], nan_policy="omit").statistic
    return float(r) if np.isfinite(r) else np.nan

def block_ids(z: pd.DataFrame):
    e, n = transformer.transform(
        z["lon"].to_numpy(float),
        z["lat"].to_numpy(float),
    )
    bx = np.floor(np.asarray(e) / BLOCK_M).astype(np.int64)
    by = np.floor(np.asarray(n) / BLOCK_M).astype(np.int64)
    return pd.Series(
        [f"{a}:{b}" for a, b in zip(bx, by)],
        index=z.index,
        dtype="string",
    )

def spatial_block_bootstrap(z: pd.DataFrame, feature: str, seed: int):
    if len(z) < min_cell_n:
        return np.nan, np.nan, np.nan, 0, 0

    z = z.copy()
    z["block"] = block_ids(z)
    blocks = sorted(z["block"].dropna().unique().tolist())
    if len(blocks) < 2:
        return np.nan, np.nan, np.nan, 0, len(blocks)

    by_block = {b: z[z["block"].eq(b)] for b in blocks}
    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(B):
        draw = rng.choice(blocks, size=len(blocks), replace=True)
        pieces = []
        for j, b in enumerate(draw):
            part = by_block[b].copy()
            # Distinguish repeated bootstrap copies without changing values.
            part["_bootcopy"] = j
            pieces.append(part)
        zz = pd.concat(pieces, ignore_index=True)
        r = rho_of(zz, feature)
        if np.isfinite(r):
            vals.append(r)

    if not vals:
        return np.nan, np.nan, np.nan, 0, len(blocks)

    a = np.asarray(vals, dtype=float)
    return (
        float(np.quantile(a, alpha / 2)),
        float(np.quantile(a, 1 - alpha / 2)),
        float(np.median(a)),
        int(len(a)),
        int(len(blocks)),
    )

# -------------------------------------------------------------------------
# Primary year-track cell validation
# -------------------------------------------------------------------------
cell_rows = []
boot_rows = []

track_groups = joined.groupby(
    ["year", "orbit_state", "relative_orbit"],
    sort=True,
)

cell_index = 0
for (year, state, orbit), g in track_groups:
    for feature in primary_features:
        z = eligible_pair(g, feature)
        n = len(z)
        rho = rho_of(z, feature) if n >= min_cell_n else np.nan
        eligible = bool(n >= min_cell_n and np.isfinite(rho))

        cell_rows.append({
            "year": int(year),
            "orbit_state": state,
            "relative_orbit": int(orbit),
            "feature": feature,
            "paired_n": int(n),
            "eligible": eligible,
            "spearman_rho": rho,
            "rho_sign": (
                "positive" if eligible and rho > 0 else
                "negative" if eligible and rho < 0 else
                "zero_or_missing"
            ),
        })

        if eligible:
            cell_index += 1
            lo, hi, med, nrep, nblocks = spatial_block_bootstrap(
                z, feature, SEED + cell_index
            )
        else:
            lo = hi = med = np.nan
            nrep = nblocks = 0

        boot_rows.append({
            "year": int(year),
            "orbit_state": state,
            "relative_orbit": int(orbit),
            "feature": feature,
            "paired_n": int(n),
            "spearman_rho": rho,
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
            "bootstrap_median_rho": med,
            "bootstrap_valid_replicates_n": int(nrep),
            "spatial_blocks_n": int(nblocks),
            "block_size_m": BLOCK_M,
            "confidence_level": CONF,
        })

cell = pd.DataFrame(cell_rows)
bootdf = pd.DataFrame(boot_rows)
cell.to_csv(CELL_OUT, index=False)
bootdf.to_csv(BOOT_OUT, index=False)

# -------------------------------------------------------------------------
# Within-track consistency summary
# -------------------------------------------------------------------------
track_rows = []
for (state, orbit, feature), g in cell.groupby(
    ["orbit_state", "relative_orbit", "feature"],
    sort=True,
):
    e = g[g["eligible"]].copy()
    r = pd.to_numeric(e["spearman_rho"], errors="coerce").dropna()
    pos = int((r > 0).sum())
    neg = int((r < 0).sum())
    nonzero = pos + neg
    majority = max(pos, neg)
    sign_agree = majority / nonzero if nonzero else np.nan
    majority_sign = (
        "positive" if pos > neg else
        "negative" if neg > pos else
        "tie"
    )
    median_rho = float(r.median()) if len(r) else np.nan
    median_abs = float(r.abs().median()) if len(r) else np.nan

    passed = bool(
        len(r) >= min_years
        and np.isfinite(sign_agree)
        and sign_agree >= min_sign_agreement
        and np.isfinite(median_abs)
        and median_abs >= min_abs_rho
        and majority_sign in {"positive", "negative"}
    )

    track_rows.append({
        "orbit_state": state,
        "relative_orbit": int(orbit),
        "feature": feature,
        "eligible_years_n": int(len(r)),
        "positive_years_n": pos,
        "negative_years_n": neg,
        "majority_sign": majority_sign,
        "sign_agreement_share": sign_agree,
        "median_rho": median_rho,
        "median_absolute_rho": median_abs,
        "within_track_consistency_pass": passed,
    })

track = pd.DataFrame(track_rows)
track.to_csv(TRACK_OUT, index=False)

# -------------------------------------------------------------------------
# Cross-track frozen decision
# -------------------------------------------------------------------------
cross_rows = []
for feature, g in track.groupby("feature", sort=True):
    p = g[g["within_track_consistency_pass"]].copy()
    pos = int((p["majority_sign"] == "positive").sum())
    neg = int((p["majority_sign"] == "negative").sum())
    passing_n = len(p)

    direction = (
        "positive" if pos > neg else
        "negative" if neg > pos else
        "tie_or_none"
    )
    consistent_direction_tracks = max(pos, neg)

    overall_pass = bool(
        passing_n >= min_tracks
        and consistent_direction_tracks >= min_tracks
        and direction in {"positive", "negative"}
    )

    cross_rows.append({
        "feature": feature,
        "tracks_passing_within_track_screen_n": int(passing_n),
        "positive_passing_tracks_n": pos,
        "negative_passing_tracks_n": neg,
        "cross_track_majority_direction": direction,
        "consistent_direction_tracks_n": int(consistent_direction_tracks),
        "construct_consistency_pass": overall_pass,
    })

cross = pd.DataFrame(cross_rows)
cross.to_csv(CROSS_OUT, index=False)

# -------------------------------------------------------------------------
# Secondary sensitivity: descriptive only, no pass/fail replacement
# -------------------------------------------------------------------------
secondary_rows = []
for (year, state, orbit), g in track_groups:
    for feature in secondary_features:
        z = eligible_pair(g, feature)
        n = len(z)
        rho = rho_of(z, feature) if n >= min_cell_n else np.nan
        secondary_rows.append({
            "year": int(year),
            "orbit_state": state,
            "relative_orbit": int(orbit),
            "feature": feature,
            "paired_n": int(n),
            "spearman_rho": rho,
            "role": "secondary_descriptive_sensitivity_only",
        })

secondary = pd.DataFrame(secondary_rows)
secondary.to_csv(SECONDARY_OUT, index=False)

passing_features = cross.loc[
    cross["construct_consistency_pass"], "feature"
].tolist()

# Linkage QA is a hard structural gate only if a year-track has zero linked FF.
hard_flags = []
zero_link = link_qa[link_qa["ff_linked_n"] == 0]
if len(zero_link):
    hard_flags.append(
        f"ZERO_RICEFLOODIT_LINKAGE_YEAR_TRACK_CELLS:{len(zero_link)}"
    )

expected_cells = len(years) * 4 * len(primary_features)
if len(cell) != expected_cells:
    hard_flags.append(
        f"PRIMARY_CELL_COUNT_MISMATCH:{len(cell)}!=expected_{expected_cells}"
    )

status = (
    "PASS_VALIDATION_EXECUTED"
    if not hard_flags
    else "FAIL_STRUCTURAL_REVIEW_REQUIRED"
)

qa = {
    "stage": "DESIGN_C_STAGE47_EXECUTE_FROZEN_RICEFLOODIT_CONSTRUCT_VALIDATION",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "validation_years": years,
    "primary_features": primary_features,
    "primary_year_track_feature_cells_n": int(len(cell)),
    "expected_primary_cells_n": int(expected_cells),
    "primary_features_passing_frozen_construct_consistency_screen":
        passing_features,
    "hard_flags": hard_flags,

    "decision_interpretation": (
        "Passing features are retained as a construct-valid family. "
        "No single best feature is selected by maximum RiceFloodIT correlation. "
        "If none pass, this frozen confirmatory path records failure/inconclusiveness "
        "rather than tuning a new feature."
    ),

    "ricefloodit_flood_values_read": True,
    "ricefloodit_role": "external_construct_validation_only",
    "groundwater_values_read": False,
    "irrigation_flow_values_read": False,
    "inundation_threshold_selected": False,
    "classifier_fitted": False,
    "feature_redesign_after_ricefloodit_performance": False,
    "groundwater_association_models_fitted": 0,
    "irrigation_association_models_fitted": 0,
    "nominal_p_values_used_for_selection": False,
}

QA_OUT.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

lines = [
    "DESIGN C - STAGE 47 FROZEN RICEFLOODIT CONSTRUCT VALIDATION",
    "=" * 78,
    "",
    f"Observed validation years: {years[0]}-{years[-1]}",
    f"Primary features: {len(primary_features)}",
    f"Primary year-track-feature cells: {len(cell)} / expected {expected_cells}",
    "",
    "RICEFLOODIT LINKAGE",
    "-------------------",
    f"Minimum year-track FF link fraction: {link_qa['ff_link_fraction'].min():.4f}",
    f"Median year-track FF link fraction: {link_qa['ff_link_fraction'].median():.4f}",
    f"Maximum year-track FF link fraction: {link_qa['ff_link_fraction'].max():.4f}",
    "",
    "FROZEN CONSTRUCT-CONSISTENCY RESULT",
    "-----------------------------------",
    f"Primary features passing: {passing_features if passing_features else 'NONE'}",
    "",
    "HARD QA FLAGS",
    "-------------",
]
lines += hard_flags if hard_flags else ["None"]
lines += [
    "",
    "FIREWALL",
    "--------",
    "RiceFloodIT flood values read: True",
    "RiceFloodIT role: external construct validation only",
    "Groundwater read: False",
    "Irrigation-flow read: False",
    "Threshold selected: False",
    "Classifier fitted: False",
    "Feature redesign from RiceFloodIT performance: False",
    "Groundwater association model fitted: False",
    "Irrigation association model fitted: False",
    "",
    f"STAGE 47 STATUS: {status}",
]

summary = "\n".join(lines) + "\n"
SUMMARY_OUT.write_text(summary, encoding="utf-8")
print(summary, end="")

if hard_flags:
    raise RuntimeError(
        "Stage 47 failed structural QA. Inspect linkage and QA outputs."
    )
