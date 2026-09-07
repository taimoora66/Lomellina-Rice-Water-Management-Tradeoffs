"""
Design C — Stage 46 freeze the RiceFloodIT construct-validation protocol.

Purpose
-------
Freeze, BEFORE reading any RiceFloodIT flooding-frequency values, the exact
external construct-validation design for the already frozen Stage-44/45
Sentinel-1 phenology representation.

This stage does NOT perform validation. It only writes the protocol.

Allowed inputs
--------------
outputs/diagnostics/design_c/c2x_phenology/
    c2x_s1_phenology_audit_qa.json
    c2x_phenology_freeze_manifest.csv
    c2x_s1_feature_dictionary.csv

Forbidden inputs
----------------
- RiceFloodIT flooding-frequency values
- groundwater observations
- irrigation-flow observations
- any downstream outcome table

Outputs
-------
outputs/diagnostics/design_c/c2y_validation_protocol/
    c2y_ricefloodit_construct_validation_protocol.json
    c2y_ricefloodit_construct_validation_protocol.txt

Run
---
python -u scripts/06_design_c/46_freeze_s1_phenology_construct_validation_protocol.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"
PH = D / "c2x_phenology"
OUT = D / "c2y_validation_protocol"
OUT.mkdir(parents=True, exist_ok=True)

AUDIT_QA = PH / "c2x_s1_phenology_audit_qa.json"
FREEZE_MANIFEST = PH / "c2x_phenology_freeze_manifest.csv"
FEATURE_DICTIONARY = PH / "c2x_s1_feature_dictionary.csv"

JSON_OUT = OUT / "c2y_ricefloodit_construct_validation_protocol.json"
TXT_OUT = OUT / "c2y_ricefloodit_construct_validation_protocol.txt"

EXPECTED_STAGE45_STATUS = "PASS_PHENOLOGY_FEATURE_FREEZE_READY"

VALIDATION_YEARS = list(range(2015, 2022))
TRACKS = [
    {"orbit_state": "ascending", "relative_orbit": 15},
    {"orbit_state": "ascending", "relative_orbit": 88},
    {"orbit_state": "descending", "relative_orbit": 66},
    {"orbit_state": "descending", "relative_orbit": 168},
]

# Prespecified primary construct-validation features.
# These were created outcome-blind in Stage 44.
PRIMARY_FEATURES = [
    "vv_p10_db",
    "vh_p10_db",
    "vv_minus_vh_p10_db",
    "early_season_vv_median_db",
    "early_season_vh_median_db",
    "early_season_vv_minus_vh_median_db",
]

# Secondary sensitivity family. These are descriptive corroboration only.
# They cannot replace the primary family after RiceFloodIT is opened.
SECONDARY_FEATURES = [
    "vv_median_db",
    "vh_median_db",
    "vv_minus_vh_median_db",
    "vv_iqr_db",
    "vh_iqr_db",
    "vv_minus_vh_iqr_db",
    "vv_range_db",
    "vh_range_db",
    "vv_minus_vh_range_db",
    "mid_season_vv_median_db",
    "mid_season_vh_median_db",
    "mid_season_vv_minus_vh_median_db",
    "late_season_vv_median_db",
    "late_season_vh_median_db",
    "late_season_vv_minus_vh_median_db",
]

# Fixed QA rules. These are not outcome-performance thresholds.
MIN_VALID_SUPPORTS_PER_CELL = 100
MIN_VALID_YEARS_PER_TRACK = 5

# Fixed construct-consistency screening.
# Direction is not assumed a priori; consistency of observed direction is tested.
MIN_SIGN_AGREEMENT_SHARE = 0.75
MIN_MEDIAN_ABS_SPEARMAN = 0.20
MIN_TRACKS_WITH_CONSISTENT_DIRECTION = 3

# Spatial block bootstrap settings.
# EPSG:32632 is appropriate for Lombardy/Lomellina.
BOOTSTRAP = {
    "enabled": True,
    "replicates": 2000,
    "seed": 46017,
    "spatial_crs": "EPSG:32632",
    "block_size_m": 10000,
    "confidence_level": 0.95,
    "resampling_unit": "fixed 10-km spatial blocks within each year-track cell",
    "p_values_computed": False,
}

for f in [AUDIT_QA, FREEZE_MANIFEST, FEATURE_DICTIONARY]:
    if not f.exists():
        raise FileNotFoundError(f)

audit = json.loads(AUDIT_QA.read_text(encoding="utf-8"))
if audit.get("status") != EXPECTED_STAGE45_STATUS:
    raise RuntimeError(
        f"Stage 45 is not freeze-ready: {audit.get('status')!r}"
    )

fd = pd.read_csv(FEATURE_DICTIONARY)
if "feature" not in fd.columns:
    raise AssertionError("Feature dictionary has no 'feature' column.")

available = set(fd["feature"].astype(str))
missing_primary = sorted(set(PRIMARY_FEATURES) - available)
missing_secondary = sorted(set(SECONDARY_FEATURES) - available)

if missing_primary or missing_secondary:
    raise AssertionError(
        "Protocol references features absent from the frozen dictionary. "
        f"Missing primary={missing_primary}; secondary={missing_secondary}"
    )

manifest_bytes = FREEZE_MANIFEST.read_bytes()
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

protocol = {
    "stage": "DESIGN_C_STAGE46_FREEZE_RICEFLOODIT_CONSTRUCT_VALIDATION_PROTOCOL",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PROTOCOL_FROZEN_NO_VALIDATION_EXECUTED",

    "frozen_upstream_state": {
        "stage45_status_required": EXPECTED_STAGE45_STATUS,
        "stage45_status_observed": audit.get("status"),
        "phenology_freeze_manifest":
            str(FREEZE_MANIFEST.relative_to(ROOT)),
        "phenology_freeze_manifest_sha256": manifest_sha256,
        "phenology_feature_dictionary":
            str(FEATURE_DICTIONARY.relative_to(ROOT)),
    },

    "validation_target": {
        "construct": "RiceFloodIT annual/seasonal flooding frequency",
        "role": (
            "external construct validation only; not feature construction, "
            "not groundwater-outcome tuning, and not irrigation-flow tuning"
        ),
        "years": VALIDATION_YEARS,
        "overlap_period_reason": (
            "Sentinel-1 frozen archive begins in 2015 and observed RiceFloodIT "
            "is used only over the predeclared overlapping observed period through 2021."
        ),
    },

    "unit_of_validation": {
        "primary_cell": "support_id x year x orbit_state x relative_orbit",
        "track_pooling_before_validation": False,
        "tracks": TRACKS,
        "minimum_valid_supports_per_year_track_feature_cell":
            MIN_VALID_SUPPORTS_PER_CELL,
    },

    "primary_features": PRIMARY_FEATURES,
    "secondary_sensitivity_features": SECONDARY_FEATURES,

    "metrics": {
        "primary": [
            "Spearman rank correlation within each year-track cell",
            "direction/sign by year-track cell",
            "median Spearman rho within track across eligible years",
            "median absolute Spearman rho within track across eligible years",
            "sign-agreement share within track across eligible years",
            "majority direction within track",
            "track-to-track direction consistency",
            "spatial-block-bootstrap confidence interval for cell-level Spearman rho",
        ],
        "not_used_for_selection": [
            "nominal p-value",
            "minimum p-value",
            "maximum correlation winner",
            "best-performing year",
            "best-performing track",
        ],
    },

    "spatial_bootstrap": BOOTSTRAP,

    "construct_consistency_screen": {
        "minimum_eligible_years_per_track": MIN_VALID_YEARS_PER_TRACK,
        "minimum_within_track_sign_agreement_share":
            MIN_SIGN_AGREEMENT_SHARE,
        "minimum_within_track_median_absolute_spearman":
            MIN_MEDIAN_ABS_SPEARMAN,
        "minimum_tracks_with_consistent_majority_direction":
            MIN_TRACKS_WITH_CONSISTENT_DIRECTION,
        "direction_predeclared": False,
        "interpretation": (
            "A feature is construct-consistent only if its association direction "
            "and magnitude are reasonably stable across years and track geometries. "
            "The sign itself is not chosen in advance because rice backscatter can "
            "depend on crop structure and acquisition geometry."
        ),
    },

    "decision_rules": {
        "primary_family_only_can_support_main_construct_freeze": True,
        "secondary_family_can_replace_failed_primary_feature": False,
        "choose_single_best_feature_by_ricefloodit_performance": False,
        "retain_all_primary_features_passing_consistency_screen": True,
        "if_no_primary_feature_passes": (
            "Do not tune a new feature or threshold. Record construct-validation "
            "failure/inconclusiveness and redesign only in a separately versioned "
            "post-validation exploratory branch."
        ),
        "if_multiple_primary_features_pass": (
            "Retain the passing set as a construct-valid measurement family; "
            "do not rank by maximum rho or minimum uncertainty."
        ),
    },

    "missingness_policy": {
        "sentinel1_missing_values": "remain missing; no imputation",
        "ricefloodit_missing_values": "complete-case within the predeclared cell",
        "minimum_cell_n": MIN_VALID_SUPPORTS_PER_CELL,
        "no_nearest_year_substitution": True,
        "no_nearest_support_substitution": True,
    },

    "firewall": {
        "groundwater_values_read": False,
        "irrigation_flow_values_read": False,
        "ricefloodit_flood_values_read": False,
        "ricefloodit_validation_executed": False,
        "inundation_threshold_selected": False,
        "classifier_fitted": False,
        "feature_selection_against_groundwater_performed": False,
        "feature_selection_against_irrigation_flow_performed": False,
        "association_models_fitted": 0,
    },
}

JSON_OUT.write_text(
    json.dumps(protocol, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 46 FROZEN RICEFLOODIT CONSTRUCT-VALIDATION PROTOCOL",
    "=" * 82,
    "",
    f"Upstream Stage-45 status: {audit.get('status')}",
    f"Freeze-manifest SHA256: {manifest_sha256}",
    "",
    "VALIDATION TARGET",
    "-----------------",
    "RiceFloodIT annual/seasonal flooding frequency.",
    "Observed overlap years: 2015-2021.",
    "Tracks remain separate: asc15, asc88, desc66, desc168.",
    "",
    "PRIMARY FEATURES",
    "----------------",
]
lines += [f"- {f}" for f in PRIMARY_FEATURES]
lines += [
    "",
    "PRIMARY METRICS",
    "---------------",
    "Within each year-track cell: Spearman rho and fixed 10-km spatial-block-bootstrap 95% CI.",
    "Across years: median rho, median |rho|, sign agreement, majority direction.",
    "Across tracks: direction consistency.",
    "Nominal p-values are not used for feature selection.",
    "",
    "CONSISTENCY SCREEN",
    "------------------",
    f"Minimum valid support points per cell: {MIN_VALID_SUPPORTS_PER_CELL}",
    f"Minimum eligible years per track: {MIN_VALID_YEARS_PER_TRACK}",
    f"Minimum within-track sign agreement: {MIN_SIGN_AGREEMENT_SHARE:.0%}",
    f"Minimum within-track median |rho|: {MIN_MEDIAN_ABS_SPEARMAN:.2f}",
    f"Minimum tracks with consistent majority direction: {MIN_TRACKS_WITH_CONSISTENT_DIRECTION}/4",
    "",
    "DECISION FIREWALL",
    "-----------------",
    "No single best feature will be chosen by maximum RiceFloodIT correlation.",
    "Secondary features cannot replace failed primary features.",
    "If no primary feature passes, no post-hoc threshold/feature tuning is allowed in this frozen confirmatory path.",
    "",
    "FIREWALL",
    "--------",
    "Groundwater read: False",
    "Irrigation-flow read: False",
    "RiceFloodIT flood values read: False",
    "RiceFloodIT validation executed: False",
    "Threshold selected: False",
    "Classifier fitted: False",
    "Association model fitted: False",
    "",
    "STAGE 46 STATUS: PROTOCOL_FROZEN_NO_VALIDATION_EXECUTED",
]

TXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
