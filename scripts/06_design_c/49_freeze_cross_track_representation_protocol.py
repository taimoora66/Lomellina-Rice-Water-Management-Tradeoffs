"""
Design C — Stage 49 freeze the downstream cross-track representation protocol.

Purpose
-------
Freeze, before any groundwater or irrigation-flow values are opened, the exact
rule that will convert the Stage-48 construct-valid track-specific Sentinel-1
feature into one support-year measurement suitable for downstream hydrological
association analysis.

This stage DOES NOT construct the composite and DOES NOT read groundwater,
irrigation flow, or any hydrological outcome.

Scientific rationale
--------------------
Stage 47/48 validated feature identity at the track level. Absolute Sentinel-1
backscatter differs by acquisition geometry, so raw dB values from different
tracks are not pooled directly.

The frozen downstream rule is:
1. use only the construct-valid feature frozen by Stage 48;
2. robust-standardize that feature separately within each track using the full
   frozen 2015–2025 support-year archive:
       z_robust = (x - median_track) / (1.4826 * MAD_track)
3. orient the sign only from the already-frozen RiceFloodIT construct direction
   so that larger values mean "more flooding-like";
4. combine available oriented track-specific standardized values by the
   unweighted median;
5. primary support-year eligibility requires at least 3 of 4 tracks;
6. exactly-4-track rows are retained as a prespecified sensitivity subset;
7. no track is weighted by RiceFloodIT correlation, bootstrap precision,
   groundwater performance, or irrigation-flow performance.

Why full-archive track standardization rather than year-wise standardization?
---------------------------------------------------------------------------
Year-wise ranking/standardization would remove common interannual shifts and
therefore weaken the temporal information needed for later hydrological
analysis. Full-archive track-specific robust standardization removes geometry-
scale differences while retaining year-to-year variation.

Inputs
------
outputs/diagnostics/design_c/c2za_construct_freeze/
    c2za_frozen_construct_valid_measurement_family.json

outputs/diagnostics/design_c/c2z_construct_validation/
    c2z_primary_feature_cross_track_summary.csv

Outputs
-------
outputs/diagnostics/design_c/c2zb_representation_protocol/
    c2zb_cross_track_representation_protocol.json
    c2zb_cross_track_representation_protocol.txt

Run
---
python -u scripts/06_design_c/49_freeze_cross_track_representation_protocol.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

FREEZE48 = (
    D / "c2za_construct_freeze"
    / "c2za_frozen_construct_valid_measurement_family.json"
)
CROSS47 = (
    D / "c2z_construct_validation"
    / "c2z_primary_feature_cross_track_summary.csv"
)

OUT = D / "c2zb_representation_protocol"
OUT.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUT / "c2zb_cross_track_representation_protocol.json"
TXT_OUT = OUT / "c2zb_cross_track_representation_protocol.txt"

for f in [FREEZE48, CROSS47]:
    if not f.exists():
        raise FileNotFoundError(f)

freeze48 = json.loads(FREEZE48.read_text(encoding="utf-8"))
cross47 = pd.read_csv(CROSS47)

if freeze48.get("status") != "PASS_CONSTRUCT_VALID_MEASUREMENT_FAMILY_FROZEN":
    raise RuntimeError(
        f"Stage 48 is not frozen/pass-ready: {freeze48.get('status')!r}"
    )

features = list(freeze48.get("construct_valid_primary_features", []))
if len(features) != 1:
    raise RuntimeError(
        "Stage 49 currently requires exactly one frozen construct-valid "
        f"primary feature; observed {features!r}"
    )

feature = features[0]

row = cross47.loc[cross47["feature"].astype(str).eq(feature)].copy()
if len(row) != 1:
    raise RuntimeError(
        f"Expected exactly one cross-track validation row for {feature}; "
        f"found {len(row)}"
    )

r = row.iloc[0]
pass_flag = str(r["construct_consistency_pass"]).strip().lower() in {
    "true", "1", "yes"
}
if not pass_flag:
    raise RuntimeError(
        f"Frozen feature {feature} is not marked construct-consistent."
    )

direction = str(r["cross_track_majority_direction"]).strip().lower()
if direction not in {"positive", "negative"}:
    raise RuntimeError(
        f"Unexpected construct direction for {feature}: {direction!r}"
    )

orientation_multiplier = 1 if direction == "positive" else -1

protocol = {
    "stage": "DESIGN_C_STAGE49_FREEZE_CROSS_TRACK_REPRESENTATION_PROTOCOL",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PROTOCOL_FROZEN_NO_HYDROLOGICAL_OUTCOME_READ",

    "source_feature": feature,
    "source_feature_status":
        "Stage-48 construct-valid primary feature",
    "frozen_ricefloodit_construct_direction": direction,

    "track_standardization": {
        "performed_separately_by":
            ["orbit_state", "relative_orbit"],
        "reference_archive_years":
            list(range(2015, 2026)),
        "center": "track-specific median across all finite support-year values",
        "scale": "1.4826 * track-specific MAD across all finite support-year values",
        "formula":
            "(x - track_median) / (1.4826 * track_MAD)",
        "year_specific_standardization": False,
        "reason": (
            "Full-archive track-specific robust standardization removes "
            "geometry-scale offsets while retaining common interannual shifts."
        ),
    },

    "construct_orientation": {
        "orientation_multiplier": orientation_multiplier,
        "formula": "flooding_like_z = orientation_multiplier * robust_z",
        "interpretation":
            "larger composite values mean more RiceFloodIT-consistent flooding-like signal",
        "orientation_uses_validation_direction_only": True,
        "orientation_does_not_use_validation_magnitude": True,
    },

    "cross_track_aggregation": {
        "method": "unweighted median",
        "weights": "none",
        "primary_minimum_available_tracks": 3,
        "maximum_tracks": 4,
        "complete_four_track_subset_retained": True,
        "no_best_track_selection": True,
        "no_ricefloodit_performance_weighting": True,
        "no_bootstrap_precision_weighting": True,
        "no_groundwater_performance_weighting": True,
        "no_irrigation_flow_performance_weighting": True,
    },

    "missingness_policy": {
        "track_specific_missing_values": "remain missing",
        "imputation": False,
        "nearest_track_substitution": False,
        "primary_support_year_requires_at_least_3_tracks": True,
        "rows_with_fewer_than_3_tracks":
            "retained in QA but excluded from primary composite eligibility",
    },

    "downstream_role": (
        "predeclared Sentinel-1 flooding-like exposure for later hydrological "
        "association analysis; not a causal treatment and not a direct measure "
        "of recharge"
    ),

    "firewall": {
        "groundwater_values_read": False,
        "irrigation_flow_values_read": False,
        "groundwater_association_model_fitted": False,
        "irrigation_association_model_fitted": False,
        "representation_tuned_to_groundwater": False,
        "representation_tuned_to_irrigation_flow": False,
    },
}

JSON_OUT.write_text(
    json.dumps(protocol, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 49 FROZEN CROSS-TRACK REPRESENTATION PROTOCOL",
    "=" * 78,
    "",
    f"Construct-valid source feature: {feature}",
    f"Frozen RiceFloodIT construct direction: {direction}",
    f"Orientation multiplier: {orientation_multiplier:+d}",
    "",
    "TRACK STANDARDIZATION",
    "---------------------",
    "Robust-standardize separately within each track over the full 2015-2025 archive.",
    "Center = track median.",
    "Scale = 1.4826 × track MAD.",
    "Do NOT standardize separately by year.",
    "",
    "CROSS-TRACK AGGREGATION",
    "-----------------------",
    "Unweighted median of oriented standardized track values.",
    "Primary eligibility: at least 3 of 4 tracks.",
    "Exactly-4-track rows retained as a prespecified sensitivity subset.",
    "No track-performance weighting.",
    "",
    "WHY NOT YEAR-WISE STANDARDIZATION?",
    "----------------------------------",
    "Year-wise standardization would remove common interannual shifts.",
    "The full-archive track-specific rule preserves temporal information.",
    "",
    "FIREWALL",
    "--------",
    "Groundwater read: False",
    "Irrigation-flow read: False",
    "Groundwater association model fitted: False",
    "Irrigation association model fitted: False",
    "Representation tuned to hydrological outcomes: False",
    "",
    "STAGE 49 STATUS: PROTOCOL_FROZEN_NO_HYDROLOGICAL_OUTCOME_READ",
]

TXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
