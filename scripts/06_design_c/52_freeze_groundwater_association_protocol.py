"""
Design C — Stage 52 freeze the groundwater association protocol BEFORE
opening groundwater values.

Purpose
-------
Predeclare the exact primary hydrological association design that will use the
Stage-51 frozen Sentinel-1 flooding-like exposure. This stage writes protocol
files only. It does not read groundwater values, irrigation-flow values, or fit
any hydrological model.

Primary scientific question
---------------------------
Among shallow ISS groundwater wells with temporally eligible observations, is
within-well interannual variation in the frozen Sentinel-1 flooding-like
exposure associated with late-season groundwater depth after accounting for
antecedent groundwater state, persistent well differences, and calendar-year
conditions common to all wells?

Frozen Sentinel-1 exposure
--------------------------
Only:
    flooding_like_s1_primary

from the Stage-51 frozen support-year exposure is permitted in the primary
confirmatory path.

Spatial linkage
---------------
Primary exposure for each well-year:
    unweighted median of flooding_like_s1_primary support points whose fixed
    coordinates fall within 10 km of the well.

Prespecified spatial sensitivities:
    5 km and 20 km.

The 10-km primary scale is inherited from the already established
RiceFloodIT-compatible groundwater exposure architecture; it is not selected
from the Design-C groundwater outcome.

Groundwater temporal definitions
--------------------------------
These are rules, not values.

Outcome:
    August groundwater depth for each well-year, defined as the median of all
    valid groundwater-depth observations dated 1–31 August.

Antecedent state:
    the last valid groundwater-depth observation dated 1 January–28/29 February
    of the same calendar year.

Eligible well-year:
    requires both an August outcome and an antecedent Jan–Feb value.

Analysis years:
    mechanically restricted to the intersection of:
      - frozen Sentinel-1 years 2015–2025
      - groundwater years satisfying the above temporal eligibility rule
    No year may be removed because of the observed association result.

Population
----------
Primary monitoring class:
    ISS shallow-groundwater wells only.

Outcome-blind station exclusions already established before this protocol:
    PO0180930U0006  (only 2013-08-20)
    PO0181870U0001  (only 2010–2013)

No further station exclusion may be based on the direction, magnitude,
significance, residual, or leverage of the Sentinel-1 association.

Primary model
-------------
gw_aug_m ~ s1_flooding_like_10km
           + gw_pre_last_janfeb_m
           + C(station)
           + C(year)

Interpretation:
- groundwater depth increases downward;
- beta < 0 means a more flooding-like S1 signal is associated with shallower
  late-season groundwater;
- beta > 0 means a more flooding-like S1 signal is associated with deeper
  late-season groundwater.

This is observational association, not a causal flooding/recharge effect.

Inference
---------
Primary coefficient:
    s1_flooding_like_10km

Primary uncertainty:
    cluster-robust covariance clustered by station/well.

Prespecified robustness inference:
    wild cluster bootstrap-t by station with Webb weights, 9,999 replications,
    fixed seed 52026.

No p-value, confidence interval, or model fit statistic may be used to redefine
the exposure, station population, temporal windows, or primary model.

Prespecified sensitivity analyses
---------------------------------
1. Same model using 5-km S1 exposure.
2. Same model using 20-km S1 exposure.
3. Same 10-km model using the Stage-51 complete-4-track S1 sensitivity exposure
   aggregated within 10 km.
4. Unadjusted fixed-effects model without antecedent groundwater:
       gw_aug_m ~ s1_flooding_like_10km + C(station) + C(year)
5. Leave-one-well-out coefficient diagnostic; diagnostic only, never a basis
   for excluding a well from the primary result.

No groundwater-driven feature selection is allowed.

Outputs
-------
outputs/diagnostics/design_c/c2ze_groundwater_protocol/
    c2ze_groundwater_association_protocol.json
    c2ze_groundwater_association_protocol.txt

Run
---
python -u scripts/06_design_c/52_freeze_groundwater_association_protocol.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

FREEZE51 = (
    D / "c2zd_exposure_freeze"
    / "c2zd_frozen_s1_exposure_manifest.json"
)

OUT = D / "c2ze_groundwater_protocol"
OUT.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUT / "c2ze_groundwater_association_protocol.json"
TXT_OUT = OUT / "c2ze_groundwater_association_protocol.txt"

if not FREEZE51.exists():
    raise FileNotFoundError(FREEZE51)

freeze51 = json.loads(FREEZE51.read_text(encoding="utf-8"))

if freeze51.get("status") != "PASS_FINAL_S1_EXPOSURE_FROZEN_FOR_HYDROLOGICAL_ANALYSIS":
    raise RuntimeError(
        "Stage 51 is not in the expected frozen/pass state: "
        f"{freeze51.get('status')!r}"
    )

frozen_measurement = freeze51["frozen_measurement"]
primary_exposure_col = frozen_measurement["support_year_exposure_column"]
complete4_col = frozen_measurement["complete4_sensitivity_column"]

if primary_exposure_col != "flooding_like_s1_primary":
    raise RuntimeError(
        f"Unexpected Stage-51 primary exposure column: {primary_exposure_col!r}"
    )

freeze51_sha256 = hashlib.sha256(FREEZE51.read_bytes()).hexdigest()

protocol = {
    "stage": "DESIGN_C_STAGE52_FREEZE_GROUNDWATER_ASSOCIATION_PROTOCOL",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PROTOCOL_FROZEN_GROUNDWATER_VALUES_UNREAD",

    "upstream_s1_freeze": {
        "stage51_manifest":
            str(FREEZE51.relative_to(ROOT)),
        "stage51_manifest_sha256": freeze51_sha256,
        "primary_support_year_exposure_column": primary_exposure_col,
        "complete4_sensitivity_column": complete4_col,
    },

    "scientific_question": (
        "Among shallow ISS groundwater wells with temporally eligible "
        "observations, is within-well interannual variation in the frozen "
        "Sentinel-1 flooding-like exposure associated with late-season "
        "groundwater depth after accounting for antecedent groundwater state, "
        "persistent well differences, and calendar-year conditions common to "
        "all wells?"
    ),

    "population": {
        "monitoring_class": "ISS",
        "interpretation": "shallow groundwater monitoring wells",
        "outcome_blind_preexisting_exclusions": [
            {
                "station": "PO0180930U0006",
                "reason": "only observation dated 2013-08-20; outside S1 era"
            },
            {
                "station": "PO0181870U0001",
                "reason": "observations only 2010-2013; outside S1 era"
            },
        ],
        "additional_outcome_based_station_exclusion_allowed": False,
    },

    "analysis_year_rule": {
        "sentinel1_available_years": list(range(2015, 2026)),
        "groundwater_years": (
            "mechanical intersection with well-years satisfying the frozen "
            "August-outcome and Jan-Feb-antecedent temporal rules"
        ),
        "association_result_based_year_exclusion_allowed": False,
    },

    "groundwater_temporal_definition": {
        "outcome_variable_name": "gw_aug_m",
        "outcome_window": "August 1 through August 31",
        "outcome_aggregation": "median of all valid groundwater-depth observations",
        "antecedent_variable_name": "gw_pre_last_janfeb_m",
        "antecedent_window": "January 1 through February 28/29",
        "antecedent_aggregation": "last valid observation in chronological order",
        "eligible_well_year_requires": [
            "at least one valid August groundwater-depth observation",
            "at least one valid January-February groundwater-depth observation",
            "valid frozen Sentinel-1 spatial exposure at the required scale",
        ],
        "value_imputation": False,
    },

    "s1_spatial_linkage": {
        "distance_crs": "EPSG:32632",
        "primary_radius_m": 10000,
        "primary_aggregation": "unweighted median",
        "primary_column_name": "s1_flooding_like_10km",
        "support_exposure_source": primary_exposure_col,
        "prespecified_sensitivity_radii_m": [5000, 20000],
        "sensitivity_column_names": [
            "s1_flooding_like_5km",
            "s1_flooding_like_20km",
        ],
        "complete4_sensitivity": {
            "radius_m": 10000,
            "support_exposure_source": complete4_col,
            "aggregation": "unweighted median",
            "column_name": "s1_flooding_like_complete4_10km",
        },
        "distance_weighting": False,
        "groundwater_performance_weighting": False,
        "minimum_support_points": 1,
        "note": (
            "Primary 10-km radius is inherited from the established "
            "RiceFloodIT-compatible groundwater exposure architecture and is "
            "not selected from Design-C groundwater outcomes."
        ),
    },

    "primary_model": {
        "formula": (
            "gw_aug_m ~ s1_flooding_like_10km "
            "+ gw_pre_last_janfeb_m + C(station) + C(year)"
        ),
        "primary_coefficient": "s1_flooding_like_10km",
        "station_fixed_effects": True,
        "year_fixed_effects": True,
        "antecedent_groundwater_adjustment": True,
        "causal_interpretation_allowed": False,
    },

    "coefficient_interpretation": {
        "groundwater_depth_direction":
            "larger groundwater-depth values mean deeper groundwater",
        "s1_exposure_direction":
            "larger values mean more RiceFloodIT-consistent flooding-like signal",
        "negative_beta":
            "more flooding-like S1 signal associated with shallower late-season groundwater",
        "positive_beta":
            "more flooding-like S1 signal associated with deeper late-season groundwater",
        "zero_beta":
            "no estimated linear partial association on the frozen exposure scale",
    },

    "inference": {
        "primary_covariance":
            "cluster-robust covariance clustered by station/well",
        "wild_cluster_bootstrap_t": {
            "enabled": True,
            "cluster": "station",
            "weights": "Webb",
            "replications": 9999,
            "seed": 52026,
            "role": "prespecified robustness inference",
        },
        "p_value_driven_redesign_allowed": False,
    },

    "prespecified_sensitivity_analyses": [
        {
            "name": "5km_spatial_scale",
            "formula": (
                "gw_aug_m ~ s1_flooding_like_5km "
                "+ gw_pre_last_janfeb_m + C(station) + C(year)"
            ),
        },
        {
            "name": "20km_spatial_scale",
            "formula": (
                "gw_aug_m ~ s1_flooding_like_20km "
                "+ gw_pre_last_janfeb_m + C(station) + C(year)"
            ),
        },
        {
            "name": "complete4_track_support",
            "formula": (
                "gw_aug_m ~ s1_flooding_like_complete4_10km "
                "+ gw_pre_last_janfeb_m + C(station) + C(year)"
            ),
        },
        {
            "name": "without_antecedent_groundwater",
            "formula": (
                "gw_aug_m ~ s1_flooding_like_10km "
                "+ C(station) + C(year)"
            ),
        },
        {
            "name": "leave_one_well_out",
            "role": (
                "coefficient influence diagnostic only; never a basis for "
                "excluding a well from the primary analysis"
            ),
        },
    ],

    "reporting_requirements": [
        "report exact N well-years and number of wells",
        "report years represented",
        "report primary beta with sign interpretation",
        "report cluster-robust uncertainty",
        "report wild-cluster-bootstrap result",
        "report within-well identifying variation of S1 exposure",
        "report sensitivity-scale coefficients",
        "report complete-4-track sensitivity",
        "report leave-one-well-out coefficient range",
        "state explicitly that the analysis is observational and non-causal",
    ],

    "firewall": {
        "groundwater_values_read": False,
        "irrigation_flow_values_read": False,
        "groundwater_model_fitted": False,
        "irrigation_model_fitted": False,
        "sentinel1_feature_reselection_allowed": False,
        "sentinel1_orientation_reselection_allowed": False,
        "sentinel1_track_weight_reselection_allowed": False,
        "spatial_radius_reselection_from_groundwater_allowed": False,
        "temporal_window_reselection_from_groundwater_allowed": False,
        "station_exclusion_from_association_result_allowed": False,
    },
}

JSON_OUT.write_text(
    json.dumps(protocol, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 52 FROZEN GROUNDWATER ASSOCIATION PROTOCOL",
    "=" * 80,
    "",
    "PRIMARY QUESTION",
    "----------------",
    protocol["scientific_question"],
    "",
    "POPULATION",
    "----------",
    "ISS shallow-groundwater wells only.",
    "Pre-existing S1-era exclusions:",
    "  PO0180930U0006",
    "  PO0181870U0001",
    "",
    "FROZEN SENTINEL-1 EXPOSURE",
    "--------------------------",
    f"Support-year source: {primary_exposure_col}",
    "Primary well exposure: unweighted median within 10 km.",
    "Prespecified spatial sensitivities: 5 km and 20 km.",
    "Complete-4-track 10-km sensitivity retained.",
    "",
    "GROUNDWATER TEMPORAL RULE",
    "-------------------------",
    "Outcome: median valid groundwater depth during August.",
    "Antecedent state: last valid groundwater depth during January-February.",
    "Eligible well-year requires both values.",
    "",
    "PRIMARY MODEL",
    "-------------",
    "gw_aug_m ~ s1_flooding_like_10km + gw_pre_last_janfeb_m + C(station) + C(year)",
    "",
    "SIGN INTERPRETATION",
    "-------------------",
    "beta < 0: more flooding-like S1 signal associated with shallower groundwater.",
    "beta > 0: more flooding-like S1 signal associated with deeper groundwater.",
    "Association only; no causal flooding/recharge interpretation.",
    "",
    "INFERENCE",
    "---------",
    "Primary covariance: cluster-robust by station.",
    "Robustness: wild cluster bootstrap-t, Webb weights, 9,999 reps, seed 52026.",
    "",
    "FIREWALL",
    "--------",
    "Groundwater values read: False",
    "Irrigation-flow values read: False",
    "Groundwater model fitted: False",
    "S1 feature/track/orientation redesign from groundwater: False",
    "Spatial-radius redesign from groundwater: False",
    "Temporal-window redesign from groundwater: False",
    "",
    "STAGE 52 STATUS: PROTOCOL_FROZEN_GROUNDWATER_VALUES_UNREAD",
]

TXT_OUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))
