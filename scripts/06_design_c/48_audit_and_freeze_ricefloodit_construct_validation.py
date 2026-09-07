
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

P = D / "c2y_validation_protocol" / "c2y_ricefloodit_construct_validation_protocol.json"
Z = D / "c2z_construct_validation"
CELL = Z / "c2z_primary_feature_year_track_correlations.csv"
TRACK = Z / "c2z_primary_feature_track_summary.csv"
CROSS = Z / "c2z_primary_feature_cross_track_summary.csv"
QA47 = Z / "c2z_construct_validation_qa.json"

OUT = D / "c2za_construct_freeze"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT_OUT = OUT / "c2za_construct_validation_independent_audit.csv"
FREEZE_JSON = OUT / "c2za_frozen_construct_valid_measurement_family.json"
FREEZE_TXT = OUT / "c2za_frozen_construct_valid_measurement_family.txt"

for f in [P, CELL, TRACK, CROSS, QA47]:
    if not f.exists():
        raise FileNotFoundError(f)

protocol = json.loads(P.read_text(encoding="utf-8"))
qa47 = json.loads(QA47.read_text(encoding="utf-8"))
cell = pd.read_csv(CELL)
track = pd.read_csv(TRACK)
cross = pd.read_csv(CROSS)

flags = []

if qa47.get("status") != "PASS_VALIDATION_EXECUTED":
    flags.append("QA47_STATUS_NOT_PASS")

primary = list(protocol["primary_features"])
years = [int(y) for y in protocol["validation_target"]["years"]]
tracks = {
    (str(x["orbit_state"]), int(x["relative_orbit"]))
    for x in protocol["unit_of_validation"]["tracks"]
}
screen = protocol["construct_consistency_screen"]
min_years = int(screen["minimum_eligible_years_per_track"])
min_sign = float(screen["minimum_within_track_sign_agreement_share"])
min_abs_rho = float(screen["minimum_within_track_median_absolute_spearman"])
min_tracks = int(screen["minimum_tracks_with_consistent_majority_direction"])

expected_cells = len(primary) * len(years) * len(tracks)
if len(cell) != expected_cells:
    flags.append(f"PRIMARY_CELL_COUNT_MISMATCH:{len(cell)}")

track_re = []
for (state, orbit, feature), g in cell.groupby(
    ["orbit_state", "relative_orbit", "feature"], sort=True
):
    e = g[g["eligible"].astype(str).str.lower().isin(["true", "1"])].copy()
    r = pd.to_numeric(e["spearman_rho"], errors="coerce").dropna()
    pos = int((r > 0).sum())
    neg = int((r < 0).sum())
    nonzero = pos + neg
    sign_agree = max(pos, neg) / nonzero if nonzero else float("nan")
    majority_sign = (
        "positive" if pos > neg else
        "negative" if neg > pos else
        "tie"
    )
    med_abs = float(r.abs().median()) if len(r) else float("nan")
    passed = bool(
        len(r) >= min_years
        and pd.notna(sign_agree)
        and sign_agree >= min_sign
        and pd.notna(med_abs)
        and med_abs >= min_abs_rho
        and majority_sign in {"positive", "negative"}
    )
    track_re.append({
        "orbit_state": state,
        "relative_orbit": int(orbit),
        "feature": feature,
        "majority_sign": majority_sign,
        "within_track_consistency_pass": passed,
    })

track_re = pd.DataFrame(track_re)

cross_re = []
for feature, g in track_re.groupby("feature", sort=True):
    p = g[g["within_track_consistency_pass"]].copy()
    pos = int((p["majority_sign"] == "positive").sum())
    neg = int((p["majority_sign"] == "negative").sum())
    passing_n = len(p)
    direction = (
        "positive" if pos > neg else
        "negative" if neg > pos else
        "tie_or_none"
    )
    consistent_n = max(pos, neg)
    passed = bool(
        passing_n >= min_tracks
        and consistent_n >= min_tracks
        and direction in {"positive", "negative"}
    )
    cross_re.append({
        "feature": feature,
        "construct_consistency_pass": passed,
    })

cross_re = pd.DataFrame(cross_re)
passing = sorted(
    cross_re.loc[cross_re["construct_consistency_pass"], "feature"]
    .astype(str)
    .tolist()
)

reported = sorted(
    map(str, qa47.get(
        "primary_features_passing_frozen_construct_consistency_screen", []
    ))
)
if passing != reported:
    flags.append(
        f"PASSING_FEATURE_SET_MISMATCH:recomputed={passing};reported={reported}"
    )

pd.DataFrame(
    [{"severity": "HARD", "flag": f} for f in flags]
    if flags else [{"severity": "NONE", "flag": "NONE"}]
).to_csv(AUDIT_OUT, index=False)

status = (
    "PASS_CONSTRUCT_VALID_MEASUREMENT_FAMILY_FROZEN"
    if not flags else
    "FAIL_REVIEW_REQUIRED"
)

freeze = {
    "stage": "DESIGN_C_STAGE48_AUDIT_AND_FREEZE_RICEFLOODIT_CONSTRUCT_VALIDATION",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "construct_valid_primary_features": passing,
    "selection_basis": (
        "predeclared Stage-46 construct-consistency screen; "
        "not maximum observed correlation"
    ),
    "track_specific_values_retained": True,
    "cross_track_pooling_rule_selected": False,
    "cross_track_standardization_rule_selected": False,
    "groundwater_values_read": False,
    "irrigation_flow_values_read": False,
    "inundation_threshold_selected": False,
    "classifier_fitted": False,
    "groundwater_association_models_fitted": 0,
    "hard_flags": flags,
}
FREEZE_JSON.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")

lines = [
    "DESIGN C - STAGE 48 CONSTRUCT-VALID MEASUREMENT FAMILY FREEZE",
    "=" * 78,
    "",
    f"Stage-47 status: {qa47.get('status')}",
    f"Independent audit hard flags: {flags if flags else 'None'}",
    "",
    "FROZEN CONSTRUCT-VALID PRIMARY FEATURE FAMILY",
    "---------------------------------------------",
    f"{passing if passing else 'NONE'}",
    "",
    "No cross-track pooling or standardization rule is frozen here.",
    "The validated feature remains track-specific.",
    "",
    "FIREWALL",
    "--------",
    "RiceFloodIT values were read in Stage 47 for construct validation only.",
    "Groundwater read: False",
    "Irrigation-flow read: False",
    "Threshold selected: False",
    "Classifier fitted: False",
    "Groundwater association model fitted: False",
    "",
    f"STAGE 48 STATUS: {status}",
]
FREEZE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

if flags:
    raise RuntimeError("Stage 48 independent validation audit failed.")
