"""
Design C — Stage 58
Descriptive audit of prespecified sensitivity-sample composition and spatial
exposure overlap AFTER Stage 57.

Purpose
-------
Explain why the 5-km, 10-km, 20-km, and complete-4 sensitivities use different
samples, without fitting any new groundwater association model.

This stage is descriptive only. It does NOT:
- refit the primary model;
- fit common-sample radius models;
- search new radii;
- select/exclude wells;
- inspect alternate outcomes;
- change the primary or sensitivity conclusions.

Outputs
-------
outputs/diagnostics/design_c/c2zm_sensitivity_sample_audit/
    c2zm_radius_sample_membership.csv
    c2zm_radius_pairwise_overlap.csv
    c2zm_well_membership_summary.csv
    c2zm_exposure_common_sample_correlations.csv
    c2zm_complete4_membership_summary.csv
    c2zm_loo_extreme_omissions.csv
    c2zm_sensitivity_sample_audit.json
    c2zm_sensitivity_sample_audit_summary.txt
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

FULL_PANEL = (
    D / "c2zg_groundwater_panel"
    / "c2zg_groundwater_s1_analysis_panel_2015_2025.csv"
)
PRIMARY_PANEL = (
    D / "c2zh_groundwater_panel_freeze"
    / "c2zh_frozen_primary_groundwater_s1_panel.csv"
)
STAGE57_RESULTS = (
    D / "c2zl_prespecified_sensitivities"
    / "c2zl_prespecified_sensitivity_results.csv"
)
STAGE57_LOO = (
    D / "c2zl_prespecified_sensitivities"
    / "c2zl_primary_leave_one_well_out.csv"
)
STAGE57_QA = (
    D / "c2zl_prespecified_sensitivities"
    / "c2zl_prespecified_sensitivity_qa.json"
)

OUT = D / "c2zm_sensitivity_sample_audit"
OUT.mkdir(parents=True, exist_ok=True)

MEMBERSHIP_OUT = OUT / "c2zm_radius_sample_membership.csv"
PAIRWISE_OUT = OUT / "c2zm_radius_pairwise_overlap.csv"
WELL_OUT = OUT / "c2zm_well_membership_summary.csv"
CORR_OUT = OUT / "c2zm_exposure_common_sample_correlations.csv"
COMPLETE4_OUT = OUT / "c2zm_complete4_membership_summary.csv"
LOO_EXTREME_OUT = OUT / "c2zm_loo_extreme_omissions.csv"
QA_OUT = OUT / "c2zm_sensitivity_sample_audit.json"
SUMMARY_OUT = OUT / "c2zm_sensitivity_sample_audit_summary.txt"

for f in [
    FULL_PANEL,
    PRIMARY_PANEL,
    STAGE57_RESULTS,
    STAGE57_LOO,
    STAGE57_QA,
]:
    if not f.exists():
        raise FileNotFoundError(f)

qa57 = json.loads(STAGE57_QA.read_text(encoding="utf-8"))
if qa57.get("status") != \
        "PASS_PRESPECIFIED_SENSITIVITIES_EXECUTED_NO_NEW_MODEL_SEARCH":
    raise RuntimeError(
        f"Unexpected Stage-57 status: {qa57.get('status')!r}"
    )

# Read only identifiers and frozen exposure columns. No groundwater outcome is
# used in this descriptive audit.
usecols = [
    "station",
    "year",
    "s1_flooding_like_5km",
    "s1_flooding_like_10km",
    "s1_flooding_like_20km",
    "s1_flooding_like_complete4_10km",
]
full = pd.read_csv(FULL_PANEL, usecols=usecols, low_memory=False)
primary = pd.read_csv(
    PRIMARY_PANEL,
    usecols=["station", "year", "s1_flooding_like_10km"],
    low_memory=False,
)

for d in [full, primary]:
    d["station"] = d["station"].astype(str)
    d["year"] = pd.to_numeric(d["year"], errors="raise").astype(int)

for col in usecols[2:]:
    if col in full.columns:
        full[col] = pd.to_numeric(full[col], errors="coerce")

if full.duplicated(["station", "year"]).any():
    raise RuntimeError("Full panel contains duplicate station-year keys.")
if primary.duplicated(["station", "year"]).any():
    raise RuntimeError("Primary panel contains duplicate station-year keys.")

primary_keys = set(map(tuple, primary[["station", "year"]].to_numpy()))
if len(primary_keys) != 193:
    raise RuntimeError(f"Expected 193 frozen primary keys; found {len(primary_keys)}")

# Membership is determined purely by availability of each frozen exposure.
full["eligible_5km"] = np.isfinite(full["s1_flooding_like_5km"])
full["eligible_10km"] = np.isfinite(full["s1_flooding_like_10km"])
full["eligible_20km"] = np.isfinite(full["s1_flooding_like_20km"])
full["eligible_complete4_10km"] = np.isfinite(
    full["s1_flooding_like_complete4_10km"]
)
full["in_frozen_primary_10km"] = [
    (s, y) in primary_keys
    for s, y in zip(full["station"], full["year"])
]

membership_cols = [
    "station",
    "year",
    "eligible_5km",
    "eligible_10km",
    "eligible_20km",
    "eligible_complete4_10km",
    "in_frozen_primary_10km",
]
full[membership_cols].sort_values(["station", "year"]).to_csv(
    MEMBERSHIP_OUT, index=False
)

sets = {}
for label, col in [
    ("5km", "eligible_5km"),
    ("10km", "eligible_10km"),
    ("20km", "eligible_20km"),
    ("complete4_10km", "eligible_complete4_10km"),
    ("frozen_primary_10km", "in_frozen_primary_10km"),
]:
    sets[label] = set(
        map(
            tuple,
            full.loc[full[col], ["station", "year"]].to_numpy(),
        )
    )

# Pairwise overlap / differences.
pairwise_rows = []
labels = list(sets)
for i, a in enumerate(labels):
    for b in labels[i + 1:]:
        A, B = sets[a], sets[b]
        pairwise_rows.append({
            "sample_a": a,
            "sample_b": b,
            "n_a": len(A),
            "n_b": len(B),
            "intersection_n": len(A & B),
            "union_n": len(A | B),
            "a_only_n": len(A - B),
            "b_only_n": len(B - A),
            "jaccard": (
                len(A & B) / len(A | B) if len(A | B) else np.nan
            ),
        })

pairwise = pd.DataFrame(pairwise_rows)
pairwise.to_csv(PAIRWISE_OUT, index=False)

# Well-level membership counts.
well_rows = []
for station, g in full.groupby("station", sort=True):
    rec = {"station": station}
    for label, col in [
        ("5km", "eligible_5km"),
        ("10km", "eligible_10km"),
        ("20km", "eligible_20km"),
        ("complete4_10km", "eligible_complete4_10km"),
        ("frozen_primary_10km", "in_frozen_primary_10km"),
    ]:
        rec[f"{label}_well_years"] = int(g[col].sum())
        rec[f"{label}_ever_eligible"] = bool(g[col].any())
    well_rows.append(rec)

well = pd.DataFrame(well_rows)
well.to_csv(WELL_OUT, index=False)

# Correlation of frozen exposure representations on exactly common rows.
# This is exposure-only descriptive comparison; no groundwater outcome is used.
corr_rows = []
pairs = [
    ("5km", "s1_flooding_like_5km",
     "10km", "s1_flooding_like_10km"),
    ("10km", "s1_flooding_like_10km",
     "20km", "s1_flooding_like_20km"),
    ("5km", "s1_flooding_like_5km",
     "20km", "s1_flooding_like_20km"),
    ("10km", "s1_flooding_like_10km",
     "complete4_10km", "s1_flooding_like_complete4_10km"),
]
for la, ca, lb, cb in pairs:
    d = full.loc[np.isfinite(full[ca]) & np.isfinite(full[cb]), [ca, cb]]
    if len(d) >= 3:
        pearson = float(d[ca].corr(d[cb], method="pearson"))
        spearman = float(d[ca].corr(d[cb], method="spearman"))
    else:
        pearson = np.nan
        spearman = np.nan
    corr_rows.append({
        "exposure_a": la,
        "exposure_b": lb,
        "common_n": int(len(d)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    })

corr = pd.DataFrame(corr_rows)
corr.to_csv(CORR_OUT, index=False)

# Complete-4 restriction relative to frozen 10-km primary.
P = sets["frozen_primary_10km"]
C4 = sets["complete4_10km"]
complete4_rows = [{
    "frozen_primary_n": len(P),
    "complete4_n": len(C4),
    "intersection_n": len(P & C4),
    "primary_only_n": len(P - C4),
    "complete4_only_n": len(C4 - P),
}]
pd.DataFrame(complete4_rows).to_csv(COMPLETE4_OUT, index=False)

# Identify which LOO omission yielded min/max and any sign reversal.
loo = pd.read_csv(STAGE57_LOO, low_memory=False)
loo["omitted_station"] = loo["omitted_station"].astype(str)
loo["beta_hat"] = pd.to_numeric(loo["beta_hat"], errors="raise")

extreme = pd.concat([
    loo.loc[[loo["beta_hat"].idxmin()]].assign(role="minimum_beta"),
    loo.loc[[loo["beta_hat"].idxmax()]].assign(role="maximum_beta"),
    loo.loc[loo["beta_hat"] > 0].assign(role="positive_beta"),
], ignore_index=True).drop_duplicates(
    subset=["omitted_station", "role"]
)
extreme.to_csv(LOO_EXTREME_OUT, index=False)

def unique_wells(S):
    return sorted({s for s, _ in S})

summary = {
    "stage": "DESIGN_C_STAGE58_SENSITIVITY_SAMPLE_COMPOSITION_AUDIT",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS_DESCRIPTIVE_SAMPLE_AUDIT_NO_NEW_ASSOCIATION_MODEL",
    "groundwater_outcome_used": False,
    "new_association_model_fitted": False,
    "sample_sizes": {
        k: len(v) for k, v in sets.items()
    },
    "well_counts": {
        k: len(unique_wells(v)) for k, v in sets.items()
    },
    "loo_positive_omission_count": int((loo["beta_hat"] > 0).sum()),
    "new_radius_added": False,
    "result_driven_exclusion": False,
}
QA_OUT.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

# Human-readable highlights.
def pair(a, b):
    row = pairwise.loc[
        ((pairwise["sample_a"] == a) & (pairwise["sample_b"] == b))
        | ((pairwise["sample_a"] == b) & (pairwise["sample_b"] == a))
    ]
    if len(row) != 1:
        raise RuntimeError(f"Pairwise row not unique for {a}, {b}")
    return row.iloc[0]

p5_10 = pair("5km", "10km")
p10_20 = pair("10km", "20km")
p5_20 = pair("5km", "20km")

pos = loo.loc[loo["beta_hat"] > 0].sort_values("beta_hat", ascending=False)

lines = [
    "DESIGN C - STAGE 58 SENSITIVITY SAMPLE COMPOSITION AUDIT",
    "=" * 74,
    "",
    "DESCRIPTIVE ONLY",
    "----------------",
    "Groundwater outcome used: False",
    "New association model fitted: False",
    "",
    "SAMPLE SIZES FROM FROZEN EXPOSURE AVAILABILITY",
    "------------------------------------------------",
    f"5 km: {len(sets['5km'])} well-years; "
    f"{len(unique_wells(sets['5km']))} wells",
    f"10 km available: {len(sets['10km'])} well-years; "
    f"{len(unique_wells(sets['10km']))} wells",
    f"Frozen primary 10 km: {len(P)} well-years; "
    f"{len(unique_wells(P))} wells",
    f"20 km: {len(sets['20km'])} well-years; "
    f"{len(unique_wells(sets['20km']))} wells",
    f"Complete-4 10 km: {len(C4)} well-years; "
    f"{len(unique_wells(C4))} wells",
    "",
    "PAIRWISE WELL-YEAR OVERLAP",
    "--------------------------",
    f"5 vs 10 km: intersection={int(p5_10['intersection_n'])}, "
    f"5-only={int(p5_10['a_only_n'] if p5_10['sample_a']=='5km' else p5_10['b_only_n'])}, "
    f"10-only={int(p5_10['b_only_n'] if p5_10['sample_b']=='10km' else p5_10['a_only_n'])}, "
    f"Jaccard={p5_10['jaccard']:.6f}",
    f"10 vs 20 km: intersection={int(p10_20['intersection_n'])}, "
    f"Jaccard={p10_20['jaccard']:.6f}",
    f"5 vs 20 km: intersection={int(p5_20['intersection_n'])}, "
    f"Jaccard={p5_20['jaccard']:.6f}",
    "",
    "EXPOSURE AGREEMENT ON COMMON WELL-YEARS",
    "---------------------------------------",
]
for r in corr.itertuples(index=False):
    lines.append(
        f"{r.exposure_a} vs {r.exposure_b}: "
        f"N={r.common_n}, Pearson r={r.pearson_r:.6f}, "
        f"Spearman rho={r.spearman_rho:.6f}"
    )

lines += [
    "",
    "COMPLETE-4 RESTRICTION",
    "----------------------",
    f"Primary 10-km rows retained in complete-4: {len(P & C4)}/{len(P)}",
    f"Primary 10-km rows lost under complete-4: {len(P - C4)}",
    "",
    "LEAVE-ONE-WELL-OUT SIGN REVERSAL",
    "--------------------------------",
    f"Positive LOO coefficients: {len(pos)}",
]
if len(pos):
    for r in pos.itertuples(index=False):
        lines.append(
            f"omitted station={r.omitted_station}; beta={r.beta_hat:.12g}"
        )
else:
    lines.append("None")

lines += [
    "",
    "INTERPRETIVE LIMIT",
    "------------------",
    "This stage describes sample/exposure differences only.",
    "It does not attribute radius-sensitivity coefficient changes to either",
    "sample composition or exposure-scale transformation causally.",
    "No common-sample association model was fitted.",
    "",
    "STAGE 58 STATUS: PASS_DESCRIPTIVE_SAMPLE_AUDIT_NO_NEW_ASSOCIATION_MODEL",
]

SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
