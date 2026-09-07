"""
Design C — Stage 59
Freeze the final groundwater–Sentinel-1 scientific synthesis after the
confirmatory primary analysis, prespecified sensitivities, and descriptive
sample-composition audit.

This stage fits NO new association model and reads no new raw data.

Its purpose is to:
- preserve the fixed Stage-56 primary result;
- preserve the Stage-57 prespecified sensitivity results;
- preserve the Stage-58 descriptive interpretation of sensitivity-sample
  differences;
- generate a concise, auditable statement of what the evidence supports and
  what it does not support.

No post-result model search, exclusion, radius change, alternate outcome,
alternate threshold, or causal claim is permitted.
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

R56 = D / "c2zk_primary_association" / "c2zk_primary_association_result.csv"
Q56 = D / "c2zk_primary_association" / "c2zk_primary_association_qa.json"

R57 = (
    D / "c2zl_prespecified_sensitivities"
    / "c2zl_prespecified_sensitivity_results.csv"
)
L57 = (
    D / "c2zl_prespecified_sensitivities"
    / "c2zl_primary_leave_one_well_out.csv"
)
Q57 = (
    D / "c2zl_prespecified_sensitivities"
    / "c2zl_prespecified_sensitivity_qa.json"
)

Q58 = (
    D / "c2zm_sensitivity_sample_audit"
    / "c2zm_sensitivity_sample_audit.json"
)
P58 = (
    D / "c2zm_sensitivity_sample_audit"
    / "c2zm_radius_pairwise_overlap.csv"
)
C58 = (
    D / "c2zm_sensitivity_sample_audit"
    / "c2zm_exposure_common_sample_correlations.csv"
)

OUT = D / "c2zn_final_groundwater_s1_synthesis"
OUT.mkdir(parents=True, exist_ok=True)

TABLE_OUT = OUT / "c2zn_final_groundwater_s1_result_table.csv"
JSON_OUT = OUT / "c2zn_final_groundwater_s1_synthesis.json"
TEXT_OUT = OUT / "c2zn_final_groundwater_s1_synthesis.txt"

for f in [R56, Q56, R57, L57, Q57, Q58, P58, C58]:
    if not f.exists():
        raise FileNotFoundError(f)

r56 = pd.read_csv(R56)
q56 = json.loads(Q56.read_text(encoding="utf-8"))
r57 = pd.read_csv(R57)
loo = pd.read_csv(L57)
q57 = json.loads(Q57.read_text(encoding="utf-8"))
q58 = json.loads(Q58.read_text(encoding="utf-8"))
pairwise = pd.read_csv(P58)
corr = pd.read_csv(C58)

if q56.get("status") != "PASS_PRIMARY_ASSOCIATION_REVEALED_NO_MODEL_SEARCH":
    raise RuntimeError(f"Unexpected Stage-56 status: {q56.get('status')!r}")

if q57.get("status") != \
        "PASS_PRESPECIFIED_SENSITIVITIES_EXECUTED_NO_NEW_MODEL_SEARCH":
    raise RuntimeError(f"Unexpected Stage-57 status: {q57.get('status')!r}")

if q58.get("status") != \
        "PASS_DESCRIPTIVE_SAMPLE_AUDIT_NO_NEW_ASSOCIATION_MODEL":
    raise RuntimeError(f"Unexpected Stage-58 status: {q58.get('status')!r}")

if len(r56) != 1:
    raise RuntimeError("Stage-56 result must contain exactly one primary row.")

required_sens = {
    "5km_spatial_scale",
    "20km_spatial_scale",
    "complete4_track_support",
    "without_antecedent_groundwater",
}
if set(r57["analysis"].astype(str)) != required_sens:
    raise RuntimeError(
        "Stage-57 sensitivity set differs from the frozen expected set."
    )

p = r56.iloc[0]

# Exact primary values are inherited, never re-estimated here.
primary = {
    "beta_hat": float(p["beta_hat"]),
    "crv1_se": float(p["crv1_se"]),
    "crv1_t": float(p["crv1_t"]),
    "crv1_df": int(p["crv1_df"]),
    "crv1_p_two_sided": float(p["crv1_p_two_sided"]),
    "crv1_ci95_low": float(p["crv1_ci95_low"]),
    "crv1_ci95_high": float(p["crv1_ci95_high"]),
    "wcr31_webb_p_two_sided": float(p["wcr31_webb_p_two_sided"]),
    "n": int(p["n"]),
    "n_wells": int(p["n_wells"]),
    "first_year": int(p["first_year"]),
    "last_year": int(p["last_year"]),
}

# Guard against accidental mutation of the known fixed primary reveal.
expected_primary = {
    "beta_hat": -0.30670915657549086,
    "crv1_se": 0.22605067516926328,
    "crv1_p_two_sided": 0.18463834766455897,
    "crv1_ci95_low": -0.76774254815019,
    "crv1_ci95_high": 0.15432423499920833,
    "wcr31_webb_p_two_sided": 0.49414941494149417,
    "n": 193,
    "n_wells": 32,
}
for k, expected in expected_primary.items():
    actual = primary[k]
    if isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"Fixed primary value changed for {k}: {actual} != {expected}"
            )
    elif actual != expected:
        raise RuntimeError(
            f"Fixed primary value changed for {k}: {actual} != {expected}"
        )

# Sensitivity lookup.
sens = {
    row.analysis: row
    for row in r57.itertuples(index=False)
}

# Leave-one-well-out summary.
loo["beta_hat"] = pd.to_numeric(loo["beta_hat"], errors="raise")
loo_negative = int((loo["beta_hat"] < 0).sum())
loo_positive = int((loo["beta_hat"] > 0).sum())
loo_zero = int((loo["beta_hat"] == 0).sum())
loo_min = float(loo["beta_hat"].min())
loo_max = float(loo["beta_hat"].max())
loo_positive_stations = (
    loo.loc[loo["beta_hat"] > 0, "omitted_station"].astype(str).tolist()
)

# Exact exposure/sample overlap evidence from Stage 58.
def pair(a: str, b: str):
    d = pairwise.loc[
        ((pairwise["sample_a"] == a) & (pairwise["sample_b"] == b))
        | ((pairwise["sample_a"] == b) & (pairwise["sample_b"] == a))
    ]
    if len(d) != 1:
        raise RuntimeError(f"Missing/nonunique pairwise overlap: {a}, {b}")
    return d.iloc[0]

p5_10 = pair("5km", "10km")
p10_20 = pair("10km", "20km")

def exposure_corr(a: str, b: str):
    d = corr.loc[
        ((corr["exposure_a"] == a) & (corr["exposure_b"] == b))
        | ((corr["exposure_a"] == b) & (corr["exposure_b"] == a))
    ]
    if len(d) != 1:
        raise RuntimeError(f"Missing/nonunique exposure correlation: {a}, {b}")
    return d.iloc[0]

c5_10 = exposure_corr("5km", "10km")
c10_20 = exposure_corr("10km", "20km")

# Compact result table.
rows = [{
    "analysis_role": "primary",
    "analysis": "10km_primary",
    "n": primary["n"],
    "n_wells": primary["n_wells"],
    "beta_hat": primary["beta_hat"],
    "crv1_se": primary["crv1_se"],
    "crv1_p_two_sided": primary["crv1_p_two_sided"],
    "crv1_ci95_low": primary["crv1_ci95_low"],
    "crv1_ci95_high": primary["crv1_ci95_high"],
    "wcr31_webb_p_two_sided": primary["wcr31_webb_p_two_sided"],
}]
for name in [
    "5km_spatial_scale",
    "20km_spatial_scale",
    "complete4_track_support",
    "without_antecedent_groundwater",
]:
    r = sens[name]
    rows.append({
        "analysis_role": "prespecified_sensitivity",
        "analysis": name,
        "n": int(r.n),
        "n_wells": int(r.n_wells),
        "beta_hat": float(r.beta_hat),
        "crv1_se": float(r.crv1_se),
        "crv1_p_two_sided": float(r.crv1_p_two_sided),
        "crv1_ci95_low": float(r.crv1_ci95_low),
        "crv1_ci95_high": float(r.crv1_ci95_high),
        "wcr31_webb_p_two_sided": float(r.wcr31_webb_p_two_sided),
    })

pd.DataFrame(rows).to_csv(TABLE_OUT, index=False)

# Evidence classification is descriptive and deterministic from frozen outputs.
all_model_wcr31_p = [primary["wcr31_webb_p_two_sided"]] + [
    float(sens[k].wcr31_webb_p_two_sided)
    for k in required_sens
]
all_wcr31_above_005 = all(x > 0.05 for x in all_model_wcr31_p)

direction_stability = {
    "primary_negative": primary["beta_hat"] < 0,
    "complete4_negative":
        float(sens["complete4_track_support"].beta_hat) < 0,
    "without_antecedent_negative":
        float(sens["without_antecedent_groundwater"].beta_hat) < 0,
    "five_km_negative":
        float(sens["5km_spatial_scale"].beta_hat) < 0,
    "twenty_km_negative":
        float(sens["20km_spatial_scale"].beta_hat) < 0,
    "loo_negative_n": loo_negative,
    "loo_positive_n": loo_positive,
    "loo_zero_n": loo_zero,
    "loo_total_n": int(len(loo)),
}

claims = {
    "primary_result": (
        "The frozen primary estimate is negative, corresponding to more "
        "flooding-like Sentinel-1 exposure being associated with shallower "
        "August groundwater after adjustment for antecedent groundwater, "
        "well fixed effects, and year fixed effects."
    ),
    "uncertainty": (
        "The primary confidence interval crosses zero and neither the primary "
        "CRV1 test nor the prespecified WCR31-Webb robustness test provides "
        "clear evidence of a non-zero association."
    ),
    "within_architecture_stability": (
        "The negative direction is retained under complete-four-track "
        "restriction, removal of antecedent-groundwater adjustment, and "
        "31 of 32 leave-one-well-out fits."
    ),
    "spatial_scale_instability": (
        "The sign is not stable across the prespecified spatial-radius "
        "sensitivities: the 5-km estimate is positive whereas the 10-km and "
        "20-km estimates are negative."
    ),
    "radius_interpretation_limit": (
        "The radius sensitivities change both exposure representation and "
        "sample composition, so coefficient differences cannot be attributed "
        "to spatial scale alone."
    ),
    "causal_limit": (
        "All estimates are observational. They do not identify causal effects "
        "of irrigation, inundation, recharge, canal delivery, or pumping."
    ),
    "overall": (
        "The evidence is directionally suggestive at the frozen 10-km scale "
        "but not statistically conclusive and not spatial-scale robust."
    ),
}

synthesis = {
    "stage": "DESIGN_C_STAGE59_FINAL_GROUNDWATER_S1_SYNTHESIS",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": "FINAL_SYNTHESIS_FROZEN_NO_NEW_MODEL_FIT",
    "new_association_model_fitted": False,
    "new_raw_data_read": False,
    "model_search_performed": False,
    "result_driven_exclusion_performed": False,
    "primary_result_replaced": False,
    "primary": primary,
    "sensitivities": {
        name: {
            "n": int(sens[name].n),
            "n_wells": int(sens[name].n_wells),
            "beta_hat": float(sens[name].beta_hat),
            "crv1_p_two_sided": float(sens[name].crv1_p_two_sided),
            "wcr31_webb_p_two_sided":
                float(sens[name].wcr31_webb_p_two_sided),
        }
        for name in sorted(required_sens)
    },
    "leave_one_well_out": {
        "fits": int(len(loo)),
        "negative": loo_negative,
        "positive": loo_positive,
        "zero": loo_zero,
        "beta_min": loo_min,
        "beta_max": loo_max,
        "positive_omitted_stations": loo_positive_stations,
    },
    "sample_and_exposure_audit": {
        "5_vs_10_intersection_n": int(p5_10["intersection_n"]),
        "5_vs_10_jaccard": float(p5_10["jaccard"]),
        "10_vs_20_intersection_n": int(p10_20["intersection_n"]),
        "10_vs_20_jaccard": float(p10_20["jaccard"]),
        "5_vs_10_pearson_r": float(c5_10["pearson_r"]),
        "5_vs_10_spearman_rho": float(c5_10["spearman_rho"]),
        "10_vs_20_pearson_r": float(c10_20["pearson_r"]),
        "10_vs_20_spearman_rho": float(c10_20["spearman_rho"]),
    },
    "direction_stability": direction_stability,
    "all_primary_and_sensitivity_wcr31_p_above_0_05":
        all_wcr31_above_005,
    "claims": claims,
    "sha256": {
        "stage56_result": hashlib.sha256(R56.read_bytes()).hexdigest(),
        "stage56_qa": hashlib.sha256(Q56.read_bytes()).hexdigest(),
        "stage57_results": hashlib.sha256(R57.read_bytes()).hexdigest(),
        "stage57_loo": hashlib.sha256(L57.read_bytes()).hexdigest(),
        "stage57_qa": hashlib.sha256(Q57.read_bytes()).hexdigest(),
        "stage58_qa": hashlib.sha256(Q58.read_bytes()).hexdigest(),
        "stage58_pairwise": hashlib.sha256(P58.read_bytes()).hexdigest(),
        "stage58_correlations": hashlib.sha256(C58.read_bytes()).hexdigest(),
        "final_result_table": hashlib.sha256(TABLE_OUT.read_bytes()).hexdigest(),
    },
}

JSON_OUT.write_text(
    json.dumps(synthesis, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 59 FINAL GROUNDWATER-SENTINEL-1 SYNTHESIS",
    "=" * 78,
    "",
    "PRIMARY RESULT — FIXED",
    "----------------------",
    f"N={primary['n']} well-years; wells={primary['n_wells']}; "
    f"years={primary['first_year']}-{primary['last_year']}",
    f"beta={primary['beta_hat']:.12g}",
    f"CRV1 SE={primary['crv1_se']:.12g}",
    f"CRV1 p={primary['crv1_p_two_sided']:.12g}",
    f"CRV1 95% CI=[{primary['crv1_ci95_low']:.12g}, "
    f"{primary['crv1_ci95_high']:.12g}]",
    f"WCR31-Webb p={primary['wcr31_webb_p_two_sided']:.12g}",
    "",
    "PRESPECIFIED SENSITIVITIES",
    "---------------------------",
]
for name in [
    "5km_spatial_scale",
    "20km_spatial_scale",
    "complete4_track_support",
    "without_antecedent_groundwater",
]:
    r = sens[name]
    lines.append(
        f"{name}: N={int(r.n)}, wells={int(r.n_wells)}, "
        f"beta={float(r.beta_hat):.12g}, "
        f"CRV1 p={float(r.crv1_p_two_sided):.12g}, "
        f"WCR31 p={float(r.wcr31_webb_p_two_sided):.12g}"
    )

lines += [
    "",
    "LEAVE-ONE-WELL-OUT",
    "------------------",
    f"Negative coefficients: {loo_negative}/{len(loo)}",
    f"Positive coefficients: {loo_positive}/{len(loo)}",
    f"Coefficient range: [{loo_min:.12g}, {loo_max:.12g}]",
    "",
    "SPATIAL-SCALE / SAMPLE AUDIT",
    "----------------------------",
    f"5 vs 10 km overlap: {int(p5_10['intersection_n'])} well-years; "
    f"Jaccard={float(p5_10['jaccard']):.6f}",
    f"5 vs 10 km exposure agreement: "
    f"Pearson r={float(c5_10['pearson_r']):.6f}; "
    f"Spearman rho={float(c5_10['spearman_rho']):.6f}",
    f"10 vs 20 km overlap: {int(p10_20['intersection_n'])} well-years; "
    f"Jaccard={float(p10_20['jaccard']):.6f}",
    f"10 vs 20 km exposure agreement: "
    f"Pearson r={float(c10_20['pearson_r']):.6f}; "
    f"Spearman rho={float(c10_20['spearman_rho']):.6f}",
    "",
    "FROZEN SCIENTIFIC SYNTHESIS",
    "---------------------------",
    "1. " + claims["primary_result"],
    "2. " + claims["uncertainty"],
    "3. " + claims["within_architecture_stability"],
    "4. " + claims["spatial_scale_instability"],
    "5. " + claims["radius_interpretation_limit"],
    "6. " + claims["causal_limit"],
    "",
    "OVERALL CONCLUSION",
    "------------------",
    claims["overall"],
    "",
    "FIREWALL",
    "--------",
    "New association model fitted: False",
    "New raw data read: False",
    "Model search performed: False",
    "Result-driven exclusion performed: False",
    "Primary result replaced: False",
    "",
    "STAGE 59 STATUS: FINAL_SYNTHESIS_FROZEN_NO_NEW_MODEL_FIT",
]

TEXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
