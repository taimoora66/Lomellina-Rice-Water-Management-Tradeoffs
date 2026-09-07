"""
Design C — Stage 57
Execute ONLY the sensitivity analyses prespecified in the frozen Stage-52
groundwater association protocol, after the Stage-56 primary result has been
revealed and fixed.

Primary result remains unchanged and is not re-ranked or replaced.

Prespecified sensitivities
--------------------------
1. 5-km spatial scale
2. 20-km spatial scale
3. complete-4-track 10-km exposure
4. primary 10-km model without antecedent groundwater
5. leave-one-well-out coefficient influence diagnostic on the frozen primary panel

Inference
---------
For model-based sensitivities, reuse the exact Stage-55 implementation:
- statsmodels OLS
- CRV1 clustered by station
- use_correction=True
- df_correction=True
- two-sided t inference with df=G-1
- WCR31 Webb, null imposed, B=9999, seed=52026

Important sample rule
---------------------
Spatial/measurement sensitivities use the mechanically eligible well-years for
that sensitivity exposure while preserving the same frozen groundwater temporal
eligibility rules.

The "without antecedent groundwater" sensitivity deliberately keeps the exact
193-row Stage-56 primary sample, so changes reflect covariate specification
rather than sample composition.

Leave-one-well-out is a coefficient-influence diagnostic only. It may never be
used to remove a well from the primary analysis.

No new models, radii, outcomes, controls, thresholds, station exclusions, or
temporal windows are introduced.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from wildboottest.wildboottest import wildboottest


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

P52 = D / "c2ze_groundwater_protocol" / "c2ze_groundwater_association_protocol.json"
I55 = D / "c2zj_inference_freeze" / "c2zj_exact_inference_implementation.json"
R56 = D / "c2zk_primary_association" / "c2zk_primary_association_result.csv"
Q56 = D / "c2zk_primary_association" / "c2zk_primary_association_qa.json"

FULL_PANEL = (
    D / "c2zg_groundwater_panel"
    / "c2zg_groundwater_s1_analysis_panel_2015_2025.csv"
)
PRIMARY_PANEL = (
    D / "c2zh_groundwater_panel_freeze"
    / "c2zh_frozen_primary_groundwater_s1_panel.csv"
)

OUT = D / "c2zl_prespecified_sensitivities"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = OUT / "c2zl_prespecified_sensitivity_results.csv"
LOO_OUT = OUT / "c2zl_primary_leave_one_well_out.csv"
QA_OUT = OUT / "c2zl_prespecified_sensitivity_qa.json"
SUMMARY_OUT = OUT / "c2zl_prespecified_sensitivity_summary.txt"

for f in [P52, I55, R56, Q56, FULL_PANEL, PRIMARY_PANEL]:
    if not f.exists():
        raise FileNotFoundError(f)

p52 = json.loads(P52.read_text(encoding="utf-8"))
i55 = json.loads(I55.read_text(encoding="utf-8"))
q56 = json.loads(Q56.read_text(encoding="utf-8"))
r56 = pd.read_csv(R56)

if q56.get("status") != "PASS_PRIMARY_ASSOCIATION_REVEALED_NO_MODEL_SEARCH":
    raise RuntimeError(f"Unexpected Stage-56 status: {q56.get('status')!r}")

if len(r56) != 1:
    raise RuntimeError("Stage-56 primary result must contain exactly one row.")

PRIMARY_BETA = float(r56.loc[0, "beta_hat"])

SEED = int(i55["wild_cluster_bootstrap"]["seed"])
B = int(i55["wild_cluster_bootstrap"]["replications"])
BOOTSTRAP_TYPE = str(i55["wild_cluster_bootstrap"]["bootstrap_type"])
WEIGHTS_TYPE = str(i55["wild_cluster_bootstrap"]["weights_type"])
IMPOSE_NULL = bool(i55["wild_cluster_bootstrap"]["impose_null"])

if (SEED, B, BOOTSTRAP_TYPE, WEIGHTS_TYPE, IMPOSE_NULL) != (
    52026, 9999, "31", "webb", True
):
    raise RuntimeError("Frozen Stage-55 inference configuration mismatch.")

OUTCOME = "gw_aug_m"
ANTECEDENT = "gw_pre_last_janfeb_m"
PRIMARY_EXPOSURE = "s1_flooding_like_10km"

full = pd.read_csv(FULL_PANEL, low_memory=False)
primary = pd.read_csv(PRIMARY_PANEL, low_memory=False)

for d in [full, primary]:
    d["station"] = d["station"].astype(str)
    d["year"] = pd.to_numeric(d["year"], errors="raise").astype(int)

numeric_cols = [
    OUTCOME,
    ANTECEDENT,
    "s1_flooding_like_5km",
    "s1_flooding_like_10km",
    "s1_flooding_like_20km",
    "s1_flooding_like_complete4_10km",
]
for col in numeric_cols:
    if col in full.columns:
        full[col] = pd.to_numeric(full[col], errors="coerce")
    if col in primary.columns:
        primary[col] = pd.to_numeric(primary[col], errors="coerce")

def extract_bootstrap_pvalue(result: pd.DataFrame, param: str) -> float:
    if param in result.index and "p-value" in result.columns:
        p = float(result.loc[param, "p-value"])
    elif "param" in result.columns and "p-value" in result.columns:
        row = result.loc[result["param"].astype(str) == param]
        if len(row) != 1:
            raise RuntimeError(f"Bootstrap parameter {param!r} not unique.")
        p = float(row["p-value"].iloc[0])
    else:
        raise RuntimeError("Unsupported wildboottest output layout.")
    if not np.isfinite(p) or not (0 <= p <= 1):
        raise RuntimeError(f"Invalid bootstrap p-value: {p!r}")
    return p

def run_model(name: str, d: pd.DataFrame, formula: str, exposure: str) -> dict:
    d = d.sort_values(["station", "year"]).reset_index(drop=True)

    if d.duplicated(["station", "year"]).any():
        raise RuntimeError(f"{name}: duplicate station-year keys.")

    needed = [OUTCOME, exposure]
    if ANTECEDENT in formula:
        needed.append(ANTECEDENT)

    if d[needed].isna().any().any():
        raise RuntimeError(f"{name}: missing model value after eligibility filter.")

    model = smf.ols(formula, data=d)
    fit = model.fit()

    X = np.asarray(fit.model.exog, dtype=float)
    rank = int(np.linalg.matrix_rank(X))
    ncols = int(X.shape[1])
    if rank != ncols:
        raise RuntimeError(f"{name}: design rank failure {rank}/{ncols}")

    if exposure not in fit.params.index:
        raise RuntimeError(f"{name}: exposure coefficient missing.")

    beta = float(fit.params[exposure])
    G = int(d["station"].nunique())
    if G < 2:
        raise RuntimeError(f"{name}: fewer than 2 clusters.")
    df_cluster = G - 1

    crv1 = fit.get_robustcov_results(
        cov_type="cluster",
        groups=d["station"],
        use_correction=True,
        df_correction=True,
    )
    names = list(fit.model.exog_names)
    idx = names.index(exposure)
    se = float(np.asarray(crv1.bse)[idx])
    if not np.isfinite(se) or se <= 0:
        raise RuntimeError(f"{name}: invalid CRV1 SE.")

    t = beta / se
    p = float(2 * stats.t.sf(abs(t), df_cluster))
    tcrit = float(stats.t.ppf(0.975, df_cluster))
    ci_lo = float(beta - tcrit * se)
    ci_hi = float(beta + tcrit * se)

    cluster_codes, levels = pd.factorize(d["station"], sort=True)
    cluster = np.asarray(cluster_codes, dtype=np.int64)
    if len(levels) != G:
        raise RuntimeError(f"{name}: cluster coding mismatch.")

    wb = wildboottest(
        model,
        param=exposure,
        cluster=cluster,
        B=B,
        bootstrap_type=BOOTSTRAP_TYPE,
        impose_null=IMPOSE_NULL,
        weights_type=WEIGHTS_TYPE,
        seed=SEED,
        parallel=False,
        show=False,
    )
    wb_p = extract_bootstrap_pvalue(wb, exposure)

    return {
        "analysis": name,
        "formula": formula,
        "exposure": exposure,
        "n": int(fit.nobs),
        "n_wells": G,
        "n_years": int(d["year"].nunique()),
        "first_year": int(d["year"].min()),
        "last_year": int(d["year"].max()),
        "design_rank": rank,
        "design_columns": ncols,
        "beta_hat": beta,
        "change_from_primary_beta": beta - PRIMARY_BETA,
        "crv1_se": se,
        "crv1_t": t,
        "crv1_df": df_cluster,
        "crv1_p_two_sided": p,
        "crv1_ci95_low": ci_lo,
        "crv1_ci95_high": ci_hi,
        "wcr31_webb_p_two_sided": wb_p,
        "wcr31_B": B,
        "wcr31_seed": SEED,
    }

# Verify Stage-52 sensitivity names/formulas are still exactly the expected set.
frozen = {
    x["name"]: x for x in p52["prespecified_sensitivity_analyses"]
    if "formula" in x
}

expected_formulas = {
    "5km_spatial_scale":
        "gw_aug_m ~ s1_flooding_like_5km + gw_pre_last_janfeb_m + C(station) + C(year)",
    "20km_spatial_scale":
        "gw_aug_m ~ s1_flooding_like_20km + gw_pre_last_janfeb_m + C(station) + C(year)",
    "complete4_track_support":
        "gw_aug_m ~ s1_flooding_like_complete4_10km + gw_pre_last_janfeb_m + C(station) + C(year)",
    "without_antecedent_groundwater":
        "gw_aug_m ~ s1_flooding_like_10km + C(station) + C(year)",
}

for k, formula in expected_formulas.items():
    if k not in frozen:
        raise RuntimeError(f"Missing frozen Stage-52 sensitivity {k!r}.")
    if frozen[k]["formula"] != formula:
        raise RuntimeError(f"Frozen formula mismatch for {k!r}.")

results = []

# 1. 5 km — mechanically eligible sensitivity sample.
d5 = full.loc[
    np.isfinite(full[OUTCOME])
    & np.isfinite(full[ANTECEDENT])
    & np.isfinite(full["s1_flooding_like_5km"])
].copy()
results.append(
    run_model(
        "5km_spatial_scale",
        d5,
        expected_formulas["5km_spatial_scale"],
        "s1_flooding_like_5km",
    )
)

# 2. 20 km — mechanically eligible sensitivity sample.
d20 = full.loc[
    np.isfinite(full[OUTCOME])
    & np.isfinite(full[ANTECEDENT])
    & np.isfinite(full["s1_flooding_like_20km"])
].copy()
results.append(
    run_model(
        "20km_spatial_scale",
        d20,
        expected_formulas["20km_spatial_scale"],
        "s1_flooding_like_20km",
    )
)

# 3. Complete-four-track measurement support at 10 km.
d4 = full.loc[
    np.isfinite(full[OUTCOME])
    & np.isfinite(full[ANTECEDENT])
    & np.isfinite(full["s1_flooding_like_complete4_10km"])
].copy()
results.append(
    run_model(
        "complete4_track_support",
        d4,
        expected_formulas["complete4_track_support"],
        "s1_flooding_like_complete4_10km",
    )
)

# 4. No-antecedent specification, exact frozen primary sample retained.
if len(primary) != 193 or primary["station"].nunique() != 32:
    raise RuntimeError("Frozen primary panel structure changed.")
if primary[[OUTCOME, PRIMARY_EXPOSURE]].isna().any().any():
    raise RuntimeError("Primary sample has missing Y/exposure.")
results.append(
    run_model(
        "without_antecedent_groundwater",
        primary.copy(),
        expected_formulas["without_antecedent_groundwater"],
        PRIMARY_EXPOSURE,
    )
)

res = pd.DataFrame(results)
res.to_csv(RESULTS_OUT, index=False)

# 5. Leave-one-well-out primary coefficient diagnostic.
primary_formula = p52["primary_model"]["formula"]
loo_rows = []
stations = sorted(primary["station"].unique())

for omitted in stations:
    d = primary.loc[primary["station"] != omitted].copy()
    fit = smf.ols(primary_formula, data=d).fit()
    if PRIMARY_EXPOSURE not in fit.params.index:
        raise RuntimeError(f"LOO omitting {omitted}: coefficient missing.")
    b = float(fit.params[PRIMARY_EXPOSURE])
    if not np.isfinite(b):
        raise RuntimeError(f"LOO omitting {omitted}: non-finite beta.")
    loo_rows.append({
        "omitted_station": omitted,
        "remaining_n": int(fit.nobs),
        "remaining_wells": int(d["station"].nunique()),
        "beta_hat": b,
        "change_from_primary_beta": b - PRIMARY_BETA,
        "sign": "negative" if b < 0 else "positive" if b > 0 else "zero",
    })

loo = pd.DataFrame(loo_rows).sort_values("omitted_station")
loo.to_csv(LOO_OUT, index=False)

loo_min = float(loo["beta_hat"].min())
loo_max = float(loo["beta_hat"].max())
loo_sign_negative_n = int((loo["beta_hat"] < 0).sum())
loo_sign_positive_n = int((loo["beta_hat"] > 0).sum())

qa = {
    "stage": "DESIGN_C_STAGE57_PRESPECIFIED_SENSITIVITY_EXECUTION",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS_PRESPECIFIED_SENSITIVITIES_EXECUTED_NO_NEW_MODEL_SEARCH",
    "primary_result_unchanged": True,
    "primary_beta_reference": PRIMARY_BETA,
    "analyses_executed": res["analysis"].tolist(),
    "leave_one_well_out_fits_n": int(len(loo)),
    "new_model_or_radius_added_after_primary_reveal": False,
    "result_driven_station_exclusion": False,
    "sensitivity_result_promoted_over_primary": False,
    "sha256": {
        "stage52_protocol": hashlib.sha256(P52.read_bytes()).hexdigest(),
        "stage55_inference_freeze": hashlib.sha256(I55.read_bytes()).hexdigest(),
        "stage56_primary_result": hashlib.sha256(R56.read_bytes()).hexdigest(),
        "stage56_primary_qa": hashlib.sha256(Q56.read_bytes()).hexdigest(),
        "full_groundwater_s1_panel": hashlib.sha256(FULL_PANEL.read_bytes()).hexdigest(),
        "frozen_primary_panel": hashlib.sha256(PRIMARY_PANEL.read_bytes()).hexdigest(),
        "sensitivity_results": hashlib.sha256(RESULTS_OUT.read_bytes()).hexdigest(),
        "loo_results": hashlib.sha256(LOO_OUT.read_bytes()).hexdigest(),
    },
}
QA_OUT.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "DESIGN C - STAGE 57 PRESPECIFIED SENSITIVITY ANALYSES",
    "=" * 72,
    "",
    "PRIMARY RESULT REFERENCE — FIXED, NOT RE-ESTIMATED",
    "--------------------------------------------------",
    f"Stage-56 beta_hat: {PRIMARY_BETA:.12g}",
    "",
    "PRESPECIFIED MODEL SENSITIVITIES",
    "--------------------------------",
]

for r in results:
    lines += [
        f"{r['analysis']}:",
        f"  N={r['n']}, wells={r['n_wells']}, years={r['first_year']}-{r['last_year']}",
        f"  beta={r['beta_hat']:.12g}",
        f"  change from primary beta={r['change_from_primary_beta']:.12g}",
        f"  CRV1 SE={r['crv1_se']:.12g}",
        f"  CRV1 p={r['crv1_p_two_sided']:.12g}",
        f"  CRV1 95% CI=[{r['crv1_ci95_low']:.12g}, {r['crv1_ci95_high']:.12g}]",
        f"  WCR31-Webb p={r['wcr31_webb_p_two_sided']:.12g}",
    ]

lines += [
    "",
    "LEAVE-ONE-WELL-OUT PRIMARY COEFFICIENT DIAGNOSTIC",
    "-------------------------------------------------",
    f"Fits: {len(loo)}",
    f"Coefficient range: [{loo_min:.12g}, {loo_max:.12g}]",
    f"Negative coefficients: {loo_sign_negative_n}/{len(loo)}",
    f"Positive coefficients: {loo_sign_positive_n}/{len(loo)}",
    "",
    "INTERPRETIVE GUARDRAILS",
    "-----------------------",
    "Primary Stage-56 result remains primary regardless of these sensitivities.",
    "No sensitivity is a basis for excluding wells or redesigning the model.",
    "All estimates remain observational and non-causal.",
    "",
    "STAGE 57 STATUS: PASS_PRESPECIFIED_SENSITIVITIES_EXECUTED_NO_NEW_MODEL_SEARCH",
]

SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
