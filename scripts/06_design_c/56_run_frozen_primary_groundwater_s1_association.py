"""
Design C — Stage 56
Controlled first real-data reveal of the frozen groundwater–Sentinel-1
association.

This is the FIRST script allowed to fit and report the Design-C primary
groundwater association coefficient.

Hard gates
----------
- Stage 53C frozen panel manifest must have passed.
- Stage 54 independent panel audit must have passed.
- Stage 55 exact inference implementation freeze must have passed.
- Stage 55 synthetic smoke test must have passed.
- Frozen primary panel must contain exactly 193 rows, 32 wells, 2015–2025.
- Exact Stage-52 primary formula only.
- No alternate outcome, exposure, radius, controls, station exclusions,
  temporal windows, or inference choices are searched.

Primary model
-------------
gw_aug_m
    ~ s1_flooding_like_10km
    + gw_pre_last_janfeb_m
    + C(station)
    + C(year)

Primary inference
-----------------
- central OLS coefficient on s1_flooding_like_10km
- CRV1 cluster-robust SE by station:
    cov_type="cluster"
    use_correction=True
    df_correction=True
- two-sided Student-t inference with df = G - 1 = 31
- 95% t confidence interval
- prespecified WCR31 wild-cluster bootstrap robustness test:
    Webb weights
    impose_null=True
    B=9999
    seed=52026
    cluster=station
    parallel=False

Interpretation
--------------
Groundwater depth larger = deeper groundwater.
S1 exposure larger = more RiceFloodIT-consistent flooding-like signal.

beta < 0:
    more flooding-like signal associated with shallower late-season groundwater.
beta > 0:
    more flooding-like signal associated with deeper late-season groundwater.

Observational association only; no causal interpretation.

Outputs
-------
outputs/diagnostics/design_c/c2zk_primary_association/
    c2zk_primary_association_result.csv
    c2zk_primary_association_qa.json
    c2zk_primary_association_summary.txt
    c2zk_primary_model_software.json
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf
from wildboottest.wildboottest import wildboottest


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
A54 = (
    D / "c2zi_primary_panel_audit"
    / "c2zi_primary_panel_audit.json"
)
I55 = (
    D / "c2zj_inference_freeze"
    / "c2zj_exact_inference_implementation.json"
)
S55 = (
    D / "c2zj_inference_freeze"
    / "c2zj_synthetic_inference_smoke_test.json"
)
PANEL = (
    D / "c2zh_groundwater_panel_freeze"
    / "c2zh_frozen_primary_groundwater_s1_panel.csv"
)

OUT = D / "c2zk_primary_association"
OUT.mkdir(parents=True, exist_ok=True)

RESULT_OUT = OUT / "c2zk_primary_association_result.csv"
QA_OUT = OUT / "c2zk_primary_association_qa.json"
SUMMARY_OUT = OUT / "c2zk_primary_association_summary.txt"
SOFTWARE_OUT = OUT / "c2zk_primary_model_software.json"

for f in [P52, M53, A54, I55, S55, PANEL]:
    if not f.exists():
        raise FileNotFoundError(f)

p52 = json.loads(P52.read_text(encoding="utf-8"))
m53 = json.loads(M53.read_text(encoding="utf-8"))
a54 = json.loads(A54.read_text(encoding="utf-8"))
i55 = json.loads(I55.read_text(encoding="utf-8"))
s55 = json.loads(S55.read_text(encoding="utf-8"))

if p52.get("status") != "PROTOCOL_FROZEN_GROUNDWATER_VALUES_UNREAD":
    raise RuntimeError(f"Unexpected Stage-52 status: {p52.get('status')!r}")

if m53.get("status") != \
        "PASS_PRIMARY_PANEL_FROZEN_WITH_DOCUMENTED_STRUCTURAL_NONOVERLAP":
    raise RuntimeError(f"Unexpected Stage-53C status: {m53.get('status')!r}")

if a54.get("status") != \
        "PASS_PRIMARY_PANEL_AUDITED_AND_READY_FOR_FIRST_ASSOCIATION_FIT":
    raise RuntimeError(f"Unexpected Stage-54 status: {a54.get('status')!r}")

if i55.get("status") != \
        "INFERENCE_IMPLEMENTATION_FROZEN_NO_REAL_ASSOCIATION_FIT":
    raise RuntimeError(f"Unexpected Stage-55 protocol status: {i55.get('status')!r}")

if s55.get("status") != "PASS_SYNTHETIC_ONLY":
    raise RuntimeError(f"Unexpected Stage-55 smoke status: {s55.get('status')!r}")

if s55.get("fixed_seed_exactly_reproducible") is not True:
    raise RuntimeError("Stage-55 bootstrap reproducibility gate did not pass.")

OUTCOME = "gw_aug_m"
EXPOSURE = "s1_flooding_like_10km"
ANTECEDENT = "gw_pre_last_janfeb_m"

FORMULA = (
    f"{OUTCOME} ~ {EXPOSURE} + {ANTECEDENT} "
    "+ C(station) + C(year)"
)

if p52["primary_model"]["formula"] != FORMULA:
    raise RuntimeError(
        "Local formula differs from frozen Stage-52 primary formula."
    )

SEED = int(i55["wild_cluster_bootstrap"]["seed"])
B = int(i55["wild_cluster_bootstrap"]["replications"])
BOOTSTRAP_TYPE = str(i55["wild_cluster_bootstrap"]["bootstrap_type"])
WEIGHTS_TYPE = str(i55["wild_cluster_bootstrap"]["weights_type"])
IMPOSE_NULL = bool(i55["wild_cluster_bootstrap"]["impose_null"])

if (SEED, B, BOOTSTRAP_TYPE, WEIGHTS_TYPE, IMPOSE_NULL) != \
        (52026, 9999, "31", "webb", True):
    raise RuntimeError("Stage-55 inference configuration mismatch.")

def pkgver(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Required package {name!r} missing.") from exc

def extract_bootstrap_pvalue(result: pd.DataFrame, param: str) -> float:
    if param in result.index and "p-value" in result.columns:
        p = float(result.loc[param, "p-value"])
    elif "param" in result.columns and "p-value" in result.columns:
        row = result.loc[result["param"].astype(str) == param]
        if len(row) != 1:
            raise RuntimeError(
                f"Could not uniquely identify bootstrap result for {param!r}."
            )
        p = float(row["p-value"].iloc[0])
    else:
        raise RuntimeError("Unsupported wildboottest output layout.")

    if not np.isfinite(p) or not 0 <= p <= 1:
        raise RuntimeError(f"Invalid bootstrap p-value: {p!r}")
    return p

panel = pd.read_csv(PANEL, low_memory=False)
panel["station"] = panel["station"].astype(str)
panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)

for col in [OUTCOME, EXPOSURE, ANTECEDENT]:
    panel[col] = pd.to_numeric(panel[col], errors="coerce")

hard_flags = []

if len(panel) != 193:
    hard_flags.append(f"N:{len(panel)}!=193")
if panel["station"].nunique() != 32:
    hard_flags.append(f"WELLS:{panel['station'].nunique()}!=32")
if sorted(panel["year"].unique().tolist()) != list(range(2015, 2026)):
    hard_flags.append("YEAR_SET_MISMATCH")
if panel.duplicated(["station", "year"]).any():
    hard_flags.append("DUPLICATE_STATION_YEAR")
if panel[[OUTCOME, EXPOSURE, ANTECEDENT]].isna().any().any():
    hard_flags.append("MISSING_PRIMARY_MODEL_VALUE")
if not np.isfinite(panel[[OUTCOME, EXPOSURE, ANTECEDENT]].to_numpy(float)).all():
    hard_flags.append("NONFINITE_PRIMARY_MODEL_VALUE")

if hard_flags:
    raise RuntimeError("Pre-fit hard QA failed: " + "; ".join(hard_flags))

# Freeze exact row order for deterministic execution.
panel = panel.sort_values(["station", "year"]).reset_index(drop=True)

model = smf.ols(FORMULA, data=panel)
fit = model.fit()

X = np.asarray(fit.model.exog, dtype=float)
rank = int(np.linalg.matrix_rank(X))
ncols = int(X.shape[1])

if rank != ncols:
    raise RuntimeError(f"Primary design rank failure: {rank}/{ncols}")

if EXPOSURE not in fit.params.index:
    raise RuntimeError(f"Primary coefficient {EXPOSURE!r} absent from fit.")

beta = float(fit.params[EXPOSURE])
if not np.isfinite(beta):
    raise RuntimeError("Primary beta is non-finite.")

G = int(panel["station"].nunique())
df_cluster = G - 1

crv1 = fit.get_robustcov_results(
    cov_type="cluster",
    groups=panel["station"],
    use_correction=True,
    df_correction=True,
)

names = list(fit.model.exog_names)
idx = names.index(EXPOSURE)

crv1_se = float(np.asarray(crv1.bse)[idx])
if not np.isfinite(crv1_se) or crv1_se <= 0:
    raise RuntimeError(f"Invalid CRV1 SE: {crv1_se!r}")

crv1_t = beta / crv1_se
crv1_p = float(2.0 * stats.t.sf(abs(crv1_t), df_cluster))
tcrit = float(stats.t.ppf(0.975, df_cluster))
crv1_ci_low = float(beta - tcrit * crv1_se)
crv1_ci_high = float(beta + tcrit * crv1_se)

# Prespecified WCR31 robustness inference.
cluster_codes, cluster_levels = pd.factorize(
    panel["station"], sort=True
)
cluster = np.asarray(cluster_codes, dtype=np.int64)

if len(cluster_levels) != G:
    raise RuntimeError("Cluster factorization mismatch.")

wb = wildboottest(
    model,
    param=EXPOSURE,
    cluster=cluster,
    B=B,
    bootstrap_type=BOOTSTRAP_TYPE,
    impose_null=IMPOSE_NULL,
    weights_type=WEIGHTS_TYPE,
    seed=SEED,
    parallel=False,
    show=False,
)

wcr31_p = extract_bootstrap_pvalue(wb, EXPOSURE)

if beta < 0:
    direction = (
        "more flooding-like S1 signal associated with shallower "
        "late-season groundwater"
    )
elif beta > 0:
    direction = (
        "more flooding-like S1 signal associated with deeper "
        "late-season groundwater"
    )
else:
    direction = "estimated linear partial association exactly zero"

result = pd.DataFrame([{
    "stage": "DESIGN_C_STAGE56_PRIMARY_ASSOCIATION_REVEAL",
    "formula": FORMULA,
    "outcome": OUTCOME,
    "exposure": EXPOSURE,
    "antecedent": ANTECEDENT,
    "n": int(fit.nobs),
    "n_wells": G,
    "n_years": int(panel["year"].nunique()),
    "first_year": int(panel["year"].min()),
    "last_year": int(panel["year"].max()),
    "beta_hat": beta,
    "crv1_se": crv1_se,
    "crv1_t": crv1_t,
    "crv1_df": df_cluster,
    "crv1_p_two_sided": crv1_p,
    "crv1_ci95_low": crv1_ci_low,
    "crv1_ci95_high": crv1_ci_high,
    "wcr31_webb_p_two_sided": wcr31_p,
    "wcr31_B": B,
    "wcr31_seed": SEED,
    "direction_interpretation": direction,
    "causal_interpretation_allowed": False,
}])

result.to_csv(RESULT_OUT, index=False)

software = {
    "python": sys.version,
    "python_implementation": platform.python_implementation(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "wildboottest": pkgver("wildboottest"),
}
SOFTWARE_OUT.write_text(
    json.dumps(software, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

qa = {
    "stage": "DESIGN_C_STAGE56_PRIMARY_ASSOCIATION_REVEAL",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS_PRIMARY_ASSOCIATION_REVEALED_NO_MODEL_SEARCH",
    "panel": {
        "n": int(fit.nobs),
        "wells": G,
        "years": sorted(panel["year"].unique().astype(int).tolist()),
        "duplicate_station_years": int(
            panel.duplicated(["station", "year"]).sum()
        ),
    },
    "design_matrix": {
        "columns": ncols,
        "rank": rank,
        "full_rank": rank == ncols,
    },
    "inference": {
        "primary": "CRV1 cluster-robust by station",
        "use_correction": True,
        "df_correction": True,
        "df": df_cluster,
        "bootstrap": "WCR31 Webb null-imposed",
        "bootstrap_B": B,
        "bootstrap_seed": SEED,
    },
    "model_search_performed": False,
    "alternate_radius_examined_in_this_stage": False,
    "alternate_outcome_examined_in_this_stage": False,
    "alternate_controls_examined_in_this_stage": False,
    "station_exclusion_based_on_result": False,
    "causal_interpretation_allowed": False,
    "sha256": {
        "stage52_protocol": hashlib.sha256(P52.read_bytes()).hexdigest(),
        "stage53c_manifest": hashlib.sha256(M53.read_bytes()).hexdigest(),
        "stage54_audit": hashlib.sha256(A54.read_bytes()).hexdigest(),
        "stage55_inference_freeze": hashlib.sha256(I55.read_bytes()).hexdigest(),
        "stage55_smoke_test": hashlib.sha256(S55.read_bytes()).hexdigest(),
        "frozen_primary_panel": hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        "primary_result": hashlib.sha256(RESULT_OUT.read_bytes()).hexdigest(),
    },
}

QA_OUT.write_text(
    json.dumps(qa, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 56 CONTROLLED PRIMARY ASSOCIATION REVEAL",
    "=" * 76,
    "",
    "FROZEN MODEL",
    "------------",
    FORMULA,
    "",
    "SAMPLE",
    "------",
    f"N well-years: {int(fit.nobs)}",
    f"Wells: {G}",
    f"Years: {int(panel['year'].min())}-{int(panel['year'].max())}",
    f"Design rank: {rank}/{ncols}",
    "",
    "PRIMARY ASSOCIATION",
    "-------------------",
    f"beta_hat: {beta:.12g}",
    f"CRV1 SE: {crv1_se:.12g}",
    f"CRV1 t({df_cluster}): {crv1_t:.12g}",
    f"CRV1 two-sided p: {crv1_p:.12g}",
    f"CRV1 95% CI: [{crv1_ci_low:.12g}, {crv1_ci_high:.12g}]",
    f"WCR31-Webb two-sided p: {wcr31_p:.12g}",
    "",
    "SIGN INTERPRETATION",
    "-------------------",
    direction,
    "",
    "SCIENTIFIC INTERPRETATION",
    "-------------------------",
    "Observational partial association only.",
    "No causal interpretation is permitted.",
    "",
    "FIREWALL / SEARCH AUDIT",
    "-----------------------",
    "Model search performed: False",
    "Alternate radius examined in Stage 56: False",
    "Alternate outcome examined in Stage 56: False",
    "Alternate controls examined in Stage 56: False",
    "Result-driven station exclusion: False",
    "",
    "STAGE 56 STATUS: PASS_PRIMARY_ASSOCIATION_REVEALED_NO_MODEL_SEARCH",
]

SUMMARY_OUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))
