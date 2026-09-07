"""
Design C — Stage 55
Freeze and smoke-test the exact inferential implementation BEFORE the first
Design-C groundwater–Sentinel-1 association coefficient is fitted.

This stage uses SYNTHETIC DATA ONLY. It does not read the frozen Design-C
groundwater panel or any real groundwater/S1 values.

Relationship to Stage 52
------------------------
Stage 52 already froze the scientific inferential roles:
- primary covariance: cluster-robust, clustered by station/well;
- wild-cluster bootstrap-t robustness inference:
  Webb weights, 9,999 replications, seed 52026.

Stage 55 resolves only implementation details that Stage 52 left unspecified,
using the already validated post-2021 production implementation as the
outcome-independent precedent.

Frozen implementation
---------------------
Primary cluster-robust covariance:
- statsmodels OLS;
- cov_type="cluster";
- groups=station;
- use_correction=True;
- df_correction=True;
- two-sided t inference with df = G - 1.

Wild-cluster bootstrap robustness:
- package: wildboottest (exact installed version recorded);
- bootstrap_type="31" (WCR31);
- impose_null=True;
- weights_type="webb";
- B=9999;
- seed=52026;
- cluster=station numeric codes;
- parallel=False;
- package-returned two-sided p-value is reported without analyst modification.

Fixed effects:
- statsmodels formula categorical indicators C(station) + C(year).

No association coefficient is fitted or reported in this stage.
"""

from __future__ import annotations

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
OUT = (
    ROOT / "outputs" / "diagnostics" / "design_c"
    / "c2zj_inference_freeze"
)
OUT.mkdir(parents=True, exist_ok=True)

PROTOCOL_OUT = OUT / "c2zj_exact_inference_implementation.json"
SUMMARY_OUT = OUT / "c2zj_exact_inference_implementation.txt"
SMOKE_OUT = OUT / "c2zj_synthetic_inference_smoke_test.json"

SEED = 52026
B = 9999
BOOTSTRAP_TYPE = "31"
WEIGHTS_TYPE = "webb"
IMPOSE_NULL = True

# Synthetic-only dimensions chosen to match the real number of clusters and
# approximately match the real row count, while never reading real panel keys.
G = 32
YEARS = tuple(range(2015, 2026))
TARGET_N = 193

PARAM = "s1_synth"
FORMULA = (
    "gw_synth ~ s1_synth + antecedent_synth "
    "+ C(station) + C(year)"
)


def pkgver(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Required package {name!r} is not installed.") from exc


def build_synthetic_panel() -> pd.DataFrame:
    """Deterministic irregular 32-cluster synthetic panel with 193 rows."""
    rng = np.random.default_rng(SEED)

    stations = [f"S{i:02d}" for i in range(1, G + 1)]
    all_pairs = [(s, y) for s in stations for y in YEARS]

    # Deterministically retain 193 synthetic station-year pairs.
    # Guarantee >=2 rows per station and all years represented.
    mandatory = []
    for i, s in enumerate(stations):
        mandatory.append((s, YEARS[i % len(YEARS)]))
        mandatory.append((s, YEARS[(i + 3) % len(YEARS)]))

    mandatory = list(dict.fromkeys(mandatory))
    remaining = [p for p in all_pairs if p not in set(mandatory)]
    rng.shuffle(remaining)
    pairs = mandatory + remaining[: TARGET_N - len(mandatory)]

    df = pd.DataFrame(pairs, columns=["station", "year"])
    df = df.sort_values(["station", "year"]).reset_index(drop=True)

    station_effect = {
        s: v for s, v in zip(stations, rng.normal(0, 0.6, G))
    }
    year_effect = {
        y: v for y, v in zip(YEARS, rng.normal(0, 0.25, len(YEARS)))
    }

    rows = []
    for r in df.itertuples(index=False):
        s1 = rng.normal(0, 0.8)
        a = rng.normal(0, 1.0)
        eps = rng.normal(0, 0.55)
        gw = (
            0.35 * s1
            + 0.40 * a
            + station_effect[r.station]
            + year_effect[r.year]
            + eps
        )
        rows.append(
            {
                "station": r.station,
                "year": int(r.year),
                "s1_synth": float(s1),
                "antecedent_synth": float(a),
                "gw_synth": float(gw),
            }
        )

    d = pd.DataFrame(rows)

    if len(d) != TARGET_N:
        raise AssertionError(f"Synthetic N {len(d)} != {TARGET_N}")
    if d["station"].nunique() != G:
        raise AssertionError("Synthetic cluster count mismatch.")
    if sorted(d["year"].unique()) != list(YEARS):
        raise AssertionError("Synthetic years are incomplete.")
    if d.duplicated(["station", "year"]).any():
        raise AssertionError("Synthetic station-year duplication.")
    if (d.groupby("station").size() < 2).any():
        raise AssertionError("Synthetic station has <2 rows.")

    return d


def extract_pvalue(result: pd.DataFrame, param: str) -> float:
    if param in result.index and "p-value" in result.columns:
        p = float(result.loc[param, "p-value"])
    elif "param" in result.columns and "p-value" in result.columns:
        row = result.loc[result["param"].astype(str) == param]
        if len(row) != 1:
            raise RuntimeError("Could not uniquely identify bootstrap parameter.")
        p = float(row["p-value"].iloc[0])
    else:
        raise RuntimeError("Unsupported wildboottest output layout.")

    if not np.isfinite(p) or not 0 <= p <= 1:
        raise RuntimeError(f"Invalid bootstrap p-value: {p!r}")
    return p


def run_wcr31(model, cluster: np.ndarray) -> float:
    res = wildboottest(
        model,
        param=PARAM,
        cluster=cluster,
        B=B,
        bootstrap_type=BOOTSTRAP_TYPE,
        impose_null=IMPOSE_NULL,
        weights_type=WEIGHTS_TYPE,
        seed=SEED,
        parallel=False,
        show=False,
    )
    return extract_pvalue(res, PARAM)


def main() -> None:
    wb_version = pkgver("wildboottest")

    d = build_synthetic_panel()

    model = smf.ols(FORMULA, data=d)
    fit = model.fit()

    exog = np.asarray(fit.model.exog, dtype=float)
    rank = int(np.linalg.matrix_rank(exog))
    ncols = int(exog.shape[1])

    if rank != ncols:
        raise RuntimeError(
            f"Synthetic design rank deficient: {rank}/{ncols}"
        )

    # Exact primary CRV1 implementation.
    crv1 = fit.get_robustcov_results(
        cov_type="cluster",
        groups=d["station"],
        use_correction=True,
        df_correction=True,
    )

    names = list(fit.model.exog_names)
    idx = names.index(PARAM)
    se = float(np.asarray(crv1.bse)[idx])

    if not np.isfinite(se) or se <= 0:
        raise RuntimeError(f"Invalid synthetic CRV1 SE: {se!r}")

    df_cluster = G - 1
    beta = float(fit.params[PARAM])
    tstat = beta / se
    p_crv1 = float(2 * stats.t.sf(abs(tstat), df_cluster))
    tcrit = float(stats.t.ppf(0.975, df_cluster))
    ci = [float(beta - tcrit * se), float(beta + tcrit * se)]

    cluster_codes, levels = pd.factorize(d["station"], sort=True)
    cluster = np.asarray(cluster_codes, dtype=np.int64)

    if len(levels) != G:
        raise RuntimeError("Synthetic bootstrap cluster count mismatch.")

    p_wb1 = run_wcr31(model, cluster)
    p_wb2 = run_wcr31(model, cluster)

    if p_wb1 != p_wb2:
        raise RuntimeError(
            f"WCR31 fixed-seed reproducibility failed: {p_wb1} vs {p_wb2}"
        )

    protocol = {
        "stage": "DESIGN_C_STAGE55_EXACT_INFERENCE_IMPLEMENTATION_FREEZE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INFERENCE_IMPLEMENTATION_FROZEN_NO_REAL_ASSOCIATION_FIT",
        "scientific_parent_protocol": "Stage 52",
        "outcome_values_read": False,
        "real_groundwater_s1_panel_read": False,
        "association_coefficient_fitted": False,
        "primary_cluster_robust_inference": {
            "implementation": "statsmodels OLS robust covariance",
            "cov_type": "cluster",
            "cluster_variable": "station",
            "use_correction": True,
            "df_correction": True,
            "reference_distribution": "Student t",
            "degrees_of_freedom_rule": "G - 1",
            "two_sided_pvalue_rule": "2 * scipy.stats.t.sf(abs(t), G - 1)",
            "confidence_interval": "beta +/- t_0.975,G-1 * SE",
            "role": "primary cluster-robust uncertainty",
        },
        "wild_cluster_bootstrap": {
            "implementation": "wildboottest.wildboottest",
            "bootstrap_type": BOOTSTRAP_TYPE,
            "label": "WCR31",
            "cluster_variable": "station",
            "cluster_encoding": "deterministic numeric factor codes sorted by station",
            "weights_type": WEIGHTS_TYPE,
            "impose_null": IMPOSE_NULL,
            "replications": B,
            "seed": SEED,
            "parallel": False,
            "pvalue_source": (
                "two-sided p-value returned by the pinned wildboottest "
                "implementation; no analyst-side modification"
            ),
            "role": "prespecified robustness inference",
        },
        "fixed_effects": {
            "implementation": "statsmodels formula categorical indicators",
            "station": "C(station)",
            "year": "C(year)",
        },
        "software": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "wildboottest": wb_version,
        },
        "precedent": {
            "source_script":
                "scripts/05_post2021_flooding/"
                "23_run_primary_groundwater_extension_2022_2025.py",
            "reason":
                "reuse previously validated outcome-independent inferential "
                "implementation rather than choose a new method after seeing "
                "Design-C groundwater results",
        },
    }

    smoke = {
        "status": "PASS_SYNTHETIC_ONLY",
        "n": int(len(d)),
        "clusters": int(d["station"].nunique()),
        "years": [int(y) for y in sorted(d["year"].unique())],
        "design_columns": ncols,
        "design_rank": rank,
        "synthetic_crv1_se": se,
        "synthetic_crv1_p": p_crv1,
        "synthetic_crv1_ci": ci,
        "synthetic_wcr31_p_run1": p_wb1,
        "synthetic_wcr31_p_run2": p_wb2,
        "fixed_seed_exactly_reproducible": p_wb1 == p_wb2,
        "scientific_interpretation":
            "NONE_SYNTHETIC_SMOKE_TEST_ONLY",
        "real_association_coefficient": "NOT_FITTED_NOT_REVEALED",
    }

    PROTOCOL_OUT.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    SMOKE_OUT.write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "DESIGN C - STAGE 55 EXACT INFERENCE IMPLEMENTATION FREEZE",
        "=" * 72,
        "",
        "PRIMARY CLUSTER-ROBUST INFERENCE",
        "--------------------------------",
        'cov_type="cluster"',
        'groups=station',
        "use_correction=True",
        "df_correction=True",
        f"reference df = G - 1",
        "",
        "WILD CLUSTER BOOTSTRAP",
        "----------------------",
        f"type=WCR{BOOTSTRAP_TYPE}",
        f"weights={WEIGHTS_TYPE}",
        f"impose_null={IMPOSE_NULL}",
        f"B={B}",
        f"seed={SEED}",
        "parallel=False",
        f"wildboottest version={wb_version}",
        "",
        "SYNTHETIC SMOKE TEST",
        "--------------------",
        f"rows={len(d)}",
        f"clusters={d['station'].nunique()}",
        f"design rank={rank}/{ncols}",
        f"WCR31 p run 1={p_wb1:.12g}",
        f"WCR31 p run 2={p_wb2:.12g}",
        f"fixed-seed reproducible={p_wb1 == p_wb2}",
        "",
        "FIREWALL",
        "--------",
        "Real Design-C groundwater-S1 panel read: False",
        "Association coefficient fitted: False",
        "Association coefficient reported: False",
        "",
        "STAGE 55 STATUS: "
        "INFERENCE_IMPLEMENTATION_FROZEN_NO_REAL_ASSOCIATION_FIT",
    ]

    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
