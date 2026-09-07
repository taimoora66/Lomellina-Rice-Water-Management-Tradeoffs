"""
Design C — Stage 51 independently audit and freeze the final Sentinel-1
support-year flooding-like exposure before groundwater is opened.

Purpose
-------
Verify that Stage 50 exactly implemented the frozen Stage-49 representation
protocol and freeze the resulting support-year exposure as the sole Sentinel-1
measurement allowed into downstream hydrological analysis.

This stage reads:
- frozen Stage-49 representation protocol
- frozen Stage-44/45 phenology table
- Stage-50 exposure outputs

This stage does NOT read:
- groundwater values
- irrigation-flow values
- raw RiceFloodIT values

Checks
------
1. 47,641 exact support-year rows (4,331 x 11 years)
2. no duplicate support-year keys
3. all support coordinates stable
4. >=3-track primary eligibility logic exact
5. complete-4-track sensitivity logic exact
6. track-specific robust-standardization parameters exactly recomputed
7. deterministic recomputation of 7 fixed support IDs x 11 years = 77 cases
8. Stage-50 output SHA256 frozen
9. no hydrological-outcome firewall violations

Outputs
-------
outputs/diagnostics/design_c/c2zd_exposure_freeze/
    c2zd_cross_track_exposure_independent_audit.csv
    c2zd_cross_track_exposure_recompute_checks.csv
    c2zd_frozen_s1_exposure_manifest.json
    c2zd_frozen_s1_exposure_summary.txt

Run
---
python -u scripts/06_design_c/51_audit_and_freeze_cross_track_s1_exposure.py
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

PHENO = (
    D / "c2x_phenology"
    / "c2x_s1_track_year_phenology_features.csv.gz"
)
P49 = (
    D / "c2zb_representation_protocol"
    / "c2zb_cross_track_representation_protocol.json"
)
S50 = D / "c2zc_cross_track_exposure"
PARAM50 = S50 / "c2zc_track_standardization_parameters.csv"
EXPOSURE50 = S50 / "c2zc_support_year_flooding_like_exposure.csv.gz"
QA50 = S50 / "c2zc_cross_track_exposure_qa.json"

OUT = D / "c2zd_exposure_freeze"
OUT.mkdir(parents=True, exist_ok=True)

AUDIT_OUT = OUT / "c2zd_cross_track_exposure_independent_audit.csv"
RECHECK_OUT = OUT / "c2zd_cross_track_exposure_recompute_checks.csv"
MANIFEST_OUT = OUT / "c2zd_frozen_s1_exposure_manifest.json"
SUMMARY_OUT = OUT / "c2zd_frozen_s1_exposure_summary.txt"

EXPECTED_SUPPORT_N = 4331
EXPECTED_YEARS = list(range(2015, 2026))
EXPECTED_TRACKS = {
    ("ascending", 15): "asc15",
    ("ascending", 88): "asc88",
    ("descending", 66): "desc66",
    ("descending", 168): "desc168",
}
EXPECTED_SUPPORT_YEAR_ROWS = EXPECTED_SUPPORT_N * len(EXPECTED_YEARS)
TOL = 1e-9

for f in [PHENO, P49, PARAM50, EXPOSURE50, QA50]:
    if not f.exists():
        raise FileNotFoundError(f)

protocol = json.loads(P49.read_text(encoding="utf-8"))
qa50 = json.loads(QA50.read_text(encoding="utf-8"))
params50 = pd.read_csv(PARAM50)
exp = pd.read_csv(EXPOSURE50, compression="gzip", low_memory=False)
ph = pd.read_csv(PHENO, compression="gzip", low_memory=False)

flags = []

def flag(code, condition):
    if bool(condition):
        flags.append(code)

def same(a, b, tol=TOL):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tol

# ---------------------------------------------------------------------
# Protocol / Stage-50 status
# ---------------------------------------------------------------------
flag(
    "STAGE49_PROTOCOL_STATUS_INVALID",
    protocol.get("status") != "PROTOCOL_FROZEN_NO_HYDROLOGICAL_OUTCOME_READ"
)
flag(
    "STAGE50_STATUS_INVALID",
    qa50.get("status") != "PASS_FROZEN_CROSS_TRACK_EXPOSURE_CONSTRUCTED"
)

feature = str(protocol["source_feature"])
orientation = int(
    protocol["construct_orientation"]["orientation_multiplier"]
)
min_tracks = int(
    protocol["cross_track_aggregation"]["primary_minimum_available_tracks"]
)

# ---------------------------------------------------------------------
# Structural audit of exposure
# ---------------------------------------------------------------------
exp["support_id"] = exp["support_id"].astype(str)
exp["year"] = pd.to_numeric(exp["year"], errors="raise").astype(int)

flag(
    f"SUPPORT_YEAR_ROW_COUNT_MISMATCH:{len(exp)}",
    len(exp) != EXPECTED_SUPPORT_YEAR_ROWS
)
flag(
    "DUPLICATE_SUPPORT_YEAR_KEYS",
    exp.duplicated(["support_id", "year"], keep=False).any()
)
flag(
    "SUPPORT_UNIVERSE_MISMATCH",
    exp["support_id"].nunique() != EXPECTED_SUPPORT_N
)
flag(
    "YEAR_UNIVERSE_MISMATCH",
    sorted(exp["year"].unique().tolist()) != EXPECTED_YEARS
)

per_support = exp.groupby("support_id")["year"].nunique()
flag(
    "SUPPORT_NOT_PRESENT_IN_ALL_11_YEARS",
    (per_support != len(EXPECTED_YEARS)).any()
)

coord_n = exp.groupby("support_id")[["lon", "lat"]].nunique(dropna=False)
flag(
    "SUPPORT_COORDINATE_INSTABILITY",
    ((coord_n["lon"] != 1) | (coord_n["lat"] != 1)).any()
)

track_cols = ["asc15", "asc88", "desc66", "desc168"]
avail_re = exp[track_cols].notna().sum(axis=1).astype(int)
flag(
    "AVAILABLE_TRACK_COUNT_MISMATCH",
    (avail_re != pd.to_numeric(exp["available_tracks_n"], errors="coerce")).any()
)

primary_eligible_re = avail_re >= min_tracks
complete4_re = avail_re == 4

reported_primary = exp["primary_eligible_ge3_tracks"].astype(str).str.lower().isin(
    ["true", "1"]
)
reported_complete4 = exp["complete_4track_subset"].astype(str).str.lower().isin(
    ["true", "1"]
)

flag(
    "PRIMARY_ELIGIBILITY_FLAG_MISMATCH",
    (reported_primary != primary_eligible_re).any()
)
flag(
    "COMPLETE4_FLAG_MISMATCH",
    (reported_complete4 != complete4_re).any()
)

median_re = exp[track_cols].median(axis=1, skipna=True)

primary_re = median_re.where(primary_eligible_re)
complete4_val_re = median_re.where(complete4_re)

for col, calc in [
    ("flooding_like_s1_median_all_available", median_re),
    ("flooding_like_s1_primary", primary_re),
    ("flooding_like_s1_complete4", complete4_val_re),
]:
    got = pd.to_numeric(exp[col], errors="coerce")
    bad = ~(
        (got.isna() & calc.isna())
        | ((got - calc).abs() <= TOL)
    )
    flag(f"{col.upper()}_MISMATCH", bad.any())

# ---------------------------------------------------------------------
# Independently recompute track standardization parameters
# ---------------------------------------------------------------------
ph["support_id"] = ph["support_id"].astype(str)
ph["year"] = pd.to_numeric(ph["year"], errors="raise").astype(int)
ph["orbit_state"] = ph["orbit_state"].astype(str).str.lower()
ph["relative_orbit"] = pd.to_numeric(
    ph["relative_orbit"], errors="raise"
).astype(int)
ph[feature] = pd.to_numeric(ph[feature], errors="coerce")

param_checks = []

for (state, orbit), label in EXPECTED_TRACKS.items():
    g = ph[
        ph["orbit_state"].eq(state)
        & ph["relative_orbit"].eq(orbit)
    ].copy()
    v = pd.to_numeric(g[feature], errors="coerce")
    vf = v[np.isfinite(v)]

    center = float(vf.median())
    mad = float((vf - center).abs().median())
    scale = float(1.4826 * mad)

    p = params50[
        params50["orbit_state"].astype(str).str.lower().eq(state)
        & pd.to_numeric(
            params50["relative_orbit"], errors="coerce"
        ).eq(orbit)
    ]

    ok = len(p) == 1
    if ok:
        pr = p.iloc[0]
        ok = (
            same(pr["track_median"], center)
            and same(pr["track_mad"], mad)
            and same(pr["robust_scale_1p4826_mad"], scale)
            and int(pr["orientation_multiplier"]) == orientation
        )

    param_checks.append({
        "orbit_state": state,
        "relative_orbit": orbit,
        "track_label": label,
        "passed": bool(ok),
        "recomputed_track_median": center,
        "recomputed_track_mad": mad,
        "recomputed_scale": scale,
    })

flag(
    "TRACK_STANDARDIZATION_PARAMETER_RECOMPUTE_FAILURE",
    not all(r["passed"] for r in param_checks)
)

# ---------------------------------------------------------------------
# Deterministic 77-case recomputation from phenology
# ---------------------------------------------------------------------
supports = sorted(exp["support_id"].unique())
idx = np.linspace(0, len(supports)-1, 7).round().astype(int)
audit_supports = [supports[i] for i in sorted(set(idx.tolist()))]

# Recomputed parameters lookup
param_lookup = {}
for r in param_checks:
    param_lookup[(r["orbit_state"], r["relative_orbit"])] = (
        r["recomputed_track_median"],
        r["recomputed_scale"],
    )

recheck = []

for sid in audit_supports:
    for year in EXPECTED_YEARS:
        vals = {}

        for (state, orbit), label in EXPECTED_TRACKS.items():
            g = ph[
                ph["support_id"].eq(sid)
                & ph["year"].eq(year)
                & ph["orbit_state"].eq(state)
                & ph["relative_orbit"].eq(orbit)
            ]

            if len(g) != 1:
                vals[label] = np.nan
                continue

            raw = pd.to_numeric(g.iloc[0][feature], errors="coerce")
            if not np.isfinite(raw):
                vals[label] = np.nan
                continue

            center, scale = param_lookup[(state, orbit)]
            z = (float(raw) - center) / scale
            vals[label] = orientation * z

        arr = np.asarray(
            [vals[k] for k in track_cols],
            dtype=float,
        )
        finite_arr = arr[np.isfinite(arr)]
        n = len(finite_arr)
        med = float(np.median(finite_arr)) if n else np.nan
        primary = med if n >= min_tracks else np.nan

        row = exp[
            exp["support_id"].eq(sid)
            & exp["year"].eq(year)
        ]

        ok = len(row) == 1
        maxerr = 0.0

        if ok:
            rr = row.iloc[0]

            for label in track_cols:
                ok = ok and same(rr[label], vals[label])
                if (
                    pd.notna(rr[label])
                    and np.isfinite(vals[label])
                ):
                    maxerr = max(
                        maxerr,
                        abs(float(rr[label]) - float(vals[label]))
                    )

            ok = ok and int(rr["available_tracks_n"]) == n
            ok = ok and same(
                rr["flooding_like_s1_median_all_available"],
                med
            )
            ok = ok and same(
                rr["flooding_like_s1_primary"],
                primary
            )

        recheck.append({
            "support_id": sid,
            "year": year,
            "passed": bool(ok),
            "available_tracks_recomputed_n": int(n),
            "max_abs_error": float(maxerr),
        })

recheck_df = pd.DataFrame(recheck)
recheck_df.to_csv(RECHECK_OUT, index=False)

failed_recheck = int((~recheck_df["passed"]).sum())
flag(
    f"DETERMINISTIC_RECOMPUTE_FAILURES:{failed_recheck}",
    failed_recheck > 0
)

# ---------------------------------------------------------------------
# Freeze hashes / manifest
# ---------------------------------------------------------------------
exp_sha = hashlib.sha256(EXPOSURE50.read_bytes()).hexdigest()
params_sha = hashlib.sha256(PARAM50.read_bytes()).hexdigest()
protocol_sha = hashlib.sha256(P49.read_bytes()).hexdigest()
pheno_sha = hashlib.sha256(PHENO.read_bytes()).hexdigest()

status = (
    "PASS_FINAL_S1_EXPOSURE_FROZEN_FOR_HYDROLOGICAL_ANALYSIS"
    if not flags else
    "FAIL_REVIEW_REQUIRED"
)

audit_rows = (
    [{"severity": "HARD", "flag": f} for f in flags]
    if flags
    else [{"severity": "NONE", "flag": "NONE"}]
)
pd.DataFrame(audit_rows).to_csv(AUDIT_OUT, index=False)

manifest = {
    "stage": "DESIGN_C_STAGE51_AUDIT_AND_FREEZE_FINAL_S1_EXPOSURE",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,

    "frozen_measurement": {
        "source_feature": feature,
        "orientation_multiplier": orientation,
        "support_year_exposure_column": "flooding_like_s1_primary",
        "primary_minimum_available_tracks": min_tracks,
        "complete4_sensitivity_column": "flooding_like_s1_complete4",
    },

    "universe": {
        "support_ids_n": int(exp["support_id"].nunique()),
        "years": EXPECTED_YEARS,
        "support_year_rows_n": int(len(exp)),
        "primary_eligible_rows_n": int(reported_primary.sum()),
        "complete4_rows_n": int(reported_complete4.sum()),
    },

    "independent_recompute": {
        "support_ids": audit_supports,
        "cases_n": int(len(recheck_df)),
        "failures_n": failed_recheck,
    },

    "sha256": {
        "phenology_input": pheno_sha,
        "stage49_protocol": protocol_sha,
        "stage50_standardization_parameters": params_sha,
        "stage50_support_year_exposure": exp_sha,
    },

    "hard_flags": flags,

    "groundwater_values_read": False,
    "irrigation_flow_values_read": False,
    "raw_ricefloodit_values_read": False,
    "hydrological_association_models_fitted": 0,

    "downstream_permission": (
        "The frozen column flooding_like_s1_primary may now be linked to "
        "groundwater under a separately frozen hydrological-analysis protocol. "
        "No alternative Sentinel-1 feature, orientation, standardization, "
        "track weighting, or aggregation rule may be selected from groundwater "
        "performance in the confirmatory path."
    ),
}

MANIFEST_OUT.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 51 FINAL SENTINEL-1 EXPOSURE FREEZE",
    "=" * 76,
    "",
    f"Source feature: {feature}",
    "Frozen primary exposure column: flooding_like_s1_primary",
    "",
    f"Support-year rows: {len(exp)} / expected {EXPECTED_SUPPORT_YEAR_ROWS}",
    f"Primary eligible rows: {int(reported_primary.sum())}",
    f"Complete-4-track sensitivity rows: {int(reported_complete4.sum())}",
    f"Independent recompute cases: {len(recheck_df)}",
    f"Independent recompute failures: {failed_recheck}",
    "",
    "HARD QA FLAGS",
    "-------------",
]
lines += flags if flags else ["None"]
lines += [
    "",
    "FIREWALL",
    "--------",
    "Raw RiceFloodIT values read: False",
    "Groundwater read: False",
    "Irrigation-flow read: False",
    "Hydrological association model fitted: False",
    "",
    "DOWNSTREAM RULE",
    "---------------",
    "Only flooding_like_s1_primary is permitted as the primary Sentinel-1",
    "exposure in the confirmatory hydrological path.",
    "No groundwater-driven re-selection of feature/track/weighting is allowed.",
    "",
    f"STAGE 51 STATUS: {status}",
]

SUMMARY_OUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
print("\n".join(lines))

if flags:
    raise RuntimeError(
        "Stage 51 failed independent QA; inspect the audit outputs."
    )
