"""
Stage 45 — independent audit of Stage-44 Sentinel-1 phenology features.
Outcome-blind: no groundwater, irrigation-flow, RiceFloodIT flooding values,
threshold selection, classifier fitting, or association modelling.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"
PH = D / "c2x_phenology"
SRC = D / "c2v" / "primary_mosaic"

FEATURE_IN = PH / "c2x_s1_track_year_phenology_features.csv.gz"
DICT_IN = PH / "c2x_s1_feature_dictionary.csv"
QA44_IN = PH / "c2x_s1_phenology_qa.json"

AUDIT_OUT = PH / "c2x_s1_phenology_feature_audit.csv"
COVERAGE_OUT = PH / "c2x_s1_phenology_coverage_by_track_year.csv"
RECHECK_OUT = PH / "c2x_s1_phenology_recompute_checks.csv"
QA_OUT = PH / "c2x_s1_phenology_audit_qa.json"
SUMMARY_OUT = PH / "c2x_s1_phenology_audit_summary.txt"

EXPECTED_SUPPORT_N = 4331
EXPECTED_YEARS = list(range(2015, 2026))
EXPECTED_TRACKS = {
    ("ascending", 15), ("ascending", 88),
    ("descending", 66), ("descending", 168)
}
EXPECTED_TY_N = 44
EXPECTED_ROWS = EXPECTED_SUPPORT_N * EXPECTED_TY_N
TOL = 1e-9

SIGNALS = {
    "vv": "vv_sigma0_corrected_db",
    "vh": "vh_sigma0_corrected_db",
    "vv_minus_vh": "vv_minus_vh_db",
}
PHASES = {
    "early_season": ("04-01", "05-31"),
    "mid_season": ("06-01", "07-15"),
    "late_season": ("07-16", "09-30"),
}

def add(flags, code, n):
    n = int(n)
    if n:
        flags.append({"severity": "HARD", "flag": code, "n": n})

def finite(s):
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return x[np.isfinite(x)]

def same(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= TOL

def phase_mask(d, phase):
    lo, hi = PHASES[phase]
    d = pd.to_datetime(d, errors="coerce")
    md = d.dt.strftime("%m-%d")
    return md.ge(lo) & md.le(hi)

for f in (FEATURE_IN, DICT_IN, QA44_IN):
    if not f.exists():
        raise FileNotFoundError(f)

print("DESIGN C - STAGE 45 INDEPENDENT SENTINEL-1 PHENOLOGY AUDIT")
print("=" * 76)
print("Groundwater read: False")
print("Irrigation-flow read: False")
print("RiceFloodIT flooding values read: False")
print("Threshold/classifier: False")
print("Association model: False\n")

x = pd.read_csv(FEATURE_IN, compression="gzip", low_memory=False)
fd = pd.read_csv(DICT_IN)
qa44 = json.loads(QA44_IN.read_text(encoding="utf-8"))
flags = []

# ---- universe and keys
x["year"] = pd.to_numeric(x["year"], errors="raise").astype(int)
x["relative_orbit"] = pd.to_numeric(x["relative_orbit"], errors="raise").astype(int)
x["orbit_state"] = x["orbit_state"].astype(str).str.lower()
x["support_id"] = x["support_id"].astype(str)

add(flags, "ROW_COUNT_MISMATCH", len(x) != EXPECTED_ROWS)
add(flags, "DUPLICATE_KEYS", x.duplicated(
    ["support_id","year","orbit_state","relative_orbit"], keep=False).sum())

supports = sorted(x["support_id"].unique())
add(flags, "SUPPORT_UNIVERSE_MISMATCH", len(supports) != EXPECTED_SUPPORT_N)
add(flags, "YEAR_UNIVERSE_MISMATCH", sorted(x["year"].unique()) != EXPECTED_YEARS)

obs_tracks = set(x[["orbit_state","relative_orbit"]]
                 .drop_duplicates().itertuples(index=False, name=None))
add(flags, "TRACK_UNIVERSE_MISMATCH", obs_tracks != EXPECTED_TRACKS)

obs_ty = set(x[["year","orbit_state","relative_orbit"]]
             .drop_duplicates().itertuples(index=False, name=None))
expected_ty = {(y,s,o) for y in EXPECTED_YEARS for s,o in EXPECTED_TRACKS}
add(flags, "TRACK_YEAR_UNIVERSE_MISMATCH", obs_ty != expected_ty)

per_support = x.groupby("support_id").size()
add(flags, "SUPPORT_NOT_PRESENT_IN_ALL_44_TRACK_YEARS",
    (per_support != EXPECTED_TY_N).sum())

coord = x.groupby("support_id")[["lon","lat"]].nunique(dropna=False)
add(flags, "COORDINATE_INSTABILITY",
    ((coord["lon"] != 1) | (coord["lat"] != 1)).sum())

# ---- sampling invariants
sd = pd.to_numeric(x["support_dates_n"], errors="coerce")
vvn = pd.to_numeric(x["vv_observed_dates_n"], errors="coerce")
vhn = pd.to_numeric(x["vh_observed_dates_n"], errors="coerce")
bn = pd.to_numeric(x["both_observed_dates_n"], errors="coerce")

add(flags, "VV_COUNT_EXCEEDS_SUPPORT", (vvn > sd).sum())
add(flags, "VH_COUNT_EXCEEDS_SUPPORT", (vhn > sd).sum())
add(flags, "BOTH_COUNT_EXCEEDS_VV", (bn > vvn).sum())
add(flags, "BOTH_COUNT_EXCEEDS_VH", (bn > vhn).sum())

for frac, count in [
    ("vv_observed_fraction", vvn),
    ("vh_observed_fraction", vhn),
    ("both_observed_fraction", bn),
]:
    f = pd.to_numeric(x[frac], errors="coerce")
    add(flags, frac.upper()+"_OUT_OF_RANGE", ((f < 0) | (f > 1)).sum())
    exp = count / sd.replace(0, np.nan)
    bad = ~((f.isna() & exp.isna()) | ((f-exp).abs() <= TOL))
    add(flags, frac.upper()+"_COUNT_MISMATCH", bad.sum())

# ---- signal invariants
for sig, exp_count in [
    ("vv", vvn), ("vh", vhn), ("vv_minus_vh", bn)
]:
    n = pd.to_numeric(x[f"{sig}_n"], errors="coerce")
    add(flags, sig.upper()+"_N_MISMATCH", (n != exp_count).sum())

    cols = [f"{sig}_{k}_db" for k in
            ["min","p10","p25","median","p75","p90","max"]]
    vals = [pd.to_numeric(x[c], errors="coerce") for c in cols]
    bad_order = False
    bad = pd.Series(False, index=x.index)
    for a,b in zip(vals[:-1], vals[1:]):
        bad |= (a > b + TOL).fillna(False)
    add(flags, sig.upper()+"_QUANTILE_ORDER_VIOLATION", bad.sum())

    iqr = pd.to_numeric(x[f"{sig}_iqr_db"], errors="coerce")
    rng = pd.to_numeric(x[f"{sig}_range_db"], errors="coerce")
    add(flags, sig.upper()+"_NEGATIVE_IQR", (iqr < -TOL).sum())
    add(flags, sig.upper()+"_NEGATIVE_RANGE", (rng < -TOL).sum())

    steps = pd.to_numeric(x[f"{sig}_valid_steps_n"], errors="coerce")
    add(flags, sig.upper()+"_STEP_COUNT_MISMATCH",
        (steps != np.maximum(n-1, 0)).sum())

# ---- phase invariants
for phase in PHASES:
    ps = pd.to_numeric(x[f"{phase}_support_dates_n"], errors="coerce")
    pb = pd.to_numeric(x[f"{phase}_both_observed_dates_n"], errors="coerce")
    add(flags, phase.upper()+"_SUPPORT_EXCEEDS_TOTAL", (ps > sd).sum())
    add(flags, phase.upper()+"_BOTH_EXCEEDS_PHASE_SUPPORT", (pb > ps).sum())
    add(flags, phase.upper()+"_BOTH_EXCEEDS_TOTAL_BOTH", (pb > bn).sum())

    diffmed = pd.to_numeric(x[f"{phase}_vv_minus_vh_median_db"], errors="coerce")
    mismatch = ((pb == 0) & diffmed.notna()) | ((pb > 0) & diffmed.isna())
    add(flags, phase.upper()+"_DIFF_MEDIAN_PRESENCE_MISMATCH", mismatch.sum())

# ---- feature dictionary / Stage-44 firewall
if "feature" not in fd.columns:
    add(flags, "FEATURE_DICTIONARY_MISSING_FEATURE_COLUMN", 1)
else:
    add(flags, "FEATURE_DICTIONARY_DUPLICATES", fd["feature"].duplicated().sum())
    add(flags, "OUTPUT_COLUMNS_MISSING_FROM_DICTIONARY",
        len(set(x.columns) - set(fd["feature"].astype(str))))
    add(flags, "DICTIONARY_EXTRA_FEATURES",
        len(set(fd["feature"].astype(str)) - set(x.columns)))

for key in [
    "groundwater_values_read",
    "irrigation_flow_values_read",
    "ricefloodit_flood_values_read",
    "inundation_threshold_selected",
    "classifier_fitted",
    "feature_selection_against_outcomes_performed",
]:
    add(flags, "STAGE44_FIREWALL_"+key.upper(), qa44.get(key) is not False)
add(flags, "STAGE44_ASSOCIATION_MODEL_FIREWALL",
    qa44.get("association_models_fitted") != 0)

# ---- descriptive track-year coverage
coverage = (x.groupby(["year","orbit_state","relative_orbit"], as_index=False)
              .agg(
                  support_rows_n=("support_id","size"),
                  support_dates_median=("support_dates_n","median"),
                  support_dates_min=("support_dates_n","min"),
                  support_dates_max=("support_dates_n","max"),
                  both_fraction_median=("both_observed_fraction","median"),
                  both_fraction_min=("both_observed_fraction","min"),
                  both_fraction_max=("both_observed_fraction","max"),
                  max_gap_days_median=("max_gap_days","median"),
                  max_gap_days_max=("max_gap_days","max"),
              )
              .sort_values(["year","orbit_state","relative_orbit"]))
coverage.to_csv(COVERAGE_OUT, index=False)

# ---- independent deterministic recomputation on 7 fixed support IDs x 44 files
idx = np.linspace(0, len(supports)-1, 7).round().astype(int)
audit_supports = [supports[i] for i in sorted(set(idx.tolist()))]
checks = []

for f in sorted(SRC.rglob("*.csv.gz")):
    s = pd.read_csv(f, compression="gzip", low_memory=False)
    s["support_id"] = s["support_id"].astype(str)
    s["year"] = pd.to_numeric(s["year"], errors="raise").astype(int)
    s["relative_orbit"] = pd.to_numeric(s["relative_orbit"], errors="raise").astype(int)
    s["orbit_state"] = s["orbit_state"].astype(str).str.lower()
    s["acquisition_date"] = pd.to_datetime(s["acquisition_date"], errors="raise")
    y = int(s["year"].iloc[0]); st = s["orbit_state"].iloc[0]
    o = int(s["relative_orbit"].iloc[0])

    for sid in audit_supports:
        g = s[s["support_id"].eq(sid)].sort_values("acquisition_date")
        r = x[(x["support_id"].eq(sid)) & (x["year"].eq(y)) &
              (x["orbit_state"].eq(st)) & (x["relative_orbit"].eq(o))]
        ok = len(g) > 0 and len(r) == 1
        maxerr = 0.0
        if ok:
            r = r.iloc[0]
            for sig, col in SIGNALS.items():
                v = finite(g[col])
                exp = {
                    f"{sig}_n": len(v),
                    f"{sig}_median_db": np.median(v) if len(v) else np.nan,
                    f"{sig}_p10_db": np.quantile(v,.10) if len(v) else np.nan,
                    f"{sig}_p25_db": np.quantile(v,.25) if len(v) else np.nan,
                    f"{sig}_p75_db": np.quantile(v,.75) if len(v) else np.nan,
                    f"{sig}_p90_db": np.quantile(v,.90) if len(v) else np.nan,
                    f"{sig}_min_db": np.min(v) if len(v) else np.nan,
                    f"{sig}_max_db": np.max(v) if len(v) else np.nan,
                }
                for c,e in exp.items():
                    ok &= same(r[c], e)
                    if not (pd.isna(r[c]) or pd.isna(e)):
                        maxerr = max(maxerr, abs(float(r[c])-float(e)))

            for phase in PHASES:
                gp = g[phase_mask(g["acquisition_date"], phase)]
                for sig,col in SIGNALS.items():
                    v = finite(gp[col])
                    e = np.median(v) if len(v) else np.nan
                    c = f"{phase}_{sig}_median_db"
                    ok &= same(r[c], e)
                    if not (pd.isna(r[c]) or pd.isna(e)):
                        maxerr = max(maxerr, abs(float(r[c])-float(e)))

        checks.append({
            "year": y, "orbit_state": st, "relative_orbit": o,
            "support_id": sid, "passed": bool(ok), "max_abs_error": maxerr
        })

recheck = pd.DataFrame(checks)
recheck.to_csv(RECHECK_OUT, index=False)
failed_recheck = int((~recheck["passed"]).sum())
add(flags, "INDEPENDENT_RECOMPUTE_FAILURES", failed_recheck)

audit = pd.DataFrame(flags) if flags else pd.DataFrame(
    [{"severity":"NONE","flag":"NONE","n":0}]
)
audit.to_csv(AUDIT_OUT, index=False)

hard = [d for d in flags if d["n"] > 0]
status = "PASS_PHENOLOGY_FEATURE_FREEZE_READY" if not hard else "FAIL_REVIEW_REQUIRED"

qa = {
    "stage": "DESIGN_C_STAGE45_INDEPENDENT_SENTINEL1_PHENOLOGY_AUDIT",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "feature_rows_n": int(len(x)),
    "expected_feature_rows_n": EXPECTED_ROWS,
    "support_ids_n": int(len(supports)),
    "track_year_combinations_n": int(len(obs_ty)),
    "deterministic_recompute_support_ids": audit_supports,
    "deterministic_recompute_cases_n": int(len(recheck)),
    "deterministic_recompute_failures_n": failed_recheck,
    "hard_flags": hard,
    "groundwater_values_read": False,
    "irrigation_flow_values_read": False,
    "ricefloodit_flood_values_read": False,
    "inundation_threshold_selected": False,
    "classifier_fitted": False,
    "outcome_based_feature_selection_performed": False,
    "association_models_fitted": 0,
    "track_pooling_performed": False,
    "missing_signal_imputation_performed": False,
}
QA_OUT.write_text(json.dumps(qa, indent=2)+"\n", encoding="utf-8")

lines = [
    "DESIGN C - STAGE 45 INDEPENDENT SENTINEL-1 PHENOLOGY FEATURE AUDIT",
    "="*78, "",
    f"Feature rows: {len(x)} / expected {EXPECTED_ROWS}",
    f"Support IDs: {len(supports)} / expected {EXPECTED_SUPPORT_N}",
    f"Track-year combinations: {len(obs_ty)} / expected {EXPECTED_TY_N}",
    f"Independent recompute cases: {len(recheck)}",
    f"Independent recompute failures: {failed_recheck}",
    "",
    "HARD QA FLAGS", "-------------",
]
lines += [f"{d['flag']}:{d['n']}" for d in hard] if hard else ["None"]
lines += [
    "", "FIREWALL", "--------",
    "Groundwater read: False",
    "Irrigation-flow read: False",
    "RiceFloodIT flood values read: False",
    "Threshold selected: False",
    "Classifier fitted: False",
    "Outcome-based feature selection: False",
    "Association model fitted: False",
    "", f"STAGE 45 STATUS: {status}",
]
summary = "\n".join(lines)+"\n"
SUMMARY_OUT.write_text(summary, encoding="utf-8")
print(summary, end="")

if hard:
    raise RuntimeError("Stage 45 failed hard QA; inspect the audit outputs.")
