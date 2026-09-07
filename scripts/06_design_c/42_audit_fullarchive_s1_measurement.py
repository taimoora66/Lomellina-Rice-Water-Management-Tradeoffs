"""
Design C — C2W full-archive Sentinel-1 measurement QA.

Purpose
-------
Independently audit the frozen primary Sentinel-1 corrected-signal checkpoint
archive produced by:

    scripts/06_design_c/41_extract_fullarchive_s1_corrected_signal_optimized.py

This audit is outcome-blind. It does not read groundwater, irrigation-flow
outcomes, RiceFloodIT flooding values, inundation thresholds/classifiers, or
association-model results.

Scientific decision encoded here
--------------------------------
After an outcome-blind recovery investigation, the remaining thermal-noise
support failures are accepted only when they match the exact documented
same-acquisition source-missingness pattern:

    scene:
      S1C_IW_GRDH_1SDV_20250421T053447_20250421T053516_
      001987_003FE6_9043_COG

    VV: 390 support points
    VH: 390 support points

These points lie outside every declared azimuth-noise applicability block.
The upstream original 9B29 SAFE and the derived 9043 COG SAFE were checked
and their VV/VH thermal-noise XML files were byte-identical. No additional
same-acquisition authoritative recovery source was therefore available.

The observations remain missing. They are never imputed.

Any calibration-support failure, any unexpected thermal-noise failure count,
or any residual thermal-noise failure outside the documented scene remains a
hard QA failure.

Run
---
python -u scripts/06_design_c/42_audit_fullarchive_s1_measurement.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths and frozen expectations
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

D = ROOT / "outputs" / "diagnostics" / "design_c"
C2V = D / "c2v"

CHECKPOINT_DIR = C2V / "scene_checkpoints" / "primary"
PRIMARY_PLAN = D / "c2uf_primary_vvvh_canonical_asset_plan.csv"

OUT = D / "c2w"
OUT.mkdir(parents=True, exist_ok=True)

SCENE_AUDIT_OUT = OUT / "c2w_scene_checkpoint_audit.csv"
FAILURE_SAMPLE_OUT = OUT / "c2w_noise_support_failure_samples.csv"
ZERO_SUPPORT_OUT = OUT / "c2w_zero_support_scenes.csv"
SUMMARY_OUT = OUT / "c2w_fullarchive_qa_summary.txt"
QA_JSON_OUT = OUT / "c2w_fullarchive_qa.json"

EXPECTED_SCENES = 2134
EXPECTED_SUPPORT = 4331
YEARS = list(range(2015, 2026))
POLARS = ["vv", "vh"]

DOCUMENTED_SCENE = (
    "S1C_IW_GRDH_1SDV_20250421T053447_20250421T053516_"
    "001987_003FE6_9043_COG"
)

EXPECTED_DOCUMENTED_NOISE_FAILURE = {
    "vv": 390,
    "vh": 390,
}


# =============================================================================
# Helpers
# =============================================================================

def to_bool_series(x: pd.Series) -> pd.Series:
    """
    Convert a heterogeneous CSV-loaded series to a nullable-safe boolean series.

    Handles booleans, 0/1, and common string representations.
    Missing values become False.
    """
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(x):
        return x.fillna(0).astype(float).ne(0)

    s = x.astype("string").str.strip().str.lower()
    return s.isin({"true", "1", "yes", "y", "t"}).fillna(False)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return to_bool_series(df[col])


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{context}: missing required checkpoint column(s): "
            + ", ".join(missing)
        )


def safe_int(x) -> int:
    if pd.isna(x):
        return 0
    return int(x)


def finite_count(x: pd.Series) -> int:
    v = pd.to_numeric(x, errors="coerce").to_numpy(float)
    return int(np.isfinite(v).sum())


def relpath(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


# =============================================================================
# Frozen plan
# =============================================================================

if not PRIMARY_PLAN.exists():
    raise FileNotFoundError(
        f"Frozen primary plan not found: {PRIMARY_PLAN}"
    )

plan = pd.read_csv(PRIMARY_PLAN, low_memory=False)

required_plan = [
    "canonical_scene_id",
    "year",
    "acquisition_date",
    "orbit_state",
    "relative_orbit",
]
require_columns(plan, required_plan, "primary plan")

plan["canonical_scene_id"] = plan["canonical_scene_id"].astype(str)
plan["year"] = pd.to_numeric(plan["year"], errors="raise").astype(int)
plan["relative_orbit"] = pd.to_numeric(
    plan["relative_orbit"], errors="raise"
).astype(int)

# One record per frozen canonical scene.
plan_scene = (
    plan.sort_values(
        ["year", "acquisition_date", "orbit_state", "relative_orbit",
         "canonical_scene_id"]
    )
    .drop_duplicates("canonical_scene_id", keep="first")
    .copy()
)

if len(plan_scene) != EXPECTED_SCENES:
    raise RuntimeError(
        f"Frozen primary plan contains {len(plan_scene)} canonical scenes; "
        f"expected {EXPECTED_SCENES}."
    )

plan_lookup = {
    str(r.canonical_scene_id): r
    for r in plan_scene.itertuples(index=False)
}

expected_scene_ids = set(plan_lookup)


# =============================================================================
# Checkpoint inventory
# =============================================================================

if not CHECKPOINT_DIR.exists():
    raise FileNotFoundError(
        f"Primary checkpoint directory not found: {CHECKPOINT_DIR}"
    )

checkpoint_files = sorted(CHECKPOINT_DIR.glob("*.csv.gz"))

print(
    f"Auditing {len(checkpoint_files)} primary checkpoints...",
    flush=True,
)

# Filename stem is expected to be safe_scene_filename(scene) + ".csv.gz".
# Because canonical scene IDs already contain only safe characters, stripping
# ".csv.gz" recovers the scene ID.
cp_scene_ids = []
for cp in checkpoint_files:
    name = cp.name
    if not name.endswith(".csv.gz"):
        continue
    cp_scene_ids.append(name[:-7])

cp_scene_set = set(cp_scene_ids)

missing_checkpoint_scenes = sorted(expected_scene_ids - cp_scene_set)
unexpected_checkpoint_scenes = sorted(cp_scene_set - expected_scene_ids)


# =============================================================================
# A. Per-checkpoint independent audit
# =============================================================================

scene_rows: list[dict] = []
failure_rows: list[dict] = []
support_ids_seen: set[str] = set()

for i, cp in enumerate(checkpoint_files, 1):

    if i % 100 == 0:
        print(f"  {i}/{len(checkpoint_files)}", flush=True)

    scene_id = cp.name[:-7]

    if scene_id not in plan_lookup:
        # Still inspect enough to preserve provenance, but mark as unexpected.
        meta = None
    else:
        meta = plan_lookup[scene_id]

    try:
        x = pd.read_csv(
            cp,
            compression="gzip",
            low_memory=False,
        )
    except Exception as e:
        scene_rows.append({
            "scene_id": scene_id,
            "checkpoint_path": relpath(cp),
            "checkpoint_read_ok": False,
            "checkpoint_read_error": repr(e),
            "rows_n": np.nan,
            "zero_support_checkpoint": False,
        })
        continue

    # Every valid checkpoint, including a header-only zero-support checkpoint,
    # must carry the production schema.
    base_required = [
        "scene_id",
        "support_id",
        "year",
        "acquisition_date",
        "orbit_state",
        "relative_orbit",
    ]
    for pol in POLARS:
        base_required += [
            f"{pol}_raster_inside",
            f"{pol}_calibration_support_ok",
            f"{pol}_noise_support_ok",
            f"{pol}_nonpositive_corrected_power",
            f"{pol}_sigma0_corrected_linear",
        ]

    require_columns(
        x,
        base_required,
        f"checkpoint {cp.name}",
    )

    zero_support = len(x) == 0

    # Empty checkpoints still have their metadata available from the frozen plan.
    if meta is not None:
        year = int(meta.year)
        acquisition_date = str(meta.acquisition_date)
        orbit_state = str(meta.orbit_state)
        relative_orbit = int(meta.relative_orbit)
        platform = (
            str(meta.platform)
            if hasattr(meta, "platform")
            else ""
        )
    elif len(x):
        year = safe_int(pd.to_numeric(x["year"], errors="coerce").iloc[0])
        acquisition_date = str(x["acquisition_date"].iloc[0])
        orbit_state = str(x["orbit_state"].iloc[0])
        relative_orbit = safe_int(
            pd.to_numeric(x["relative_orbit"], errors="coerce").iloc[0]
        )
        platform = str(x["platform"].iloc[0]) if "platform" in x.columns else ""
    else:
        year = np.nan
        acquisition_date = ""
        orbit_state = ""
        relative_orbit = np.nan
        platform = ""

    row = {
        "scene_id": scene_id,
        "checkpoint_path": relpath(cp),
        "checkpoint_read_ok": True,
        "checkpoint_read_error": "",
        "year": year,
        "acquisition_date": acquisition_date,
        "orbit_state": orbit_state,
        "relative_orbit": relative_orbit,
        "platform": platform,
        "rows_n": int(len(x)),
        "zero_support_checkpoint": bool(zero_support),
    }

    if len(x):
        scene_values = set(x["scene_id"].dropna().astype(str).unique())
        row["scene_id_values_n"] = len(scene_values)
        row["scene_id_matches_filename"] = (
            scene_values == {scene_id}
        )
        support_ids_seen.update(
            x["support_id"].dropna().astype(str).tolist()
        )
        row["support_ids_n"] = int(
            x["support_id"].dropna().astype(str).nunique()
        )
        row["duplicate_support_rows_n"] = int(
            x["support_id"].astype(str).duplicated().sum()
        )
    else:
        row["scene_id_values_n"] = 0
        row["scene_id_matches_filename"] = True
        row["support_ids_n"] = 0
        row["duplicate_support_rows_n"] = 0

    for pol in POLARS:

        inside = bool_series(x, f"{pol}_raster_inside")
        cal_ok = bool_series(x, f"{pol}_calibration_support_ok")
        noise_ok = bool_series(x, f"{pol}_noise_support_ok")
        nonpos = bool_series(x, f"{pol}_nonpositive_corrected_power")
        sigma = numeric_series(x, f"{pol}_sigma0_corrected_linear")

        cal_fail = inside & ~cal_ok
        noise_fail = inside & cal_ok & ~noise_ok
        valid_sigma = (
            inside
            & cal_ok
            & noise_ok
            & np.isfinite(sigma.to_numpy(float))
            & sigma.gt(0)
        )

        row[f"{pol}_raster_inside_n"] = int(inside.sum())
        row[f"{pol}_calibration_fail_n"] = int(cal_fail.sum())
        row[f"{pol}_noise_fail_n"] = int(noise_fail.sum())
        row[f"{pol}_nonpositive_n"] = int(nonpos.sum())
        row[f"{pol}_valid_corrected_sigma0_n"] = int(valid_sigma.sum())

        if noise_fail.any():
            cols = [
                c for c in [
                    "support_id", "lon", "lat",
                    f"{pol}_image_row",
                    f"{pol}_image_col",
                    f"{pol}_raster_inside",
                    f"{pol}_calibration_support_ok",
                    f"{pol}_noise_support_ok",
                    f"{pol}_noise_power_eta",
                    f"{pol}_noise_range_component",
                    f"{pol}_noise_azimuth_component",
                    f"{pol}_noise_azimuth_candidate_blocks_n",
                ]
                if c in x.columns
            ]

            z = x.loc[noise_fail, cols].copy()
            z.insert(0, "polarization", pol.upper())
            z.insert(0, "relative_orbit", relative_orbit)
            z.insert(0, "orbit_state", orbit_state)
            z.insert(0, "acquisition_date", acquisition_date)
            z.insert(0, "year", year)
            z.insert(0, "scene_id", scene_id)
            failure_rows.extend(z.to_dict("records"))

    scene_rows.append(row)


scene = pd.DataFrame(scene_rows)

if len(scene):
    scene = scene.sort_values(
        ["year", "acquisition_date", "orbit_state",
         "relative_orbit", "scene_id"],
        na_position="last",
    ).reset_index(drop=True)

scene.to_csv(SCENE_AUDIT_OUT, index=False)

failures = pd.DataFrame(failure_rows)
if len(failures):
    failures = failures.sort_values(
        ["scene_id", "polarization", "support_id"]
    ).reset_index(drop=True)
failures.to_csv(FAILURE_SAMPLE_OUT, index=False)


# =============================================================================
# B. Zero-support archive
# =============================================================================

if len(scene):
    zero = scene.loc[
        scene["zero_support_checkpoint"].fillna(False).astype(bool)
    ].copy()
else:
    zero = pd.DataFrame()

zero.to_csv(ZERO_SUPPORT_OUT, index=False)


# =============================================================================
# C. Archive-level totals
# =============================================================================

years_observed = sorted(
    int(y)
    for y in scene["year"].dropna().astype(int).unique().tolist()
) if len(scene) else []

archive_totals = {}

for pol in POLARS:
    archive_totals[pol] = {
        "raster_inside_n": int(
            pd.to_numeric(
                scene.get(
                    f"{pol}_raster_inside_n",
                    pd.Series(dtype=float)
                ),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "valid_corrected_sigma0_n": int(
            pd.to_numeric(
                scene.get(
                    f"{pol}_valid_corrected_sigma0_n",
                    pd.Series(dtype=float)
                ),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "nonpositive_n": int(
            pd.to_numeric(
                scene.get(
                    f"{pol}_nonpositive_n",
                    pd.Series(dtype=float)
                ),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "calibration_fail_n": int(
            pd.to_numeric(
                scene.get(
                    f"{pol}_calibration_fail_n",
                    pd.Series(dtype=float)
                ),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "noise_fail_n": int(
            pd.to_numeric(
                scene.get(
                    f"{pol}_noise_fail_n",
                    pd.Series(dtype=float)
                ),
                errors="coerce",
            ).fillna(0).sum()
        ),
    }

    inside = archive_totals[pol]["raster_inside_n"]
    nonpos = archive_totals[pol]["nonpositive_n"]

    archive_totals[pol]["nonpositive_rate_among_raster_inside"] = (
        float(nonpos / inside)
        if inside > 0
        else math.nan
    )


# =============================================================================
# D. Hard QA gates versus documented source missingness
# =============================================================================

hard_flags: list[str] = []
documented_source_missingness: list[str] = []

if len(checkpoint_files) != EXPECTED_SCENES:
    hard_flags.append(
        f"FAIL_CHECKPOINT_COUNT:"
        f"{len(checkpoint_files)}!=expected_{EXPECTED_SCENES}"
    )

if missing_checkpoint_scenes:
    hard_flags.append(
        f"FAIL_MISSING_FROZEN_SCENES:{len(missing_checkpoint_scenes)}"
    )

if unexpected_checkpoint_scenes:
    hard_flags.append(
        f"FAIL_UNEXPECTED_CHECKPOINT_SCENES:"
        f"{len(unexpected_checkpoint_scenes)}"
    )

if years_observed != YEARS:
    hard_flags.append(
        f"FAIL_YEAR_UNIVERSE:{years_observed}"
    )

if len(support_ids_seen) != EXPECTED_SUPPORT:
    hard_flags.append(
        f"FAIL_SUPPORT_ID_UNIVERSE:"
        f"{len(support_ids_seen)}!=expected_{EXPECTED_SUPPORT}"
    )

if len(scene):
    unreadable = int(
        (~scene["checkpoint_read_ok"].fillna(False).astype(bool)).sum()
    )
    if unreadable:
        hard_flags.append(
            f"FAIL_UNREADABLE_CHECKPOINTS:{unreadable}"
        )

    scene_id_mismatch = int(
        (~scene["scene_id_matches_filename"]
         .fillna(False)
         .astype(bool)).sum()
    )
    if scene_id_mismatch:
        hard_flags.append(
            f"FAIL_SCENE_ID_CHECKPOINT_MISMATCH:{scene_id_mismatch}"
        )

    duplicate_rows = int(
        pd.to_numeric(
            scene["duplicate_support_rows_n"],
            errors="coerce",
        ).fillna(0).sum()
    )
    if duplicate_rows:
        hard_flags.append(
            f"FAIL_DUPLICATE_SUPPORT_ROWS:{duplicate_rows}"
        )

for pol in POLARS:

    cal = archive_totals[pol]["calibration_fail_n"]
    noise = archive_totals[pol]["noise_fail_n"]

    if cal > 0:
        hard_flags.append(
            f"CALIBRATION_SUPPORT_FAILURE_{pol.upper()}:{cal}"
        )

    expected_noise = EXPECTED_DOCUMENTED_NOISE_FAILURE[pol]

    pol_fail = (
        failures.loc[
            failures["polarization"].astype(str).str.upper()
            == pol.upper()
        ].copy()
        if len(failures)
        else pd.DataFrame()
    )

    fail_scenes = (
        sorted(pol_fail["scene_id"].astype(str).unique().tolist())
        if len(pol_fail)
        else []
    )

    exact_documented_pattern = (
        noise == expected_noise
        and fail_scenes == [DOCUMENTED_SCENE]
        and len(pol_fail) == expected_noise
    )

    if exact_documented_pattern:
        documented_source_missingness.append(
            f"DOCUMENTED_NOISE_SUPPORT_MISSINGNESS_"
            f"{pol.upper()}:{noise}"
        )
    elif noise > 0:
        hard_flags.append(
            f"UNEXPECTED_NOISE_SUPPORT_FAILURE_"
            f"{pol.upper()}:{noise};scenes={fail_scenes}"
        )
    elif expected_noise > 0:
        # A disappearance of the prespecified residual pattern is not silently
        # treated as PASS because it would imply the underlying archive or
        # processing method changed after this audit rule was frozen.
        hard_flags.append(
            f"FAIL_DOCUMENTED_MISSINGNESS_PATTERN_CHANGED_"
            f"{pol.upper()}:observed_0_expected_{expected_noise}"
        )


# Combined list retained for compatibility with older downstream consumers.
flags = hard_flags + documented_source_missingness

if hard_flags:
    status = "REVIEW_REQUIRED"
elif documented_source_missingness:
    status = "PASS_WITH_DOCUMENTED_SOURCE_MISSINGNESS"
else:
    status = "PASS_ARCHIVE_STRUCTURAL_QA"


# =============================================================================
# E. Human-readable summary
# =============================================================================

summary: list[str] = []

summary.append(
    "DESIGN C - C2W FULL-ARCHIVE SENTINEL-1 MEASUREMENT QA"
)
summary.append("=" * 78)
summary.append("")

summary.append(f"Checkpoint files: {len(checkpoint_files)}")
summary.append(f"Expected frozen scenes: {EXPECTED_SCENES}")
summary.append(f"Years represented: {years_observed}")
summary.append(
    f"Zero-study-support checkpoints: {len(zero)}"
)
summary.append(
    f"Support IDs represented: {len(support_ids_seen)}"
)
summary.append("")

for pol in POLARS:

    z = archive_totals[pol]
    rate = z["nonpositive_rate_among_raster_inside"]

    summary.append(
        f"{pol.upper()} raster-inside samples: "
        f"{z['raster_inside_n']}"
    )
    summary.append(
        f"{pol.upper()} valid corrected sigma0: "
        f"{z['valid_corrected_sigma0_n']}"
    )
    summary.append(
        f"{pol.upper()} non-positive corrected power: "
        f"{z['nonpositive_n']}"
    )
    summary.append(
        f"{pol.upper()} non-positive rate among raster-inside: "
        f"{rate:.8f}"
    )
    summary.append("")

summary.append("HARD QA FLAGS")
summary.append("-------------")

if hard_flags:
    summary.extend(hard_flags)
else:
    summary.append("None")

summary.append("")

summary.append("DOCUMENTED SOURCE MISSINGNESS")
summary.append("-----------------------------")

if documented_source_missingness:
    summary.extend(documented_source_missingness)
    summary.append("")
    summary.append(
        "Remaining thermal-noise unsupported samples are retained "
        "as missing and are not imputed."
    )
    summary.append(
        "The upstream original SAFE and derived COG SAFE contain "
        "byte-identical thermal-noise annotations for the affected "
        "2025-04-21 S1C acquisition; no additional same-acquisition "
        "authoritative recovery source was available."
    )
else:
    summary.append("None")

summary.append("")
summary.append("FIREWALL")
summary.append("--------")
summary.append("Groundwater read: False")
summary.append("Irrigation-flow read: False")
summary.append("RiceFloodIT flood values read: False")
summary.append("Threshold/classifier selected: False")
summary.append("Association model fitted: False")
summary.append("")
summary.append(f"C2W STATUS: {status}")

summary_text = "\n".join(summary) + "\n"

SUMMARY_OUT.write_text(
    summary_text,
    encoding="utf-8",
)


# =============================================================================
# F. Machine-readable QA
# =============================================================================

qa_json = {
    "stage":
        "DESIGN_C_C2W_FULLARCHIVE_SENTINEL1_MEASUREMENT_QA",
    "status": status,

    "checkpoint_directory": relpath(CHECKPOINT_DIR),
    "checkpoint_files_n": int(len(checkpoint_files)),
    "expected_frozen_scenes_n": EXPECTED_SCENES,

    "frozen_plan_path": relpath(PRIMARY_PLAN),
    "frozen_plan_scenes_n": int(len(plan_scene)),

    "expected_support_coordinates_n": EXPECTED_SUPPORT,
    "support_ids_represented_n": int(len(support_ids_seen)),

    "years_expected": YEARS,
    "years_observed": years_observed,

    "zero_support_checkpoints_n": int(len(zero)),

    "missing_checkpoint_scenes_n":
        int(len(missing_checkpoint_scenes)),
    "missing_checkpoint_scenes":
        missing_checkpoint_scenes,

    "unexpected_checkpoint_scenes_n":
        int(len(unexpected_checkpoint_scenes)),
    "unexpected_checkpoint_scenes":
        unexpected_checkpoint_scenes,

    "archive_totals": {
        pol.upper(): archive_totals[pol]
        for pol in POLARS
    },

    "hard_flags": hard_flags,
    "documented_source_missingness":
        documented_source_missingness,
    "flags": flags,

    "documented_noise_missingness_expected": {
        pol.upper(): int(n)
        for pol, n in EXPECTED_DOCUMENTED_NOISE_FAILURE.items()
    },
    "documented_noise_missingness_scene":
        DOCUMENTED_SCENE,
    "documented_noise_missingness_policy":
        "retain_as_missing_no_imputation",
    "documented_noise_missingness_provenance": (
        "Remaining cases occur after validated declared-block "
        "azimuth-LUT edge recovery. The affected 2025-04-21 "
        "S1C upstream original 9B29 SAFE and derived 9043 COG SAFE "
        "contain byte-identical VV/VH thermal-noise annotations, "
        "so no additional same-acquisition authoritative recovery "
        "source was available."
    ),

    "outputs": {
        "scene_checkpoint_audit_csv":
            relpath(SCENE_AUDIT_OUT),
        "noise_support_failure_samples_csv":
            relpath(FAILURE_SAMPLE_OUT),
        "zero_support_scenes_csv":
            relpath(ZERO_SUPPORT_OUT),
        "summary_txt":
            relpath(SUMMARY_OUT),
        "qa_json":
            relpath(QA_JSON_OUT),
    },

    "groundwater_values_read": False,
    "irrigation_flow_values_read": False,
    "ricefloodit_flood_values_read": False,
    "threshold_selected": False,
    "classifier_fitted": False,
    "association_models_fitted": 0,
}

QA_JSON_OUT.write_text(
    json.dumps(qa_json, indent=2) + "\n",
    encoding="utf-8",
)


# =============================================================================
# G. Console
# =============================================================================

print("")
print(summary_text, end="")

# Do not raise merely because the accepted, documented source missingness
# remains present. Raise only for a genuine hard QA failure.
if hard_flags:
    raise RuntimeError(
        "C2W hard QA gates failed; inspect "
        f"{QA_JSON_OUT} and {SCENE_AUDIT_OUT}."
    )
