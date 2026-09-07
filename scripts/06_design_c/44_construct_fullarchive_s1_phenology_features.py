"""
Design C — C2X-PH outcome-blind full-archive Sentinel-1 phenology construction.

Purpose
-------
Transform the frozen C2V primary Sentinel-1 VV/VH mosaics into prespecified
multi-temporal track-year phenology descriptors at the 4,331 fixed
RiceFloodIT support coordinates.

This stage is intentionally outcome-blind.

It DOES:
- read only frozen C2V primary VV/VH same-date/track mosaics;
- preserve the four stable orbit/track families separately;
- summarize corrected VV, VH, and VV-VH dB time series;
- compute fixed calendar-window summaries;
- compute sampling/missingness diagnostics;
- retain missing corrected observations as missing.

It DOES NOT:
- read groundwater values;
- read irrigation-flow outcomes;
- read RiceFloodIT flooding-frequency values;
- select or tune an inundation threshold;
- fit a flood/no-flood classifier;
- optimize features against any outcome;
- fit association models;
- impute missing Sentinel-1 observations.

Scientific architecture
-----------------------
The earlier outcome-blind architecture audit rejected a single-snapshot
inundation-classifier backbone and selected multi-temporal Sentinel-1 phenology
as the primary measurement architecture. Accordingly, this script constructs
descriptive temporal features only.

Calendar strata are fixed a priori and are NOT claimed to be field-observed
phenological stages:
- early_season : 01 April through 31 May
- mid_season   : 01 June through 15 July
- late_season  : 16 July through 30 September

Track families remain separate:
- ascending 15
- ascending 88
- descending 66
- descending 168

Input
-----
outputs/diagnostics/design_c/c2v/primary_mosaic/year=YYYY/<orbit>_<track>.csv.gz

Expected frozen mosaic schema
-----------------------------
year
orbit_state
relative_orbit
stream
acquisition_date
support_id
lon
lat
vv_sigma0_corrected_linear
vv_contributing_scene_rows_n
vv_candidate_scene_rows_n
vv_sigma0_corrected_db
vv_observed
vh_sigma0_corrected_linear
vh_contributing_scene_rows_n
vh_candidate_scene_rows_n
vh_sigma0_corrected_db
vh_observed
vv_minus_vh_db
vv_vh_both_observed

Outputs
-------
outputs/diagnostics/design_c/c2x_phenology/
    c2x_s1_track_year_phenology_features.csv.gz
    c2x_s1_track_year_sampling_qa.csv
    c2x_s1_feature_dictionary.csv
    c2x_s1_phenology_qa.json
    c2x_s1_phenology_summary.txt

Run
---
python -u scripts/06_design_c/44_construct_fullarchive_s1_phenology_features.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths and frozen design
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

D = ROOT / "outputs" / "diagnostics" / "design_c"
IN_ROOT = D / "c2v" / "primary_mosaic"

OUT = D / "c2x_phenology"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_OUT = OUT / "c2x_s1_track_year_phenology_features.csv.gz"
SAMPLING_OUT = OUT / "c2x_s1_track_year_sampling_qa.csv"
DICT_OUT = OUT / "c2x_s1_feature_dictionary.csv"
QA_OUT = OUT / "c2x_s1_phenology_qa.json"
SUMMARY_OUT = OUT / "c2x_s1_phenology_summary.txt"

EXPECTED_SUPPORT_N = 4331
EXPECTED_FILES_N = 44
EXPECTED_YEARS = list(range(2015, 2026))

EXPECTED_TRACKS = {
    ("ascending", 15),
    ("ascending", 88),
    ("descending", 66),
    ("descending", 168),
}

EXPECTED_SCHEMA = [
    "year",
    "orbit_state",
    "relative_orbit",
    "stream",
    "acquisition_date",
    "support_id",
    "lon",
    "lat",
    "vv_sigma0_corrected_linear",
    "vv_contributing_scene_rows_n",
    "vv_candidate_scene_rows_n",
    "vv_sigma0_corrected_db",
    "vv_observed",
    "vh_sigma0_corrected_linear",
    "vh_contributing_scene_rows_n",
    "vh_candidate_scene_rows_n",
    "vh_sigma0_corrected_db",
    "vh_observed",
    "vv_minus_vh_db",
    "vv_vh_both_observed",
]

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


# =============================================================================
# Generic helpers
# =============================================================================

def to_bool_series(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce").fillna(0).ne(0)

    s = x.astype("string").str.strip().str.lower()
    return s.isin({"true", "1", "yes", "y", "t"}).fillna(False)


def q(x: np.ndarray, p: float) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, p))


def finite_values(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return x[np.isfinite(x)]


def summarize_signal(s: pd.Series, prefix: str) -> dict:
    """
    Prespecified distributional summary of one dB time series.
    """
    x = finite_values(s)

    if len(x) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_median_db": np.nan,
            f"{prefix}_p10_db": np.nan,
            f"{prefix}_p25_db": np.nan,
            f"{prefix}_p75_db": np.nan,
            f"{prefix}_p90_db": np.nan,
            f"{prefix}_min_db": np.nan,
            f"{prefix}_max_db": np.nan,
            f"{prefix}_iqr_db": np.nan,
            f"{prefix}_range_db": np.nan,
        }

    p10 = q(x, 0.10)
    p25 = q(x, 0.25)
    p75 = q(x, 0.75)
    p90 = q(x, 0.90)
    xmin = float(np.min(x))
    xmax = float(np.max(x))

    return {
        f"{prefix}_n": int(len(x)),
        f"{prefix}_median_db": float(np.median(x)),
        f"{prefix}_p10_db": p10,
        f"{prefix}_p25_db": p25,
        f"{prefix}_p75_db": p75,
        f"{prefix}_p90_db": p90,
        f"{prefix}_min_db": xmin,
        f"{prefix}_max_db": xmax,
        f"{prefix}_iqr_db": float(p75 - p25),
        f"{prefix}_range_db": float(xmax - xmin),
    }


def summarize_temporal_changes(
    dates: pd.Series,
    values: pd.Series,
    prefix: str,
) -> dict:
    """
    Summarize changes between consecutive AVAILABLE valid observations.

    Raw step deltas are retained because they describe the observed temporal
    sequence. Per-day rates are also reported so irregular revisit intervals
    are explicit rather than silently ignored.

    No interpolation is performed.
    """
    z = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce"),
        "value": pd.to_numeric(values, errors="coerce"),
    })

    z = (
        z.loc[
            z["date"].notna()
            & np.isfinite(z["value"].to_numpy(float))
        ]
        .sort_values("date")
        .drop_duplicates("date", keep="first")
        .reset_index(drop=True)
    )

    if len(z) < 2:
        return {
            f"{prefix}_valid_steps_n": 0,
            f"{prefix}_max_negative_step_db": np.nan,
            f"{prefix}_max_positive_step_db": np.nan,
            f"{prefix}_median_abs_step_db": np.nan,
            f"{prefix}_max_negative_rate_db_per_day": np.nan,
            f"{prefix}_max_positive_rate_db_per_day": np.nan,
            f"{prefix}_median_abs_rate_db_per_day": np.nan,
        }

    dv = np.diff(z["value"].to_numpy(float))
    dd = (
        np.diff(z["date"].to_numpy(dtype="datetime64[D]"))
        .astype("timedelta64[D]")
        .astype(float)
    )

    valid = np.isfinite(dv) & np.isfinite(dd) & (dd > 0)
    dv = dv[valid]
    dd = dd[valid]

    if len(dv) == 0:
        return {
            f"{prefix}_valid_steps_n": 0,
            f"{prefix}_max_negative_step_db": np.nan,
            f"{prefix}_max_positive_step_db": np.nan,
            f"{prefix}_median_abs_step_db": np.nan,
            f"{prefix}_max_negative_rate_db_per_day": np.nan,
            f"{prefix}_max_positive_rate_db_per_day": np.nan,
            f"{prefix}_median_abs_rate_db_per_day": np.nan,
        }

    rates = dv / dd

    return {
        f"{prefix}_valid_steps_n": int(len(dv)),
        f"{prefix}_max_negative_step_db": float(np.min(dv)),
        f"{prefix}_max_positive_step_db": float(np.max(dv)),
        f"{prefix}_median_abs_step_db": float(np.median(np.abs(dv))),
        f"{prefix}_max_negative_rate_db_per_day": float(np.min(rates)),
        f"{prefix}_max_positive_rate_db_per_day": float(np.max(rates)),
        f"{prefix}_median_abs_rate_db_per_day":
            float(np.median(np.abs(rates))),
    }


def phase_mask(dates: pd.Series, phase: str) -> pd.Series:
    if phase not in PHASES:
        raise KeyError(phase)

    start_md, end_md = PHASES[phase]
    d = pd.to_datetime(dates, errors="coerce")

    md = d.dt.strftime("%m-%d")
    return md.ge(start_md) & md.le(end_md)


def date_gap_summary(dates: pd.Series) -> dict:
    d = (
        pd.to_datetime(dates, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(d) < 2:
        return {
            "median_gap_days": np.nan,
            "max_gap_days": np.nan,
        }

    gaps = d.diff().dt.days.dropna().to_numpy(float)

    return {
        "median_gap_days": float(np.median(gaps)),
        "max_gap_days": float(np.max(gaps)),
    }


def make_feature_dictionary() -> pd.DataFrame:
    rows = []

    base = [
        ("support_id", "identifier",
         "Fixed RiceFloodIT-compatible support-coordinate identifier."),
        ("year", "identifier",
         "Calendar year of the frozen rice-season track-year series."),
        ("orbit_state", "identifier",
         "Sentinel-1 orbit state; track families are not pooled."),
        ("relative_orbit", "identifier",
         "Sentinel-1 relative orbit; track families are not pooled."),
        ("lon", "identifier",
         "Fixed support longitude."),
        ("lat", "identifier",
         "Fixed support latitude."),
        ("support_dates_n", "sampling",
         "Unique acquisition dates represented by rows for this support/track/year."),
        ("vv_observed_dates_n", "sampling",
         "Dates with valid corrected VV observation."),
        ("vh_observed_dates_n", "sampling",
         "Dates with valid corrected VH observation."),
        ("both_observed_dates_n", "sampling",
         "Dates with both corrected VV and VH observed."),
        ("vv_observed_fraction", "sampling",
         "VV observed dates divided by support_dates_n; no imputation."),
        ("vh_observed_fraction", "sampling",
         "VH observed dates divided by support_dates_n; no imputation."),
        ("both_observed_fraction", "sampling",
         "Both-polarization observed dates divided by support_dates_n; no imputation."),
        ("median_gap_days", "sampling",
         "Median gap between represented acquisition dates."),
        ("max_gap_days", "sampling",
         "Maximum gap between represented acquisition dates."),
    ]

    for name, family, desc in base:
        rows.append({
            "feature": name,
            "family": family,
            "description": desc,
            "outcome_tuned": False,
            "imputation_used": False,
        })

    for sig in SIGNALS:
        label = {
            "vv": "VV corrected sigma0 dB",
            "vh": "VH corrected sigma0 dB",
            "vv_minus_vh": "VV minus VH dB",
        }[sig]

        distribution = [
            ("n", "Number of finite observations."),
            ("median_db", "Median across finite rice-season observations."),
            ("p10_db", "10th percentile."),
            ("p25_db", "25th percentile."),
            ("p75_db", "75th percentile."),
            ("p90_db", "90th percentile."),
            ("min_db", "Minimum."),
            ("max_db", "Maximum."),
            ("iqr_db", "75th minus 25th percentile."),
            ("range_db", "Maximum minus minimum."),
        ]

        for suffix, desc in distribution:
            rows.append({
                "feature": f"{sig}_{suffix}",
                "family": "seasonal_distribution",
                "description": f"{label}: {desc}",
                "outcome_tuned": False,
                "imputation_used": False,
            })

        changes = [
            ("valid_steps_n",
             "Number of consecutive valid-observation intervals."),
            ("max_negative_step_db",
             "Most negative change between consecutive available valid observations."),
            ("max_positive_step_db",
             "Most positive change between consecutive available valid observations."),
            ("median_abs_step_db",
             "Median absolute change between consecutive available valid observations."),
            ("max_negative_rate_db_per_day",
             "Most negative consecutive-observation change normalized by elapsed days."),
            ("max_positive_rate_db_per_day",
             "Most positive consecutive-observation change normalized by elapsed days."),
            ("median_abs_rate_db_per_day",
             "Median absolute consecutive-observation change normalized by elapsed days."),
        ]

        for suffix, desc in changes:
            rows.append({
                "feature": f"{sig}_{suffix}",
                "family": "temporal_change",
                "description": f"{label}: {desc}",
                "outcome_tuned": False,
                "imputation_used": False,
            })

    for phase in PHASES:
        for sig in SIGNALS:
            rows.append({
                "feature": f"{phase}_{sig}_median_db",
                "family": "fixed_calendar_window",
                "description": (
                    f"Median {sig} dB in fixed {phase} calendar window; "
                    "calendar stratum, not field-observed crop stage."
                ),
                "outcome_tuned": False,
                "imputation_used": False,
            })

        rows.append({
            "feature": f"{phase}_support_dates_n",
            "family": "fixed_calendar_window_sampling",
            "description": (
                f"Unique represented acquisition dates in fixed {phase} window."
            ),
            "outcome_tuned": False,
            "imputation_used": False,
        })
        rows.append({
            "feature": f"{phase}_both_observed_dates_n",
            "family": "fixed_calendar_window_sampling",
            "description": (
                f"Dates with both VV and VH observed in fixed {phase} window."
            ),
            "outcome_tuned": False,
            "imputation_used": False,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Inventory and structural audit
# =============================================================================

if not IN_ROOT.exists():
    raise FileNotFoundError(
        f"Frozen C2V primary mosaic directory not found: {IN_ROOT}"
    )

files = sorted(IN_ROOT.rglob("*.csv.gz"))

print("DESIGN C - C2X-PH OUTCOME-BLIND SENTINEL-1 PHENOLOGY CONSTRUCTION")
print("=" * 86)
print(f"Input mosaic files: {len(files)}")
print("Groundwater read: False")
print("Irrigation-flow read: False")
print("RiceFloodIT flooding values read: False")
print("Threshold/classifier: False")
print("Association model: False")
print()

hard_flags = []

if len(files) != EXPECTED_FILES_N:
    hard_flags.append(
        f"FAIL_MOSAIC_FILE_COUNT:{len(files)}!=expected_{EXPECTED_FILES_N}"
    )

inventory_rows = []
feature_parts = []
sampling_rows = []
all_support_ids = set()
schemas = set()
track_year_seen = set()


# =============================================================================
# Construct track-year features
# =============================================================================

for j, f in enumerate(files, 1):

    print(f"[{j:02d}/{len(files):02d}] {f}", flush=True)

    x = pd.read_csv(f, compression="gzip", low_memory=False)

    schema = tuple(x.columns.tolist())
    schemas.add(schema)

    missing = [c for c in EXPECTED_SCHEMA if c not in x.columns]
    extra = [c for c in x.columns if c not in EXPECTED_SCHEMA]

    if missing:
        hard_flags.append(
            f"FAIL_SCHEMA_MISSING:{f}:{missing}"
        )

    if extra:
        hard_flags.append(
            f"FAIL_SCHEMA_EXTRA:{f}:{extra}"
        )

    if missing:
        continue

    x["year"] = pd.to_numeric(x["year"], errors="raise").astype(int)
    x["relative_orbit"] = pd.to_numeric(
        x["relative_orbit"], errors="raise"
    ).astype(int)
    x["acquisition_date"] = pd.to_datetime(
        x["acquisition_date"], errors="raise"
    )

    x["orbit_state"] = x["orbit_state"].astype(str).str.lower()
    x["stream"] = x["stream"].astype(str).str.lower()
    x["support_id"] = x["support_id"].astype(str)

    for c in ["vv_observed", "vh_observed", "vv_vh_both_observed"]:
        x[c] = to_bool_series(x[c])

    # Exact primary-stream gate.
    streams = sorted(x["stream"].dropna().unique().tolist())
    if streams != ["primary"]:
        hard_flags.append(
            f"FAIL_NONPRIMARY_STREAM:{f}:{streams}"
        )

    years = sorted(x["year"].unique().tolist())
    states = sorted(x["orbit_state"].unique().tolist())
    robs = sorted(x["relative_orbit"].unique().tolist())

    if len(years) != 1 or len(states) != 1 or len(robs) != 1:
        hard_flags.append(
            f"FAIL_FILE_NOT_SINGLE_TRACK_YEAR:{f}:"
            f"years={years};states={states};orbits={robs}"
        )
        continue

    year = int(years[0])
    orbit_state = str(states[0])
    relative_orbit = int(robs[0])

    ty = (year, orbit_state, relative_orbit)
    track_year_seen.add(ty)

    if year not in EXPECTED_YEARS:
        hard_flags.append(
            f"FAIL_UNEXPECTED_YEAR:{f}:{year}"
        )

    if (orbit_state, relative_orbit) not in EXPECTED_TRACKS:
        hard_flags.append(
            f"FAIL_UNEXPECTED_TRACK:{f}:{orbit_state}_{relative_orbit}"
        )

    # Duplicate date/support rows would violate the frozen mosaic contract.
    dup = int(
        x.duplicated(
            ["acquisition_date", "support_id"],
            keep=False,
        ).sum()
    )

    if dup:
        hard_flags.append(
            f"FAIL_DUPLICATE_DATE_SUPPORT_ROWS:{f}:{dup}"
        )

    file_dates_n = int(x["acquisition_date"].nunique())
    file_support_n = int(x["support_id"].nunique())

    inventory_rows.append({
        "path": str(f.relative_to(ROOT)),
        "year": year,
        "orbit_state": orbit_state,
        "relative_orbit": relative_orbit,
        "rows_n": int(len(x)),
        "acquisition_dates_n": file_dates_n,
        "support_ids_n": file_support_n,
        "duplicate_date_support_rows_n": dup,
        "first_date": (
            x["acquisition_date"].min().date().isoformat()
            if len(x) else ""
        ),
        "last_date": (
            x["acquisition_date"].max().date().isoformat()
            if len(x) else ""
        ),
    })

    all_support_ids.update(x["support_id"].unique().tolist())

    # Sort before group operations.
    x = x.sort_values(
        ["support_id", "acquisition_date"]
    ).reset_index(drop=True)

    out_rows = []

    for support_id, g in x.groupby("support_id", sort=False):

        g = g.sort_values("acquisition_date").copy()

        lon_vals = pd.to_numeric(
            g["lon"], errors="coerce"
        ).dropna().unique()
        lat_vals = pd.to_numeric(
            g["lat"], errors="coerce"
        ).dropna().unique()

        if len(lon_vals) != 1 or len(lat_vals) != 1:
            hard_flags.append(
                f"FAIL_SUPPORT_COORDINATE_INSTABILITY:"
                f"{support_id}:{year}:{orbit_state}:{relative_orbit}"
            )
            lon = float(lon_vals[0]) if len(lon_vals) else np.nan
            lat = float(lat_vals[0]) if len(lat_vals) else np.nan
        else:
            lon = float(lon_vals[0])
            lat = float(lat_vals[0])

        support_dates_n = int(g["acquisition_date"].nunique())
        vv_n = int(g["vv_observed"].sum())
        vh_n = int(g["vh_observed"].sum())
        both_n = int(g["vv_vh_both_observed"].sum())

        row = {
            "support_id": support_id,
            "year": year,
            "orbit_state": orbit_state,
            "relative_orbit": relative_orbit,
            "lon": lon,
            "lat": lat,
            "support_dates_n": support_dates_n,
            "vv_observed_dates_n": vv_n,
            "vh_observed_dates_n": vh_n,
            "both_observed_dates_n": both_n,
            "vv_observed_fraction":
                float(vv_n / support_dates_n)
                if support_dates_n else np.nan,
            "vh_observed_fraction":
                float(vh_n / support_dates_n)
                if support_dates_n else np.nan,
            "both_observed_fraction":
                float(both_n / support_dates_n)
                if support_dates_n else np.nan,
            **date_gap_summary(g["acquisition_date"]),
        }

        # Whole-season distribution and temporal-change descriptors.
        for sig, col in SIGNALS.items():
            row.update(
                summarize_signal(
                    g[col],
                    sig,
                )
            )
            row.update(
                summarize_temporal_changes(
                    g["acquisition_date"],
                    g[col],
                    sig,
                )
            )

        # Prespecified calendar strata.
        for phase in PHASES:

            m = phase_mask(g["acquisition_date"], phase)
            gp = g.loc[m].copy()

            row[f"{phase}_support_dates_n"] = int(
                gp["acquisition_date"].nunique()
            )
            row[f"{phase}_both_observed_dates_n"] = int(
                gp["vv_vh_both_observed"].sum()
            )

            for sig, col in SIGNALS.items():
                vals = finite_values(gp[col])
                row[f"{phase}_{sig}_median_db"] = (
                    float(np.median(vals))
                    if len(vals)
                    else np.nan
                )

        out_rows.append(row)

    fy = pd.DataFrame(out_rows)

    if len(fy):
        fy = fy.sort_values("support_id").reset_index(drop=True)

    feature_parts.append(fy)

    sampling_rows.append({
        "year": year,
        "orbit_state": orbit_state,
        "relative_orbit": relative_orbit,
        "file_path": str(f.relative_to(ROOT)),
        "file_acquisition_dates_n": file_dates_n,
        "file_support_ids_n": file_support_n,
        "feature_rows_n": int(len(fy)),
        "median_support_dates_n": (
            float(fy["support_dates_n"].median())
            if len(fy) else np.nan
        ),
        "min_support_dates_n": (
            int(fy["support_dates_n"].min())
            if len(fy) else 0
        ),
        "max_support_dates_n": (
            int(fy["support_dates_n"].max())
            if len(fy) else 0
        ),
        "median_both_observed_fraction": (
            float(fy["both_observed_fraction"].median())
            if len(fy) else np.nan
        ),
        "support_rows_with_zero_both_observed_n": (
            int((fy["both_observed_dates_n"] == 0).sum())
            if len(fy) else 0
        ),
    })


# =============================================================================
# Final structural gates
# =============================================================================

expected_track_year = {
    (year, state, orbit)
    for year in EXPECTED_YEARS
    for state, orbit in EXPECTED_TRACKS
}

missing_ty = sorted(expected_track_year - track_year_seen)
unexpected_ty = sorted(track_year_seen - expected_track_year)

if missing_ty:
    hard_flags.append(
        f"FAIL_MISSING_TRACK_YEAR_COMBINATIONS:{missing_ty}"
    )

if unexpected_ty:
    hard_flags.append(
        f"FAIL_UNEXPECTED_TRACK_YEAR_COMBINATIONS:{unexpected_ty}"
    )

if len(schemas) != 1:
    hard_flags.append(
        f"FAIL_MULTIPLE_INPUT_SCHEMAS:{len(schemas)}"
    )
else:
    only_schema = list(next(iter(schemas)))
    if only_schema != EXPECTED_SCHEMA:
        hard_flags.append(
            "FAIL_INPUT_SCHEMA_NOT_EXACT_FROZEN_SCHEMA"
        )

if len(all_support_ids) != EXPECTED_SUPPORT_N:
    hard_flags.append(
        f"FAIL_SUPPORT_UNIVERSE:"
        f"{len(all_support_ids)}!=expected_{EXPECTED_SUPPORT_N}"
    )


# =============================================================================
# Write outputs
# =============================================================================

features = (
    pd.concat(feature_parts, ignore_index=True)
    if feature_parts
    else pd.DataFrame()
)

if len(features):
    features = features.sort_values(
        ["year", "orbit_state", "relative_orbit", "support_id"]
    ).reset_index(drop=True)

sampling = pd.DataFrame(sampling_rows)
if len(sampling):
    sampling = sampling.sort_values(
        ["year", "orbit_state", "relative_orbit"]
    ).reset_index(drop=True)

dictionary = make_feature_dictionary()

features.to_csv(
    FEATURE_OUT,
    index=False,
    compression="gzip",
)

sampling.to_csv(
    SAMPLING_OUT,
    index=False,
)

dictionary.to_csv(
    DICT_OUT,
    index=False,
)


# =============================================================================
# QA
# =============================================================================

feature_rows_expected_max = EXPECTED_SUPPORT_N * EXPECTED_FILES_N

status = (
    "PASS_OUTCOME_BLIND_PHENOLOGY_CONSTRUCTION"
    if not hard_flags
    else "FAIL_REVIEW_REQUIRED"
)

qa = {
    "stage":
        "DESIGN_C_C2X_PH_OUTCOME_BLIND_SENTINEL1_PHENOLOGY_CONSTRUCTION",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,

    "input_root": str(IN_ROOT.relative_to(ROOT)),
    "input_mosaic_files_n": int(len(files)),
    "expected_mosaic_files_n": EXPECTED_FILES_N,
    "unique_input_schemas_n": int(len(schemas)),

    "years_expected": EXPECTED_YEARS,
    "tracks_expected": [
        {
            "orbit_state": state,
            "relative_orbit": orbit,
        }
        for state, orbit in sorted(EXPECTED_TRACKS)
    ],
    "track_year_combinations_expected_n":
        int(len(expected_track_year)),
    "track_year_combinations_observed_n":
        int(len(track_year_seen)),
    "missing_track_year_combinations": missing_ty,
    "unexpected_track_year_combinations": unexpected_ty,

    "support_ids_expected_n": EXPECTED_SUPPORT_N,
    "support_ids_observed_n": int(len(all_support_ids)),

    "feature_rows_n": int(len(features)),
    "maximum_possible_support_track_year_rows_n":
        int(feature_rows_expected_max),

    "calendar_windows": {
        k: {
            "start_mm_dd": v[0],
            "end_mm_dd": v[1],
            "interpretation":
                "fixed calendar stratum; not field-observed crop stage",
        }
        for k, v in PHASES.items()
    },

    "primary_signals": {
        "VV": "vv_sigma0_corrected_db",
        "VH": "vh_sigma0_corrected_db",
        "VV_minus_VH": "vv_minus_vh_db",
    },

    "track_pooling_performed": False,
    "cross_track_standardization_performed": False,
    "missing_signal_imputation_performed": False,
    "temporal_interpolation_performed": False,
    "nonpositive_corrected_power_recovered_or_clipped": False,

    "groundwater_values_read": False,
    "irrigation_flow_values_read": False,
    "ricefloodit_flood_values_read": False,
    "inundation_threshold_selected": False,
    "classifier_fitted": False,
    "feature_selection_against_outcomes_performed": False,
    "association_models_fitted": 0,

    "hard_flags": hard_flags,

    "outputs": {
        "track_year_phenology_features":
            str(FEATURE_OUT.relative_to(ROOT)),
        "track_year_sampling_qa":
            str(SAMPLING_OUT.relative_to(ROOT)),
        "feature_dictionary":
            str(DICT_OUT.relative_to(ROOT)),
        "qa_json":
            str(QA_OUT.relative_to(ROOT)),
        "summary_txt":
            str(SUMMARY_OUT.relative_to(ROOT)),
    },
}

QA_OUT.write_text(
    json.dumps(qa, indent=2) + "\n",
    encoding="utf-8",
)


# =============================================================================
# Summary
# =============================================================================

lines = [
    "DESIGN C - C2X-PH OUTCOME-BLIND SENTINEL-1 PHENOLOGY CONSTRUCTION",
    "=" * 86,
    "",
    f"Input mosaic files: {len(files)} / expected {EXPECTED_FILES_N}",
    f"Unique input schemas: {len(schemas)}",
    f"Track-year combinations: {len(track_year_seen)} / expected {len(expected_track_year)}",
    f"Support IDs represented: {len(all_support_ids)} / expected {EXPECTED_SUPPORT_N}",
    f"Feature rows produced: {len(features)}",
    "",
    "FROZEN FEATURE ARCHITECTURE",
    "---------------------------",
    "Signals: corrected VV dB, corrected VH dB, VV-VH dB.",
    "Tracks remain separate: asc15, asc88, desc66, desc168.",
    "Whole-season distributional descriptors are prespecified.",
    "Consecutive-observation change descriptors are prespecified.",
    "Calendar strata: early 04-01..05-31; mid 06-01..07-15; late 07-16..09-30.",
    "Calendar strata are not treated as field-observed crop-development stages.",
    "",
    "MISSINGNESS",
    "-----------",
    "Missing corrected Sentinel-1 observations remain missing.",
    "No temporal interpolation, nearest-date replacement, clipping, or imputation.",
    "Sampling density and temporal gaps are retained as QA features.",
    "",
    "HARD QA FLAGS",
    "-------------",
]

if hard_flags:
    lines.extend(hard_flags)
else:
    lines.append("None")

lines += [
    "",
    "FIREWALL",
    "--------",
    "Groundwater read: False",
    "Irrigation-flow read: False",
    "RiceFloodIT flood values read: False",
    "Threshold selected: False",
    "Classifier fitted: False",
    "Outcome-based feature selection: False",
    "Association model fitted: False",
    "",
    f"C2X-PH STATUS: {status}",
]

summary = "\n".join(lines) + "\n"

SUMMARY_OUT.write_text(
    summary,
    encoding="utf-8",
)

print()
print(summary, end="")

if hard_flags:
    raise RuntimeError(
        "C2X-PH failed structural QA; inspect "
        f"{QA_OUT} and {SAMPLING_OUT}."
    )
