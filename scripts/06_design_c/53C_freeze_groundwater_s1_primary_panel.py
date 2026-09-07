"""
Design C — Stage 53C
Resolve documented 10-km structural spatial non-overlap and freeze the
groundwater–Sentinel-1 analysis panel WITHOUT fitting any association model.

Scientific basis
----------------
Stage 53B-v2 constructed the complete 2015–2025 groundwater/S1 panel and failed
only because two wells had zero support points inside the already-frozen 10-km
primary radius.

Those same two wells were documented previously, before the present
groundwater–S1 association analysis, as geometry-zero wells in the historical
10-km flooding-support architecture:

    PO0181220U0001
    PO0181550U0001

Stage 52 requires a valid frozen S1 exposure for primary well-year eligibility.
Therefore station-years from a well with zero 10-km support are mechanically
ineligible for the primary analysis. They are NOT removed based on the
groundwater–S1 coefficient or significance.

No radius is changed. 20-km values remain sensitivity-only and must never
rescue missing 10-km primary exposures.

Inputs
------
outputs/diagnostics/design_c/c2zg_groundwater_panel/
    c2zg_groundwater_s1_analysis_panel_2015_2025.csv
    c2zg_well_s1_spatial_linkage_qa.csv
    c2zg_groundwater_s1_panel_qa.json
    c2zg_groundwater_s1_panel_coverage.csv

outputs/diagnostics/design_c/c2ze_groundwater_protocol/
    c2ze_groundwater_association_protocol.json

Outputs
-------
outputs/diagnostics/design_c/c2zh_groundwater_panel_freeze/
    c2zh_structural_spatial_nonoverlap_audit.csv
    c2zh_frozen_primary_groundwater_s1_panel.csv
    c2zh_frozen_primary_panel_manifest.json
    c2zh_frozen_primary_panel_summary.txt

No hydrological association model is fitted.
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

P52 = (
    D / "c2ze_groundwater_protocol"
    / "c2ze_groundwater_association_protocol.json"
)

SRC_DIR = D / "c2zg_groundwater_panel"
PANEL_IN = SRC_DIR / "c2zg_groundwater_s1_analysis_panel_2015_2025.csv"
LINK_IN = SRC_DIR / "c2zg_well_s1_spatial_linkage_qa.csv"
QA53B = SRC_DIR / "c2zg_groundwater_s1_panel_qa.json"
COVERAGE_IN = SRC_DIR / "c2zg_groundwater_s1_panel_coverage.csv"

OUT = D / "c2zh_groundwater_panel_freeze"
OUT.mkdir(parents=True, exist_ok=True)

AUDIT_OUT = OUT / "c2zh_structural_spatial_nonoverlap_audit.csv"
PRIMARY_OUT = OUT / "c2zh_frozen_primary_groundwater_s1_panel.csv"
MANIFEST_OUT = OUT / "c2zh_frozen_primary_panel_manifest.json"
SUMMARY_OUT = OUT / "c2zh_frozen_primary_panel_summary.txt"

for f in [P52, PANEL_IN, LINK_IN, QA53B, COVERAGE_IN]:
    if not f.exists():
        raise FileNotFoundError(f)

p52 = json.loads(P52.read_text(encoding="utf-8"))
qa53 = json.loads(QA53B.read_text(encoding="utf-8"))

if p52.get("status") != "PROTOCOL_FROZEN_GROUNDWATER_VALUES_UNREAD":
    raise RuntimeError(
        f"Unexpected Stage-52 protocol status: {p52.get('status')!r}"
    )

# Stage 53B-v2 must have failed only because of the structural 10-km support gate.
flags53 = list(qa53.get("hard_flags", []))
allowed_prior_flag = "ONE_OR_MORE_WELLS_HAVE_ZERO_10KM_S1_SUPPORT"
unexpected_prior_flags = [x for x in flags53 if x != allowed_prior_flag]
if unexpected_prior_flags:
    raise RuntimeError(
        "Stage 53B-v2 had additional hard flags; cannot resolve automatically: "
        + ", ".join(unexpected_prior_flags)
    )

EXPECTED_GEOMETRY_ZERO_WELLS = {
    "PO0181220U0001",
    "PO0181550U0001",
}

panel = pd.read_csv(PANEL_IN, low_memory=False)
link = pd.read_csv(LINK_IN, low_memory=False)
coverage = pd.read_csv(COVERAGE_IN, low_memory=False)

required_panel = {
    "station", "year",
    "gw_aug_m", "gw_pre_last_janfeb_m",
    "s1_flooding_like_10km",
    "s1_support_points_10km_n",
    "s1_flooding_like_20km",
    "primary_complete",
}
missing = sorted(required_panel - set(panel.columns))
if missing:
    raise AssertionError(f"Panel missing columns: {missing}")

required_link = {
    "station", "radius_m", "support_points_n",
    "nearest_support_distance_m",
}
missing = sorted(required_link - set(link.columns))
if missing:
    raise AssertionError(f"Link QA missing columns: {missing}")

panel["station"] = panel["station"].astype(str)
panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
panel["s1_support_points_10km_n"] = pd.to_numeric(
    panel["s1_support_points_10km_n"], errors="coerce"
)
panel["s1_flooding_like_10km"] = pd.to_numeric(
    panel["s1_flooding_like_10km"], errors="coerce"
)
panel["s1_flooding_like_20km"] = pd.to_numeric(
    panel["s1_flooding_like_20km"], errors="coerce"
)

# Normalize bool if CSV parser read strings.
if panel["primary_complete"].dtype != bool:
    panel["primary_complete"] = (
        panel["primary_complete"].astype(str).str.lower().eq("true")
    )

link["station"] = link["station"].astype(str)
link["radius_m"] = pd.to_numeric(link["radius_m"], errors="raise").astype(int)
link["support_points_n"] = pd.to_numeric(
    link["support_points_n"], errors="raise"
).astype(int)
link["nearest_support_distance_m"] = pd.to_numeric(
    link["nearest_support_distance_m"], errors="coerce"
)

hard_flags = []

# ---------------------------------------------------------------------
# 1. Exact fixed geometry-zero wells at 10 km
# ---------------------------------------------------------------------
zero10 = link[
    link["radius_m"].eq(10000)
    & link["support_points_n"].eq(0)
].copy()

observed_zero_wells = set(zero10["station"])

if observed_zero_wells != EXPECTED_GEOMETRY_ZERO_WELLS:
    hard_flags.append(
        "UNEXPECTED_10KM_GEOMETRY_ZERO_SET:"
        f"{sorted(observed_zero_wells)}"
    )

if not (zero10["nearest_support_distance_m"] > 10000).all():
    hard_flags.append(
        "ZERO_SUPPORT_WELL_HAS_NEAREST_SUPPORT_WITHIN_10KM"
    )

# ---------------------------------------------------------------------
# 2. Every primary-missing row must arise only from these fixed wells
#    and must have exactly zero 10-km supports.
# ---------------------------------------------------------------------
missing_primary_s1 = panel[
    ~np.isfinite(panel["s1_flooding_like_10km"])
].copy()

missing_wells = set(missing_primary_s1["station"])

if missing_wells != EXPECTED_GEOMETRY_ZERO_WELLS:
    hard_flags.append(
        "PRIMARY_S1_MISSINGNESS_NOT_RESTRICTED_TO_EXPECTED_GEOMETRY_ZERO_WELLS:"
        f"{sorted(missing_wells)}"
    )

if not missing_primary_s1["s1_support_points_10km_n"].eq(0).all():
    hard_flags.append(
        "PRIMARY_S1_MISSING_WITH_NONZERO_10KM_SUPPORT"
    )

# Frozen Stage 52 mechanical eligibility requires a valid S1 exposure.
# Verify no groundwater variable caused these rows to be lost.
if not np.isfinite(missing_primary_s1["gw_aug_m"]).all():
    hard_flags.append("STRUCTURAL_NONOVERLAP_ROWS_HAVE_MISSING_GW_OUTCOME")

if not np.isfinite(
    missing_primary_s1["gw_pre_last_janfeb_m"]
).all():
    hard_flags.append("STRUCTURAL_NONOVERLAP_ROWS_HAVE_MISSING_ANTECEDENT_GW")

# 20-km exposure may exist, but it is sensitivity-only.
if not np.isfinite(missing_primary_s1["s1_flooding_like_20km"]).all():
    hard_flags.append(
        "STRUCTURAL_NONOVERLAP_ROWS_ALSO_MISSING_20KM_SENSITIVITY"
    )

# ---------------------------------------------------------------------
# 3. No other 10-km exposure loss among positive-support well-years
# ---------------------------------------------------------------------
positive_support = panel["s1_support_points_10km_n"] > 0
bad_positive = panel[
    positive_support
    & ~np.isfinite(panel["s1_flooding_like_10km"])
].copy()

if len(bad_positive):
    hard_flags.append(
        f"POSITIVE_10KM_SUPPORT_BUT_MISSING_PRIMARY_EXPOSURE:{len(bad_positive)}"
    )

# ---------------------------------------------------------------------
# 4. Freeze primary mechanically complete analysis universe
# ---------------------------------------------------------------------
expected_complete = (
    np.isfinite(panel["gw_aug_m"])
    & np.isfinite(panel["gw_pre_last_janfeb_m"])
    & np.isfinite(panel["s1_flooding_like_10km"])
)

if not (expected_complete.to_numpy() == panel["primary_complete"].to_numpy()).all():
    hard_flags.append("PRIMARY_COMPLETE_FLAG_DOES_NOT_MATCH_FROZEN_RULE")

primary = (
    panel.loc[expected_complete]
    .sort_values(["station", "year"])
    .reset_index(drop=True)
)

if primary.duplicated(["station", "year"]).any():
    hard_flags.append("DUPLICATE_STATION_YEAR_IN_FROZEN_PRIMARY_PANEL")

# Expected Stage 53B-v2 counts.
if len(panel) != 203:
    hard_flags.append(f"FULL_TEMPORAL_PANEL_ROWS:{len(panel)}!=203")
if len(primary) != 193:
    hard_flags.append(f"PRIMARY_ROWS:{len(primary)}!=193")
if primary["station"].nunique() != 32:
    hard_flags.append(
        f"PRIMARY_WELLS:{primary['station'].nunique()}!=32"
    )

expected_years = list(range(2015, 2026))
years = sorted(primary["year"].unique().tolist())
if years != expected_years:
    hard_flags.append(f"PRIMARY_YEAR_SET:{years}")

# Critically: there must be no 10-km spatial-overlap loss in 2022-2025.
post = panel[panel["year"].between(2022, 2025)]
post_missing10 = post[~np.isfinite(post["s1_flooding_like_10km"])]
if len(post_missing10):
    hard_flags.append(
        f"POST2021_PRIMARY_10KM_SPATIAL_LOSS:{len(post_missing10)}"
    )

# Reconfirm frozen post-2021 groundwater eligibility counts and full S1 retention.
expected_post_counts = {2022: 17, 2023: 13, 2024: 15, 2025: 17}
for yr, exp in expected_post_counts.items():
    y_full = panel[panel["year"].eq(yr)]
    y_primary = primary[primary["year"].eq(yr)]
    if len(y_full) != exp:
        hard_flags.append(f"TEMPORAL_COUNT_{yr}:{len(y_full)}!={exp}")
    if len(y_primary) != exp:
        hard_flags.append(f"PRIMARY_COUNT_{yr}:{len(y_primary)}!={exp}")

# ---------------------------------------------------------------------
# 5. Structural non-overlap audit artifact
# ---------------------------------------------------------------------
audit_rows = []

for _, r in zero10.sort_values("station").iterrows():
    st = r["station"]
    lost = missing_primary_s1[
        missing_primary_s1["station"].eq(st)
    ].sort_values("year")

    audit_rows.append({
        "station": st,
        "nearest_s1_support_distance_m":
            float(r["nearest_support_distance_m"]),
        "support_points_10km_n": int(r["support_points_n"]),
        "lost_primary_well_years_n": int(len(lost)),
        "lost_years": ";".join(map(str, lost["year"].astype(int).tolist())),
        "all_lost_rows_have_valid_groundwater":
            bool(
                np.isfinite(lost["gw_aug_m"]).all()
                and np.isfinite(lost["gw_pre_last_janfeb_m"]).all()
            ),
        "all_lost_rows_have_20km_sensitivity":
            bool(np.isfinite(lost["s1_flooding_like_20km"]).all()),
        "classification": "DOCUMENTED_STRUCTURAL_SPATIAL_NONOVERLAP",
        "primary_action":
            "mechanically ineligible at frozen 10-km radius; no substitution",
        "20km_action":
            "retain only as prespecified sensitivity; never rescue primary",
    })

audit = pd.DataFrame(audit_rows)
audit.to_csv(AUDIT_OUT, index=False)

primary.to_csv(PRIMARY_OUT, index=False)

status = (
    "PASS_PRIMARY_PANEL_FROZEN_WITH_DOCUMENTED_STRUCTURAL_NONOVERLAP"
    if not hard_flags
    else "FAIL_REVIEW_REQUIRED"
)

manifest = {
    "stage": "DESIGN_C_STAGE53C_GROUNDWATER_PANEL_FREEZE",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,

    "primary_panel_rule": (
        "valid August-median groundwater outcome + valid last-JanFeb "
        "antecedent groundwater + valid frozen 10-km S1 exposure"
    ),

    "documented_structural_spatial_nonoverlap": {
        "radius_m": 10000,
        "wells": sorted(EXPECTED_GEOMETRY_ZERO_WELLS),
        "well_years_lost_n": int(len(missing_primary_s1)),
        "lost_years_by_well": {
            st: sorted(
                missing_primary_s1.loc[
                    missing_primary_s1["station"].eq(st), "year"
                ].astype(int).tolist()
            )
            for st in sorted(EXPECTED_GEOMETRY_ZERO_WELLS)
        },
        "classification":
            "mechanical exposure ineligibility, not groundwater-driven exclusion",
        "20km_substitution_allowed": False,
    },

    "counts": {
        "temporally_eligible_groundwater_well_years_n": int(len(panel)),
        "primary_complete_well_years_n": int(len(primary)),
        "primary_wells_n": int(primary["station"].nunique()),
        "primary_wells_with_ge2_years_n":
            int((primary.groupby("station").size() >= 2).sum()),
        "primary_years": years,
        "post2021_primary_counts": {
            str(yr): int((primary["year"] == yr).sum())
            for yr in [2022, 2023, 2024, 2025]
        },
    },

    "hard_flags": hard_flags,

    "firewall": {
        "groundwater_values_read": True,
        "frozen_s1_exposure_read": True,
        "irrigation_flow_values_read": False,
        "hydrological_association_model_fitted": False,
        "groundwater_driven_station_exclusion": False,
        "groundwater_driven_radius_selection": False,
        "20km_used_to_rescue_primary": False,
    },

    "sha256": {
        "stage52_protocol":
            hashlib.sha256(P52.read_bytes()).hexdigest(),
        "stage53b_panel":
            hashlib.sha256(PANEL_IN.read_bytes()).hexdigest(),
        "stage53b_linkage_qa":
            hashlib.sha256(LINK_IN.read_bytes()).hexdigest(),
        "frozen_primary_panel":
            hashlib.sha256(PRIMARY_OUT.read_bytes()).hexdigest(),
    },
}

MANIFEST_OUT.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "DESIGN C - STAGE 53C FROZEN GROUNDWATER-S1 PRIMARY PANEL",
    "=" * 78,
    "",
    "DOCUMENTED STRUCTURAL 10-KM NON-OVERLAP",
    "---------------------------------------",
]

for _, r in audit.iterrows():
    lines += [
        f"{r['station']}:",
        f"  nearest S1 support = {r['nearest_s1_support_distance_m']:.3f} m",
        f"  10-km support points = {int(r['support_points_10km_n'])}",
        f"  mechanically lost years = {r['lost_years']}",
        f"  lost well-years = {int(r['lost_primary_well_years_n'])}",
    ]

lines += [
    "",
    "PRIMARY PANEL",
    "-------------",
    f"Temporally eligible groundwater well-years: {len(panel)}",
    f"Structural 10-km non-overlap well-years: {len(missing_primary_s1)}",
    f"Frozen primary well-years: {len(primary)}",
    f"Frozen primary wells: {primary['station'].nunique()}",
    f"Wells with >=2 primary years: "
    f"{int((primary.groupby('station').size() >= 2).sum())}",
    f"Years: {years}",
    "",
    "POST-2021 PRIMARY RETENTION",
    "---------------------------",
]

for yr in [2022, 2023, 2024, 2025]:
    lines.append(
        f"{yr}: {int((primary['year'] == yr).sum())} / "
        f"{expected_post_counts[yr]} retained"
    )

lines += [
    "",
    "RULE",
    "----",
    "The 10-km radius remains unchanged.",
    "The two zero-support wells are not outcome-selected exclusions.",
    "Their affected station-years are mechanically ineligible because the",
    "frozen primary exposure does not exist at the frozen 10-km radius.",
    "20-km exposure remains sensitivity-only and does not replace primary.",
    "",
    "HARD QA FLAGS",
    "-------------",
]
lines += hard_flags if hard_flags else ["None"]

lines += [
    "",
    "FIREWALL",
    "--------",
    "Groundwater values read: True",
    "Frozen Sentinel-1 exposure read: True",
    "Irrigation-flow values read: False",
    "Hydrological association model fitted: False",
    "Groundwater-driven station exclusion: False",
    "Groundwater-driven radius selection: False",
    "20-km rescue of primary exposure: False",
    "",
    f"STAGE 53C STATUS: {status}",
]

SUMMARY_OUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))

if hard_flags:
    raise RuntimeError(
        "Stage 53C failed hard QA; do not proceed to model fitting."
    )
