"""
Design C — Stage 53A groundwater source preflight.

Purpose
-------
Open the authoritative cleaned groundwater observation table for the first time
under the frozen Stage-52 protocol, verify temporal/population coverage, and
determine whether the raw observation record is sufficient to construct the
frozen August-median outcome and last-Jan-Feb antecedent state through the
Sentinel-1 era.

NO Sentinel-1 association model is fitted here.

Authoritative observation source
--------------------------------
data/processed/publication_groundwater/groundwater_clean.csv

Why this file?
--------------
It contains observation-level groundwater depth and dates. The annual tables
are derived products and contain August means, whereas Stage 52 froze the
primary outcome as the median of valid August observations. Therefore the
primary outcome must be reconstructed from observation-level records rather
than substituted with a mean.

Metadata source
---------------
data/processed/publication_groundwater/groundwater_station_metadata.csv

Outputs
-------
outputs/diagnostics/design_c/c2zf_groundwater_preflight/
    c2zf_groundwater_year_coverage.csv
    c2zf_groundwater_iss_station_coverage.csv
    c2zf_groundwater_preflight_qa.json
    c2zf_groundwater_preflight_summary.txt

Run
---
python -u scripts/06_design_c/53A_groundwater_source_preflight.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"

GW = ROOT / "data" / "processed" / "publication_groundwater" / "groundwater_clean.csv"
META = ROOT / "data" / "processed" / "publication_groundwater" / "groundwater_station_metadata.csv"
P52 = D / "c2ze_groundwater_protocol" / "c2ze_groundwater_association_protocol.json"

OUT = D / "c2zf_groundwater_preflight"
OUT.mkdir(parents=True, exist_ok=True)

YEAR_OUT = OUT / "c2zf_groundwater_year_coverage.csv"
STATION_OUT = OUT / "c2zf_groundwater_iss_station_coverage.csv"
QA_OUT = OUT / "c2zf_groundwater_preflight_qa.json"
SUMMARY_OUT = OUT / "c2zf_groundwater_preflight_summary.txt"

for f in [GW, META, P52]:
    if not f.exists():
        raise FileNotFoundError(f)

p52 = json.loads(P52.read_text(encoding="utf-8"))
if p52.get("status") != "PROTOCOL_FROZEN_GROUNDWATER_VALUES_UNREAD":
    raise RuntimeError(
        f"Unexpected Stage-52 protocol status: {p52.get('status')!r}"
    )

g = pd.read_csv(GW, low_memory=False)
m = pd.read_csv(META, low_memory=False)

required_gw = {
    "station", "gw_depth_m", "date", "utm_e", "utm_n",
    "year", "month", "aquifer_group"
}
missing = sorted(required_gw - set(g.columns))
if missing:
    raise AssertionError(f"groundwater_clean.csv missing columns: {missing}")

required_meta = {"station", "utm_e", "utm_n", "aquifer_group"}
missing = sorted(required_meta - set(m.columns))
if missing:
    raise AssertionError(f"groundwater_station_metadata.csv missing columns: {missing}")

g["station"] = g["station"].astype(str)
g["date"] = pd.to_datetime(g["date"], errors="coerce")
g["gw_depth_m"] = pd.to_numeric(g["gw_depth_m"], errors="coerce")
g["year"] = g["date"].dt.year
g["month"] = g["date"].dt.month
g["aquifer_group"] = g["aquifer_group"].astype(str).str.upper().str.strip()

# Frozen Stage-52 population.
excluded = {"PO0180930U0006", "PO0181870U0001"}
iss = g[g["aquifer_group"].eq("ISS")].copy()
iss = iss[~iss["station"].isin(excluded)].copy()

# Valid groundwater observations only.
iss["valid_depth"] = np.isfinite(iss["gw_depth_m"])

year_rows = []
for year in range(2015, 2026):
    y = iss[iss["year"].eq(year)].copy()
    aug = y[y["month"].eq(8) & y["valid_depth"]]
    jf = y[y["month"].isin([1, 2]) & y["valid_depth"]]

    # Frozen temporal eligibility: a well-year requires both Aug and Jan-Feb.
    aug_stations = set(aug["station"])
    jf_stations = set(jf["station"])
    eligible = aug_stations & jf_stations

    year_rows.append({
        "year": year,
        "iss_observation_rows_n": int(len(y)),
        "iss_stations_any_obs_n": int(y["station"].nunique()),
        "iss_aug_valid_obs_n": int(len(aug)),
        "iss_aug_stations_n": int(len(aug_stations)),
        "iss_janfeb_valid_obs_n": int(len(jf)),
        "iss_janfeb_stations_n": int(len(jf_stations)),
        "temporally_eligible_stations_n": int(len(eligible)),
    })

year_df = pd.DataFrame(year_rows)
year_df.to_csv(YEAR_OUT, index=False)

station_rows = []
for station, s in iss.groupby("station", sort=True):
    years_any = sorted(s.loc[s["valid_depth"], "year"].dropna().astype(int).unique().tolist())
    aug_years = sorted(
        s.loc[s["valid_depth"] & s["month"].eq(8), "year"]
        .dropna().astype(int).unique().tolist()
    )
    jf_years = sorted(
        s.loc[s["valid_depth"] & s["month"].isin([1, 2]), "year"]
        .dropna().astype(int).unique().tolist()
    )
    eligible_years = sorted(set(aug_years) & set(jf_years) & set(range(2015, 2026)))

    station_rows.append({
        "station": station,
        "first_valid_year": min(years_any) if years_any else np.nan,
        "last_valid_year": max(years_any) if years_any else np.nan,
        "s1_era_valid_years_n": len(set(years_any) & set(range(2015, 2026))),
        "s1_era_aug_years_n": len(set(aug_years) & set(range(2015, 2026))),
        "s1_era_janfeb_years_n": len(set(jf_years) & set(range(2015, 2026))),
        "s1_era_temporally_eligible_years_n": len(eligible_years),
        "eligible_years": ";".join(map(str, eligible_years)),
    })

station_df = pd.DataFrame(station_rows)
station_df.to_csv(STATION_OUT, index=False)

valid_all = g[np.isfinite(g["gw_depth_m"]) & g["date"].notna()]
valid_iss = iss[iss["valid_depth"] & iss["date"].notna()]

raw_min_date = valid_all["date"].min()
raw_max_date = valid_all["date"].max()
iss_min_date = valid_iss["date"].min()
iss_max_date = valid_iss["date"].max()

coverage_2022_2025 = year_df[
    year_df["year"].between(2022, 2025)
]["iss_observation_rows_n"].sum()

hard_flags = []
if len(valid_iss) == 0:
    hard_flags.append("NO_VALID_ISS_GROUNDWATER_OBSERVATIONS")
if year_df["temporally_eligible_stations_n"].sum() == 0:
    hard_flags.append("NO_TEMPORALLY_ELIGIBLE_S1_ERA_ISS_WELL_YEARS")

# This does not fail the stage; it tells us whether the observation-level source
# can support the full S1 era or whether a later raw source must be located.
full_era_raw_support = bool(coverage_2022_2025 > 0)

status = (
    "PASS_RAW_SOURCE_SUPPORTS_POST2021"
    if full_era_raw_support and not hard_flags
    else
    "PASS_PREFLIGHT_BUT_POST2021_RAW_SOURCE_REQUIRED"
    if not hard_flags
    else
    "FAIL_PREFLIGHT"
)

qa = {
    "stage": "DESIGN_C_STAGE53A_GROUNDWATER_SOURCE_PREFLIGHT",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "source": str(GW.relative_to(ROOT)),
    "metadata_source": str(META.relative_to(ROOT)),
    "raw_all_valid_date_min": None if pd.isna(raw_min_date) else str(raw_min_date.date()),
    "raw_all_valid_date_max": None if pd.isna(raw_max_date) else str(raw_max_date.date()),
    "iss_valid_date_min": None if pd.isna(iss_min_date) else str(iss_min_date.date()),
    "iss_valid_date_max": None if pd.isna(iss_max_date) else str(iss_max_date.date()),
    "iss_stations_after_frozen_exclusions_n": int(iss["station"].nunique()),
    "temporally_eligible_station_years_2015_2025_n":
        int(year_df["temporally_eligible_stations_n"].sum()),
    "post2021_observation_rows_present": bool(full_era_raw_support),
    "hard_flags": hard_flags,

    "groundwater_values_read": True,
    "groundwater_values_role": "source coverage and temporal eligibility preflight only",
    "sentinel1_values_read": False,
    "irrigation_flow_values_read": False,
    "hydrological_association_model_fitted": False,
}

QA_OUT.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

lines = [
    "DESIGN C - STAGE 53A GROUNDWATER SOURCE PREFLIGHT",
    "=" * 74,
    "",
    f"Observation source: {GW.relative_to(ROOT)}",
    f"Valid groundwater date range (all aquifer groups): "
    f"{qa['raw_all_valid_date_min']} to {qa['raw_all_valid_date_max']}",
    f"Valid ISS date range: {qa['iss_valid_date_min']} to {qa['iss_valid_date_max']}",
    f"ISS stations after frozen exclusions: {qa['iss_stations_after_frozen_exclusions_n']}",
    f"Temporally eligible ISS station-years, 2015-2025: "
    f"{qa['temporally_eligible_station_years_2015_2025_n']}",
    "",
    "YEAR COVERAGE",
    "-------------",
    year_df.to_string(index=False),
    "",
    "POST-2021 RAW SUPPORT",
    "---------------------",
    f"Observation-level 2022-2025 rows present: {full_era_raw_support}",
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
    "Sentinel-1 values read: False",
    "Irrigation-flow values read: False",
    "Hydrological association model fitted: False",
    "",
    f"STAGE 53A STATUS: {status}",
]

SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

if hard_flags:
    raise RuntimeError("Stage 53A groundwater source preflight failed.")
