from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CP_DIR = (
    ROOT
    / "outputs"
    / "diagnostics"
    / "design_c"
    / "c2v"
    / "scene_checkpoints"
    / "primary"
)

OUT = (
    ROOT
    / "outputs"
    / "diagnostics"
    / "design_c"
    / "c2w"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "S1C_IW_GRDH_1SDV_20250421T053447_20250421T053516_001987_003FE6_9043_COG",
    "S1B_IW_GRDH_1SDV_20180704T172205_20180704T172229_011664_01573E_AD95_COG",
    "S1B_IW_GRDH_1SDV_20180622T172204_20180622T172228_011489_0151C9_5837_COG",
]


def read_checkpoint(path):
    try:
        return pd.read_csv(path, compression="gzip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def bool_true(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


results = []
detail_rows = []

all_files = sorted(CP_DIR.glob("*.csv.gz"))

for target_scene in TARGETS:

    target_path = CP_DIR / f"{target_scene}.csv.gz"

    if not target_path.exists():
        raise FileNotFoundError(target_path)

    target = read_checkpoint(target_path)

    if target.empty:
        raise RuntimeError(f"Target unexpectedly empty: {target_scene}")

    acquisition_date = str(target["acquisition_date"].iloc[0])
    orbit_state = str(target["orbit_state"].iloc[0])
    relative_orbit = int(target["relative_orbit"].iloc[0])

    print()
    print("=" * 100)
    print(target_scene)
    print("=" * 100)
    print(
        f"Date={acquisition_date} "
        f"orbit={orbit_state} "
        f"relative_orbit={relative_orbit}"
    )

    # ---------------------------------------------------------
    # Find every checkpoint belonging to this same date/track
    # ---------------------------------------------------------
    companions = []

    for path in all_files:

        # Fast filename date filter
        if acquisition_date.replace("-", "") not in path.name:
            continue

        d = read_checkpoint(path)

        if d.empty:
            continue

        if (
            str(d["acquisition_date"].iloc[0]) == acquisition_date
            and str(d["orbit_state"].iloc[0]) == orbit_state
            and int(d["relative_orbit"].iloc[0]) == relative_orbit
        ):
            companions.append((path.name.replace(".csv.gz", ""), d))

    print(f"Same-date/track scene checkpoints: {len(companions)}")

    for scene_id, _ in companions:
        print(f"  {scene_id}")

    for pol in ["vv", "vh"]:

        target_inside = bool_true(
            target[f"{pol}_raster_inside"]
        )

        target_noise_ok = bool_true(
            target[f"{pol}_noise_support_ok"]
        )

        target_fail = target_inside & ~target_noise_ok

        failed = target.loc[
            target_fail,
            ["support_id", "lon", "lat"]
        ].copy()

        failed_ids = set(failed["support_id"])

        # Map each failed support point to all alternative observations
        rescue_records = []

        for scene_id, d in companions:

            # do not count target itself as rescue
            if scene_id == target_scene:
                continue

            q = d[d["support_id"].isin(failed_ids)].copy()

            if q.empty:
                continue

            inside = bool_true(
                q[f"{pol}_raster_inside"]
            )

            noise_ok = bool_true(
                q[f"{pol}_noise_support_ok"]
            )

            sigma = pd.to_numeric(
                q[f"{pol}_sigma0_corrected_linear"],
                errors="coerce"
            )

            valid = inside & noise_ok & np.isfinite(sigma)

            tmp = pd.DataFrame(
                {
                    "support_id": q["support_id"],
                    "alternative_scene_id": scene_id,
                    "alternative_inside": inside,
                    "alternative_noise_ok": noise_ok,
                    "alternative_sigma0_valid": valid,
                    "alternative_sigma0_linear": sigma,
                }
            )

            rescue_records.append(tmp)

        if rescue_records:

            rescue = pd.concat(
                rescue_records,
                ignore_index=True
            )

            valid_by_support = (
                rescue.groupby("support_id")
                ["alternative_sigma0_valid"]
                .any()
            )

            alternatives_by_support = (
                rescue.groupby("support_id")
                .size()
            )

            valid_alternatives_n = (
                rescue[
                    rescue["alternative_sigma0_valid"]
                ]
                .groupby("support_id")
                .size()
            )

        else:

            rescue = pd.DataFrame()

            valid_by_support = pd.Series(dtype=bool)
            alternatives_by_support = pd.Series(dtype=int)
            valid_alternatives_n = pd.Series(dtype=int)

        failed["rescued_by_overlap"] = (
            failed["support_id"]
            .map(valid_by_support)
            .fillna(False)
        )

        failed["alternative_records_n"] = (
            failed["support_id"]
            .map(alternatives_by_support)
            .fillna(0)
            .astype(int)
        )

        failed["valid_alternative_observations_n"] = (
            failed["support_id"]
            .map(valid_alternatives_n)
            .fillna(0)
            .astype(int)
        )

        failed["target_scene_id"] = target_scene
        failed["year"] = int(target["year"].iloc[0])
        failed["acquisition_date"] = acquisition_date
        failed["orbit_state"] = orbit_state
        failed["relative_orbit"] = relative_orbit
        failed["polarization"] = pol.upper()

        detail_rows.append(failed)

        fail_n = len(failed)
        rescued_n = int(
            failed["rescued_by_overlap"].sum()
        )
        unresolved_n = fail_n - rescued_n

        results.append(
            {
                "target_scene_id": target_scene,
                "year": int(target["year"].iloc[0]),
                "acquisition_date": acquisition_date,
                "orbit_state": orbit_state,
                "relative_orbit": relative_orbit,
                "polarization": pol.upper(),
                "same_date_track_scenes_n": len(companions),
                "target_noise_fail_supports_n": fail_n,
                "rescued_by_overlap_n": rescued_n,
                "unresolved_after_overlap_n": unresolved_n,
                "rescue_fraction": (
                    rescued_n / fail_n
                    if fail_n > 0 else np.nan
                ),
            }
        )

        print()
        print(pol.upper())
        print("-" * 50)
        print(f"Target failures:           {fail_n}")
        print(f"Recovered by overlap:      {rescued_n}")
        print(f"Unresolved after overlap:  {unresolved_n}")
        print(
            f"Rescue fraction:           "
            f"{rescued_n / fail_n:.6f}"
            if fail_n else
            "Rescue fraction:           NA"
        )


summary = pd.DataFrame(results)

details = pd.concat(
    detail_rows,
    ignore_index=True
)

summary.to_csv(
    OUT / "c2w_noise_overlap_rescue_summary.csv",
    index=False
)

details.to_csv(
    OUT / "c2w_noise_overlap_rescue_points.csv",
    index=False
)

print()
print("=" * 100)
print("OVERLAP RESCUE SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))

print()
print("TOTALS")
print("-" * 50)

for pol in ["VV", "VH"]:

    g = summary[
        summary["polarization"] == pol
    ]

    total_fail = int(
        g["target_noise_fail_supports_n"].sum()
    )

    total_rescued = int(
        g["rescued_by_overlap_n"].sum()
    )

    total_unresolved = int(
        g["unresolved_after_overlap_n"].sum()
    )

    print(
        f"{pol}: failures={total_fail}, "
        f"rescued={total_rescued}, "
        f"unresolved={total_unresolved}"
    )

print()
print(
    "Outputs written to:",
    OUT
)