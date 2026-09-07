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


def read_cp(path):
    try:
        return pd.read_csv(path, compression="gzip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def btrue(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


# ------------------------------------------------------------
# Build lightweight metadata catalogue for all checkpoints
# ------------------------------------------------------------

catalog = []

files = sorted(CP_DIR.glob("*.csv.gz"))

print(f"Indexing {len(files)} checkpoints...")

for i, path in enumerate(files, 1):

    if i % 200 == 0:
        print(f"  {i}/{len(files)}")

    d = read_cp(path)

    if d.empty:
        continue

    catalog.append(
        {
            "path": path,
            "scene_id": str(d["scene_id"].iloc[0]),
            "year": int(d["year"].iloc[0]),
            "date": pd.to_datetime(d["acquisition_date"].iloc[0]),
            "orbit_state": str(d["orbit_state"].iloc[0]),
            "relative_orbit": int(d["relative_orbit"].iloc[0]),
            "platform": str(d["platform"].iloc[0]),
        }
    )

catalog = pd.DataFrame(catalog)

summary_rows = []
temporal_rows = []
edge_rows = []


for target_scene in TARGETS:

    target_path = CP_DIR / f"{target_scene}.csv.gz"
    d = read_cp(target_path)

    target_date = pd.to_datetime(d["acquisition_date"].iloc[0])
    orbit_state = str(d["orbit_state"].iloc[0])
    relative_orbit = int(d["relative_orbit"].iloc[0])

    same_track = catalog[
        (catalog["orbit_state"] == orbit_state)
        & (catalog["relative_orbit"] == relative_orbit)
    ].copy()

    same_track["day_offset"] = (
        same_track["date"] - target_date
    ).dt.days

    # nearest previous and next acquisition date on same track
    previous = same_track[
        same_track["day_offset"] < 0
    ].sort_values("day_offset", ascending=False)

    following = same_track[
        same_track["day_offset"] > 0
    ].sort_values("day_offset")

    previous_date = (
        previous["date"].iloc[0]
        if len(previous)
        else pd.NaT
    )

    following_date = (
        following["date"].iloc[0]
        if len(following)
        else pd.NaT
    )

    print()
    print("=" * 100)
    print(target_scene)
    print("=" * 100)
    print(f"Target date:    {target_date.date()}")
    print(f"Track:          {orbit_state} {relative_orbit}")
    print(
        "Previous date:  "
        + (
            str(previous_date.date())
            if pd.notna(previous_date)
            else "NONE"
        )
    )
    print(
        "Following date: "
        + (
            str(following_date.date())
            if pd.notna(following_date)
            else "NONE"
        )
    )

    for pol in ["vv", "vh"]:

        inside = btrue(d[f"{pol}_raster_inside"])
        noise_ok = btrue(d[f"{pol}_noise_support_ok"])

        sigma = pd.to_numeric(
            d[f"{pol}_sigma0_corrected_linear"],
            errors="coerce"
        )

        fail = inside & ~noise_ok

        failed = d.loc[
            fail,
            [
                "support_id",
                "lon",
                "lat",
                f"{pol}_image_row",
                f"{pol}_image_col",
            ]
        ].copy()

        failed_ids = set(failed["support_id"])

        # ----------------------------------------------------
        # Spatial/image-coordinate comparison
        # ----------------------------------------------------

        rows_all = pd.to_numeric(
            d.loc[inside, f"{pol}_image_row"],
            errors="coerce"
        )

        cols_all = pd.to_numeric(
            d.loc[inside, f"{pol}_image_col"],
            errors="coerce"
        )

        rows_fail = pd.to_numeric(
            d.loc[fail, f"{pol}_image_row"],
            errors="coerce"
        )

        cols_fail = pd.to_numeric(
            d.loc[fail, f"{pol}_image_col"],
            errors="coerce"
        )

        edge_rows.append(
            {
                "scene_id": target_scene,
                "polarization": pol.upper(),
                "inside_n": int(inside.sum()),
                "fail_n": int(fail.sum()),

                "all_row_min": rows_all.min(),
                "all_row_p01": rows_all.quantile(0.01),
                "all_row_median": rows_all.median(),
                "all_row_p99": rows_all.quantile(0.99),
                "all_row_max": rows_all.max(),

                "fail_row_min": rows_fail.min(),
                "fail_row_p01": rows_fail.quantile(0.01),
                "fail_row_median": rows_fail.median(),
                "fail_row_p99": rows_fail.quantile(0.99),
                "fail_row_max": rows_fail.max(),

                "all_col_min": cols_all.min(),
                "all_col_p01": cols_all.quantile(0.01),
                "all_col_median": cols_all.median(),
                "all_col_p99": cols_all.quantile(0.99),
                "all_col_max": cols_all.max(),

                "fail_col_min": cols_fail.min(),
                "fail_col_p01": cols_fail.quantile(0.01),
                "fail_col_median": cols_fail.median(),
                "fail_col_p99": cols_fail.quantile(0.99),
                "fail_col_max": cols_fail.max(),
            }
        )

        # ----------------------------------------------------
        # Check nearest previous/next same-track acquisition
        # ----------------------------------------------------

        temporal_result = {
            "scene_id": target_scene,
            "polarization": pol.upper(),
            "target_date": target_date.date().isoformat(),
            "orbit_state": orbit_state,
            "relative_orbit": relative_orbit,
            "failed_support_n": len(failed_ids),
            "previous_date": (
                previous_date.date().isoformat()
                if pd.notna(previous_date)
                else None
            ),
            "following_date": (
                following_date.date().isoformat()
                if pd.notna(following_date)
                else None
            ),
        }

        for label, date_value in [
            ("previous", previous_date),
            ("following", following_date),
        ]:

            if pd.isna(date_value):
                temporal_result[f"{label}_valid_n"] = 0
                temporal_result[f"{label}_valid_share"] = np.nan
                continue

            date_scenes = same_track[
                same_track["date"] == date_value
            ]

            valid_ids = set()

            for _, rec in date_scenes.iterrows():

                q = read_cp(rec["path"])

                if q.empty:
                    continue

                q = q[
                    q["support_id"].isin(failed_ids)
                ].copy()

                if q.empty:
                    continue

                q_inside = btrue(
                    q[f"{pol}_raster_inside"]
                )

                q_noise = btrue(
                    q[f"{pol}_noise_support_ok"]
                )

                q_sigma = pd.to_numeric(
                    q[f"{pol}_sigma0_corrected_linear"],
                    errors="coerce"
                )

                valid = (
                    q_inside
                    & q_noise
                    & np.isfinite(q_sigma)
                )

                valid_ids.update(
                    q.loc[valid, "support_id"]
                    .astype(str)
                    .tolist()
                )

            n_valid = len(
                valid_ids.intersection(failed_ids)
            )

            temporal_result[f"{label}_valid_n"] = n_valid

            temporal_result[f"{label}_valid_share"] = (
                n_valid / len(failed_ids)
                if failed_ids else np.nan
            )

        temporal_rows.append(temporal_result)

        # ----------------------------------------------------
        # Persistent failure across immediate neighbours
        # ----------------------------------------------------

        prev_n = temporal_result.get(
            "previous_valid_n", 0
        )

        next_n = temporal_result.get(
            "following_valid_n", 0
        )

        summary_rows.append(
            {
                "scene_id": target_scene,
                "year": int(d["year"].iloc[0]),
                "target_date": target_date.date().isoformat(),
                "orbit_state": orbit_state,
                "relative_orbit": relative_orbit,
                "polarization": pol.upper(),
                "failed_support_n": len(failed_ids),
                "previous_valid_n": prev_n,
                "following_valid_n": next_n,
                "previous_valid_share": (
                    prev_n / len(failed_ids)
                    if failed_ids else np.nan
                ),
                "following_valid_share": (
                    next_n / len(failed_ids)
                    if failed_ids else np.nan
                ),
            }
        )


edge = pd.DataFrame(edge_rows)
temporal = pd.DataFrame(temporal_rows)
summary = pd.DataFrame(summary_rows)

edge.to_csv(
    OUT / "c2w_noise_failure_image_edge_audit.csv",
    index=False
)

temporal.to_csv(
    OUT / "c2w_noise_failure_temporal_redundancy.csv",
    index=False
)

summary.to_csv(
    OUT / "c2w_noise_failure_final_impact.csv",
    index=False
)

print()
print("=" * 100)
print("TEMPORAL REDUNDANCY")
print("=" * 100)
print(
    temporal.to_string(index=False)
)

print()
print("=" * 100)
print("IMAGE-COORDINATE EDGE DIAGNOSTIC")
print("=" * 100)
print(
    edge.to_string(index=False)
)

print()
print("Outputs written to:", OUT)