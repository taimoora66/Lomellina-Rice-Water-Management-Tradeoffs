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

all_points = []
summaries = []

for scene_id in TARGETS:

    path = CP_DIR / f"{scene_id}.csv.gz"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, compression="gzip")

    print()
    print("=" * 90)
    print(scene_id)
    print("=" * 90)
    print(f"Rows: {len(df)}")

    for pol in ["vv", "vh"]:

        inside = (
            df[f"{pol}_raster_inside"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )

        noise_ok = (
            df[f"{pol}_noise_support_ok"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )

        fail = inside & ~noise_ok

        sigma = pd.to_numeric(
            df[f"{pol}_sigma0_corrected_linear"],
            errors="coerce"
        )

        eta = pd.to_numeric(
            df[f"{pol}_noise_power_eta"],
            errors="coerce"
        )

        range_component = pd.to_numeric(
            df[f"{pol}_noise_range_component"],
            errors="coerce"
        )

        az_component = pd.to_numeric(
            df[f"{pol}_noise_azimuth_component"],
            errors="coerce"
        )

        candidates = pd.to_numeric(
            df[f"{pol}_noise_azimuth_candidate_blocks_n"],
            errors="coerce"
        )

        print()
        print(pol.upper())
        print("-" * 40)
        print(f"Raster inside:        {int(inside.sum())}")
        print(f"Noise support fail:   {int(fail.sum())}")
        print(f"Fail with sigma0:     {int((fail & sigma.notna()).sum())}")
        print(f"Fail with eta:        {int((fail & eta.notna()).sum())}")
        print(f"Fail with range comp: {int((fail & range_component.notna()).sum())}")
        print(f"Fail with az comp:    {int((fail & az_component.notna()).sum())}")

        if fail.any():

            x = df.loc[
                fail,
                [
                    "scene_id",
                    "year",
                    "acquisition_date",
                    "orbit_state",
                    "relative_orbit",
                    "platform",
                    "support_id",
                    "lon",
                    "lat",
                    f"{pol}_image_row",
                    f"{pol}_image_col",
                    f"{pol}_noise_power_eta",
                    f"{pol}_noise_range_component",
                    f"{pol}_noise_azimuth_component",
                    f"{pol}_noise_azimuth_candidate_blocks_n",
                    f"{pol}_corrected_detected_power",
                    f"{pol}_nonpositive_corrected_power",
                    f"{pol}_sigma0_corrected_linear",
                    f"{pol}_sigma0_corrected_db",
                ]
            ].copy()

            x["polarization"] = pol.upper()

            all_points.append(x)

            summaries.append(
                {
                    "scene_id": scene_id,
                    "polarization": pol.upper(),
                    "inside_n": int(inside.sum()),
                    "noise_fail_n": int(fail.sum()),
                    "fail_sigma0_present_n": int(
                        (fail & sigma.notna()).sum()
                    ),
                    "fail_eta_present_n": int(
                        (fail & eta.notna()).sum()
                    ),
                    "fail_range_component_present_n": int(
                        (fail & range_component.notna()).sum()
                    ),
                    "fail_azimuth_component_present_n": int(
                        (fail & az_component.notna()).sum()
                    ),
                    "candidate_blocks_min": (
                        float(candidates[fail].min())
                        if candidates[fail].notna().any()
                        else np.nan
                    ),
                    "candidate_blocks_max": (
                        float(candidates[fail].max())
                        if candidates[fail].notna().any()
                        else np.nan
                    ),
                    "failed_lon_min": float(
                        pd.to_numeric(df.loc[fail, "lon"]).min()
                    ),
                    "failed_lon_max": float(
                        pd.to_numeric(df.loc[fail, "lon"]).max()
                    ),
                    "failed_lat_min": float(
                        pd.to_numeric(df.loc[fail, "lat"]).min()
                    ),
                    "failed_lat_max": float(
                        pd.to_numeric(df.loc[fail, "lat"]).max()
                    ),
                }
            )

summary = pd.DataFrame(summaries)
points = pd.concat(all_points, ignore_index=True)

summary.to_csv(
    OUT / "c2w_three_scene_noise_support_summary.csv",
    index=False
)

points.to_csv(
    OUT / "c2w_three_scene_noise_failure_points.csv",
    index=False
)

print()
print("=" * 90)
print("THREE-SCENE SUMMARY")
print("=" * 90)
print(summary.to_string(index=False))

print()
print("Unique failing support IDs by scene:")

for scene_id in TARGETS:
    q = points[points["scene_id"] == scene_id]
    print(
        scene_id,
        q["support_id"].nunique()
    )

print()
print("Outputs written to:", OUT)