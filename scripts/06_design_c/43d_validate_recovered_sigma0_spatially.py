from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CP = (
    ROOT / "outputs" / "diagnostics" / "design_c"
    / "c2v" / "scene_checkpoints" / "primary"
)

REC = (
    ROOT / "outputs" / "diagnostics" / "design_c"
    / "c2x_recovery"
    / "c2x_shadow_full_sigma0_recovery.csv"
)

OUT = (
    ROOT / "outputs" / "diagnostics" / "design_c"
    / "c2x_recovery"
)

OUT.mkdir(parents=True, exist_ok=True)

rec = pd.read_csv(REC, low_memory=False)

rec = rec[
    rec["status"].eq("RECOVERED_VALID")
].copy()

rows = []

for scene_id in rec["scene_id"].unique():

    cp_path = CP / f"{scene_id}.csv.gz"

    cp = pd.read_csv(
        cp_path,
        compression="gzip",
        low_memory=False,
    )

    print()
    print("=" * 100)
    print(scene_id)
    print("=" * 100)

    for pol in ["VV", "VH"]:

        p = pol.lower()

        rpol = rec[
            (rec["scene_id"] == scene_id)
            & (rec["polarization"] == pol)
        ].copy()

        if rpol.empty:
            continue

        valid = cp[
            pd.to_numeric(
                cp[f"{p}_sigma0_corrected_db"],
                errors="coerce"
            ).notna()
        ].copy()

        valid[f"{p}_image_row"] = pd.to_numeric(
            valid[f"{p}_image_row"],
            errors="coerce"
        )

        valid[f"{p}_image_col"] = pd.to_numeric(
            valid[f"{p}_image_col"],
            errors="coerce"
        )

        valid[f"{p}_sigma0_corrected_db"] = pd.to_numeric(
            valid[f"{p}_sigma0_corrected_db"],
            errors="coerce"
        )

        print(
            f"{pol}: recovered={len(rpol)}, "
            f"original_valid={len(valid)}"
        )

        for _, rr in rpol.iterrows():

            row = float(rr["image_row"])
            col = float(rr["image_col"])
            recovered_db = float(
                rr["recovered_sigma0_corrected_db"]
            )

            dx = (
                valid[f"{p}_image_row"] - row
            )

            dy = (
                valid[f"{p}_image_col"] - col
            )

            dist = np.sqrt(
                dx * dx + dy * dy
            )

            z = valid.copy()
            z["pixel_distance"] = dist.values

            z = z.sort_values(
                "pixel_distance"
            )

            # nearest 5 and 10 originally-valid support points
            n5 = z.head(5)
            n10 = z.head(10)

            med5 = (
                float(
                    n5[f"{p}_sigma0_corrected_db"].median()
                )
                if len(n5)
                else np.nan
            )

            med10 = (
                float(
                    n10[f"{p}_sigma0_corrected_db"].median()
                )
                if len(n10)
                else np.nan
            )

            nearest_distance = (
                float(n5["pixel_distance"].iloc[0])
                if len(n5)
                else np.nan
            )

            nearest_db = (
                float(
                    n5[f"{p}_sigma0_corrected_db"].iloc[0]
                )
                if len(n5)
                else np.nan
            )

            rows.append(
                {
                    "scene_id": scene_id,
                    "polarization": pol,
                    "support_id": rr["support_id"],
                    "image_row": row,
                    "image_col": col,
                    "edge_distance_lines":
                        rr["edge_distance_lines"],
                    "recovered_sigma0_db":
                        recovered_db,
                    "nearest_valid_pixel_distance":
                        nearest_distance,
                    "nearest_valid_sigma0_db":
                        nearest_db,
                    "nearest5_median_sigma0_db":
                        med5,
                    "nearest10_median_sigma0_db":
                        med10,
                    "delta_vs_nearest_db":
                        recovered_db - nearest_db
                        if np.isfinite(nearest_db)
                        else np.nan,
                    "delta_vs_nearest5_median_db":
                        recovered_db - med5
                        if np.isfinite(med5)
                        else np.nan,
                    "delta_vs_nearest10_median_db":
                        recovered_db - med10
                        if np.isfinite(med10)
                        else np.nan,
                }
            )

res = pd.DataFrame(rows)

res.to_csv(
    OUT / "c2x_recovery_spatial_validation_points.csv",
    index=False,
)

print()
print("=" * 100)
print("SPATIAL VALIDATION SUMMARY")
print("=" * 100)

summary = (
    res.groupby(
        [
            "scene_id",
            "polarization",
        ]
    )
    .agg(
        recovered_n=(
            "support_id",
            "nunique"
        ),
        median_abs_delta_vs_nearest_db=(
            "delta_vs_nearest_db",
            lambda x: np.nanmedian(np.abs(x))
        ),
        p95_abs_delta_vs_nearest_db=(
            "delta_vs_nearest_db",
            lambda x: np.nanpercentile(
                np.abs(x.dropna()),
                95
            )
            if x.notna().any()
            else np.nan
        ),
        median_abs_delta_vs_nearest5_db=(
            "delta_vs_nearest5_median_db",
            lambda x: np.nanmedian(np.abs(x))
        ),
        p95_abs_delta_vs_nearest5_db=(
            "delta_vs_nearest5_median_db",
            lambda x: np.nanpercentile(
                np.abs(x.dropna()),
                95
            )
            if x.notna().any()
            else np.nan
        ),
        nearest_distance_median=(
            "nearest_valid_pixel_distance",
            "median"
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUT / "c2x_recovery_spatial_validation_summary.csv",
    index=False,
)

print(
    summary.to_string(
        index=False
    )
)