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

files = sorted(CP_DIR.glob("*.csv.gz"))

rows = []
problem_rows = []

print(f"Diagnosing {len(files)} checkpoints...")

for i, path in enumerate(files, 1):

    if i % 100 == 0:
        print(f"  {i}/{len(files)}")

    try:
        df = pd.read_csv(path, compression="gzip")
    except pd.errors.EmptyDataError:
        continue

    if df.empty:
        continue

    base = {
        "scene_id": df["scene_id"].iloc[0],
        "year": int(df["year"].iloc[0]),
        "acquisition_date": df["acquisition_date"].iloc[0],
        "orbit_state": df["orbit_state"].iloc[0],
        "relative_orbit": int(df["relative_orbit"].iloc[0]),
        "platform": df["platform"].iloc[0],
    }

    for pol in ["vv", "vh"]:

        inside_col = f"{pol}_raster_inside"
        noise_col = f"{pol}_noise_support_ok"
        cal_col = f"{pol}_calibration_support_ok"

        inside_raw = (
            df[inside_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        inside = inside_raw.isin(["true", "1"])

        noise_raw = (
            df[noise_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        cal_raw = (
            df[cal_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Explicit states only
        noise_true = noise_raw.isin(["true", "1"])
        noise_false = noise_raw.isin(["false", "0"])
        noise_missing = ~(noise_true | noise_false)

        cal_true = cal_raw.isin(["true", "1"])
        cal_false = cal_raw.isin(["false", "0"])
        cal_missing = ~(cal_true | cal_false)

        result = {
            **base,
            "polarization": pol.upper(),
            "inside_n": int(inside.sum()),

            "noise_true_inside_n": int(
                (inside & noise_true).sum()
            ),
            "noise_false_inside_n": int(
                (inside & noise_false).sum()
            ),
            "noise_missing_inside_n": int(
                (inside & noise_missing).sum()
            ),

            "calibration_true_inside_n": int(
                (inside & cal_true).sum()
            ),
            "calibration_false_inside_n": int(
                (inside & cal_false).sum()
            ),
            "calibration_missing_inside_n": int(
                (inside & cal_missing).sum()
            ),
        }

        rows.append(result)

        bad = inside & (noise_false | noise_missing)

        if bad.any():

            q = df.loc[
                bad,
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
                    inside_col,
                    noise_col,
                    f"{pol}_noise_power_eta",
                    f"{pol}_noise_range_component",
                    f"{pol}_noise_azimuth_component",
                    f"{pol}_noise_azimuth_candidate_blocks_n",
                    f"{pol}_nonpositive_corrected_power",
                    f"{pol}_sigma0_corrected_linear",
                    f"{pol}_sigma0_corrected_db",
                ],
            ].copy()

            q["polarization"] = pol.upper()

            q["noise_flag_type"] = np.where(
                noise_false[bad],
                "EXPLICIT_FALSE",
                "MISSING_OR_UNRECOGNIZED",
            )

            problem_rows.append(q)


summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "c2w_noise_support_state_diagnostic.csv",
    index=False,
)

if problem_rows:
    problems = pd.concat(
        problem_rows,
        ignore_index=True
    )
else:
    problems = pd.DataFrame()

problems.to_csv(
    OUT / "c2w_noise_support_problem_points.csv",
    index=False,
)


print()
print("=" * 78)
print("C2W NOISE-SUPPORT DIAGNOSTIC")
print("=" * 78)

for pol in ["VV", "VH"]:

    g = summary[summary["polarization"] == pol]

    print()
    print(pol)
    print("-" * 30)

    for c in [
        "inside_n",
        "noise_true_inside_n",
        "noise_false_inside_n",
        "noise_missing_inside_n",
        "calibration_false_inside_n",
        "calibration_missing_inside_n",
    ]:
        print(f"{c}: {int(g[c].sum())}")

print()
print("Scenes with explicit/missing noise issues:")

if len(problems):

    x = (
        problems
        .groupby(
            [
                "year",
                "acquisition_date",
                "orbit_state",
                "relative_orbit",
                "scene_id",
                "polarization",
                "noise_flag_type",
            ],
            dropna=False
        )
        .size()
        .reset_index(name="points_n")
        .sort_values(
            ["points_n"],
            ascending=False
        )
    )

    print(
        x.head(30).to_string(index=False)
    )

    x.to_csv(
        OUT / "c2w_noise_problem_scene_ranking.csv",
        index=False,
    )

else:
    print("None")

print()
print("Diagnostic complete.")