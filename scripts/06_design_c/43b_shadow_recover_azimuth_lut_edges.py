from pathlib import Path
import json
import xml.etree.ElementTree as ET
import boto3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CACHE = (
    ROOT / "outputs" / "diagnostics" / "design_c"
    / "c2u_item_cache"
)

CP = (
    ROOT / "outputs" / "diagnostics" / "design_c"
    / "c2v" / "scene_checkpoints" / "primary"
)

OUT = (
    ROOT / "outputs" / "diagnostics" / "design_c"
    / "c2x_recovery"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "S1C_IW_GRDH_1SDV_20250421T053447_20250421T053516_001987_003FE6_9043_COG",
    "S1B_IW_GRDH_1SDV_20180704T172205_20180704T172229_011664_01573E_AD95_COG",
    "S1B_IW_GRDH_1SDV_20180622T172204_20180622T172228_011489_0151C9_5837_COG",
]


def lname(tag):
    return tag.split("}")[-1]


def first(root, name):
    for e in root.iter():
        if lname(e.tag) == name:
            return e
    return None


def child_text(node, name):
    for c in list(node):
        if lname(c.tag) == name:
            return c.text
    return None


def parse_s3(uri):
    x = uri[5:]
    return x.split("/", 1)


def fetch(s3, uri):
    b, k = parse_s3(uri)
    return s3.get_object(Bucket=b, Key=k)["Body"].read()


def btrue(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


def linear_edge_extend(xp, fp, x):
    """
    Linear interpolation inside sampled support.
    Linear extension using nearest two LUT knots only when x is
    still inside the declared applicability block.
    """
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    x = float(x)

    if len(xp) == 1:
        return float(fp[0])

    if xp[0] <= x <= xp[-1]:
        return float(np.interp(x, xp, fp))

    if x < xp[0]:
        x0, x1 = xp[0], xp[1]
        y0, y1 = fp[0], fp[1]
    else:
        x0, x1 = xp[-2], xp[-1]
        y0, y1 = fp[-2], fp[-1]

    return float(
        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    )


s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu",
)

rows = []

for scene_id in TARGETS:

    print()
    print("=" * 100)
    print(scene_id)
    print("=" * 100)

    meta = json.loads(
        (CACHE / f"{scene_id}.json").read_text()
    )

    cp = pd.read_csv(
        CP / f"{scene_id}.csv.gz",
        compression="gzip",
    )

    for pol in ["vv", "vh"]:

        href = meta["assets"][f"schema-noise-{pol}"]["href"]
        root = ET.fromstring(fetch(s3, href))

        azlist = first(root, "noiseAzimuthVectorList")
        blocks = []

        for v in list(azlist):

            if lname(v.tag) != "noiseAzimuthVector":
                continue

            fal = int(child_text(v, "firstAzimuthLine"))
            lal = int(child_text(v, "lastAzimuthLine"))
            frs = int(child_text(v, "firstRangeSample"))
            lrs = int(child_text(v, "lastRangeSample"))

            lines = np.array(
                [int(z) for z in child_text(v, "line").split()],
                dtype=float,
            )

            lut = np.array(
                [
                    float(z)
                    for z in child_text(
                        v, "noiseAzimuthLut"
                    ).split()
                ],
                dtype=float,
            )

            blocks.append(
                {
                    "fal": fal,
                    "lal": lal,
                    "frs": frs,
                    "lrs": lrs,
                    "lines": lines,
                    "lut": lut,
                }
            )

        inside = btrue(cp[f"{pol}_raster_inside"])
        ok = btrue(cp[f"{pol}_noise_support_ok"])
        fail = inside & ~ok

        q = cp.loc[fail].copy()

        recovered = 0
        no_block = 0

        for _, r in q.iterrows():

            row = int(r[f"{pol}_image_row"])
            col = int(r[f"{pol}_image_col"])

            applicable = [
                b for b in blocks
                if (
                    b["fal"] <= row <= b["lal"]
                    and b["frs"] <= col <= b["lrs"]
                )
            ]

            if len(applicable) != 1:
                no_block += 1

                rows.append(
                    {
                        "scene_id": scene_id,
                        "polarization": pol.upper(),
                        "support_id": r["support_id"],
                        "image_row": row,
                        "image_col": col,
                        "status": "NO_UNIQUE_DECLARED_BLOCK",
                    }
                )
                continue

            b = applicable[0]

            az = linear_edge_extend(
                b["lines"],
                b["lut"],
                row,
            )

            dist = 0

            if row < b["lines"][0]:
                dist = int(b["lines"][0] - row)

            elif row > b["lines"][-1]:
                dist = int(row - b["lines"][-1])

            recovered += 1

            rows.append(
                {
                    "scene_id": scene_id,
                    "polarization": pol.upper(),
                    "support_id": r["support_id"],
                    "image_row": row,
                    "image_col": col,
                    "status":
                        "RECOVERABLE_WITHIN_DECLARED_BLOCK",
                    "firstAzimuthLine": b["fal"],
                    "lastAzimuthLine": b["lal"],
                    "lut_line_min": int(b["lines"][0]),
                    "lut_line_max": int(b["lines"][-1]),
                    "edge_distance_lines": dist,
                    "recovered_azimuth_factor": az,
                }
            )

        print(
            f"{pol.upper()}: "
            f"recoverable={recovered}, "
            f"no_unique_block={no_block}"
        )


res = pd.DataFrame(rows)

res.to_csv(
    OUT / "c2x_shadow_azimuth_edge_recovery.csv",
    index=False,
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

summary = (
    res.groupby(
        ["scene_id", "polarization", "status"]
    )
    .agg(
        supports_n=("support_id", "nunique"),
        max_edge_distance_lines=(
            "edge_distance_lines", "max"
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUT / "c2x_shadow_azimuth_edge_recovery_summary.csv",
    index=False,
)

print(summary.to_string(index=False))