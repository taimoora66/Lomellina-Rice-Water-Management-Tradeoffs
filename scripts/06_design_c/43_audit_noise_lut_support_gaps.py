from pathlib import Path
import json
import xml.etree.ElementTree as ET
import boto3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

C2U_CACHE = (
    ROOT
    / "outputs"
    / "diagnostics"
    / "design_c"
    / "c2u_item_cache"
)

C2V_CP = (
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
    assert uri.startswith("s3://")
    x = uri[5:]
    bucket, key = x.split("/", 1)
    return bucket, key


def read_s3_bytes(s3, uri):
    bucket, key = parse_s3(uri)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def btrue(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
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

    meta_path = C2U_CACHE / f"{scene_id}.json"
    cp_path = C2V_CP / f"{scene_id}.csv.gz"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cp = pd.read_csv(cp_path, compression="gzip")

    for pol in ["vv", "vh"]:

        print()
        print(pol.upper())

        noise_href = meta["assets"][f"schema-noise-{pol}"]["href"]

        xml_bytes = read_s3_bytes(s3, noise_href)
        root = ET.fromstring(xml_bytes)

        azlist = first(root, "noiseAzimuthVectorList")

        blocks = []

        if azlist is not None:

            for idx, v in enumerate(list(azlist)):

                if lname(v.tag) != "noiseAzimuthVector":
                    continue

                fal = child_text(v, "firstAzimuthLine")
                lal = child_text(v, "lastAzimuthLine")
                frs = child_text(v, "firstRangeSample")
                lrs = child_text(v, "lastRangeSample")
                line_text = child_text(v, "line")
                lut_text = child_text(v, "noiseAzimuthLut")
                swath = child_text(v, "swath")

                if line_text is None or lut_text is None:
                    continue

                lines = np.array(
                    [int(x) for x in line_text.split()],
                    dtype=int
                )

                blocks.append(
                    {
                        "block_index": idx,
                        "swath": swath,
                        "firstAzimuthLine":
                            int(fal) if fal is not None else None,
                        "lastAzimuthLine":
                            int(lal) if lal is not None else None,
                        "firstRangeSample":
                            int(frs) if frs is not None else None,
                        "lastRangeSample":
                            int(lrs) if lrs is not None else None,
                        "lut_line_min": int(lines.min()),
                        "lut_line_max": int(lines.max()),
                        "lut_lines_n": len(lines),
                    }
                )

        inside = btrue(cp[f"{pol}_raster_inside"])
        noise_ok = btrue(cp[f"{pol}_noise_support_ok"])

        fail = inside & ~noise_ok

        q = cp.loc[
            fail,
            [
                "support_id",
                "lon",
                "lat",
                f"{pol}_image_row",
                f"{pol}_image_col",
            ],
        ].copy()

        print(f"Failed points: {len(q)}")
        print(f"Azimuth blocks: {len(blocks)}")

        for _, r in q.iterrows():

            row = int(r[f"{pol}_image_row"])
            col = int(r[f"{pol}_image_col"])

            applicable = []

            for b in blocks:

                row_ok = (
                    (b["firstAzimuthLine"] is None
                     or row >= b["firstAzimuthLine"])
                    and
                    (b["lastAzimuthLine"] is None
                     or row <= b["lastAzimuthLine"])
                )

                col_ok = (
                    (b["firstRangeSample"] is None
                     or col >= b["firstRangeSample"])
                    and
                    (b["lastRangeSample"] is None
                     or col <= b["lastRangeSample"])
                )

                if row_ok and col_ok:
                    applicable.append(b)

            if len(applicable) == 0:

                rows.append(
                    {
                        "scene_id": scene_id,
                        "polarization": pol.upper(),
                        "support_id": r["support_id"],
                        "lon": r["lon"],
                        "lat": r["lat"],
                        "image_row": row,
                        "image_col": col,
                        "failure_class":
                            "NO_APPLICABLE_DECLARED_BLOCK",
                        "applicable_blocks_n": 0,
                        "block_index": np.nan,
                        "firstAzimuthLine": np.nan,
                        "lastAzimuthLine": np.nan,
                        "lut_line_min": np.nan,
                        "lut_line_max": np.nan,
                        "distance_below_lut_min": np.nan,
                        "distance_above_lut_max": np.nan,
                        "inside_lut_sample_span": False,
                    }
                )

                continue

            for b in applicable:

                below = max(
                    b["lut_line_min"] - row,
                    0
                )

                above = max(
                    row - b["lut_line_max"],
                    0
                )

                inside_lut = (
                    b["lut_line_min"]
                    <= row
                    <= b["lut_line_max"]
                )

                rows.append(
                    {
                        "scene_id": scene_id,
                        "polarization": pol.upper(),
                        "support_id": r["support_id"],
                        "lon": r["lon"],
                        "lat": r["lat"],
                        "image_row": row,
                        "image_col": col,
                        "failure_class":
                            "DECLARED_BLOCK_BUT_LUT_SPAN_CHECK",
                        "applicable_blocks_n": len(applicable),
                        "block_index": b["block_index"],
                        "swath": b["swath"],
                        "firstAzimuthLine":
                            b["firstAzimuthLine"],
                        "lastAzimuthLine":
                            b["lastAzimuthLine"],
                        "firstRangeSample":
                            b["firstRangeSample"],
                        "lastRangeSample":
                            b["lastRangeSample"],
                        "lut_line_min":
                            b["lut_line_min"],
                        "lut_line_max":
                            b["lut_line_max"],
                        "lut_lines_n":
                            b["lut_lines_n"],
                        "distance_below_lut_min":
                            below,
                        "distance_above_lut_max":
                            above,
                        "inside_lut_sample_span":
                            inside_lut,
                    }
                )


res = pd.DataFrame(rows)

res.to_csv(
    OUT / "c2x_noise_lut_support_gap_points.csv",
    index=False,
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

summary = (
    res.groupby(
        [
            "scene_id",
            "polarization",
            "failure_class",
        ],
        dropna=False
    )
    .agg(
        rows_n=("support_id", "size"),
        support_ids_n=("support_id", "nunique"),
        max_below_lut_min=("distance_below_lut_min", "max"),
        max_above_lut_max=("distance_above_lut_max", "max"),
        median_below_lut_min=("distance_below_lut_min", "median"),
        median_above_lut_max=("distance_above_lut_max", "median"),
        inside_lut_sample_span_n=("inside_lut_sample_span", "sum"),
    )
    .reset_index()
)

summary.to_csv(
    OUT / "c2x_noise_lut_support_gap_summary.csv",
    index=False,
)

print(summary.to_string(index=False))

print()
print("DISTANCE DISTRIBUTION FOR DECLARED-BLOCK CASES")
print("-" * 100)

d = res[
    res["failure_class"]
    == "DECLARED_BLOCK_BUT_LUT_SPAN_CHECK"
].copy()

if len(d):

    d["outside_distance"] = (
        d["distance_below_lut_min"]
        .fillna(0)
        +
        d["distance_above_lut_max"]
        .fillna(0)
    )

    print(
        d.groupby(
            ["scene_id", "polarization"]
        )["outside_distance"]
        .describe(
            percentiles=[0.5, 0.9, 0.95, 0.99]
        )
        .to_string()
    )

print()
print("Output:", OUT)