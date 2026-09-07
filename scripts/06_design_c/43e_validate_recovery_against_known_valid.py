from pathlib import Path
import json
import xml.etree.ElementTree as ET

import boto3
import numpy as np
import pandas as pd


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
    bucket, key = parse_s3(uri)
    return s3.get_object(
        Bucket=bucket,
        Key=key
    )["Body"].read()


def btrue(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )


def interp1_strict(xp, fp, x):
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    x = float(x)

    if len(xp) < 2:
        raise ValueError("too few support points")

    if x < xp[0] or x > xp[-1]:
        raise ValueError("outside interpolation support")

    return float(np.interp(x, xp, fp))


def interp_range_vectors_strict(vectors, row, col):

    vecs = sorted(
        vectors,
        key=lambda z: z["line"]
    )

    lines = np.asarray(
        [v["line"] for v in vecs],
        dtype=float
    )

    r = float(row)

    if r < lines[0] or r > lines[-1]:
        raise ValueError("range row outside support")

    exact = np.where(lines == r)[0]

    if len(exact):
        v = vecs[int(exact[0])]
        return interp1_strict(
            v["pixel"],
            v["noise"],
            col
        )

    hi = int(np.searchsorted(lines, r))
    lo = hi - 1

    v0 = vecs[lo]
    v1 = vecs[hi]

    n0 = interp1_strict(
        v0["pixel"],
        v0["noise"],
        col
    )

    n1 = interp1_strict(
        v1["pixel"],
        v1["noise"],
        col
    )

    l0 = float(v0["line"])
    l1 = float(v1["line"])

    w = (r - l0) / (l1 - l0)

    return float(
        n0 + w * (n1 - n0)
    )


def azimuth_value(lines, lut, row):
    """
    For known-valid validation only:
    interpolate strictly inside sampled LUT support.
    """
    lines = np.asarray(lines, dtype=float)
    lut = np.asarray(lut, dtype=float)
    row = float(row)

    if len(lines) == 1:
        return float(lut[0])

    if row < lines[0] or row > lines[-1]:
        raise ValueError("outside sampled azimuth LUT span")

    return float(
        np.interp(row, lines, lut)
    )


def parse_noise(root):

    rlist = first(
        root,
        "noiseRangeVectorList"
    )

    alist = first(
        root,
        "noiseAzimuthVectorList"
    )

    range_vectors = []
    blocks = []

    for v in list(rlist):

        if lname(v.tag) != "noiseRangeVector":
            continue

        range_vectors.append(
            {
                "line": int(
                    child_text(v, "line")
                ),
                "pixel": np.asarray(
                    [
                        int(z)
                        for z in child_text(
                            v, "pixel"
                        ).split()
                    ],
                    dtype=float
                ),
                "noise": np.asarray(
                    [
                        float(z)
                        for z in child_text(
                            v, "noiseRangeLut"
                        ).split()
                    ],
                    dtype=float
                ),
            }
        )

    for v in list(alist):

        if lname(v.tag) != "noiseAzimuthVector":
            continue

        fal = child_text(v, "firstAzimuthLine")
        lal = child_text(v, "lastAzimuthLine")
        frs = child_text(v, "firstRangeSample")
        lrs = child_text(v, "lastRangeSample")

        lines = np.asarray(
            [
                int(z)
                for z in child_text(
                    v, "line"
                ).split()
            ],
            dtype=float
        )

        lut = np.asarray(
            [
                float(z)
                for z in child_text(
                    v, "noiseAzimuthLut"
                ).split()
            ],
            dtype=float
        )

        blocks.append(
            {
                "fal": int(fal),
                "lal": int(lal),
                "frs": int(frs),
                "lrs": int(lrs),
                "lines": lines,
                "lut": lut,
            }
        )

    return range_vectors, blocks


s3 = boto3.client(
    "s3",
    endpoint_url=(
        "https://eodata.dataspace.copernicus.eu"
    ),
)

records = []

for scene_id in TARGETS:

    print()
    print("=" * 100)
    print(scene_id)
    print("=" * 100)

    meta = json.loads(
        (
            CACHE / f"{scene_id}.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    cp = pd.read_csv(
        CP / f"{scene_id}.csv.gz",
        compression="gzip",
        low_memory=False,
    )

    for pol in ["vv", "vh"]:

        href = (
            meta["assets"]
            [f"schema-noise-{pol}"]
            ["href"]
        )

        root = ET.fromstring(
            fetch(s3, href)
        )

        range_vectors, blocks = parse_noise(root)

        original_db = pd.to_numeric(
            cp[f"{pol}_sigma0_corrected_db"],
            errors="coerce"
        )

        original_lin = pd.to_numeric(
            cp[f"{pol}_sigma0_corrected_linear"],
            errors="coerce"
        )

        raw_power = pd.to_numeric(
            cp[f"{pol}_raw_detected_power"],
            errors="coerce"
        )

        sigma_lut = pd.to_numeric(
            cp[f"{pol}_sigma_lut"],
            errors="coerce"
        )

        valid = (
            btrue(cp[f"{pol}_raster_inside"])
            & btrue(cp[f"{pol}_noise_support_ok"])
            & original_db.notna()
            & raw_power.notna()
            & sigma_lut.notna()
        )

        q = cp.loc[valid].copy()

        candidates = []

        for idx, r in q.iterrows():

            row = int(
                r[f"{pol}_image_row"]
            )

            col = int(
                r[f"{pol}_image_col"]
            )

            applicable = [
                b for b in blocks
                if (
                    b["fal"] <= row <= b["lal"]
                    and b["frs"] <= col <= b["lrs"]
                )
            ]

            if len(applicable) != 1:
                continue

            b = applicable[0]

            if len(b["lines"]) < 2:
                continue

            if not (
                b["lines"][0]
                <= row
                <= b["lines"][-1]
            ):
                continue

            edge_dist = min(
                row - b["lines"][0],
                b["lines"][-1] - row
            )

            if edge_dist > 20:
                continue

            candidates.append(
                (
                    idx,
                    b,
                    edge_dist
                )
            )

        print(
            f"{pol.upper()}: "
            f"known-valid edge candidates={len(candidates)}"
        )

        for idx, b, edge_dist in candidates:

            r = cp.loc[idx]

            row = int(
                r[f"{pol}_image_row"]
            )

            col = int(
                r[f"{pol}_image_col"]
            )

            try:
                nr = interp_range_vectors_strict(
                    range_vectors,
                    row,
                    col,
                )

                na = azimuth_value(
                    b["lines"],
                    b["lut"],
                    row,
                )

                eta = nr * na

                rp = float(
                    r[f"{pol}_raw_detected_power"]
                )

                sl = float(
                    r[f"{pol}_sigma_lut"]
                )

                cpow = rp - eta

                if cpow <= 0:
                    continue

                rec_lin = (
                    cpow / (sl ** 2)
                )

                rec_db = (
                    10.0
                    * np.log10(rec_lin)
                )

                orig_db = float(
                    r[
                        f"{pol}_sigma0_corrected_db"
                    ]
                )

                orig_lin = float(
                    r[
                        f"{pol}_sigma0_corrected_linear"
                    ]
                )

                records.append(
                    {
                        "scene_id": scene_id,
                        "polarization": pol.upper(),
                        "support_id": r["support_id"],
                        "image_row": row,
                        "image_col": col,
                        "edge_distance_lines":
                            edge_dist,
                        "original_sigma0_db":
                            orig_db,
                        "recomputed_sigma0_db":
                            rec_db,
                        "delta_db":
                            rec_db - orig_db,
                        "original_sigma0_linear":
                            orig_lin,
                        "recomputed_sigma0_linear":
                            rec_lin,
                        "relative_error_linear":
                            (
                                rec_lin - orig_lin
                            ) / orig_lin,
                    }
                )

            except Exception:
                continue


res = pd.DataFrame(records)

res.to_csv(
    OUT
    / "c2x_recovery_known_valid_validation.csv",
    index=False,
)

print()
print("=" * 100)
print("KNOWN-VALID REPRODUCTION SUMMARY")
print("=" * 100)

if res.empty:

    print("NO VALIDATION RECORDS PRODUCED")

else:

    summary = (
        res.groupby(
            [
                "scene_id",
                "polarization",
            ]
        )
        .agg(
            validation_n=(
                "support_id",
                "nunique"
            ),
            max_abs_delta_db=(
                "delta_db",
                lambda x:
                    np.nanmax(
                        np.abs(x)
                    )
            ),
            median_abs_delta_db=(
                "delta_db",
                lambda x:
                    np.nanmedian(
                        np.abs(x)
                    )
            ),
            max_abs_relative_error_linear=(
                "relative_error_linear",
                lambda x:
                    np.nanmax(
                        np.abs(x)
                    )
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT
        / "c2x_recovery_known_valid_validation_summary.csv",
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )