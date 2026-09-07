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


def interp1_strict(xp, fp, x, label):
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    x = float(x)

    if len(xp) != len(fp):
        raise ValueError(f"{label}: length mismatch")

    if len(xp) < 2:
        raise ValueError(
            f"{label}: fewer than two support points"
        )

    if np.any(np.diff(xp) <= 0):
        raise ValueError(
            f"{label}: support not strictly increasing"
        )

    if x < xp[0] or x > xp[-1]:
        raise ValueError(
            f"{label}: coordinate {x} outside "
            f"[{xp[0]}, {xp[-1]}]"
        )

    return float(np.interp(x, xp, fp))


def interp_range_vectors_strict(vectors, row, col):
    """
    Reproduce the existing extractor's strict 2-D interpolation
    for the modern noiseRangeVectorList.
    """

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
        raise ValueError(
            f"range row {r} outside "
            f"[{lines[0]}, {lines[-1]}]"
        )

    # Exact range-vector line
    exact = np.where(lines == r)[0]

    if len(exact):
        v = vecs[int(exact[0])]
        return interp1_strict(
            v["pixel"],
            v["noise"],
            col,
            "modern range noise range"
        )

    hi = int(np.searchsorted(lines, r))
    lo = hi - 1

    v0 = vecs[lo]
    v1 = vecs[hi]

    n0 = interp1_strict(
        v0["pixel"],
        v0["noise"],
        col,
        "modern range noise range"
    )

    n1 = interp1_strict(
        v1["pixel"],
        v1["noise"],
        col,
        "modern range noise range"
    )

    l0 = float(v0["line"])
    l1 = float(v1["line"])

    w = (r - l0) / (l1 - l0)

    return float(
        n0 + w * (n1 - n0)
    )


def azimuth_edge_value(lines, lut, row):
    """
    Inside sampled LUT support:
        normal interpolation.

    Between sampled LUT edge and the officially declared
    applicability boundary:
        linear continuation using the nearest two LUT knots.

    This function must only be called AFTER the row/col has
    been shown to lie within exactly one declared block.
    """

    lines = np.asarray(lines, dtype=float)
    lut = np.asarray(lut, dtype=float)

    row = float(row)

    if len(lines) != len(lut):
        raise ValueError("Azimuth LUT length mismatch")

    if len(lines) == 0:
        raise ValueError("Empty azimuth LUT")

    if len(lines) == 1:
        return float(lut[0])

    if np.any(np.diff(lines) <= 0):
        raise ValueError(
            "Azimuth line support not increasing"
        )

    if lines[0] <= row <= lines[-1]:
        return float(
            np.interp(row, lines, lut)
        )

    if row < lines[0]:
        x0, x1 = lines[0], lines[1]
        y0, y1 = lut[0], lut[1]
    else:
        x0, x1 = lines[-2], lines[-1]
        y0, y1 = lut[-2], lut[-1]

    return float(
        y0
        + (row - x0)
        * (y1 - y0)
        / (x1 - x0)
    )


def parse_modern_noise(root):

    rlist = first(
        root,
        "noiseRangeVectorList"
    )

    alist = first(
        root,
        "noiseAzimuthVectorList"
    )

    range_vectors = []
    azimuth_blocks = []

    if rlist is None:
        raise RuntimeError(
            "No modern noiseRangeVectorList found"
        )

    for v in list(rlist):

        if lname(v.tag) != "noiseRangeVector":
            continue

        line = int(
            child_text(v, "line")
        )

        pixels = np.asarray(
            [
                int(z)
                for z in child_text(
                    v, "pixel"
                ).split()
            ],
            dtype=float
        )

        noise = np.asarray(
            [
                float(z)
                for z in child_text(
                    v, "noiseRangeLut"
                ).split()
            ],
            dtype=float
        )

        range_vectors.append(
            {
                "line": line,
                "pixel": pixels,
                "noise": noise,
            }
        )

    if alist is not None:

        for v in list(alist):

            if lname(v.tag) != "noiseAzimuthVector":
                continue

            fal = child_text(
                v, "firstAzimuthLine"
            )

            lal = child_text(
                v, "lastAzimuthLine"
            )

            frs = child_text(
                v, "firstRangeSample"
            )

            lrs = child_text(
                v, "lastRangeSample"
            )

            line_text = child_text(
                v, "line"
            )

            lut_text = child_text(
                v, "noiseAzimuthLut"
            )

            if (
                line_text is None
                or lut_text is None
            ):
                continue

            azimuth_blocks.append(
                {
                    "fal":
                        int(fal)
                        if fal is not None
                        else None,

                    "lal":
                        int(lal)
                        if lal is not None
                        else None,

                    "frs":
                        int(frs)
                        if frs is not None
                        else None,

                    "lrs":
                        int(lrs)
                        if lrs is not None
                        else None,

                    "lines": np.asarray(
                        [
                            int(z)
                            for z
                            in line_text.split()
                        ],
                        dtype=float
                    ),

                    "lut": np.asarray(
                        [
                            float(z)
                            for z
                            in lut_text.split()
                        ],
                        dtype=float
                    ),
                }
            )

    return range_vectors, azimuth_blocks


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
            CACHE
            / f"{scene_id}.json"
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

        noise_href = (
            meta["assets"]
            [f"schema-noise-{pol}"]
            ["href"]
        )

        root = ET.fromstring(
            fetch(s3, noise_href)
        )

        range_vectors, blocks = (
            parse_modern_noise(root)
        )

        inside = btrue(
            cp[f"{pol}_raster_inside"]
        )

        noise_ok = btrue(
            cp[f"{pol}_noise_support_ok"]
        )

        fail = inside & ~noise_ok

        q = cp.loc[fail].copy()

        recovered_valid = 0
        recovered_nonpositive = 0
        unresolved = 0
        failed_other = 0

        for _, r in q.iterrows():

            image_row = int(
                r[f"{pol}_image_row"]
            )

            image_col = int(
                r[f"{pol}_image_col"]
            )

            base = {
                "scene_id": scene_id,
                "polarization": pol.upper(),
                "support_id": r["support_id"],
                "lon": r["lon"],
                "lat": r["lat"],
                "image_row": image_row,
                "image_col": image_col,
                "edge_distance_lines": np.nan,
                "recovered_sigma0_corrected_db":
                    np.nan,
                "recovered_sigma0_corrected_linear":
                    np.nan,
            }

            applicable = []

            for b in blocks:

                row_ok = (
                    (
                        b["fal"] is None
                        or image_row >= b["fal"]
                    )
                    and
                    (
                        b["lal"] is None
                        or image_row <= b["lal"]
                    )
                )

                col_ok = (
                    (
                        b["frs"] is None
                        or image_col >= b["frs"]
                    )
                    and
                    (
                        b["lrs"] is None
                        or image_col <= b["lrs"]
                    )
                )

                if row_ok and col_ok:
                    applicable.append(b)

            if len(applicable) != 1:

                records.append(
                    {
                        **base,
                        "status":
                            "UNRESOLVED_NO_UNIQUE_DECLARED_BLOCK",
                    }
                )

                unresolved += 1
                continue

            b = applicable[0]

            # This recovery stage is ONLY for the
            # observed LUT-edge failure class.
            if (
                b["lines"][0]
                <= image_row
                <= b["lines"][-1]
            ):

                records.append(
                    {
                        **base,
                        "status":
                            "UNEXPECTED_INSIDE_AZIMUTH_LUT_SPAN",
                    }
                )

                failed_other += 1
                continue

            try:
                noise_range = (
                    interp_range_vectors_strict(
                        range_vectors,
                        image_row,
                        image_col,
                    )
                )

                noise_azimuth = (
                    azimuth_edge_value(
                        b["lines"],
                        b["lut"],
                        image_row,
                    )
                )

            except Exception as exc:

                records.append(
                    {
                        **base,
                        "status":
                            "RECOVERY_COMPONENT_FAILURE",
                        "error": str(exc),
                    }
                )

                failed_other += 1
                continue

            raw_power = pd.to_numeric(
                pd.Series(
                    [
                        r[
                            f"{pol}_raw_detected_power"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            sigma_lut = pd.to_numeric(
                pd.Series(
                    [
                        r[
                            f"{pol}_sigma_lut"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if not np.isfinite(raw_power):

                records.append(
                    {
                        **base,
                        "status":
                            "FAIL_RAW_POWER_MISSING",
                    }
                )

                failed_other += 1
                continue

            if not np.isfinite(sigma_lut):

                records.append(
                    {
                        **base,
                        "status":
                            "FAIL_SIGMA_LUT_MISSING",
                    }
                )

                failed_other += 1
                continue

            eta = float(
                noise_range
                * noise_azimuth
            )

            corrected_power = float(
                raw_power - eta
            )

            if image_row < b["lines"][0]:

                edge_distance = int(
                    b["lines"][0]
                    - image_row
                )

            else:

                edge_distance = int(
                    image_row
                    - b["lines"][-1]
                )

            sigma_corr = np.nan
            db_corr = np.nan

            if corrected_power <= 0:

                status = (
                    "RECOVERED_NONPOSITIVE_POWER"
                )

                recovered_nonpositive += 1

            else:

                sigma_corr = float(
                    corrected_power
                    / (sigma_lut ** 2)
                )

                db_corr = float(
                    10.0
                    * np.log10(
                        sigma_corr
                    )
                )

                status = "RECOVERED_VALID"

                recovered_valid += 1

            records.append(
                {
                    **base,

                    "status": status,

                    "firstAzimuthLine":
                        b["fal"],

                    "lastAzimuthLine":
                        b["lal"],

                    "lut_line_min":
                        int(b["lines"][0]),

                    "lut_line_max":
                        int(b["lines"][-1]),

                    "edge_distance_lines":
                        edge_distance,

                    "raw_detected_power":
                        raw_power,

                    "sigma_lut":
                        sigma_lut,

                    "recomputed_noise_range_component":
                        noise_range,

                    "recovered_noise_azimuth_component":
                        noise_azimuth,

                    "recovered_noise_power_eta":
                        eta,

                    "recovered_corrected_detected_power":
                        corrected_power,

                    "recovered_sigma0_corrected_linear":
                        sigma_corr,

                    "recovered_sigma0_corrected_db":
                        db_corr,
                }
            )

        print(
            f"{pol.upper()}: "
            f"valid={recovered_valid}, "
            f"nonpositive={recovered_nonpositive}, "
            f"unresolved={unresolved}, "
            f"other_fail={failed_other}"
        )


res = pd.DataFrame(records)

detail_path = (
    OUT
    / "c2x_shadow_full_sigma0_recovery.csv"
)

res.to_csv(
    detail_path,
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
            "status",
        ],
        dropna=False,
    )
    .agg(
        supports_n=(
            "support_id",
            "nunique"
        ),

        max_edge_distance_lines=(
            "edge_distance_lines",
            "max"
        ),

        sigma0_db_min=(
            "recovered_sigma0_corrected_db",
            "min"
        ),

        sigma0_db_median=(
            "recovered_sigma0_corrected_db",
            "median"
        ),

        sigma0_db_max=(
            "recovered_sigma0_corrected_db",
            "max"
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUT
    / "c2x_shadow_full_sigma0_recovery_summary.csv",
    index=False,
)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "Output:",
    OUT
)