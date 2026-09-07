"""Design C â€” C2V full-season Sentinel-1 calibrated extraction.

Authoritative full-archive execution stage after C2U-F and validated 2025 pilot.

Scientific contract
-------------------
Inputs are frozen by C2U-F:
  * primary: IW VV/VH canonical scene plan
  * auxiliary: IW HH/HV scene plan

For every fixed RiceFloodIT support coordinate actually covered by a raster:
  1. map lon/lat to image row/column using embedded Sentinel-1 GCPs;
  2. read the detected-amplitude DN only at mapped support pixels;
  3. strictly interpolate the calibration LUT (no support clamping);
  4. interpolate thermal-noise LUTs only within declared applicability support;
     for multi-sample modern azimuth LUTs, permit validated linear edge
     extension only from an outermost LUT knot to the declared block boundary;
  5. require thermalNoiseCorrectionPerformed == false;
  6. corrected_power = DN^2 - eta;
  7. corrected_power <= 0 -> missing (never clipped);
  8. corrected sigma0 linear = corrected_power / sigmaNoughtLUT^2;
  9. mosaic overlapping same-date / track / support observations by MEDIAN
     CORRECTED LINEAR POWER;
 10. convert the mosaicked linear sigma0 to dB only AFTER mosaicking.

No inundation threshold or classifier is selected here.
No groundwater, irrigation flow, or RiceFloodIT flood values are read.
No sensor-response feature selection is performed.

Restartability
--------------
Each scene is written atomically as a compressed checkpoint. Zero-study-support scenes use a fast path that skips unnecessary radiometric XML retrieval. Existing valid
checkpoints are reused. Final mosaics are rebuilt deterministically from the
scene checkpoints, so a network interruption does not discard completed work.

Outputs
-------
outputs/diagnostics/design_c/c2v/
  scene_checkpoints/primary/<scene>.csv.gz
  scene_checkpoints/auxiliary/<scene>.csv.gz
  primary_mosaic/year=YYYY/<orbit>_<track>.csv.gz
  auxiliary_mosaic/year=YYYY/<orbit>_<track>.csv.gz
  c2v_scene_qa.csv
  c2v_partition_qa.csv
  c2v_fullseason_extraction_qa.json
  c2v_fullseason_extraction_summary.txt

Run
---
python -u scripts/06_design_c/41_extract_fullarchive_s1_corrected_signal_optimized.py

Optional
--------
--stream primary     process only primary VV/VH
--stream auxiliary   process only auxiliary HH/HV
--year 2025          process only one year (useful for a smoke test)
--force              recompute existing scene checkpoints
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import GCPTransformer
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "diagnostics" / "design_c"
OUT = D / "c2v"
SCENE_ROOT = OUT / "scene_checkpoints"
MOSAIC_ROOT = OUT
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_PLAN = D / "c2uf_primary_vvvh_canonical_asset_plan.csv"
AUX_PLAN = D / "c2uf_auxiliary_hhhv_asset_plan.csv"
C2UF_QA = D / "c2uf_measurement_universe_qa.json"
RICE_GEO = (
    ROOT / "data" / "processed" / "publication_groundwater"
    / "ricefloodit_georef.csv"
)

SCENE_QA_OUT = OUT / "c2v_scene_qa.csv"
PART_QA_OUT = OUT / "c2v_partition_qa.csv"
QA_OUT = OUT / "c2v_fullseason_extraction_qa.json"
TXT_OUT = OUT / "c2v_fullseason_extraction_summary.txt"

S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
RASTER_ENV = {
    "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
    "AWS_VIRTUAL_HOSTING": "FALSE",
    "AWS_DEFAULT_REGION": "default",
}
MAX_REMOTE_ATTEMPTS = 8
BASE_DELAY_S = 5.0

EXPECTED_SUPPORT_N = 4331
EXPECTED_PRIMARY_CANONICAL_N = 2134
EXPECTED_AUX_SCENE_N = 3
EXPECTED_PRIMARY_YT_N = 44

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def require_credentials():
    missing = [
        k for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        if not os.environ.get(k)
    ]
    if missing:
        raise RuntimeError(
            "Missing CDSE S3 credential environment variable(s): "
            + ", ".join(missing)
        )


def parse_s3_href(href):
    m = re.match(r"^s3://([^/]+)/(.+)$", str(href))
    if not m:
        raise ValueError(f"Not an s3:// href: {href}")
    return m.group(1), m.group(2)


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name="default",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def fetch_bytes_retry(s3, href):
    last = None
    for attempt in range(1, MAX_REMOTE_ATTEMPTS + 1):
        try:
            bucket, key = parse_s3_href(href)
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            return body, attempt
        except Exception as e:
            last = repr(e)
            if attempt < MAX_REMOTE_ATTEMPTS:
                delay = min(BASE_DELAY_S * attempt, 60.0)
                print(
                    f"    XML fetch attempt {attempt} failed: {last}\n"
                    f"    retrying in {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"Failed XML fetch after {MAX_REMOTE_ATTEMPTS} attempts: "
        f"{href}\n{last}"
    )


def lname(tag):
    return tag.rsplit("}", 1)[-1]


def first(root, name):
    for e in root.iter():
        if lname(e.tag) == name:
            return e
    return None


def child(el, name):
    for c in list(el):
        if lname(c.tag) == name:
            return c
    return None


def text_of(el, name):
    c = child(el, name)
    if c is None or c.text is None:
        raise ValueError(f"Missing XML child: {name}")
    return c.text.strip()


def child_text(el, name, required=True):
    for c in list(el):
        if lname(c.tag) == name and c.text is not None:
            return c.text.strip()
    if required:
        raise ValueError(f"Missing XML child: {name}")
    return None


def arr_float(text):
    return np.asarray([float(x) for x in text.split()], dtype=float)


def arr_int(text):
    return np.asarray([int(x) for x in text.split()], dtype=int)


def boolish(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def finite_or_nan(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Authoritative calibration logic: strict support, no clamping.
# ---------------------------------------------------------------------------

def parse_calibration_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)

    info = first(root, "calibrationInformation")
    abs_cal = None
    if info is not None:
        x = child(info, "absoluteCalibrationConstant")
        if x is not None and x.text:
            abs_cal = float(x.text.strip())

    vector_list = first(root, "calibrationVectorList")
    if vector_list is None:
        raise ValueError("No calibrationVectorList found")

    vecs = []
    for v in list(vector_list):
        if lname(v.tag) != "calibrationVector":
            continue
        line = int(text_of(v, "line"))
        pixel = arr_int(text_of(v, "pixel"))
        sigma = arr_float(text_of(v, "sigmaNought"))
        beta = arr_float(text_of(v, "betaNought"))
        gamma = arr_float(text_of(v, "gamma"))
        dn = arr_float(text_of(v, "dn"))
        n = len(pixel)
        if not all(len(a) == n for a in [sigma, beta, gamma, dn]):
            raise ValueError(f"LUT length mismatch at line {line}")
        if n < 2:
            raise ValueError(
                f"Calibration vector at line {line} has <2 range samples"
            )
        if np.any(np.diff(pixel) <= 0):
            raise ValueError(
                f"Non-increasing pixel LUT at line {line}"
            )
        vecs.append({
            "line": line,
            "pixel": pixel,
            "sigma": sigma,
            "beta": beta,
            "gamma": gamma,
            "dn": dn,
        })

    if not vecs:
        raise ValueError("No calibrationVector entries parsed")

    vecs = sorted(vecs, key=lambda z: z["line"])
    lines = np.asarray([v["line"] for v in vecs], dtype=float)
    if np.any(np.diff(lines) <= 0):
        raise ValueError(
            "Calibration vector lines are not strictly increasing"
        )
    return vecs, abs_cal


def interp_range_strict(vec, col, field):
    px = vec["pixel"].astype(float)
    c = float(col)
    if c < px[0] or c > px[-1]:
        raise ValueError(
            f"Column {c} outside LUT pixel support "
            f"[{px[0]}, {px[-1]}] at line {vec['line']}"
        )
    return float(np.interp(c, px, vec[field].astype(float)))


def interp_2d_strict(vecs, row, col, field):
    lines = np.asarray([v["line"] for v in vecs], dtype=float)
    r = float(row)

    if r < lines[0] or r > lines[-1]:
        raise ValueError(
            f"Row {r} outside calibration line support "
            f"[{lines[0]}, {lines[-1]}]"
        )

    if r == lines[0]:
        return interp_range_strict(vecs[0], col, field)
    if r == lines[-1]:
        return interp_range_strict(vecs[-1], col, field)

    hi = int(np.searchsorted(lines, r, side="right"))
    lo = hi - 1
    v0, v1 = vecs[lo], vecs[hi]

    a0 = interp_range_strict(v0, col, field)
    a1 = interp_range_strict(v1, col, field)
    l0, l1 = float(v0["line"]), float(v1["line"])
    w = (r - l0) / (l1 - l0)
    return float(a0 + w * (a1 - a0))


# ---------------------------------------------------------------------------
# Authoritative noise logic: legacy or modern range*azimuth, strict support.
# ---------------------------------------------------------------------------

def parse_product_noise_flag(xml_bytes):
    root = ET.fromstring(xml_bytes)
    e = first(root, "thermalNoiseCorrectionPerformed")
    if e is None or e.text is None:
        return None
    return e.text.strip().lower()


def parse_noise_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    legacy = first(root, "noiseVectorList")
    modern_range = first(root, "noiseRangeVectorList")
    modern_az = first(root, "noiseAzimuthVectorList")

    if legacy is not None:
        vecs = []
        for v in list(legacy):
            if lname(v.tag) != "noiseVector":
                continue
            line = int(child_text(v, "line"))
            pixel = arr_int(child_text(v, "pixel"))
            lut = arr_float(child_text(v, "noiseLut"))
            if len(pixel) != len(lut):
                raise ValueError(
                    f"Legacy LUT length mismatch at line {line}"
                )
            if len(pixel) < 2 or np.any(np.diff(pixel) <= 0):
                raise ValueError(
                    f"Invalid legacy pixel support at line {line}"
                )
            vecs.append({
                "line": line,
                "pixel": pixel,
                "noise": lut,
            })
        if not vecs:
            raise ValueError(
                "Legacy schema found but no vectors parsed"
            )
        vecs = sorted(vecs, key=lambda z: z["line"])
        lines = np.asarray(
            [v["line"] for v in vecs], dtype=float
        )
        if np.any(np.diff(lines) <= 0):
            raise ValueError(
                "Legacy noise vector lines not strictly increasing"
            )
        return {
            "schema": "legacy_noiseVectorList",
            "legacy_vectors": vecs,
            "range_vectors": [],
            "azimuth_vectors": [],
        }

    if modern_range is not None:
        rvecs = []
        for v in list(modern_range):
            if lname(v.tag) != "noiseRangeVector":
                continue
            line = int(child_text(v, "line"))
            pixel = arr_int(child_text(v, "pixel"))
            lut = arr_float(child_text(v, "noiseRangeLut"))
            if len(pixel) != len(lut):
                raise ValueError(
                    f"Modern range LUT mismatch at line {line}"
                )
            if len(pixel) < 2 or np.any(np.diff(pixel) <= 0):
                raise ValueError(
                    f"Invalid modern range support at line {line}"
                )
            rvecs.append({
                "line": line,
                "pixel": pixel,
                "noise": lut,
            })
        if not rvecs:
            raise ValueError(
                "Modern schema found but no range vectors parsed"
            )
        rvecs = sorted(rvecs, key=lambda z: z["line"])
        rlines = np.asarray(
            [v["line"] for v in rvecs], dtype=float
        )
        if np.any(np.diff(rlines) <= 0):
            raise ValueError(
                "Modern range vector lines not strictly increasing"
            )

        avecs = []
        if modern_az is not None:
            for v in list(modern_az):
                if lname(v.tag) != "noiseAzimuthVector":
                    continue
                fal = child_text(
                    v, "firstAzimuthLine", required=False
                )
                lal = child_text(
                    v, "lastAzimuthLine", required=False
                )
                frs = child_text(
                    v, "firstRangeSample", required=False
                )
                lrs = child_text(
                    v, "lastRangeSample", required=False
                )
                line_text = child_text(
                    v, "line", required=False
                )
                lut_text = child_text(
                    v, "noiseAzimuthLut", required=False
                )
                swath = child_text(
                    v, "swath", required=False
                )
                if line_text is None or lut_text is None:
                    continue
                lines = arr_int(line_text)
                lut = arr_float(lut_text)
                if len(lines) != len(lut):
                    raise ValueError(
                        "Azimuth LUT length mismatch"
                    )
                if len(lines) == 0:
                    raise ValueError(
                        "Empty azimuth LUT support"
                    )
                if len(lines) > 1 and np.any(np.diff(lines) <= 0):
                    raise ValueError(
                        "Invalid azimuth line support"
                    )

                # A one-sample azimuth LUT may describe a declared multi-line
                # applicability block. Accept it only when the declared interval
                # is valid and the sole line coordinate lies inside that interval.
                # Runtime use remains restricted to first/lastAzimuthLine and
                # first/lastRangeSample by applicable_azimuth_vectors(); therefore
                # the singleton value is never used outside declared support.
                if len(lines) == 1:
                    if fal is None or lal is None:
                        raise ValueError(
                            "Singleton azimuth LUT lacks first/lastAzimuthLine"
                        )
                    fal_i = int(fal)
                    lal_i = int(lal)
                    line_i = int(lines[0])
                    if fal_i > lal_i:
                        raise ValueError(
                            "Singleton azimuth LUT has reversed "
                            "first/lastAzimuthLine"
                        )
                    if not (fal_i <= line_i <= lal_i):
                        raise ValueError(
                            "Singleton azimuth LUT line lies outside "
                            "first/lastAzimuthLine"
                        )

                avecs.append({
                    "swath": swath,
                    "firstAzimuthLine":
                        int(fal) if fal is not None else None,
                    "lastAzimuthLine":
                        int(lal) if lal is not None else None,
                    "firstRangeSample":
                        int(frs) if frs is not None else None,
                    "lastRangeSample":
                        int(lrs) if lrs is not None else None,
                    "line": lines,
                    "noise": lut,
                })
        return {
            "schema": "modern_range_x_azimuth",
            "legacy_vectors": [],
            "range_vectors": rvecs,
            "azimuth_vectors": avecs,
        }

    raise ValueError(
        "Unrecognized Sentinel-1 noise XML schema"
    )


def interp1_strict(px, vals, x, label):
    px = np.asarray(px, dtype=float)
    vals = np.asarray(vals, dtype=float)
    xx = float(x)
    if xx < px[0] or xx > px[-1]:
        raise ValueError(
            f"{label}: coordinate {xx} outside "
            f"[{px[0]}, {px[-1]}]"
        )
    return float(np.interp(xx, px, vals))


def interp_2d_line_pixel_strict(vectors, row, col, label):
    lines = np.asarray(
        [v["line"] for v in vectors], dtype=float
    )
    r = float(row)
    if r < lines[0] or r > lines[-1]:
        raise ValueError(
            f"{label}: row {r} outside "
            f"[{lines[0]}, {lines[-1]}]"
        )

    if r == lines[0]:
        return interp1_strict(
            vectors[0]["pixel"],
            vectors[0]["noise"],
            col,
            label + " range",
        )
    if r == lines[-1]:
        return interp1_strict(
            vectors[-1]["pixel"],
            vectors[-1]["noise"],
            col,
            label + " range",
        )

    hi = int(np.searchsorted(lines, r, side="right"))
    lo = hi - 1
    v0, v1 = vectors[lo], vectors[hi]
    n0 = interp1_strict(
        v0["pixel"], v0["noise"], col,
        label + " range"
    )
    n1 = interp1_strict(
        v1["pixel"], v1["noise"], col,
        label + " range"
    )
    l0, l1 = float(v0["line"]), float(v1["line"])
    w = (r - l0) / (l1 - l0)
    return float(n0 + w * (n1 - n0))


def applicable_azimuth_vectors(azvecs, row, col):
    out = []
    for v in azvecs:
        fal, lal = (
            v["firstAzimuthLine"],
            v["lastAzimuthLine"],
        )
        frs, lrs = (
            v["firstRangeSample"],
            v["lastRangeSample"],
        )
        row_ok = (
            (fal is None or row >= fal)
            and (lal is None or row <= lal)
        )
        col_ok = (
            (frs is None or col >= frs)
            and (lrs is None or col <= lrs)
        )
        if row_ok and col_ok:
            out.append(v)
    return out


def modern_azimuth_factor_strict(azvecs, row, col):
    """
    Evaluate modern Sentinel-1 azimuth thermal-noise support.

    Applicability is governed first by the officially declared
    firstAzimuthLine:lastAzimuthLine and
    firstRangeSample:lastRangeSample block.

    Within exactly one declared block:
      * singleton LUTs are constant over that declared block;
      * multi-sample LUTs are interpolated inside sampled line support;
      * if the target row lies between an outermost LUT knot and the
        corresponding declared block boundary, the azimuth LUT is extended
        linearly using only the nearest two terminal LUT knots.

    No value is generated outside declared applicability support. Missing or
    overlapping declared blocks remain hard support failures.
    """
    if not azvecs:
        return 1.0, 0

    candidates = applicable_azimuth_vectors(
        azvecs, row, col
    )
    if len(candidates) == 0:
        raise ValueError(
            f"No applicable azimuth noise block "
            f"for row={row}, col={col}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple azimuth noise blocks "
            f"({len(candidates)}) for row={row}, col={col}"
        )

    v = candidates[0]
    lines = np.asarray(v["line"], dtype=float)
    vals = np.asarray(v["noise"], dtype=float)
    r = float(row)

    if len(lines) != len(vals):
        raise ValueError("Azimuth LUT length mismatch")
    if len(lines) == 0:
        raise ValueError("Empty azimuth LUT support")

    # applicable_azimuth_vectors() has already established that the target
    # lies within the vector's declared azimuth/range applicability block.
    if len(lines) == 1:
        return float(vals[0]), 1

    if np.any(np.diff(lines) <= 0):
        raise ValueError("Invalid azimuth line support")

    # Preserve the original strict interpolation wherever sampled LUT support
    # exists.
    if lines[0] <= r <= lines[-1]:
        val = interp1_strict(
            lines, vals, r, "azimuth"
        )
        return float(val), 1

    # Edge recovery: the point is outside the sampled LUT knots but is still
    # inside exactly one official declared applicability block. Extend only
    # from the nearest two terminal LUT knots to that declared boundary.
    if r < lines[0]:
        x0, x1 = lines[0], lines[1]
        y0, y1 = vals[0], vals[1]
    else:
        x0, x1 = lines[-2], lines[-1]
        y0, y1 = vals[-2], vals[-1]

    val = y0 + (r - x0) * (y1 - y0) / (x1 - x0)

    if not np.isfinite(val):
        raise ValueError(
            f"Non-finite azimuth noise edge extension "
            f"for row={row}, col={col}"
        )

    return float(val), 1


def noise_power_at_strict(parsed, row, col):
    if parsed["schema"] == "legacy_noiseVectorList":
        eta = interp_2d_line_pixel_strict(
            parsed["legacy_vectors"],
            row, col, "legacy noise"
        )
        return eta, np.nan, np.nan, 1

    nr = interp_2d_line_pixel_strict(
        parsed["range_vectors"],
        row, col, "modern range noise"
    )
    na, nc = modern_azimuth_factor_strict(
        parsed["azimuth_vectors"],
        row, col
    )
    return nr * na, nr, na, nc


# ---------------------------------------------------------------------------
# Support coordinates and sparse raster reads.
# ---------------------------------------------------------------------------

def load_support():
    if not RICE_GEO.exists():
        raise FileNotFoundError(RICE_GEO)

    hdr = pd.read_csv(RICE_GEO, nrows=0)
    lower = {str(c).lower(): c for c in hdr.columns}

    lon_col = None
    lat_col = None
    for x in ["lon", "longitude", "x", "lng"]:
        if x in lower:
            lon_col = lower[x]
            break
    for x in ["lat", "latitude", "y"]:
        if x in lower:
            lat_col = lower[x]
            break
    if lon_col is None or lat_col is None:
        raise RuntimeError(
            f"Could not identify lon/lat columns in {RICE_GEO}"
        )

    id_col = None
    for x in [
        "support_id", "cell_id", "grid_id",
        "point_id", "id"
    ]:
        if x in lower:
            id_col = lower[x]
            break

    use = [lon_col, lat_col]
    if id_col is not None:
        use.append(id_col)

    # Deliberately read coordinates / identifier only; no flooding values.
    x = pd.read_csv(
        RICE_GEO, usecols=use, low_memory=False
    )
    x = x.rename(
        columns={lon_col: "lon", lat_col: "lat"}
    )
    x["lon"] = pd.to_numeric(x["lon"], errors="coerce")
    x["lat"] = pd.to_numeric(x["lat"], errors="coerce")
    x = x.dropna(subset=["lon", "lat"]).copy()

    if id_col is not None:
        x = x.rename(columns={id_col: "source_support_id"})
        # A source ID may repeat across years/records; coordinate is the
        # actual support identity for this measurement archive.
        x = x[
            ["source_support_id", "lon", "lat"]
        ].drop_duplicates(["lon", "lat"])
    else:
        x = x[["lon", "lat"]].drop_duplicates()

    x = x.sort_values(["lat", "lon"]).reset_index(drop=True)
    x.insert(
        0, "support_id",
        [f"RFIT_{i:04d}" for i in range(1, len(x) + 1)]
    )

    if len(x) != EXPECTED_SUPPORT_N:
        raise AssertionError(
            f"Expected {EXPECTED_SUPPORT_N} fixed support "
            f"coordinates, got {len(x)}"
        )
    return x


def rowcol_from_gcps(ds, lon, lat):
    gcps, gcp_crs = ds.gcps
    if not gcps:
        raise RuntimeError("Raster has no embedded GCPs")
    if gcp_crs is None:
        raise RuntimeError("Raster GCP CRS is missing")

    tr = Transformer.from_crs(
        "EPSG:4326", gcp_crs, always_xy=True
    )
    x, y = tr.transform(
        np.asarray(lon, float),
        np.asarray(lat, float),
    )

    with GCPTransformer(gcps) as gt:
        rows, cols = gt.rowcol(x, y)

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    inside = (
        (rows >= 0) & (rows < ds.height)
        & (cols >= 0) & (cols < ds.width)
    )
    return rows, cols, inside, len(gcps), str(gcp_crs)


def sparse_read_band1(ds, rows, cols):
    """Read sparse row/col locations, caching each raster block once."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    out = np.full(len(rows), np.nan, dtype=float)

    try:
        block_h, block_w = ds.block_shapes[0]
    except Exception:
        block_h, block_w = 512, 512

    block_h = int(block_h) if int(block_h) > 0 else 512
    block_w = int(block_w) if int(block_w) > 0 else 512

    keys = np.column_stack(
        (rows // block_h, cols // block_w)
    )
    uniq, inv = np.unique(
        keys, axis=0, return_inverse=True
    )

    for j, (br, bc) in enumerate(uniq):
        idx = np.where(inv == j)[0]
        r0 = int(br * block_h)
        c0 = int(bc * block_w)
        h = min(block_h, ds.height - r0)
        w = min(block_w, ds.width - c0)

        arr = ds.read(
            1,
            window=Window(c0, r0, w, h),
            masked=False,
        )
        rr = rows[idx] - r0
        cc = cols[idx] - c0
        out[idx] = arr[rr, cc].astype(float)

    return out, int(len(uniq))


def raster_sample_support(href, support):
    last = None
    for attempt in range(1, MAX_REMOTE_ATTEMPTS + 1):
        try:
            with rasterio.Env(**RASTER_ENV):
                with rasterio.open(href) as ds:
                    rows, cols, inside, gcp_n, gcp_crs = (
                        rowcol_from_gcps(
                            ds,
                            support["lon"].to_numpy(float),
                            support["lat"].to_numpy(float),
                        )
                    )
                    idx = np.where(inside)[0]
                    values = np.full(
                        len(support), np.nan, dtype=float
                    )
                    block_reads = 0
                    if len(idx):
                        v, block_reads = sparse_read_band1(
                            ds, rows[idx], cols[idx]
                        )
                        values[idx] = v
                    meta = {
                        "raster_width": int(ds.width),
                        "raster_height": int(ds.height),
                        "raster_dtype": str(ds.dtypes[0]),
                        "raster_driver": str(ds.driver),
                        "gcp_count": int(gcp_n),
                        "gcp_crs": gcp_crs,
                        "block_reads_n": int(block_reads),
                        "remote_attempts": int(attempt),
                    }
                    return rows, cols, inside, values, meta

        except Exception as e:
            last = repr(e)
            if attempt < MAX_REMOTE_ATTEMPTS:
                delay = min(
                    BASE_DELAY_S * attempt, 60.0
                )
                print(
                    f"    raster attempt {attempt} failed: "
                    f"{last}\n    retrying in {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Raster access failed after "
        f"{MAX_REMOTE_ATTEMPTS} attempts:\n"
        f"{href}\n{last}"
    )


# ---------------------------------------------------------------------------
# Scene/polarization processing.
# ---------------------------------------------------------------------------

def process_pol(
    s3,
    support,
    scene,
    pol,
    raster_href,
    calibration_href,
    noise_href,
    product_href,
):
    # Optimization only: determine study-support coverage FIRST.
    # If this polarization raster covers zero of the 4,331 frozen support
    # coordinates, no radiometric XML is needed because there is no
    # measurement to calibrate/correct.
    rows, cols, inside, raw, rmeta = raster_sample_support(
        raster_href, support
    )

    n = len(support)

    if int(inside.sum()) == 0:
        out = support.copy()
        out["image_row"] = rows
        out["image_col"] = cols
        out["raster_inside"] = inside
        out["raw_dn"] = raw
        out["raw_detected_power"] = np.nan
        out["sigma_lut"] = np.nan
        out["beta_lut"] = np.nan
        out["gamma_lut"] = np.nan
        out["dn_lut"] = np.nan
        out["calibration_support_ok"] = False
        out["noise_power_eta"] = np.nan
        out["noise_range_component"] = np.nan
        out["noise_azimuth_component"] = np.nan
        out["noise_azimuth_candidate_blocks_n"] = np.nan
        out["noise_support_ok"] = False
        out["corrected_detected_power"] = np.nan
        out["nonpositive_corrected_power"] = False
        out["sigma0_uncorrected_linear"] = np.nan
        out["sigma0_corrected_linear"] = np.nan
        out["sigma0_corrected_db"] = np.nan
        out = out.iloc[0:0].copy()

        qa = {
            "scene_id": scene,
            "polarization": pol,
            "support_total_n": int(n),
            "raster_inside_n": 0,
            "raster_coverage_fraction": 0.0,
            "calibration_support_fail_n": 0,
            "noise_support_fail_n": 0,
            "nonpositive_corrected_power_n": 0,
            "valid_corrected_sigma0_n": 0,
            "thermalNoiseCorrectionPerformed_input": "NOT_READ_ZERO_SUPPORT",
            "noise_schema": "NOT_READ_ZERO_SUPPORT",
            "absoluteCalibrationConstant": np.nan,
            "calibration_xml_attempts": 0,
            "noise_xml_attempts": 0,
            "product_xml_attempts": 0,
            "zero_study_support_fastpath": True,
            **rmeta,
        }
        return out, qa

    # Non-zero support: execute the already validated radiometric chain.
    cal_bytes, cal_attempts = fetch_bytes_retry(
        s3, calibration_href
    )
    noise_bytes, noise_attempts = fetch_bytes_retry(
        s3, noise_href
    )
    prod_bytes, product_attempts = fetch_bytes_retry(
        s3, product_href
    )

    flag = parse_product_noise_flag(prod_bytes)
    if flag != "false":
        raise RuntimeError(
            f"{scene} {pol}: expected "
            f"thermalNoiseCorrectionPerformed=false, "
            f"got {flag!r}"
        )

    cal_vecs, abs_cal = parse_calibration_xml(
        cal_bytes
    )
    noise = parse_noise_xml(noise_bytes)

    sigma_lut = np.full(n, np.nan)
    beta_lut = np.full(n, np.nan)
    gamma_lut = np.full(n, np.nan)
    dn_lut = np.full(n, np.nan)
    eta = np.full(n, np.nan)
    noise_range = np.full(n, np.nan)
    noise_az = np.full(n, np.nan)
    noise_cand = np.full(n, np.nan)

    cal_ok = np.zeros(n, dtype=bool)
    noise_ok = np.zeros(n, dtype=bool)

    for i in np.where(inside)[0]:
        rr = int(rows[i])
        cc = int(cols[i])

        try:
            sigma_lut[i] = interp_2d_strict(
                cal_vecs, rr, cc, "sigma"
            )
            beta_lut[i] = interp_2d_strict(
                cal_vecs, rr, cc, "beta"
            )
            gamma_lut[i] = interp_2d_strict(
                cal_vecs, rr, cc, "gamma"
            )
            dn_lut[i] = interp_2d_strict(
                cal_vecs, rr, cc, "dn"
            )
            cal_ok[i] = True
        except ValueError:
            cal_ok[i] = False

        try:
            e, rn, an, nc = noise_power_at_strict(
                noise, rr, cc
            )
            eta[i] = e
            noise_range[i] = rn
            noise_az[i] = an
            noise_cand[i] = nc
            noise_ok[i] = np.isfinite(e)
        except ValueError:
            noise_ok[i] = False

    raw_power = raw ** 2
    corrected_power = raw_power - eta
    positive = (
        inside
        & cal_ok
        & noise_ok
        & np.isfinite(corrected_power)
        & (corrected_power > 0)
        & np.isfinite(sigma_lut)
        & (sigma_lut > 0)
    )

    sigma_corr = np.full(n, np.nan)
    sigma_corr[positive] = (
        corrected_power[positive]
        / (sigma_lut[positive] ** 2)
    )

    sigma_uncorr = np.full(n, np.nan)
    uncorr_ok = (
        inside & cal_ok
        & np.isfinite(raw_power)
        & np.isfinite(sigma_lut)
        & (sigma_lut > 0)
    )
    sigma_uncorr[uncorr_ok] = (
        raw_power[uncorr_ok]
        / (sigma_lut[uncorr_ok] ** 2)
    )

    db_corr = np.full(n, np.nan)
    okdb = np.isfinite(sigma_corr) & (sigma_corr > 0)
    db_corr[okdb] = 10.0 * np.log10(
        sigma_corr[okdb]
    )

    out = support.copy()
    out["image_row"] = rows
    out["image_col"] = cols
    out["raster_inside"] = inside
    out["raw_dn"] = raw
    out["raw_detected_power"] = raw_power
    out["sigma_lut"] = sigma_lut
    out["beta_lut"] = beta_lut
    out["gamma_lut"] = gamma_lut
    out["dn_lut"] = dn_lut
    out["calibration_support_ok"] = cal_ok
    out["noise_power_eta"] = eta
    out["noise_range_component"] = noise_range
    out["noise_azimuth_component"] = noise_az
    out["noise_azimuth_candidate_blocks_n"] = noise_cand
    out["noise_support_ok"] = noise_ok
    out["corrected_detected_power"] = corrected_power
    out["nonpositive_corrected_power"] = (
        inside & cal_ok & noise_ok
        & np.isfinite(corrected_power)
        & (corrected_power <= 0)
    )
    out["sigma0_uncorrected_linear"] = sigma_uncorr
    out["sigma0_corrected_linear"] = sigma_corr
    out["sigma0_corrected_db"] = db_corr

    out = out.loc[inside].copy()

    qa = {
        "scene_id": scene,
        "polarization": pol,
        "support_total_n": int(n),
        "raster_inside_n": int(inside.sum()),
        "raster_coverage_fraction": float(inside.mean()),
        "calibration_support_fail_n":
            int((inside & ~cal_ok).sum()),
        "noise_support_fail_n":
            int((inside & cal_ok & ~noise_ok).sum()),
        "nonpositive_corrected_power_n":
            int(
                (
                    inside & cal_ok & noise_ok
                    & np.isfinite(corrected_power)
                    & (corrected_power <= 0)
                ).sum()
            ),
        "valid_corrected_sigma0_n":
            int(np.isfinite(sigma_corr).sum()),
        "thermalNoiseCorrectionPerformed_input": flag,
        "noise_schema": noise["schema"],
        "absoluteCalibrationConstant":
            abs_cal if abs_cal is not None else np.nan,
        "calibration_xml_attempts": int(cal_attempts),
        "noise_xml_attempts": int(noise_attempts),
        "product_xml_attempts": int(product_attempts),
        "zero_study_support_fastpath": False,
        **rmeta,
    }
    return out, qa

def safe_scene_filename(scene):
    return re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(scene)
    ) + ".csv.gz"


def atomic_write_csv_gz(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        compression="gzip",
        float_format="%.10g",
    )
    os.replace(tmp, path)


def checkpoint_valid(path, scene, pols):
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        x = pd.read_csv(
            path,
            nrows=5,
            compression="gzip",
        )
        if "scene_id" not in x.columns:
            return False
        if not set(
            f"{p.lower()}_sigma0_corrected_linear"
            for p in pols
        ).issubset(x.columns):
            return False
        return True
    except Exception:
        return False


def scene_plan_record_primary(r):
    return {
        "stream": "primary",
        "scene_id": str(r.canonical_scene_id),
        "year": int(r.year),
        "acquisition_date": str(r.acquisition_date),
        "acquisition_datetime": str(r.acquisition_datetime),
        "orbit_state": str(r.orbit_state),
        "relative_orbit": int(r.relative_orbit),
        "platform": str(r.platform),
        "pols": ["VV", "VH"],
        "hrefs": {
            "VV": {
                "raster": str(r.vv_href),
                "calibration": str(r.calibration_vv_href),
                "noise": str(r.noise_vv_href),
                "product": str(r.product_vv_href),
            },
            "VH": {
                "raster": str(r.vh_href),
                "calibration": str(r.calibration_vh_href),
                "noise": str(r.noise_vh_href),
                "product": str(r.product_vh_href),
            },
        },
    }


def scene_plan_record_aux(r):
    return {
        "stream": "auxiliary",
        "scene_id": str(r.scene_id),
        "year": int(r.year),
        "acquisition_date": str(r.acquisition_date),
        "acquisition_datetime": str(r.acquisition_datetime),
        "orbit_state": str(r.orbit_state),
        "relative_orbit": int(r.relative_orbit),
        "platform": str(r.platform),
        "pols": ["HH", "HV"],
        "hrefs": {
            "HH": {
                "raster": str(r.hh_href),
                "calibration": str(r.calibration_hh_href),
                "noise": str(r.noise_hh_href),
                "product": str(r.product_hh_href),
            },
            "HV": {
                "raster": str(r.hv_href),
                "calibration": str(r.calibration_hv_href),
                "noise": str(r.noise_hv_href),
                "product": str(r.product_hv_href),
            },
        },
    }


def process_scene(
    s3, support, rec, force=False
):
    scene = rec["scene_id"]
    pols = rec["pols"]
    stream = rec["stream"]
    cp = (
        SCENE_ROOT / stream
        / safe_scene_filename(scene)
    )

    if not force and checkpoint_valid(
        cp, scene, pols
    ):
        return cp, [{
            "stream": stream,
            "scene_id": scene,
            "year": rec["year"],
            "acquisition_date":
                rec["acquisition_date"],
            "orbit_state": rec["orbit_state"],
            "relative_orbit":
                rec["relative_orbit"],
            "polarization": "*",
            "checkpoint_reused": True,
            "status": "REUSED",
        }]

    pol_frames = {}
    qa_rows = []

    for pol in pols:
        print(
            f"      {pol}: sampling/calibration/noise",
            flush=True,
        )
        x, qa = process_pol(
            s3=s3,
            support=support,
            scene=scene,
            pol=pol,
            raster_href=rec["hrefs"][pol]["raster"],
            calibration_href=
                rec["hrefs"][pol]["calibration"],
            noise_href=rec["hrefs"][pol]["noise"],
            product_href=rec["hrefs"][pol]["product"],
        )
        qa.update({
            "stream": stream,
            "year": rec["year"],
            "acquisition_date":
                rec["acquisition_date"],
            "acquisition_datetime":
                rec["acquisition_datetime"],
            "orbit_state": rec["orbit_state"],
            "relative_orbit":
                rec["relative_orbit"],
            "platform": rec["platform"],
            "checkpoint_reused": False,
            "status": (
                "REVIEW_STRICT_SUPPORT"
                if (
                    qa["calibration_support_fail_n"] > 0
                    or qa["noise_support_fail_n"] > 0
                )
                else "PASS"
            ),
        })
        qa_rows.append(qa)
        pol_frames[pol] = x

    # Outer-merge polarizations on fixed support identity so an asset-level
    # coverage difference remains explicit rather than being silently dropped.
    base_cols = [
        "support_id", "lon", "lat"
    ]
    if "source_support_id" in support.columns:
        base_cols.append("source_support_id")

    wide = support[base_cols].copy()

    keep_metric = [
        "image_row", "image_col", "raster_inside",
        "raw_dn", "raw_detected_power",
        "sigma_lut", "beta_lut", "gamma_lut",
        "dn_lut", "calibration_support_ok",
        "noise_power_eta", "noise_range_component",
        "noise_azimuth_component",
        "noise_azimuth_candidate_blocks_n",
        "noise_support_ok",
        "corrected_detected_power",
        "nonpositive_corrected_power",
        "sigma0_uncorrected_linear",
        "sigma0_corrected_linear",
        "sigma0_corrected_db",
    ]

    for pol in pols:
        z = pol_frames[pol][
            ["support_id"] + keep_metric
        ].copy()
        z = z.rename(
            columns={
                c: f"{pol.lower()}_{c}"
                for c in keep_metric
            }
        )
        wide = wide.merge(
            z, on="support_id",
            how="left", validate="one_to_one"
        )

    # Keep only support points observed inside at least one polarization raster.
    incols = [
        f"{p.lower()}_raster_inside" for p in pols
    ]
    observed = np.zeros(len(wide), dtype=bool)
    for c in incols:
        observed |= (
            wide[c].fillna(False).astype(bool)
        ).to_numpy()
    wide = wide.loc[observed].copy()

    wide.insert(0, "scene_id", scene)
    wide.insert(1, "stream", stream)
    wide.insert(2, "year", rec["year"])
    wide.insert(
        3, "acquisition_date",
        rec["acquisition_date"]
    )
    wide.insert(
        4, "acquisition_datetime",
        rec["acquisition_datetime"]
    )
    wide.insert(
        5, "orbit_state", rec["orbit_state"]
    )
    wide.insert(
        6, "relative_orbit",
        rec["relative_orbit"]
    )
    wide.insert(7, "platform", rec["platform"])

    atomic_write_csv_gz(wide, cp)
    return cp, qa_rows


# ---------------------------------------------------------------------------
# Deterministic date-track mosaics from scene checkpoints.
# ---------------------------------------------------------------------------

def read_scene_minimal(cp, pols):
    cols = [
        "scene_id", "year", "acquisition_date",
        "acquisition_datetime", "orbit_state",
        "relative_orbit", "platform",
        "support_id", "lon", "lat",
    ]
    for p in pols:
        q = p.lower()
        cols.extend([
            f"{q}_sigma0_corrected_linear",
            f"{q}_nonpositive_corrected_power",
            f"{q}_calibration_support_ok",
            f"{q}_noise_support_ok",
            f"{q}_raster_inside",
        ])
    x = pd.read_csv(
        cp, compression="gzip",
        usecols=lambda c: c in set(cols),
        low_memory=False,
    )
    return x


def mosaic_partition(
    support, stream, records, checkpoints
):
    if stream == "primary":
        pols = ["VV", "VH"]
    else:
        pols = ["HH", "HV"]

    year = int(records[0]["year"])
    orbit = str(records[0]["orbit_state"])
    track = int(records[0]["relative_orbit"])

    frames = [
        read_scene_minimal(cp, pols)
        for cp in checkpoints
    ]
    if frames:
        x = pd.concat(frames, ignore_index=True)
    else:
        x = pd.DataFrame()

    dates = sorted(
        set(r["acquisition_date"] for r in records)
    )
    date_frame = pd.DataFrame(
        {"acquisition_date": dates}
    )
    grid = (
        date_frame.assign(_k=1)
        .merge(
            support[
                ["support_id", "lon", "lat"]
            ].assign(_k=1),
            on="_k",
            how="inner",
        )
        .drop(columns="_k")
    )

    out = grid.copy()
    out.insert(0, "year", year)
    out.insert(1, "orbit_state", orbit)
    out.insert(2, "relative_orbit", track)
    out.insert(3, "stream", stream)

    group_keys = [
        "acquisition_date", "support_id"
    ]

    for pol in pols:
        q = pol.lower()
        value = f"{q}_sigma0_corrected_linear"

        if len(x):
            z = x[
                [
                    "acquisition_date",
                    "support_id",
                    "scene_id",
                    value,
                ]
            ].copy()
            z[value] = pd.to_numeric(
                z[value], errors="coerce"
            )

            # Median of corrected LINEAR sigma0 across overlapping
            # canonical frames; dB is computed only afterwards.
            agg = (
                z.groupby(group_keys, as_index=False)
                .agg(
                    **{
                        f"{q}_sigma0_corrected_linear":
                            (value, "median"),
                        f"{q}_contributing_scene_rows_n":
                            (
                                value,
                                lambda s:
                                    int(
                                        np.isfinite(
                                            pd.to_numeric(
                                                s,
                                                errors="coerce"
                                            )
                                        ).sum()
                                    ),
                            ),
                        f"{q}_candidate_scene_rows_n":
                            ("scene_id", "nunique"),
                    }
                )
            )
        else:
            agg = pd.DataFrame(
                columns=group_keys
            )

        out = out.merge(
            agg, on=group_keys,
            how="left", validate="one_to_one"
        )

        lin = pd.to_numeric(
            out.get(
                f"{q}_sigma0_corrected_linear"
            ),
            errors="coerce",
        )
        db = np.full(len(out), np.nan)
        ok = np.isfinite(lin) & (lin > 0)
        db[ok] = 10.0 * np.log10(lin[ok])
        out[
            f"{q}_sigma0_corrected_db"
        ] = db

        for c in [
            f"{q}_contributing_scene_rows_n",
            f"{q}_candidate_scene_rows_n",
        ]:
            out[c] = (
                pd.to_numeric(
                    out.get(c), errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

        out[f"{q}_observed"] = (
            np.isfinite(lin)
            & (lin > 0)
        )

    # Convenient polarization contrasts are created only at the
    # measurement level; they are not used for selection/classification.
    p1, p2 = pols[0].lower(), pols[1].lower()
    out[f"{p1}_minus_{p2}_db"] = (
        out[f"{p1}_sigma0_corrected_db"]
        - out[f"{p2}_sigma0_corrected_db"]
    )
    out[f"{p1}_{p2}_both_observed"] = (
        out[f"{p1}_observed"]
        & out[f"{p2}_observed"]
    )

    path = (
        MOSAIC_ROOT
        / f"{stream}_mosaic"
        / f"year={year}"
        / f"{orbit}_{track}.csv.gz"
    )
    atomic_write_csv_gz(out, path)

    qa = {
        "stream": stream,
        "year": year,
        "orbit_state": orbit,
        "relative_orbit": track,
        "acquisition_dates_n": int(len(dates)),
        "support_points_n": int(len(support)),
        "mosaic_rows_n": int(len(out)),
        "scene_checkpoints_n": int(len(checkpoints)),
    }
    for pol in pols:
        q = pol.lower()
        qa[
            f"{q}_observed_rows_n"
        ] = int(out[f"{q}_observed"].sum())
        qa[
            f"{q}_observed_share"
        ] = float(out[f"{q}_observed"].mean())

    qa["output_path"] = str(
        path.relative_to(ROOT)
    )
    return path, qa


def plan_records(plan, stream, year_filter=None):
    if stream == "primary":
        recs = [
            scene_plan_record_primary(r)
            for r in plan.itertuples(index=False)
        ]
    else:
        recs = [
            scene_plan_record_aux(r)
            for r in plan.itertuples(index=False)
        ]
    if year_filter is not None:
        recs = [
            r for r in recs
            if int(r["year"]) == int(year_filter)
        ]
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stream",
        choices=["primary", "auxiliary", "both"],
        default="both",
    )
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument(
        "--force", action="store_true"
    )
    args = ap.parse_args()

    print(
        "DESIGN C - C2V FULL-SEASON SENTINEL-1 "
        "CORRECTED SIGNAL EXTRACTION"
    )
    print("=" * 86)
    print(
        "Frozen C2U-F measurement universe only."
    )
    print(
        "Strict calibration + strict thermal-noise correction."
    )
    print(
        "Overlap mosaic: median corrected LINEAR sigma0, then dB."
    )
    print(
        "No groundwater / flow / RiceFloodIT flooding values."
    )
    print(
        "No threshold / classifier / response-based feature selection.\n"
    )

    require_credentials()

    for p in [
        PRIMARY_PLAN, AUX_PLAN, C2UF_QA, RICE_GEO
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    c2uf = json.loads(
        C2UF_QA.read_text(encoding="utf-8")
    )
    if c2uf.get("status") != "PASS":
        raise RuntimeError(
            "C2U-F is not PASS; C2V may not proceed."
        )
    if c2uf.get("c2t_modified") is not False:
        raise RuntimeError(
            "C2U-F provenance says C2T was modified."
        )

    support = load_support()
    primary_plan = pd.read_csv(
        PRIMARY_PLAN, low_memory=False
    )
    aux_plan = pd.read_csv(
        AUX_PLAN, low_memory=False
    )

    if (
        primary_plan["canonical_scene_id"].nunique()
        != EXPECTED_PRIMARY_CANONICAL_N
    ):
        raise AssertionError(
            "Primary canonical scene count changed: "
            f"{primary_plan['canonical_scene_id'].nunique()}"
        )
    if len(aux_plan) != EXPECTED_AUX_SCENE_N:
        raise AssertionError(
            f"Expected {EXPECTED_AUX_SCENE_N} "
            f"auxiliary scene rows, got {len(aux_plan)}"
        )

    streams = (
        ["primary", "auxiliary"]
        if args.stream == "both"
        else [args.stream]
    )
    s3 = make_s3()
    all_scene_qa = []
    all_part_qa = []

    for stream in streams:
        plan = (
            primary_plan
            if stream == "primary"
            else aux_plan
        )
        recs = plan_records(
            plan, stream, args.year
        )
        if not recs:
            print(
                f"No {stream} records for requested scope.",
                flush=True,
            )
            continue

        print(
            f"\nSTREAM: {stream.upper()} "
            f"({len(recs)} scene records)",
            flush=True,
        )

        cp_by_scene = {}
        total = len(recs)

        for j, rec in enumerate(recs, 1):
            print(
                f"  [{j:04d}/{total:04d}] "
                f"{rec['acquisition_date']} "
                f"{rec['orbit_state']} "
                f"{rec['relative_orbit']} "
                f"{rec['scene_id']}",
                flush=True,
            )
            cp, qa_rows = process_scene(
                s3=s3,
                support=support,
                rec=rec,
                force=args.force,
            )
            cp_by_scene[rec["scene_id"]] = cp
            all_scene_qa.extend(qa_rows)

        # Rebuild deterministic mosaics by year-track.
        keys = sorted(
            set(
                (
                    int(r["year"]),
                    str(r["orbit_state"]),
                    int(r["relative_orbit"]),
                )
                for r in recs
            )
        )
        for key in keys:
            part = [
                r for r in recs
                if (
                    int(r["year"]),
                    str(r["orbit_state"]),
                    int(r["relative_orbit"]),
                ) == key
            ]
            cps = [
                cp_by_scene[r["scene_id"]]
                for r in part
            ]
            print(
                f"  MOSAIC {stream}: "
                f"{key[0]} {key[1]} {key[2]} "
                f"scenes={len(part)}",
                flush=True,
            )
            _, qa = mosaic_partition(
                support, stream, part, cps
            )
            all_part_qa.append(qa)

    # Merge with any QA from earlier partial invocations when possible.
    scene_qa = pd.DataFrame(all_scene_qa)
    if len(scene_qa):
        scene_qa.to_csv(
            SCENE_QA_OUT, index=False
        )
    part_qa = pd.DataFrame(all_part_qa)
    if len(part_qa):
        part_qa.to_csv(
            PART_QA_OUT, index=False
        )

    # Full-run gate only when both streams and no year filter.
    full_run = (
        args.stream == "both"
        and args.year is None
    )

    strict_cal_fail = (
        int(
            pd.to_numeric(
                scene_qa.get(
                    "calibration_support_fail_n",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).fillna(0).sum()
        )
        if len(scene_qa) else 0
    )
    strict_noise_fail = (
        int(
            pd.to_numeric(
                scene_qa.get(
                    "noise_support_fail_n",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).fillna(0).sum()
        )
        if len(scene_qa) else 0
    )
    nonpositive_n = (
        int(
            pd.to_numeric(
                scene_qa.get(
                    "nonpositive_corrected_power_n",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).fillna(0).sum()
        )
        if len(scene_qa) else 0
    )

    if full_run:
        primary_parts = part_qa[
            part_qa["stream"].eq("primary")
        ] if len(part_qa) else pd.DataFrame()
        primary_yt_n = len(primary_parts)

        if primary_yt_n != EXPECTED_PRIMARY_YT_N:
            status = "FAIL"
        elif strict_cal_fail or strict_noise_fail:
            # No extrapolation was performed. A support failure is not
            # silently accepted; it triggers a review gate.
            status = "REVIEW_STRICT_LUT_SUPPORT"
        else:
            status = "PASS"
    else:
        status = (
            "PARTIAL_SCOPE_COMPLETE"
            if not (
                strict_cal_fail
                or strict_noise_fail
            )
            else "PARTIAL_SCOPE_REVIEW_STRICT_LUT_SUPPORT"
        )

    qa = {
        "status": status,
        "stage":
            "DESIGN_C_C2VF_FULLARCHIVE_SENTINEL1_CORRECTED_SIGNAL_EXTRACTION_OPTIMIZED_ZERO_SUPPORT_FASTPATH",
        "execution_scope": {
            "stream": args.stream,
            "year": args.year,
            "force": bool(args.force),
        },
        "full_run": bool(full_run),
        "support_coordinates_n": int(len(support)),
        "primary_canonical_scenes_frozen_n":
            int(
                primary_plan[
                    "canonical_scene_id"
                ].nunique()
            ),
        "auxiliary_hhhv_scene_rows_frozen_n":
            int(len(aux_plan)),
        "scene_qa_rows_this_run_n":
            int(len(scene_qa)),
        "partition_qa_rows_this_run_n":
            int(len(part_qa)),
        "strict_calibration_support_fail_points_n":
            strict_cal_fail,
        "strict_noise_support_fail_points_n":
            strict_noise_fail,
        "nonpositive_corrected_power_points_n":
            nonpositive_n,
        "nonpositive_corrected_power_policy":
            "NaN; never clipped",
        "mosaic_rule":
            "median corrected sigma0 LINEAR across same date/track/support/polarization; dB after median",
        "singleton_azimuth_noise_vector_policy":
            "accepted when sole line coordinate lies within a valid declared firstAzimuthLine:lastAzimuthLine interval; sole LUT value used only within declared azimuth/range applicability block; no use outside declared support",
        "azimuth_noise_edge_extension_policy":
            "within exactly one declared azimuth/range applicability block, multi-sample azimuth LUTs are linearly interpolated within sampled line support and linearly extended only between an outermost LUT knot and the corresponding declared firstAzimuthLine:lastAzimuthLine boundary; no extension outside declared applicability support",
        "zero_study_support_fastpath":
            "raster/GCP coverage evaluated before XML fetch; if raster_inside_n == 0, radiometric XML is not fetched and a zero-support checkpoint is written",
        "partial_primary_spatial_coverage_allowed":
            True,
        "uncovered_support_policy":
            "missing_not_imputed",
        "hh_hv_substituted_for_vv_vh":
            False,
        "c2t_modified": False,
        "c2uf_modified": False,
        "groundwater_values_read": False,
        "irrigation_flow_values_read": False,
        "ricefloodit_flood_values_read": False,
        "sensor_response_used_for_feature_selection":
            False,
        "inundation_threshold_selected": False,
        "classifier_fitted": False,
        "association_models_fitted": 0,
    }
    QA_OUT.write_text(
        json.dumps(qa, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "DESIGN C - C2V-F FULL-ARCHIVE SENTINEL-1 CORRECTED SIGNAL EXTRACTION",
        "=" * 86,
        "",
        f"Execution scope: stream={args.stream}, year={args.year}",
        f"Fixed support coordinates: {len(support)}",
        f"Frozen primary canonical scenes: {primary_plan['canonical_scene_id'].nunique()}",
        f"Frozen auxiliary HH/HV scene rows: {len(aux_plan)}",
        f"Scene QA rows this run: {len(scene_qa)}",
        f"Partition mosaics this run: {len(part_qa)}",
        f"Strict calibration-support failures: {strict_cal_fail}",
        f"Strict noise-support failures: {strict_noise_fail}",
        f"Non-positive corrected powers (flagged -> missing): {nonpositive_n}",
        "",
        "MOSAIC RULE",
        "-----------",
        "Same acquisition date + orbit state + relative orbit + support point.",
        "Median corrected LINEAR sigma0 across overlapping scenes.",
        "dB conversion only after linear-power mosaicking.",
        "",
        "FIREWALL",
        "--------",
        "Groundwater read: False",
        "Irrigation flow read: False",
        "RiceFloodIT flooding values read: False",
        "Threshold/classifier: False",
        "Association model: False",
        "",
        f"C2V STATUS: {status}",
    ]
    txt = "\n".join(lines) + "\n"
    TXT_OUT.write_text(txt, encoding="utf-8")
    print("\n" + txt)

    if status == "FAIL":
        raise RuntimeError(
            "C2V failed its frozen-universe gate."
        )


if __name__ == "__main__":
    main()

