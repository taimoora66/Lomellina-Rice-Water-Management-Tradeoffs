import hashlib
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu"
)

bucket = "eodata"

orig = (
    "Sentinel-1/SAR/IW_GRDH_1S/2025/04/21/"
    "S1C_IW_GRDH_1SDV_20250421T053447_20250421T053516_001987_003FE6_9B29.SAFE/"
    "annotation/calibration/"
)

cog = (
    "Sentinel-1/SAR/IW_GRDH_1S-COG/2025/04/21/"
    "S1C_IW_GRDH_1SDV_20250421T053447_20250421T053516_001987_003FE6_9043_COG.SAFE/"
    "annotation/calibration/"
)

files = {
    "ORIG_VV": orig
    + "noise-s1c-iw-grd-vv-20250421t053447-20250421t053516-001987-003fe6-001.xml",

    "ORIG_VH": orig
    + "noise-s1c-iw-grd-vh-20250421t053447-20250421t053516-001987-003fe6-002.xml",

    "COG_VV": cog
    + "noise-s1c-iw-grd-vv-20250421t053447-20250421t053516-001987-003fe6-001-cog.xml",

    "COG_VH": cog
    + "noise-s1c-iw-grd-vh-20250421t053447-20250421t053516-001987-003fe6-002-cog.xml",
}

for name, key in files.items():

    data = s3.get_object(
        Bucket=bucket,
        Key=key
    )["Body"].read()

    sha = hashlib.sha256(data).hexdigest()

    print(
        f"{name}: "
        f"bytes={len(data)} "
        f"sha256={sha}"
    )