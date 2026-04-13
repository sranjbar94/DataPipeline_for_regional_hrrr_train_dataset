"""
HRRR S3 Fetcher — streams HRRR GRIB2 from AWS Open Data and extracts
a 64×64 patch centred on a given (lat, lon) at a given datetime.

AWS bucket: s3://noaa-hrrr-bdp-pds  (anonymous access, no key needed)

Requires:
    pip install cfgrib s3fs eccodes herbie-data
"""

from __future__ import annotations
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.utils.logger import get_logger

log = get_logger("hrrr_fetcher")

# HRRR variable name → cfgrib shortName mapping
# cfgrib uses GRIB shortName or stepType to select fields
_HRRR_VAR_MAP = {
    "2t":   dict(shortName="2t",   typeOfLevel="heightAboveGround", level=2),
    "10u":  dict(shortName="10u",  typeOfLevel="heightAboveGround", level=10),
    "10v":  dict(shortName="10v",  typeOfLevel="heightAboveGround", level=10),
    "tp":   dict(shortName="tp",   typeOfLevel="surface"),
    "ssrd": dict(shortName="dswrf", typeOfLevel="surface"),
    "strd": dict(shortName="dlwrf", typeOfLevel="surface"),
    "sp":   dict(shortName="sp",   typeOfLevel="surface"),
    "q":    dict(shortName="q",    typeOfLevel="heightAboveGround", level=2),
    "sf":   dict(shortName="weasd", typeOfLevel="surface"),
}


def _s3_url(dt: datetime, fxx: int = 0) -> str:
    return (
        f"s3://noaa-hrrr-bdp-pds/hrrr.{dt.strftime('%Y%m%d')}/"
        f"conus/hrrr.t{dt.hour:02d}z.wrfsfcf{fxx:02d}.grib2"
    )


def _find_patch_indices(
    lats: np.ndarray, lons: np.ndarray,
    lat: float, lon: float, patch_size: int
) -> tuple[int, int] | None:
    """Find row/col of nearest grid point; return None if too close to edge."""
    # HRRR lons are 0–360
    lon360 = lon % 360
    dist = np.sqrt((lats - lat) ** 2 + (lons - lon360) ** 2)
    r, c  = np.unravel_index(np.argmin(dist), dist.shape)
    half  = patch_size // 2
    if r - half < 0 or c - half < 0:
        return None
    if r + half > lats.shape[0] or c + half > lats.shape[1]:
        return None
    return int(r), int(c)


def fetch_hrrr_patch(
    dt: datetime,
    lat: float,
    lon: float,
    patch_size: int = 64,
    fxx: int = 0,
    retries: int = 3,
    dry_run: bool = False,
) -> dict[str, np.ndarray] | None:
    """
    Download HRRR GRIB2 from S3 and extract a patch_size × patch_size
    window centred on (lat, lon).

    Parameters
    ----------
    dt         : target datetime (UTC)
    lat, lon   : patch centre (lon in –180..180)
    patch_size : output patch size in HRRR pixels (default 64)
    fxx        : forecast hour (0 = analysis)
    retries    : number of S3 retry attempts
    dry_run    : if True, return synthetic random data (for local testing)

    Returns
    -------
    dict  {short_name: float32 array (patch_size, patch_size)}
    None  on unrecoverable failure
    """
    if dry_run:
        log.debug(f"  [DRY_RUN] synthetic HRRR patch ({dt}, {lat:.2f}, {lon:.2f})")
        return {k: np.random.randn(patch_size, patch_size).astype(np.float32)
                for k in _HRRR_VAR_MAP}

    try:
        import s3fs
        import cfgrib
    except ImportError:
        raise ImportError(
            "cfgrib and s3fs are required for HRRR fetching.\n"
            "Run: pip install cfgrib s3fs eccodes"
        )

    url = _s3_url(dt, fxx)
    fs  = s3fs.S3FileSystem(anon=True)

    for attempt in range(1, retries + 1):
        tmp_path = None
        try:
            log.debug(f"  [HRRR] fetching {url}  (attempt {attempt}/{retries})")
            with fs.open(url, "rb") as f_s3:
                with tempfile.NamedTemporaryFile(
                    suffix=".grib2", delete=False
                ) as tmp:
                    tmp.write(f_s3.read())
                    tmp_path = tmp.name

            # Open all message groups in the GRIB2 file
            datasets = cfgrib.open_datasets(tmp_path)
            patch    = {}

            for short_name, filter_keys in _HRRR_VAR_MAP.items():
                for ds in datasets:
                    import xarray as xr
                    xds = xr.Dataset(ds)
                    # Try matching by variable name substring
                    matched = None
                    for vname in xds.data_vars:
                        if vname.lower() == short_name.lower() or \
                           vname.lower().startswith(short_name[:3].lower()):
                            matched = vname
                            break
                    if matched is None:
                        continue

                    lats = xds["latitude"].values
                    lons = xds["longitude"].values
                    rc   = _find_patch_indices(lats, lons, lat, lon, patch_size)
                    if rc is None:
                        log.debug(f"    Patch centre too close to edge: {lat},{lon}")
                        continue

                    r, c = rc
                    half = patch_size // 2
                    arr  = xds[matched].values[r - half: r + half,
                                               c - half: c + half]
                    if arr.shape == (patch_size, patch_size):
                        patch[short_name] = arr.astype(np.float32)
                        break

            return patch if patch else None

        except Exception as e:
            log.warning(f"  [HRRR] attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    log.warning(f"  [HRRR] all {retries} attempts failed for {dt} ({lat},{lon})")
    return None
