# Documentation reviewed and improved for clarity.
"""
HRRR S3 Fetcher — streams HRRR GRIB2 from AWS Open Data and extracts
a 64x64 patch centred on a given (lat, lon) at a given datetime.

AWS bucket: s3://noaa-hrrr-bdp-pds  (anonymous access, no key needed)
"""

from __future__ import annotations
import os
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger

log = get_logger("hrrr_fetcher")

# Map our output channel names -> list of candidate cfgrib variable names
_HRRR_VAR_MAP = {
    "2t":   ["t2m"],
    "10u":  ["u10"],
    "10v":  ["v10"],
    "sp":   ["sp", "pres"],
    "q":    ["sh2", "q2m", "q"],
    "ssrd": ["sdswrf", "dswrf"],
    "strd": ["sdlwrf", "dlwrf"],
    "tp":   ["tp", "prate"],
    "sf":   ["sdwe", "weasd", "sde"],
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
    lon360 = lon % 360
    dist = np.sqrt((lats - lat) ** 2 + (lons - lon360) ** 2)
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    half = patch_size // 2
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
    Download HRRR GRIB2 from S3 and extract a patch_size x patch_size
    window centred on (lat, lon).

    Returns dict {short_name: float32 array} or None on failure.
    """
    if dry_run:
        log.debug(f"  [DRY_RUN] synthetic HRRR patch ({dt}, {lat:.2f}, {lon:.2f})")
        return {k: np.random.randn(patch_size, patch_size).astype(np.float32)
                for k in _HRRR_VAR_MAP}

    try:
        import s3fs
        import cfgrib
        import xarray as xr
    except ImportError:
        raise ImportError(
            "cfgrib and s3fs are required for HRRR fetching.\n"
            "Run: pip install cfgrib s3fs eccodes"
        )

    url = _s3_url(dt, fxx)
    fs = s3fs.S3FileSystem(anon=True)

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                datasets = cfgrib.open_datasets(tmp_path)

            # Build a flat lookup: cfgrib_varname -> (xr.Dataset, varname)
            all_vars = {}
            for ds in datasets:
                xds = xr.Dataset(ds)
                for vname in xds.data_vars:
                    # Skip variables with extra dimensions (pressure levels etc)
                    shape = xds[vname].shape
                    if len(shape) == 2:  # (y, x) only
                        all_vars[vname.lower()] = (xds, vname)

            log.debug(f"  [HRRR] available 2D vars: {sorted(all_vars.keys())}")

            patch = {}
            rc_cache = {}  # cache patch indices per grid shape

            for out_name, candidates in _HRRR_VAR_MAP.items():
                for cand in candidates:
                    if cand.lower() in all_vars:
                        xds, vname = all_vars[cand.lower()]
                        lats = xds["latitude"].values
                        lons = xds["longitude"].values

                        grid_key = (lats.shape, lons.shape)
                        if grid_key not in rc_cache:
                            rc_cache[grid_key] = _find_patch_indices(
                                lats, lons, lat, lon, patch_size
                            )
                        rc = rc_cache[grid_key]

                        if rc is None:
                            log.debug(f"    Patch centre too close to edge")
                            break

                        r, c = rc
                        half = patch_size // 2
                        arr = xds[vname].values[r - half: r + half,
                                                c - half: c + half]
                        if arr.shape == (patch_size, patch_size):
                            patch[out_name] = arr.astype(np.float32)
                        break

            if patch:
                log.debug(f"  [HRRR] extracted {len(patch)}/{len(_HRRR_VAR_MAP)} vars")
            return patch if patch else None

        except Exception as e:
            log.warning(f"  [HRRR] attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    log.warning(f"  [HRRR] all {retries} attempts failed for {dt} ({lat},{lon})")
    return None
