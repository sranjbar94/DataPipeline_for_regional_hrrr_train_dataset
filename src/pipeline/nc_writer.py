# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""
NetCDF Writer — creates and writes the regional_hrrr_train_dataset.nc
in a schema that exactly matches HRRR-Mini so it drops into CorrDiff
without any dataset loader changes.

HRRR-Mini schema recap:
    Dimensions : sample, y_lr, x_lr, y_hr, x_hr, coord
    Root vars  : time(sample), coord(sample, coord)
    Group input   : (sample, y_lr, x_lr)  -- ERA5 LR channels
    Group output  : (sample, y_hr, x_hr)  -- HRRR HR targets
    Group invariant: (y_lr, x_lr)          -- static fields (written once)
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np

from src.utils.logger import get_logger

log = get_logger("nc_writer")

# Epoch aligned with the dataset period (2015-01-01)
_EPOCH = datetime(2015, 1, 1)
_EPOCH_STR = "hours since 2015-01-01 00:00:00"

# All ERA5 input channel names written to [input] group
_SL_CHANNELS = [
    "u10", "v10", "t2m", "tcwv", "sp", "msl",
    "d2m", "q",
    "ssrd", "strd", "tp", "sf", "lsm",
]
_PL_BASE = ["u", "v", "z", "t"]
_PL_LEVS = [1000, 850, 500, 250]


def _all_input_channels() -> list[str]:
    channels = list(_SL_CHANNELS)
    for var in _PL_BASE:
        for lev in _PL_LEVS:
            channels.append(f"{var}{lev}")
    # specific humidity at pressure levels
    for lev in _PL_LEVS:
        channels.append(f"q{lev}")
    return channels


# HRRR output short names (must match config hrrr.output_vars[*].short)
_OUTPUT_CHANNELS = ["2t", "10u", "10v", "tp", "ssrd", "strd", "sp", "q", "sf"]


class NCWriter:
    """
    Manages creation of and incremental writing to the output NetCDF.
    Call `open()` before the sampling loop, `write_sample()` per sample,
    `close()` when done.  Supports append mode for checkpoint recovery.
    """

    def __init__(self, path: Path | str, n_samples: int, cfg: SimpleNamespace):
        self.path      = Path(path)
        self.n_samples = n_samples
        self.patch_lr  = cfg.patches.era5_patch_size   # 8
        self.patch_hr  = cfg.patches.hrrr_patch_size   # 64
        self.compress  = cfg.storage.compression_level  # 4
        self._ds       = None
        self._invariant_written = False

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------

    def _create(self):
        """Create a fresh NetCDF file with the HRRR-Mini schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ds = nc.Dataset(str(self.path), "w", format="NETCDF4")

        # --- Dimensions ---
        ds.createDimension("sample", self.n_samples)
        ds.createDimension("y_lr",   self.patch_lr)
        ds.createDimension("x_lr",   self.patch_lr)
        ds.createDimension("y_hr",   self.patch_hr)
        ds.createDimension("x_hr",   self.patch_hr)
        ds.createDimension("coord",  2)

        # --- Root variables ---
        tv = ds.createVariable("time", "f8", ("sample",))
        tv.units     = _EPOCH_STR
        tv.calendar  = "standard"
        tv.long_name = "UTC timestamp of sample"

        cv = ds.createVariable("coord", "u2", ("sample", "coord"))
        cv.long_name = "ERA5 grid indices of patch centre (lat_idx, lon_idx)"

        # --- [input] group -- ERA5 LR ---
        grp_in = ds.createGroup("input")
        for ch in _all_input_channels():
            v = grp_in.createVariable(
                ch, "f4", ("sample", "y_lr", "x_lr"),
                zlib=True, complevel=self.compress,
                fill_value=np.float32(np.nan),
            )
            v.long_name = ch

        # --- [output] group -- HRRR HR ---
        grp_out = ds.createGroup("output")
        for ch in _OUTPUT_CHANNELS:
            v = grp_out.createVariable(
                ch, "f4", ("sample", "y_hr", "x_hr"),
                zlib=True, complevel=self.compress,
                fill_value=np.float32(np.nan),
            )
            v.long_name = ch

        # --- [invariant] group -- static fields ---
        grp_inv = ds.createGroup("invariant")
        for field in ["latitude", "longitude", "elev_mean", "lsm_mean"]:
            v = grp_inv.createVariable(
                field, "f4", ("y_lr", "x_lr"),
                zlib=True, complevel=self.compress,
            )
            v.long_name = field

        # --- Global attributes ---
        ds.title         = "CorrDiff Regional HRRR Training Dataset"
        ds.source_era5   = "ERA5 reanalysis (Copernicus CDS), 0.25 deg"
        ds.source_hrrr   = "HRRR operational (AWS Open Data), ~3 km"
        ds.era5_patch    = f"{self.patch_lr}x{self.patch_lr} pixels"
        ds.hrrr_patch    = f"{self.patch_hr}x{self.patch_hr} pixels"
        ds.schema_compat = "HRRR-Mini (PhysicsNeMo CorrDiff)"
        ds.time_period   = "2015-2025"
        ds.created_utc   = datetime.utcnow().isoformat()

        ds.close()
        log.info(f"Created output NetCDF: {self.path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self, resume: bool = False):
        """Open the NetCDF for writing.  Creates file if not resuming."""
        if not self.path.exists() or not resume:
            self._create()
            self._invariant_written = False
        else:
            log.info(f"Resuming -- appending to: {self.path}")
            self._invariant_written = True   # assume written in prior run
        self._ds = nc.Dataset(str(self.path), "a")

    def write_sample(
        self,
        idx: int,
        dt: datetime,
        lat_idx: int,
        lon_idx: int,
        era5_patch: dict,
        hrrr_patch: dict,
    ):
        """
        Write one sample at position `idx`.

        Parameters
        ----------
        idx        : sample index (0-based)
        dt         : sample datetime
        lat_idx    : ERA5 grid lat index of patch centre
        lon_idx    : ERA5 grid lon index of patch centre
        era5_patch : dict from ERA5Reader.extract_patch()
        hrrr_patch : dict from fetch_hrrr_patch() -- keys are short names
        """
        self._ds["time"][idx]     = (dt - _EPOCH).total_seconds() / 3600.0
        self._ds["coord"][idx, :] = [lat_idx, lon_idx]

        grp_in  = self._ds["input"]
        grp_out = self._ds["output"]
        grp_inv = self._ds["invariant"]

        # Input channels
        for ch in _all_input_channels():
            if ch in era5_patch:
                grp_in[ch][idx] = era5_patch[ch]

        # Output channels
        for ch in _OUTPUT_CHANNELS:
            if ch in hrrr_patch:
                grp_out[ch][idx] = hrrr_patch[ch]

        # Invariants -- written once from first valid sample
        if not self._invariant_written and "_lat" in era5_patch:
            lat_1d = era5_patch["_lat"]   # shape (8,)
            lon_1d = era5_patch["_lon"]   # shape (8,)
            lat_2d = np.tile(lat_1d[:, None], (1, len(lon_1d))).astype(np.float32)
            lon_2d = np.tile(lon_1d[None, :], (len(lat_1d), 1)).astype(np.float32)
            grp_inv["latitude"][:]  = lat_2d
            grp_inv["longitude"][:] = lon_2d
            grp_inv["lsm_mean"][:]  = era5_patch.get(
                "lsm", np.zeros((self.patch_lr, self.patch_lr), dtype=np.float32)
            )
            grp_inv["elev_mean"][:] = np.zeros(
                (self.patch_lr, self.patch_lr), dtype=np.float32
            )
            self._invariant_written = True

    def flush(self):
        """Force write buffers to disk."""
        if self._ds:
            self._ds.sync()

    def close(self):
        if self._ds:
            self._ds.close()
            self._ds = None
            log.info(f"NetCDF closed: {self.path}")
