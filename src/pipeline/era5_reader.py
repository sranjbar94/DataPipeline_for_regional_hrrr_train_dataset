# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""
ERA5Reader — loads daily ERA5 NetCDF files from disk and extracts
8x8 patches centred on a given (lat, lon) at a given datetime.

Files are named era5_sl_YYYYMMDD.nc / era5_pl_YYYYMMDD.nc (one per day),
matching the output of era5_downloader.py.
"""

from __future__ import annotations
from datetime import datetime, date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
import zipfile
import shutil

from src.utils.logger import get_logger

log = get_logger("era5_reader")

# Variable short-name mappings ERA5 -> internal name
_SL_RENAME = {
    "t2m":  "t2m",
    "d2m":  "d2m",
    "sp":   "sp",
    "msl":  "msl",
    "u10":  "u10",
    "v10":  "v10",
    "ssrd": "ssrd",
    "strd": "strd",
    "tp":   "tp",
    "sf":   "sf",
    "lsm":  "lsm",
    "tcwv": "tcwv",
}


class ERA5Reader:
    """
    Lazy, memory-efficient reader for daily ERA5 NetCDF files.

    Each file covers one calendar day x 24 hours (matching the output
    of era5_downloader.py).  Caches at most 4 days in memory at once;
    evicts the oldest when a fifth day is needed.
    """

    def __init__(self, era5_dir: Path | str, cfg: SimpleNamespace):
        self.era5_dir  = Path(era5_dir)
        self.patch_sz  = cfg.patches.era5_patch_size   # 8
        self.era5_res  = cfg.patches.era5_resolution   # 0.25
        self.cfg       = cfg
        self._sl_cache: dict[date, xr.Dataset] = {}
        self._pl_cache: dict[date, xr.Dataset] = {}
        self._lsm_cache: tuple | None = None   # (lsm_arr, lats, lons)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sl_path(self, d: date) -> Path:
        return (self.era5_dir / "single_level" /
                f"era5_sl_{d.strftime('%Y%m%d')}.nc")

    def _pl_path(self, d: date) -> Path:
        return (self.era5_dir / "pressure_level" /
                f"era5_pl_{d.strftime('%Y%m%d')}.nc")

    def _unzip_if_needed(self, fpath: Path) -> Path:
        """If fpath is actually a ZIP archive, extract the .nc inside it."""
        # Quick check: NetCDF4 files start with bytes 0x89484446 or CDF\x01/\x02
        with open(fpath, "rb") as f:
            magic = f.read(4)
        if magic[:2] == b"PK":  # ZIP magic bytes
            log.info(f"  Unzipping {fpath.name} ...")
            extract_dir = fpath.parent / f"_unzip_{fpath.stem}"
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(fpath, "r") as zf:
                zf.extractall(extract_dir)
            # Find the .nc file inside
            nc_files = list(extract_dir.glob("*.nc"))
            if not nc_files:
                raise FileNotFoundError(
                    f"No .nc file found inside ZIP: {fpath}"
                )
            # Replace the ZIP with the actual NetCDF
            nc_file = nc_files[0]
            fpath.unlink()
            shutil.move(str(nc_file), str(fpath))
            shutil.rmtree(extract_dir)
            log.info(f"  Extracted: {fpath.name}")
        return fpath

    def _load_sl(self, year: int, month: int, day: int) -> xr.Dataset:
        """Load (and cache) a single-level daily file."""
        d = date(year, month, day)
        if d not in self._sl_cache:
            fpath = self._sl_path(d)
            if not fpath.exists():
                raise FileNotFoundError(
                    f"ERA5 single-level file not found: {fpath}\n"
                    "Run `python run_pipeline.py download` first."
                )
            fpath = self._unzip_if_needed(fpath)
            self._sl_cache[d] = xr.open_dataset(str(fpath), engine="netcdf4")
            if len(self._sl_cache) > 4:
                oldest = min(self._sl_cache)
                self._sl_cache.pop(oldest).close()
        return self._sl_cache[d]

    def _load_pl(self, year: int, month: int, day: int) -> xr.Dataset:
        """Load (and cache) a pressure-level daily file."""
        d = date(year, month, day)
        if d not in self._pl_cache:
            fpath = self._pl_path(d)
            if not fpath.exists():
                raise FileNotFoundError(
                    f"ERA5 pressure-level file not found: {fpath}\n"
                    "Run `python run_pipeline.py download` first."
                )
            fpath = self._unzip_if_needed(fpath)
            self._pl_cache[d] = xr.open_dataset(str(fpath), engine="netcdf4")
            if len(self._pl_cache) > 4:
                oldest = min(self._pl_cache)
                self._pl_cache.pop(oldest).close()
        return self._pl_cache[d]

    def _lat_lon_indices(self, lats: np.ndarray, lons: np.ndarray,
                         lat: float, lon: float) -> tuple[int, int]:
        return (int(np.argmin(np.abs(lats - lat))),
                int(np.argmin(np.abs(lons - lon))))

    def _patch_slice(self, idx: int) -> slice:
        half = self.patch_sz // 2
        return slice(idx - half, idx + half)

    def _first_available_day(self) -> date:
        """
        Return the date of the first single-level file found in era5_dir.
        Falls back to cfg.time.date_start if no files exist yet.
        """
        sl_dir = self.era5_dir / "single_level"
        if sl_dir.exists():
            files = sorted(sl_dir.glob("era5_sl_????????.nc"))
            if files:
                name = files[0].stem   # era5_sl_20150101
                ds = name.split("_")[-1]
                return date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
        # Fall back to config start date
        return date.fromisoformat(self.cfg.time.date_start)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lsm(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (lsm_2d, lat_1d, lon_1d) for the full downloaded domain.
        Loaded from the first available daily file and cached in memory.
        """
        if self._lsm_cache is None:
            first_day = self._first_available_day()
            ds = self._load_sl(first_day.year, first_day.month, first_day.day)
            lsm_var = next(
                (v for v in ["lsm", "land_sea_mask"] if v in ds), None
            )
            if lsm_var is None:
                raise KeyError("land_sea_mask / lsm not found in ERA5 dataset.")
            lsm  = ds[lsm_var].isel(valid_time=0).values if "valid_time" in ds.dims else ds[lsm_var].isel(time=0).values
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            self._lsm_cache = (lsm.astype(np.float32), lats, lons)
            log.info(f"LSM loaded: grid {lsm.shape}, "
                     f"lat [{lats.min():.1f}, {lats.max():.1f}], "
                     f"lon [{lons.min():.1f}, {lons.max():.1f}]")
        return self._lsm_cache

    def get_ocean_patch_centers(
        self, lsm_threshold: float = 0.2, min_ocean_frac: float = 0.6,
        lon_min: float = -140, lon_max: float = -55,
        lat_min: float = 18,   lat_max: float = 56,
    ) -> list[tuple[float, float]]:
        """
        Return (lat, lon) pairs whose 8x8 ERA5 patch is majority ocean
        and whose centre falls within the HRRR domain (with buffer).
        """
        lsm, lats, lons = self.get_lsm()
        half   = self.patch_sz // 2
        buffer = half * self.era5_res
        centers = []

        for i, lat in enumerate(lats):
            if not (lat_min + buffer <= lat <= lat_max - buffer):
                continue
            for j, lon in enumerate(lons):
                if not (lon_min + buffer <= lon <= lon_max - buffer):
                    continue
                patch = lsm[i - half: i + half, j - half: j + half]
                if patch.shape != (self.patch_sz, self.patch_sz):
                    continue
                if np.mean(patch < lsm_threshold) >= min_ocean_frac:
                    centers.append((float(lat), float(lon)))

        log.info(f"Ocean patch centers found: {len(centers):,}")
        return centers

    def extract_patch(
        self, dt: datetime, lat: float, lon: float
    ) -> dict[str, np.ndarray]:
        """
        Extract an 8x8 ERA5 patch centred on (lat, lon) at datetime dt.

        Returns a dict mapping variable name -> float32 array (8, 8),
        plus 1-D "_lat" and "_lon" coordinate arrays.
        """
        ds_sl = self._load_sl(dt.year, dt.month, dt.day)
        ds_pl = self._load_pl(dt.year, dt.month, dt.day)

        # ERA5 CDS downloads use "valid_time" or "time" depending on API version
        time_dim = "valid_time" if "valid_time" in ds_sl.dims else "time"

        # Nearest time index
        times = ds_sl[time_dim].values
        t_idx = int(np.argmin(np.abs(times - np.datetime64(dt))))

        # Spatial indices
        lats = ds_sl["latitude"].values
        lons = ds_sl["longitude"].values
        li, lj = self._lat_lon_indices(lats, lons, lat, lon)
        rslice = self._patch_slice(li)
        cslice = self._patch_slice(lj)

        patch: dict[str, np.ndarray] = {}

        # --- Single-level ---
        for nc_var, short in _SL_RENAME.items():
            for candidate in [short, nc_var, f"var{nc_var}"]:
                if candidate in ds_sl:
                    da = ds_sl[candidate].isel({time_dim: t_idx})
                    arr = da.values[rslice, cslice]
                    if arr.shape == (self.patch_sz, self.patch_sz):
                        patch[short] = arr.astype(np.float32)
                    break

        # --- Derive specific humidity from d2m and sp ---
        if "d2m" in patch and "sp" in patch:
            d2m = patch["d2m"]   # K
            sp  = patch["sp"]    # Pa
            e_s = 611.2 * np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65))
            patch["q"] = (0.622 * e_s / (sp - 0.378 * e_s)).astype(np.float32)

        # --- Pressure-level ---
        pl_time_dim = "valid_time" if "valid_time" in ds_pl.dims else "time"
        pl_times    = ds_pl[pl_time_dim].values
        pl_t_idx    = int(np.argmin(np.abs(pl_times - np.datetime64(dt))))

        # Detect pressure coordinate name ("level" or "pressure_level")
        pl_coord = next(
            (c for c in ["level", "pressure_level", "isobaricInhPa"]
             if c in ds_pl.dims or c in ds_pl.coords),
            "level",
        )

        pl_lats = ds_pl["latitude"].values
        pl_lons = ds_pl["longitude"].values
        pli, plj = self._lat_lon_indices(pl_lats, pl_lons, lat, lon)
        pl_rslice = self._patch_slice(pli)
        pl_cslice = self._patch_slice(plj)

        for var in ["u", "v", "z", "t", "q"]:
            if var not in ds_pl:
                continue
            for lev in [1000, 850, 500, 250]:
                try:
                    arr = (ds_pl[var]
                           .sel({pl_coord: lev}, method="nearest")
                           .isel({pl_time_dim: pl_t_idx})
                           .values[pl_rslice, pl_cslice])
                    if arr.shape == (self.patch_sz, self.patch_sz):
                        # Store q at pressure levels as q1000, q850, etc.
                        key = f"{var}{lev}"
                        patch[key] = arr.astype(np.float32)
                except Exception:
                    pass

        # --- Coordinate arrays ---
        patch["_lat"] = lats[rslice].astype(np.float32)
        patch["_lon"] = lons[cslice].astype(np.float32)

        # --- Land-sea mask patch (use cached lsm grid) ---
        lsm, lsm_lats, lsm_lons = self.get_lsm()
        li2, lj2 = self._lat_lon_indices(lsm_lats, lsm_lons, lat, lon)
        patch["lsm"] = lsm[self._patch_slice(li2),
                           self._patch_slice(lj2)]

        return patch
