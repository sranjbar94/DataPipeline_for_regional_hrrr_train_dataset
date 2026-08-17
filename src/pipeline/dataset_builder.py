# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""
DatasetBuilder — the main sampling loop (Stage 2).

For each iteration:
  1. Draw a random timestamp from 2010–2020
  2. Sample `samples_per_ts` random ocean patch centres
  3. Extract ERA5 8×8 patch from disk
  4. Fetch HRRR 64×64 patch from AWS S3 (or synthetic if dry_run)
  5. Write sample to output NetCDF
  6. Save checkpoint every 1000 samples for safe resume
"""

from __future__ import annotations
import json
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import zoom

from src.pipeline.era5_reader import ERA5Reader
from src.pipeline.hrrr_fetcher import fetch_hrrr_patch
from src.pipeline.nc_writer import NCWriter
from src.utils.logger import get_logger
from src.utils.time_sampler import TimestampSampler

log = get_logger("dataset_builder")


class DatasetBuilder:
    """
    Orchestrates the full dataset build.

    Parameters
    ----------
    cfg      : loaded pipeline config namespace
    dry_run  : if True, use synthetic HRRR data (no S3 calls) for local testing
    """

    def __init__(self, cfg: SimpleNamespace, dry_run: bool = False):
        self.cfg     = cfg
        self.dry_run = dry_run

        self.era5_dir  = Path(cfg.storage.era5_raw_dir)
        self.out_path  = (Path(cfg.storage.output_dir) /
                          cfg.storage.output_filename)
        self.ckpt_dir  = Path(cfg.storage.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.ckpt_dir / "build_checkpoint.json"

        self.n_samples  = cfg.sampling.n_samples
        self.per_ts     = cfg.sampling.samples_per_ts
        self.seed       = cfg.sampling.random_seed
        self.ckpt_freq  = cfg.storage.checkpoint_freq
        self.log_freq   = cfg.logging.log_freq

        random.seed(self.seed)
        np.random.seed(self.seed)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, samples_done: int, used_ts: set):
        state = {
            "samples_done":    samples_done,
            "used_timestamps": sorted(used_ts),
        }
        with open(self.ckpt_path, "w") as f:
            json.dump(state, f, indent=2)
        log.info(f"  [CKPT] checkpoint saved at {samples_done:,} samples")

    def _load_checkpoint(self) -> tuple[int, set]:
        if not self.ckpt_path.exists():
            return 0, set()
        with open(self.ckpt_path) as f:
            state = json.load(f)
        samples_done = state.get("samples_done", 0)
        used_ts      = set(state.get("used_timestamps", []))
        log.info(f"Resuming from checkpoint: {samples_done:,} samples done")
        return samples_done, used_ts

    # ------------------------------------------------------------------
    # Main build
    # ------------------------------------------------------------------

    def _synthetic_era5_patch(self, lat: float, lon: float) -> dict:
        """Generate a synthetic ERA5 patch for dry-run testing."""
        ps = self.cfg.patches.era5_patch_size
        res = self.cfg.patches.era5_resolution
        patch = {}
        # Surface variables with realistic-ish random ranges
        synth_ranges = {
            "t2m": (270, 310), "d2m": (260, 300), "sp": (95000, 105000),
            "msl": (98000, 104000), "u10": (-15, 15), "v10": (-15, 15),
            "ssrd": (0, 3e6), "strd": (0, 2e6), "tp": (0, 0.05),
            "sf": (0, 0.01), "lsm": (0, 0.1), "tcwv": (5, 60),
            "q": (0.001, 0.02),
        }
        for var, (lo, hi) in synth_ranges.items():
            patch[var] = np.random.uniform(lo, hi, (ps, ps)).astype(np.float32)
        # Pressure-level variables
        for var in ["u", "v", "z", "t", "q"]:
            for lev in [1000, 850, 500, 250]:
                patch[f"{var}{lev}"] = np.random.uniform(
                    -50, 50, (ps, ps)
                ).astype(np.float32)
        # Coordinate arrays
        half = ps // 2
        patch["_lat"] = np.linspace(lat - half * res, lat + half * res, ps).astype(np.float32)
        patch["_lon"] = np.linspace(lon - half * res, lon + half * res, ps).astype(np.float32)
        return patch
    
    def run(self):
        cfg = self.cfg

        log.info("=" * 60)
        log.info("CorrDiff Regional Dataset Builder")
        log.info(f"  Output        : {self.out_path}")
        log.info(f"  Target samples: {self.n_samples:,}")
        log.info(f"  Samples/ts    : {self.per_ts}")
        log.info(f"  Date range    : {cfg.time.date_start} → {cfg.time.date_end}")
        log.info(f"  Dry run       : {self.dry_run}")
        log.info("=" * 60)

        # --- Checkpoint / resume ---
        samples_done, used_ts = self._load_checkpoint()
        resume = samples_done > 0

        # --- ERA5 reader ---
        reader = ERA5Reader(self.era5_dir, cfg)

        if self.dry_run:
            # --- Synthetic ocean centres (no ERA5 files needed) ---
            log.info("Dry run: generating synthetic ocean patch centres")
            lon_min, lon_max = cfg.domain.lon_min, cfg.domain.lon_max
            lat_min, lat_max = cfg.domain.lat_min, cfg.domain.lat_max
            buffer = (cfg.patches.era5_patch_size // 2) * cfg.patches.era5_resolution
            ocean_centers = [
                (round(lat, 2), round(lon, 2))
                for lat in np.arange(lat_min + buffer, lat_max - buffer, 1.0)
                for lon in np.arange(lon_min + buffer, lon_max - buffer, 1.0)
            ]
            random.shuffle(ocean_centers)
            ocean_centers = ocean_centers[:5000]
            log.info(f"Synthetic ocean patch centres: {len(ocean_centers):,}")
        else:
            # --- Discover ocean patch centres from real ERA5 ---
            ocean_centers = reader.get_ocean_patch_centers(
                lsm_threshold  = cfg.ocean.lsm_threshold,
                min_ocean_frac = cfg.ocean.min_ocean_frac,
                lon_min = cfg.domain.lon_min, lon_max = cfg.domain.lon_max,
                lat_min = cfg.domain.lat_min, lat_max = cfg.domain.lat_max,
            )
        if not ocean_centers:
            raise RuntimeError(
                "No valid ocean patch centres found. "
                "Check ERA5 files and domain bounds in config."
            )

        # --- Timestamp sampler ---
        sampler = TimestampSampler(
            date_start    = cfg.time.date_start,
            date_end      = cfg.time.date_end,
            allow_repeats = cfg.sampling.allow_ts_repeats,
            seed          = self.seed,
            used          = used_ts,
        )
        log.info(f"Timestamp pool: {sampler.pool_size:,} "
                 f"({sampler.n_used:,} already used)")

        # --- NetCDF writer ---
        writer = NCWriter(self.out_path, self.n_samples, cfg)
        writer.open(resume=resume)

        # Get LSM grid for coord index lookup
        if self.dry_run:
            ps = cfg.patches.era5_patch_size
            res = cfg.patches.era5_resolution
            lsm_lats = np.arange(cfg.domain.lat_max, cfg.domain.lat_min, -res).astype(np.float32)
            lsm_lons = np.arange(cfg.domain.lon_min, cfg.domain.lon_max, res).astype(np.float32)
            lsm = np.zeros((len(lsm_lats), len(lsm_lons)), dtype=np.float32)
        else:
            lsm, lsm_lats, lsm_lons = reader.get_lsm()

        # --- Sampling loop ---
        log.info(f"Sampling loop: {samples_done:,} → {self.n_samples:,}")
        try:
            while samples_done < self.n_samples:
                # 1. Draw timestamp
                try:
                    dt = sampler.sample()
                except StopIteration as e:
                    log.warning(str(e))
                    break

                # 2. Pick random ocean centres for this timestamp
                centres_this_ts = random.sample(
                    ocean_centers,
                    min(self.per_ts, len(ocean_centers)),
                )

                ts_successes = 0

                for lat, lon in centres_this_ts:
                    if samples_done >= self.n_samples:
                        break

                    try:
                        # 3. ERA5 patch
                        if self.dry_run:
                            era5_patch = self._synthetic_era5_patch(lat, lon)
                        else:
                            era5_patch = reader.extract_patch(dt, lat, lon)

                        # 4. HRRR patch
                        hrrr_patch = fetch_hrrr_patch(
                            dt, lat, lon,
                            patch_size = cfg.patches.hrrr_patch_size,
                            fxx        = cfg.hrrr.forecast_hour,
                            dry_run    = self.dry_run,
                        )
                        if hrrr_patch is None:
                            log.debug(
                                f"  HRRR fetch failed: {dt} ({lat:.2f}, {lon:.2f})"
                            )
                            continue

                        # 4b. Fill missing HRRR strd from ERA5 (interpolated)
                        if "strd" not in hrrr_patch and "strd" in era5_patch:
                            hr = cfg.patches.hrrr_patch_size
                            lr = cfg.patches.era5_patch_size
                            scale = hr / lr
                            strd_lr = era5_patch["strd"]
                            hrrr_patch["strd"] = zoom(
                                strd_lr, scale, order=1
                            ).astype(np.float32)

                        # 5. ERA5 coord indices
                        lat_idx = int(np.argmin(np.abs(lsm_lats - lat)))
                        lon_idx = int(np.argmin(np.abs(lsm_lons - lon)))

                        # 6. Write
                        writer.write_sample(
                            idx        = samples_done,
                            dt         = dt,
                            lat_idx    = lat_idx,
                            lon_idx    = lon_idx,
                            era5_patch = era5_patch,
                            hrrr_patch = hrrr_patch,
                        )

                        samples_done += 1
                        ts_successes += 1

                        # Logging
                        if samples_done % self.log_freq == 0:
                            pct = 100 * samples_done / self.n_samples
                            log.info(
                                f"  Progress: {samples_done:>8,} / "
                                f"{self.n_samples:,}  ({pct:.1f}%)"
                            )

                        # Checkpoint
                        if samples_done % self.ckpt_freq == 0:
                            writer.flush()
                            self._save_checkpoint(samples_done, sampler.used)

                    except Exception as e:
                        log.warning(
                            f"  Sample failed [{dt} ({lat:.2f},{lon:.2f})]: {e}"
                        )
                        continue

                if ts_successes == 0:
                    log.debug(f"  Zero successes from timestamp {dt}")

        finally:
            writer.close()
            self._save_checkpoint(samples_done, sampler.used)

        log.info("=" * 60)
        log.info(f"Build complete: {samples_done:,} / {self.n_samples:,} samples")
        log.info(f"Output: {self.out_path}")
        log.info("=" * 60)
        return samples_done
