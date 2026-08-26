# Reviewed: comments kept in sync with behavior.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""
ERA5 bulk downloader — Stage 1 of the pipeline.

Downloads DAILY ERA5 single-level and pressure-level NetCDF files
for the CONUS/HRRR domain via the Copernicus CDS API.

Requesting data by DAY (rather than by month) avoids the CDS API
volume-limit errors that occur when a single request covers too many
grid points x time steps. Each daily request is small enough to stay
well within the CDS per-request size cap (~2 GB).

Prerequisite: ~/.cdsapirc must exist with your CDS API key.
    url: https://cds.climate.copernicus.eu/api/v2
    key: UID:API-KEY
"""

from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import cdsapi

from src.utils.logger import get_logger

log = get_logger("era5_downloader")


def _day_range(start: date, end: date):
    """Yield every calendar date from start to end inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


# CDS returns a ZIP when mixing instantaneous + accumulated variables,
# so we split them into two requests and merge with xarray.
_ACCUMULATED_VARS = {
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_precipitation",
    "snowfall",
}


def _download_single_level_day(
    client: cdsapi.Client,
    day: date,
    variables: list[str],
    area: list[float],
    out_dir: Path,
    retries: int = 3,
    retry_wait: float = 30.0,
) -> str:
    """
    Download one day of single-level ERA5 data (all 24 hours).
    Splits instantaneous and accumulated variables into two CDS requests
    to avoid ZIP downloads, then merges into a single NetCDF.
    Skips gracefully if the file already exists (safe to resume).
    """
    import xarray as xr

    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"era5_sl_{day.strftime('%Y%m%d')}.nc"

    if fpath.exists():
        # Verify it is real NetCDF, not a stale ZIP
        with open(fpath, "rb") as f:
            magic = f.read(2)
        if magic != b"PK":
            log.debug(f"  [SKIP] {fpath.name} already exists")
            return str(fpath)
        else:
            log.info(f"  [REDO] {fpath.name} is a ZIP -- re-downloading")
            fpath.unlink()

    instant_vars = [v for v in variables if v not in _ACCUMULATED_VARS]
    accum_vars   = [v for v in variables if v in _ACCUMULATED_VARS]

    def _retrieve(var_list, tmp_path):
        wait = retry_wait
        for attempt in range(1, retries + 1):
            try:
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable":     var_list,
                        "year":         str(day.year),
                        "month":        f"{day.month:02d}",
                        "day":          f"{day.day:02d}",
                        "time":         [f"{h:02d}:00" for h in range(24)],
                        "area":         area,
                        "format":       "netcdf",
                    },
                    str(tmp_path),
                )
                return
            except Exception as exc:
                if attempt == retries:
                    raise
                log.warning(
                    f"  [RETRY {attempt}/{retries}] {tmp_path.name}: {exc}  "
                    f"(waiting {wait:.0f}s)"
                )
                time.sleep(wait)
                wait *= 2

    log.info(f"  [DL  ] single-level  {day} ...")

    tmp_inst = out_dir / f"_tmp_inst_{day.strftime('%Y%m%d')}.nc"
    tmp_acc  = out_dir / f"_tmp_acc_{day.strftime('%Y%m%d')}.nc"

    try:
        if instant_vars:
            _retrieve(instant_vars, tmp_inst)
        if accum_vars:
            _retrieve(accum_vars, tmp_acc)

        # Merge into single file
        datasets = []
        if instant_vars and tmp_inst.exists():
            datasets.append(xr.open_dataset(str(tmp_inst)))
        if accum_vars and tmp_acc.exists():
            datasets.append(xr.open_dataset(str(tmp_acc)))

        merged = xr.merge(datasets)
        merged.to_netcdf(str(fpath))
        merged.close()
        for ds in datasets:
            ds.close()

        log.info(f"  [OK  ] {fpath.name}")
        return str(fpath)
    finally:
        # Clean up temp files
        if tmp_inst.exists():
            tmp_inst.unlink()
        if tmp_acc.exists():
            tmp_acc.unlink()


def _download_pressure_level_day(
    client: cdsapi.Client,
    day: date,
    variables: list[str],
    levels: list[int],
    area: list[float],
    out_dir: Path,
    retries: int = 3,
    retry_wait: float = 30.0,
) -> str:
    """
    Download one day of pressure-level ERA5 data (all 24 hours).
    Skips gracefully if the file already exists (safe to resume).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"era5_pl_{day.strftime('%Y%m%d')}.nc"

    if fpath.exists():
        log.debug(f"  [SKIP] {fpath.name} already exists")
        return str(fpath)

    log.info(f"  [DL  ] pressure-level {day} ...")
    wait = retry_wait
    for attempt in range(1, retries + 1):
        try:
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type":   "reanalysis",
                    "variable":       variables,
                    "pressure_level": [str(lv) for lv in levels],
                    "year":           str(day.year),
                    "month":          f"{day.month:02d}",
                    "day":            f"{day.day:02d}",
                    "time":           [f"{h:02d}:00" for h in range(24)],
                    "area":           area,
                    "format":         "netcdf",
                },
                str(fpath),
            )
            log.info(f"  [OK  ] {fpath.name}")
            return str(fpath)
        except Exception as exc:
            if attempt == retries:
                raise
            log.warning(
                f"  [RETRY {attempt}/{retries}] {fpath.name}: {exc}  "
                f"(waiting {wait:.0f}s)"
            )
            time.sleep(wait)
            wait *= 2

    raise RuntimeError(f"Failed after {retries} attempts: {fpath.name}")


def run_downloader(
    cfg: SimpleNamespace,
    workers: int = 2,
    skip_pressure: bool = False,
    date_start_override: str | None = None,
    date_end_override: str | None = None,
):
    """
    Download ERA5 data day-by-day for the configured (or overridden) period.

    Each CDS request covers exactly one calendar day x 24 hours, keeping
    request sizes small and avoiding CDS volume-limit errors.

    Parameters
    ----------
    cfg                  : loaded pipeline config namespace
    workers              : parallel CDS API workers (keep <= 4)
    skip_pressure        : only download single-level files
    date_start_override  : ISO date string, overrides config
    date_end_override    : ISO date string, overrides config
    """
    out_root = Path(cfg.storage.era5_raw_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    start_str = date_start_override or cfg.time.date_start
    end_str   = date_end_override   or cfg.time.date_end
    d_start   = date.fromisoformat(start_str)
    d_end     = date.fromisoformat(end_str)
    days      = list(_day_range(d_start, d_end))

    area = [
        cfg.domain.lat_max,
        cfg.domain.lon_min,
        cfg.domain.lat_min,
        cfg.domain.lon_max,
    ]

    sl_vars = cfg.era5.single_level
    pl_vars = cfg.era5.pressure_level.variables
    pl_levs = cfg.era5.pressure_level.levels

    log.info("=" * 60)
    log.info("ERA5 Daily Downloader")
    log.info(f"  Period  : {d_start} -> {d_end}  ({len(days)} days)")
    log.info(f"  Domain  : lat [{cfg.domain.lat_min}, {cfg.domain.lat_max}]  "
             f"lon [{cfg.domain.lon_min}, {cfg.domain.lon_max}]")
    log.info(f"  Workers : {workers}")
    log.info(f"  Output  : {out_root}")
    log.info("=" * 60)

    client = cdsapi.Client()

    tasks: list[tuple[str, date]] = []
    for day in days:
        tasks.append(("sl", day))
        if not skip_pressure:
            tasks.append(("pl", day))

    log.info(f"Total tasks: {len(tasks)}")
    results: dict[str, list] = {"success": [], "failed": []}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict = {}
        for kind, day in tasks:
            if kind == "sl":
                f = pool.submit(
                    _download_single_level_day,
                    client, day, sl_vars, area,
                    out_root / "single_level",
                )
            else:
                f = pool.submit(
                    _download_pressure_level_day,
                    client, day, pl_vars, pl_levs, area,
                    out_root / "pressure_level",
                )
            futures[f] = (kind, day)

        done = 0
        total = len(futures)
        for future in as_completed(futures):
            kind, day = futures[future]
            done += 1
            try:
                results["success"].append(future.result())
            except Exception as exc:
                msg = f"{kind} {day}: {exc}"
                log.error(f"  [FAIL] {msg}")
                results["failed"].append(msg)

            if done % 50 == 0 or done == total:
                log.info(f"  Progress: {done}/{total}  "
                         f"({len(results['failed'])} failed so far)")

    log.info("=" * 60)
    log.info(f"Done: {len(results['success'])} succeeded, "
             f"{len(results['failed'])} failed")
    if results["failed"]:
        log.warning("Failed tasks (safe to re-run -- existing files are skipped):")
        for m in results["failed"]:
            log.warning(f"  {m}")

    result_path = out_root / "download_results.json"
    with open(result_path, "w") as fp:
        json.dump(results, fp, indent=2)
    log.info(f"Results log: {result_path}")
    return results
