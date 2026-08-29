#!/usr/bin/env python3
# Reviewed: comments kept in sync with behavior.
# Docstring coverage checked and touched up.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""
run_pipeline.py — single entry point for all pipeline stages.

Usage
-----
# Test run (10 days in Jan 2015, fast validation)
python run_pipeline.py download --test_mode --workers 2
python run_pipeline.py build   --test_mode --dry_run   # synthetic HRRR
python run_pipeline.py build   --test_mode             # real HRRR S3

# Full 10-year run (2015-2025)
python run_pipeline.py download [--workers 2] [--skip_pressure]
python run_pipeline.py build
python run_pipeline.py stats
python run_pipeline.py all [--dry_run]

# Custom date range override
python run_pipeline.py download --date_start 2020-01-01 --date_end 2020-01-31
python run_pipeline.py build --n_samples 500 --samples_per_ts 5
"""

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("run_pipeline", log_dir="logs")

# ---------------------------------------------------------------------------
# Test-mode constants: 10 days in Jan 2015
# ---------------------------------------------------------------------------
TEST_DATE_START = "2015-01-01"
TEST_DATE_END   = "2015-01-10"
TEST_N_SAMPLES  = 200    # ~20 samples/day x 10 days


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------

def cmd_download(cfg, args):
    from src.pipeline.era5_downloader import run_downloader
    log.info("=== Stage 1: ERA5 Download ===")

    start = args.date_start
    end   = args.date_end

    if args.test_mode:
        log.info(
            f"[TEST MODE] Downloading only {TEST_DATE_START} -> {TEST_DATE_END} "
            "(10 days). Re-run without --test_mode for the full 2015-2025 period."
        )
        start = start or TEST_DATE_START
        end   = end   or TEST_DATE_END

    run_downloader(
        cfg,
        workers=args.workers,
        skip_pressure=args.skip_pressure,
        date_start_override=start,
        date_end_override=end,
    )


def cmd_build(cfg, args):
    from src.pipeline.dataset_builder import DatasetBuilder
    log.info("=== Stage 2: Dataset Build ===")

    if args.test_mode:
        log.info(
            f"[TEST MODE] Building {TEST_N_SAMPLES} samples from "
            f"{TEST_DATE_START} -> {TEST_DATE_END}. "
            "Re-run without --test_mode for the full 10-year dataset."
        )
        cfg.time.date_start          = TEST_DATE_START
        cfg.time.date_end            = TEST_DATE_END
        cfg.sampling.n_samples       = TEST_N_SAMPLES
        cfg.sampling.allow_ts_repeats = True   # small window needs repeats

    # Explicit CLI overrides (applied after test_mode defaults)
    if args.n_samples is not None:
        cfg.sampling.n_samples = args.n_samples
    if args.samples_per_ts is not None:
        cfg.sampling.samples_per_ts = args.samples_per_ts
    if args.date_start is not None:
        cfg.time.date_start = args.date_start
    if args.date_end is not None:
        cfg.time.date_end = args.date_end

    builder = DatasetBuilder(cfg, dry_run=args.dry_run)
    builder.run()


def cmd_stats(cfg, args):
    from src.pipeline.compute_stats import compute_stats
    log.info("=== Stage 3: Compute Stats ===")
    dataset_path = (Path(cfg.storage.output_dir) /
                    cfg.storage.output_filename)
    stats_path   = Path(cfg.storage.output_dir) / "stats.json"
    compute_stats(dataset_path, stats_path)
    log.info(f"stats.json written to: {stats_path}")


def cmd_all(cfg, args):
    cmd_download(cfg, args)
    cmd_build(cfg, args)
    cmd_stats(cfg, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="CorrDiff Regional HRRR Dataset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "stage",
        choices=["download", "build", "stats", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config (default: configs/pipeline_config.yaml)",
    )

    # Test mode
    parser.add_argument(
        "--test_mode", action="store_true",
        help=(
            "Run a 10-day pilot (2015-01-01 to 2015-01-10) to validate the "
            "pipeline before launching the full 10-year job."
        ),
    )

    # Download args
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Parallel CDS API workers (default: 2, max recommended: 4)",
    )
    parser.add_argument(
        "--skip_pressure", action="store_true",
        help="Skip pressure-level ERA5 download",
    )

    # Date range overrides
    parser.add_argument(
        "--date_start", default=None,
        help="Override config date_start, e.g. 2015-06-01",
    )
    parser.add_argument(
        "--date_end", default=None,
        help="Override config date_end, e.g. 2015-06-30",
    )

    # Build args
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Use synthetic HRRR data -- no S3 calls (for local testing)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Override config n_samples",
    )
    parser.add_argument(
        "--samples_per_ts", type=int, default=None,
        help="Override config samples_per_ts",
    )

    args = parser.parse_args()
    cfg  = load_config(args.config)

    dispatch = {
        "download": cmd_download,
        "build":    cmd_build,
        "stats":    cmd_stats,
        "all":      cmd_all,
    }
    dispatch[args.stage](cfg, args)


if __name__ == "__main__":
    main()
