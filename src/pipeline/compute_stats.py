# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""
Compute stats.json — per-variable mean and std for CorrDiff normalization.
Uses Welford's online algorithm so the full dataset never needs to fit in RAM.
Includes invariant section with CorrDiff-standard normalization conventions.
"""

from __future__ import annotations
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

from src.utils.logger import get_logger

log = get_logger("compute_stats")

CHUNK = 2000   # samples per batch

# CorrDiff convention: lat/lon normalized by deg-to-rad factor
_DEG2RAD = float(np.pi / 180.0)


def compute_stats(dataset_path: str | Path, output_path: str | Path):
    dataset_path = Path(dataset_path)
    output_path  = Path(output_path)

    log.info(f"Computing stats: {dataset_path}")
    ds    = nc.Dataset(str(dataset_path))
    stats = {}

    # ------------------------------------------------------------------
    # input & output groups — Welford online mean/std over all samples
    # ------------------------------------------------------------------
    for grp_name in ["input", "output"]:
        if grp_name not in ds.groups:
            continue
        grp = ds.groups[grp_name]
        stats[grp_name] = {}
        log.info(f"  Group [{grp_name}] — {len(grp.variables)} variables")

        for vname, var in grp.variables.items():
            n_samples = var.shape[0]
            count, mean, M2 = 0, 0.0, 0.0

            for start in range(0, n_samples, CHUNK):
                chunk = var[start: start + CHUNK]
                if hasattr(chunk, "filled"):
                    chunk = chunk.filled(np.nan)
                flat = chunk.flatten()
                flat = flat[np.isfinite(flat)]

                for x in flat:
                    count += 1
                    delta  = x - mean
                    mean  += delta / count
                    M2    += delta * (x - mean)

            std = float(np.sqrt(M2 / max(count - 1, 1)))
            stats[grp_name][vname] = {"mean": float(mean), "std": std}
            log.info(f"    {vname:<20}  mean={mean:>12.4f}  std={std:>10.4f}")

    # ------------------------------------------------------------------
    # invariant group — CorrDiff normalization conventions
    # ------------------------------------------------------------------
    if "invariant" in ds.groups:
        grp_inv = ds.groups["invariant"]
        stats["invariant"] = {}
        log.info(f"  Group [invariant] — {len(grp_inv.variables)} variables")

        for vname, var in grp_inv.variables.items():
            data = var[:]
            if hasattr(data, "filled"):
                data = data.filled(np.nan)
            flat = data.flatten()
            flat = flat[np.isfinite(flat)]

            if vname in ("latitude", "longitude"):
                # CorrDiff convention: normalize by deg→rad factor
                vmean = 0.0
                vstd  = _DEG2RAD
            elif vname == "lsm_mean":
                # Binary mask: no shift, unit scale
                vmean = 0.0
                vstd  = 1.0
            else:
                # elev_mean and any others: compute from data
                vmean = float(np.mean(flat)) if len(flat) > 0 else 0.0
                vstd  = float(np.std(flat, ddof=1)) if len(flat) > 1 else 1.0

            stats["invariant"][vname] = {"mean": vmean, "std": vstd}
            log.info(f"    {vname:<20}  mean={vmean:>12.4f}  std={vstd:>10.4f}")

    ds.close()

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"Stats written: {output_path}")
    return stats
