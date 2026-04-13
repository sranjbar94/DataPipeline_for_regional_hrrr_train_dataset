"""
Compute stats.json — per-variable mean and std for CorrDiff normalization.
Uses Welford's online algorithm so the full dataset never needs to fit in RAM.
"""

from __future__ import annotations
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

from src.utils.logger import get_logger

log = get_logger("compute_stats")

CHUNK = 2000   # samples per batch


def compute_stats(dataset_path: str | Path, output_path: str | Path):
    dataset_path = Path(dataset_path)
    output_path  = Path(output_path)

    log.info(f"Computing stats: {dataset_path}")
    ds    = nc.Dataset(str(dataset_path))
    stats = {}

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

    ds.close()

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"Stats written: {output_path}")
    return stats
