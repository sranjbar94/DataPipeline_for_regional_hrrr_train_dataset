#!/usr/bin/env bash
# -------------------------------------------------------
# Smoke test — 50 samples, dry_run (no S3 / no ERA5 needed)
# -------------------------------------------------------
set -e

echo "=== CorrDiff Pipeline Smoke Test (dry_run) ==="
python run_pipeline.py build \
    --dry_run \
    --n_samples 50 \
    --samples_per_ts 5

echo ""
echo "=== Inspecting output NetCDF ==="
python - <<'EOF'
import netCDF4 as nc, numpy as np
from pathlib import Path
import json

cfg = __import__('yaml').safe_load(open('configs/pipeline_config.yaml'))
path = Path(cfg['storage']['output_dir']) / cfg['storage']['output_filename']

if not path.exists():
    print(f"ERROR: {path} not found"); exit(1)

ds = nc.Dataset(str(path))
print(f"File   : {path}")
print(f"Size   : {path.stat().st_size/1e6:.1f} MB  for {len(ds.dimensions['sample'])} samples")
print(f"Groups : {list(ds.groups.keys())}")
print("")
for grp in ['input','output']:
    g = ds.groups[grp]
    print(f"[{grp}]  {len(g.variables)} variables")
    for v, var in g.variables.items():
        a = var[:].flatten()
        print(f"  {v:<22} shape={var.shape}  "
              f"min={float(np.nanmin(a)):.3f}  max={float(np.nanmax(a)):.3f}")
ds.close()
proj = path.stat().st_size / 50 * 200_000 / 1e9
print(f"\nProjected full dataset (200k samples): ~{proj:.0f} GB")
print("\nSmoke test PASSED.")
EOF
