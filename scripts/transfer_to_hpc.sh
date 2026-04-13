#!/usr/bin/env bash
# -------------------------------------------------------
# Transfer regional_hrrr_train_dataset.nc to Yale Bouchet HPC
# -------------------------------------------------------
# Usage:
#   bash scripts/transfer_to_hpc.sh
#
# Requires: SSH key-based auth set up for Bouchet
# -------------------------------------------------------

set -e

HPC_USER="sr2723"
HPC_HOST="login1.bouchet.ycrc.yale.edu"
HPC_DEST="~/scratch_pi_ey239/sr2723/corrdiff/data/"

# Read output path from config
OUTPUT_DIR=$(python -c "
import yaml
c = yaml.safe_load(open('configs/pipeline_config.yaml'))
print(c['storage']['output_dir'])
")
OUTPUT_FILE=$(python -c "
import yaml
c = yaml.safe_load(open('configs/pipeline_config.yaml'))
print(c['storage']['output_filename'])
")
STATS_FILE="${OUTPUT_DIR}/stats.json"
NC_FILE="${OUTPUT_DIR}/${OUTPUT_FILE}"

echo "=== Transfer to Bouchet HPC ==="
echo "  Source NC   : ${NC_FILE}"
echo "  Source stats: ${STATS_FILE}"
echo "  Destination : ${HPC_USER}@${HPC_HOST}:${HPC_DEST}"
echo ""

if [ ! -f "$NC_FILE" ]; then
    echo "ERROR: ${NC_FILE} not found. Run the pipeline first."
    exit 1
fi

# Transfer NetCDF
rsync -avh --progress \
    "$NC_FILE" \
    "${HPC_USER}@${HPC_HOST}:${HPC_DEST}"

# Transfer stats.json if it exists
if [ -f "$STATS_FILE" ]; then
    rsync -avh --progress \
        "$STATS_FILE" \
        "${HPC_USER}@${HPC_HOST}:${HPC_DEST}"
fi

echo ""
echo "Transfer complete."
echo "On HPC, symlink the file into the CorrDiff data directory:"
echo ""
echo "  ssh ${HPC_USER}@${HPC_HOST}"
echo "  cd ~/physicsnemo/examples/weather/corrdiff"
echo "  ln -s ${HPC_DEST}${OUTPUT_FILE} data/regional_hrrr/"
echo "  ln -s ${HPC_DEST}stats.json     data/regional_hrrr/"
