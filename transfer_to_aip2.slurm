#!/bin/bash
#SBATCH --job-name=transfer_aip2
#SBATCH --output=/lus/lfs1aip1/home/a5l/shuqing.a5l/transfer_aip2_%j.log
#SBATCH --error=/lus/lfs1aip1/home/a5l/shuqing.a5l/transfer_aip2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288

# Transfer script: aip1 -> aip2 (SLURM version)
# NOTE: This may fail if compute nodes can't access external network

SOURCE="/lus/lfs1aip1/home/a5l/shuqing.a5l/"
DEST="a5l.aip2.isambard:/lus/lfs1aip2/home/a5l/shuqing.a5l/"

echo "=============================================="
echo "Transfer Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Source: $SOURCE"
echo "Dest:   $DEST"
echo "=============================================="

# Test connection first
echo "Testing connection to aip2..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=30 a5l.aip2.isambard "echo 'Connection OK'" 2>&1; then
    echo "ERROR: Cannot connect to aip2 from compute node."
    echo "Compute nodes may not have external network access."
    echo "Please run transfer_to_aip2.sh from a LOGIN NODE instead."
    exit 1
fi

echo "Connection successful! Starting transfer..."
echo ""

# Transfer with exclusions (recommended)
rsync -avz --progress \
    --exclude '.cache/' \
    --exclude '.triton/' \
    --exclude '.triton_aarch64/' \
    --exclude '.triton_build_nano/' \
    --exclude '.pip-tmp/' \
    --exclude '.tmp-build/' \
    --exclude '.nv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'slurm-*.out' \
    --exclude '.vscode-server/' \
    --exclude '.cursor-server/' \
    "$SOURCE" "$DEST"

STATUS=$?

echo ""
echo "=============================================="
echo "Transfer Job Finished: $(date)"
echo "Exit Status: $STATUS"
echo "=============================================="

exit $STATUS
