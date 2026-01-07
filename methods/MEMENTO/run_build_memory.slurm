#!/bin/bash
#SBATCH --job-name=memento_build
#SBATCH --output=memento_build_%j.out
#SBATCH --error=memento_build_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# MEMENTO: Build User Profile Memory
# This script builds the hierarchical knowledge graph-based memory
# from heuristic agent trajectories.

echo "=========================================="
echo "MEMENTO: Building User Profile Memory"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
cd /home/a5l/shuqing.a5l/partnr-planner

# Activate conda environment if needed
if [ -f /lus/home/NAT/hpc/a5l/shuqing.a5l/miniforge3/etc/profile.d/conda.sh ]; then
    source /lus/home/NAT/hpc/a5l/shuqing.a5l/miniforge3/etc/profile.d/conda.sh
    conda activate partnr
fi

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/methods"

# Configuration
DATA_DIR="${DATA_DIR:-data/trajectories/heuristic_results}"
OUTPUT_DIR="${OUTPUT_DIR:-data/memento_memory}"
USER_ID="${USER_ID:-user_0}"

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  User ID: $USER_ID"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run memory building
python methods/MEMENTO/build_memory.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --user_id "$USER_ID"

exit_code=$?

echo ""
echo "=========================================="
echo "Build complete!"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "=========================================="

exit $exit_code
