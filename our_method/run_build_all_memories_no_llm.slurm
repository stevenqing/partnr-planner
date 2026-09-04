#!/bin/bash
#SBATCH --job-name=build_skill_memory_fast
#SBATCH --output=/home/a5l/shuqing.a5l/partnr-planner/logs/build_memory_fast_%j.out
#SBATCH --error=/home/a5l/shuqing.a5l/partnr-planner/logs/build_memory_fast_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=workq

# =============================================================================
# SLURM Script: Build Hierarchical Skill Memory (Rule-Based, No LLM)
#
# This is a faster version that uses rule-based extraction without LLM.
# No GPU required.
#
# Usage:
#   sbatch run_build_all_memories_no_llm.slurm
# =============================================================================

set -e

echo "=== Build Hierarchical Skill Memory (Fast/No LLM) Started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Configuration
PROJECT_DIR="/home/a5l/shuqing.a5l/partnr-planner"
OUTPUTS_DIR="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner/outputs/habitat_llm"
DATA_DIR="${PROJECT_DIR}/data"
LOG_DIR="${PROJECT_DIR}/logs"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${DATA_DIR}/hierarchical_skill_memory"
mkdir -p "${DATA_DIR}/rag_datasets"

cd "${PROJECT_DIR}"

# Activate conda environment
source ~/.bashrc
conda activate habitat-llm 2>/dev/null || echo "Using current environment"

# =============================================================================
# Define Results Directories
# =============================================================================

RESULTS_DIRS=(
    "2025-12-30_17-02-27-rerange_only.json/results"
    "2025-12-30_21-10-23-heterogeneous+rerange.json/results"
    "2025-12-30_23-44-10-heterogeneous+rerange+spatial+temporal.json/results"
    "2025-12-30_23-42-08-spatial_only.json/results"
    "2025-12-30_23-43-27-heterogeneous+temporal.json/results"
)

# =============================================================================
# Process Each Results Directory
# =============================================================================

echo ""
echo "=== Processing Heuristic Results (Rule-Based Extraction) ==="
echo ""

PROCESSED=0
FAILED=0

for rel_path in "${RESULTS_DIRS[@]}"; do
    RESULTS_DIR="${OUTPUTS_DIR}/${rel_path}"
    DATASET_NAME=$(echo "$rel_path" | sed 's|/results||' | sed 's|\.json||' | sed 's|^[0-9_-]*||')

    echo "Processing: $DATASET_NAME"

    if [ ! -d "$RESULTS_DIR" ] || [ ! -f "${RESULTS_DIR}/episode_result_log.csv" ]; then
        echo "  SKIPPED: Directory or episode log not found"
        continue
    fi

    EPISODE_COUNT=$(($(wc -l < "${RESULTS_DIR}/episode_result_log.csv") - 1))
    echo "  Episodes: $EPISODE_COUNT"

    if [ $EPISODE_COUNT -lt 1 ]; then
        echo "  SKIPPED: No episodes"
        continue
    fi

    MEMORY_OUTPUT="${DATA_DIR}/hierarchical_skill_memory/${DATASET_NAME}"
    RAG_OUTPUT="${DATA_DIR}/rag_datasets/${DATASET_NAME}_rag"

    if python our_method/run_build_memory.py \
        --results-dir "${RESULTS_DIR}" \
        --memory-output-dir "${MEMORY_OUTPUT}" \
        --rag-output-dir "${RAG_OUTPUT}" \
        2>&1 | tee "${LOG_DIR}/build_memory_${DATASET_NAME}_${SLURM_JOB_ID}.log"; then
        echo "  SUCCESS"
        PROCESSED=$((PROCESSED + 1))
    else
        echo "  FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Processed: $PROCESSED directories"
echo "Failed: $FAILED directories"
echo ""
ls -la "${DATA_DIR}/hierarchical_skill_memory/" 2>/dev/null
echo ""
echo "=== Job Completed at $(date) ==="
