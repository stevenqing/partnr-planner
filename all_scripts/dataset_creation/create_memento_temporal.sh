#!/bin/bash

# Create MEMENTO memory from Temporal heuristic trajectories
# This creates memory_from_trajectories_success/temporal_only/

echo "=== Creating MEMENTO memory for Temporal tasks ==="

# Find the latest output directory for temporal
TRAJ_DIR=$(ls -td outputs/habitat_llm/*heterogeneous+temporal*/results/* 2>/dev/null | head -1)

if [ -z "$TRAJ_DIR" ]; then
    echo "Error: No trajectory directory found for heterogeneous+temporal"
    echo "Please run run_heuristic_temporal.sh first"
    exit 1
fi

echo "Using trajectory directory: $TRAJ_DIR"

python create_memento_memory.py "$TRAJ_DIR" \
    --output_dir memory_from_trajectories_success/temporal_only \
    --filter_successful_only
