#!/bin/bash

# Create MEMENTO memory from Rearrange heuristic trajectories
# This creates memory_from_trajectories_success/rerange_only/

echo "=== Creating MEMENTO memory for Rearrange tasks ==="

# Find the latest output directory for rerange
TRAJ_DIR=$(ls -td outputs/habitat_llm/*rerange_only*/results/* 2>/dev/null | head -1)

if [ -z "$TRAJ_DIR" ]; then
    echo "Error: No trajectory directory found for rerange_only"
    echo "Please run run_heuristic_rerange.sh first"
    exit 1
fi

echo "Using trajectory directory: $TRAJ_DIR"

python create_memento_memory.py "$TRAJ_DIR" \
    --output_dir memory_from_trajectories_success/rerange_only \
    --filter_successful_only
