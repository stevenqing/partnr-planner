#!/bin/bash

# Create MEMENTO memory from all task types heuristic trajectories
# This creates memory_from_trajectories_success/ (combined)

echo "=== Creating combined MEMENTO memory for all task types ==="

# Find the latest output directory for all tasks
TRAJ_DIR=$(ls -td outputs/habitat_llm/*heterogeneous+rerange+spatial+temporal*/results/* 2>/dev/null | head -1)

if [ -z "$TRAJ_DIR" ]; then
    echo "Error: No trajectory directory found for heterogeneous+rerange+spatial+temporal"
    echo "Please run run_heuristic_all.sh first"
    exit 1
fi

echo "Using trajectory directory: $TRAJ_DIR"

python create_memento_memory.py "$TRAJ_DIR" \
    --output_dir memory_from_trajectories_success \
    --filter_successful_only
