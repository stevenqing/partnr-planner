#!/bin/bash

# Create MEMENTO memory from Heterogeneous heuristic trajectories
# This creates memory_from_trajectories_success/heterogeneous_only/

echo "=== Creating MEMENTO memory for Heterogeneous tasks ==="

# Find the latest output directory for heterogeneous
TRAJ_DIR=$(ls -td outputs/habitat_llm/*heterogeneous+rerange*/results/* 2>/dev/null | head -1)

if [ -z "$TRAJ_DIR" ]; then
    echo "Error: No trajectory directory found for heterogeneous+rerange"
    echo "Please run run_heuristic_heterogeneous.sh first"
    exit 1
fi

echo "Using trajectory directory: $TRAJ_DIR"

python create_memento_memory.py "$TRAJ_DIR" \
    --output_dir memory_from_trajectories_success/heterogeneous_only \
    --filter_successful_only
