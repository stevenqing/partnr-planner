#!/bin/bash

# Create MEMENTO memory from Spatial heuristic trajectories
# This creates memory_from_trajectories_success/spatial_only/

echo "=== Creating MEMENTO memory for Spatial tasks ==="

# Find the latest output directory for spatial
TRAJ_DIR=$(ls -td outputs/habitat_llm/*spatial_only*/results/* 2>/dev/null | head -1)

if [ -z "$TRAJ_DIR" ]; then
    echo "Error: No trajectory directory found for spatial_only"
    echo "Please run run_heuristic_spatial.sh first"
    exit 1
fi

echo "Using trajectory directory: $TRAJ_DIR"

python create_memento_memory.py "$TRAJ_DIR" \
    --output_dir memory_from_trajectories_success/spatial_only \
    --filter_successful_only
