#!/bin/bash

# Master script to run all heuristic trajectory generations
# Run this to generate trajectories for all task types

echo "=========================================="
echo "Running all heuristic trajectory generations"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /home/a5l/shuqing.a5l/partnr-planner

echo ""
echo "[1/5] Generating Rearrange trajectories..."
bash "$SCRIPT_DIR/run_heuristic_rerange.sh"

echo ""
echo "[2/5] Generating Heterogeneous trajectories..."
bash "$SCRIPT_DIR/run_heuristic_heterogeneous.sh"

echo ""
echo "[3/5] Generating Spatial trajectories..."
bash "$SCRIPT_DIR/run_heuristic_spatial.sh"

echo ""
echo "[4/5] Generating Temporal trajectories..."
bash "$SCRIPT_DIR/run_heuristic_temporal.sh"

echo ""
echo "[5/5] Generating All Combined trajectories..."
bash "$SCRIPT_DIR/run_heuristic_all.sh"

echo ""
echo "=========================================="
echo "All heuristic trajectory generations complete!"
echo "=========================================="
