#!/bin/bash

# Master script to create all MEMENTO memory datasets
# Run this after run_all_heuristics.sh completes

echo "=========================================="
echo "Creating all MEMENTO memory datasets"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /home/a5l/shuqing.a5l/partnr-planner

# Create base directory
mkdir -p memory_from_trajectories_success

echo ""
echo "[1/5] Creating Rearrange MEMENTO memory..."
bash "$SCRIPT_DIR/create_memento_rerange.sh"

echo ""
echo "[2/5] Creating Heterogeneous MEMENTO memory..."
bash "$SCRIPT_DIR/create_memento_heterogeneous.sh"

echo ""
echo "[3/5] Creating Spatial MEMENTO memory..."
bash "$SCRIPT_DIR/create_memento_spatial.sh"

echo ""
echo "[4/5] Creating Temporal MEMENTO memory..."
bash "$SCRIPT_DIR/create_memento_temporal.sh"

echo ""
echo "[5/5] Creating Combined MEMENTO memory..."
bash "$SCRIPT_DIR/create_memento_all.sh"

echo ""
echo "=========================================="
echo "All MEMENTO memory datasets created!"
echo "=========================================="
echo ""
echo "Created directories:"
ls -la memory_from_trajectories_success/
