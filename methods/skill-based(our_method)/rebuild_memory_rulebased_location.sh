#!/bin/bash
# Rebuild all hierarchical skill memories with enhanced location extraction
# Uses rule-based extraction (fast) with rooms and object_locations

set -e

BASE_DIR="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner"
HEURISTIC_DIR="$BASE_DIR/heuristic_dataset"
OUTPUT_BASE="$BASE_DIR/data/hierarchical_skill_memory_rule_location"
SCRIPT_DIR="$BASE_DIR/our_method"

cd "$SCRIPT_DIR"

echo "=========================================="
echo "Rebuilding Hierarchical Skill Memories"
echo "Enhanced Location Extraction (rooms, object_locations)"
echo "Mode: Rule-based extraction (fast)"
echo "=========================================="

# Rerange Only (578 episodes)
echo ""
echo "=== Building hierarchical_rerange_only (578 episodes) ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_17-02-27-rerange_only.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_rerange_only" \
    --include-failed

# Heterogeneous + Rerange (196 episodes)
echo ""
echo "=== Building hierarchical_heterogeneous_rerange (196 episodes) ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_21-10-23-heterogeneous+rerange.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_rerange" \
    --include-failed

# Spatial Only (4 episodes)
echo ""
echo "=== Building hierarchical_spatial_only (4 episodes) ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_23-42-08-spatial_only.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_spatial_only" \
    --include-failed

# Heterogeneous + Temporal (2 episodes)
echo ""
echo "=== Building hierarchical_heterogeneous_temporal (2 episodes) ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_23-43-27-heterogeneous+temporal.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_temporal" \
    --include-failed

# Heterogeneous + Rerange + Spatial + Temporal (4 episodes)
echo ""
echo "=== Building hierarchical_heterogeneous_rerange_spatial_temporal (4 episodes) ==="
python build_hierarchical_skill_memory.py \
    --results-dir "$HEURISTIC_DIR/2025-12-30_23-44-10-heterogeneous+rerange+spatial+temporal.json/results" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_rerange_spatial_temporal" \
    --include-failed

echo ""
echo "=========================================="
echo "All memories rebuilt successfully!"
echo "=========================================="
