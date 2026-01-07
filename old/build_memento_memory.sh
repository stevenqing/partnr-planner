#!/bin/bash
# Build MEMENTO memory for all heuristic datasets

set -e

cd /home/a5l/shuqing.a5l/partnr-planner

# Activate conda environment
if [ -f /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
    conda activate habitat-llm
fi

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/methods"

# Output base directory
OUTPUT_BASE="/home/a5l/shuqing.a5l/partnr-planner/data/memory_memento_dataset"
mkdir -p "$OUTPUT_BASE"

# Heuristic datasets
declare -A DATASETS=(
    ["rerange_only"]="/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_17-02-27-rerange_only.json/results"
    ["heterogeneous_rerange"]="/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_21-10-23-heterogeneous+rerange.json/results"
    ["spatial_only"]="/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_23-42-08-spatial_only.json/results"
    ["heterogeneous_temporal"]="/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_23-43-27-heterogeneous+temporal.json/results"
    ["all_combined"]="/home/a5l/shuqing.a5l/partnr-planner/heuristic_dataset/2025-12-30_23-44-10-heterogeneous+rerange+spatial+temporal.json/results"
)

echo "=========================================="
echo "MEMENTO: Building User Profile Memory"
echo "=========================================="
echo "Start time: $(date)"

for name in "${!DATASETS[@]}"; do
    data_dir="${DATASETS[$name]}"
    output_dir="$OUTPUT_BASE/$name"

    echo ""
    echo "Processing: $name"
    echo "  Data: $data_dir"
    echo "  Output: $output_dir"

    if [ -d "$data_dir" ]; then
        python methods/MEMENTO/build_memory.py \
            --data_dir "$data_dir" \
            --output_dir "$output_dir" \
            --user_id "user_0"
        echo "  Completed: $name"
    else
        echo "  WARNING: Data directory not found: $data_dir"
    fi
done

echo ""
echo "=========================================="
echo "All memory builds complete!"
echo "End time: $(date)"
echo "=========================================="
