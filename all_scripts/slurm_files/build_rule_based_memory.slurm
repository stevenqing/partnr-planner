#!/bin/bash
#SBATCH --job-name=rule_memory
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rule_memory_%j.log
#SBATCH --error=logs/rule_memory_%j.log

# Build rule-based hierarchical skill memory
# This is faster than LLM-based and better than memento
#
# Key improvements over memento:
# 1. Skill-focused (not just user preferences) - captures reusable action patterns
# 2. Cooperation patterns (6 types) - detects coordination between agents
# 3. Theory of Mind - models partner beliefs and intentions
# 4. Hierarchical structure - L_ind (individual) + L_coop (cooperation) skills
# 5. Better abstraction - generalizes skills across similar objects
# 6. Action sequences - preserves full action sequences for retrieval

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate habitat-llm

BASE_DIR="/lus/lfs1aip1/home/a5l/shuqing.a5l/partnr-planner"
cd "$BASE_DIR/methods/skill-based(our_method)"

OUTPUT_BASE="$BASE_DIR/data/hierarchical_skill_memory_rule"

echo ""
echo "Building Enhanced Rule-Based Hierarchical Skill Memory..."
echo "Output base: $OUTPUT_BASE"
echo ""

# Build memory for each task type separately
echo "=========================================="
echo "Building: hierarchical_rerange_only"
echo "=========================================="
python build_rule_based_memory.py \
    --results-dirs "$BASE_DIR/heuristic_dataset/2025-12-30_17-02-27-rerange_only.json" \
    --output-dir "$OUTPUT_BASE/hierarchical_rerange_only"

echo ""
echo "=========================================="
echo "Building: hierarchical_heterogeneous_rerange"
echo "=========================================="
python build_rule_based_memory.py \
    --results-dirs "$BASE_DIR/heuristic_dataset/2025-12-30_21-10-23-heterogeneous+rerange.json" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_rerange"

echo ""
echo "=========================================="
echo "Building: hierarchical_spatial_only"
echo "=========================================="
python build_rule_based_memory.py \
    --results-dirs "$BASE_DIR/heuristic_dataset/2025-12-30_23-42-08-spatial_only.json" \
    --output-dir "$OUTPUT_BASE/hierarchical_spatial_only"

echo ""
echo "=========================================="
echo "Building: hierarchical_heterogeneous_temporal"
echo "=========================================="
python build_rule_based_memory.py \
    --results-dirs "$BASE_DIR/heuristic_dataset/2025-12-30_23-43-27-heterogeneous+temporal.json" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_temporal"

echo ""
echo "=========================================="
echo "Building: hierarchical_heterogeneous_rerange_spatial_temporal"
echo "=========================================="
python build_rule_based_memory.py \
    --results-dirs "$BASE_DIR/heuristic_dataset/2025-12-30_23-44-10-heterogeneous+rerange+spatial+temporal.json" \
    --output-dir "$OUTPUT_BASE/hierarchical_heterogeneous_rerange_spatial_temporal"

echo ""
echo "=========================================="
echo "All memories built successfully!"
echo "=========================================="
echo ""
echo "Summary of built memories:"
for dir in "$OUTPUT_BASE"/hierarchical_*; do
    if [ -f "$dir/memory_summary.json" ]; then
        echo "  $(basename $dir):"
        python3 -c "import json; d=json.load(open('$dir/memory_summary.json')); print(f'    ind: {d.get(\"num_individual_skills\",0)}, coop: {d.get(\"num_cooperation_skills\",0)}, instances: {d.get(\"num_individual_instances\",0)+d.get(\"num_cooperation_instances\",0)}')"
    fi
done

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
