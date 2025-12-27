#!/usr/bin/env python3

"""
Build memory directly from trajectory output directory.
"""

import os
import sys

# Add the project to Python path
sys.path.insert(0, "/home/shuqing/partnr-planner")

from habitat_llm.utils.build_memory import build_memory


def main():
    """Build memory from trajectory directory."""

    trajectory_path = "/home/shuqing/partnr-planner/outputs/habitat_llm/2025-09-06_15-50-48-rerange_only.json"

    print("🚀 Building Memory from Trajectory Data")
    print("=" * 60)
    print(f"📁 Source: {trajectory_path}")

    # The trajectory path structure is:
    # /outputs/habitat_llm/2025-09-06_15-50-48-rerange_only.json/results/rerange_only.json.gz/

    # For build_memory, we need:
    # output_path: path to results directory
    # data_path: path to dataset file

    output_path = os.path.join(trajectory_path, "results")
    data_path = "rerange_only.json.gz"  # This is the dataset name

    print(f"📂 Output path: {output_path}")
    print(f"📄 Dataset: {data_path}")

    # Check if the results directory exists
    if not os.path.exists(output_path):
        print(f"❌ Results directory not found: {output_path}")
        return

    try:
        # Build memory without dataset object (it will use default scene mapping)
        memory_path = build_memory(
            output_path=output_path,
            data_path=data_path,
            dataset=None,  # Let it create default scene mapping
            memory_base_dir="data/memory",
            filter_successful_only=False,  # Include all episodes since we want to build from available data
        )

        print("\n✅ Memory built successfully!")
        print(f"📍 Memory location: {memory_path}")

        # Extract the relative path parts for the script
        memory_parts = memory_path.split("/")

        if "memory" in memory_parts:
            memory_idx = memory_parts.index("memory")
            if memory_idx + 2 < len(memory_parts):
                experiment_name = memory_parts[memory_idx + 1]
                run_name = memory_parts[memory_idx + 2]

                print("\n✨ Memory is ready to use!")
                print("📁 Directory structure created:")
                print(f"   data/memory/{experiment_name}/{run_name}/")

                print("\n📋 Update your script parameters:")
                print("   rag_dataset_dir=[data/memory/]")
                print(f"   memory_path={experiment_name}/{run_name}")

                print("\n🎯 Use in planner configuration:")
                print(
                    "   evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir=[data/memory/] \\"
                )
                print(
                    "   evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir=[data/memory/] \\"
                )
                print(
                    f"   evaluation.agents.agent_0.planner.plan_config.memory_path={experiment_name}/{run_name} \\"
                )
                print(
                    f"   evaluation.agents.agent_1.planner.plan_config.memory_path={experiment_name}/{run_name}"
                )

    except Exception as e:
        print(f"❌ Failed to build memory: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
