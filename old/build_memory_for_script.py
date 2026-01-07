#!/usr/bin/env python3

"""
Build memory directory for use with your planner script.
This creates the memory structure that your script expects.
"""

import os
import shutil
import sys

# Add the project to Python path
sys.path.insert(0, "/home/shuqing/partnr-planner")

from habitat_llm.utils import build_memory, create_dataset_from_traces


def create_memory_from_rag_dataset():
    """Create memory directory from your existing RAG dataset."""

    # Your existing RAG dataset path
    rag_dataset_path = "data/rag_datasets/rerange_only_organized_by_skills_enhanced"

    if not os.path.exists(rag_dataset_path):
        print(f"❌ RAG dataset not found: {rag_dataset_path}")
        return None

    print(f"🔍 Found RAG dataset at: {rag_dataset_path}")

    # Create dataset from your existing traces
    print("📝 Creating dataset from trace files...")
    dataset = create_dataset_from_traces(rag_dataset_path)
    print(f"✅ Created dataset with {len(dataset)} episodes")

    # Create temporary structure for build_memory (needs proper experiment/run structure)
    temp_output = "outputs/temp_experiment/temp_run/results"
    os.makedirs(temp_output, exist_ok=True)

    # Copy your RAG dataset to temp location
    dataset_name = "react_trajectories"
    temp_dataset_path = os.path.join(temp_output, dataset_name)

    if os.path.exists(temp_dataset_path):
        shutil.rmtree(temp_dataset_path)

    print("📁 Setting up temporary structure...")
    shutil.copytree(rag_dataset_path, temp_dataset_path)

    # Create episode result log (assume all successful)
    result_log_path = os.path.join(temp_output, "episode_result_log.csv")
    with open(result_log_path, "w") as f:
        f.write("episode_id,success\n")
        for episode in dataset.episodes:
            f.write(f"{episode.episode_id},1.0\n")

    try:
        # Build memory with a specific experiment/run name
        memory_path = build_memory(
            output_path=temp_output,
            data_path="dataset.json",  # Not used since we provide dataset directly
            dataset=dataset,
            memory_base_dir="data/memory",  # This creates data/memory/ directory
            filter_successful_only=True,
        )

        print("✅ Memory built successfully!")
        print(f"📍 Memory location: {memory_path}")

        # Clean up temporary files
        shutil.rmtree("outputs")
        print("🧹 Cleaned up temporary files")

        return memory_path

    except Exception as e:
        print(f"❌ Failed to build memory: {e}")
        import traceback

        traceback.print_exc()

        # Clean up on failure
        if os.path.exists("outputs"):
            shutil.rmtree("outputs")
        return None


def main():
    """Build memory directory for the planner script."""
    print("🚀 Building Memory Directory for Planner Script")
    print("=" * 60)

    # Create memory directory
    memory_path = create_memory_from_rag_dataset()

    if memory_path:
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

                print("\n🎯 Complete script parameters:")
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
                    f"   evaluation.agents.agent_1.planner.plan_config.memory_path={experiment_name}/{run_name} \\"
                )

        # Show the created structure
        if os.path.exists("data/memory"):
            print("\n📂 Created memory structure:")
            for root, dirs, files in os.walk("data/memory"):
                level = root.replace("data/memory", "").count(os.sep)
                indent = "  " * level
                print(f"{indent}{os.path.basename(root)}/")
                if level < 2:  # Only show first 2 levels to avoid clutter
                    subindent = "  " * (level + 1)
                    for d in dirs[:3]:  # Show first 3 subdirs
                        print(f"{subindent}{d}/")
                    if len(dirs) > 3:
                        print(f"{subindent}... and {len(dirs)-3} more directories")
                    for f in files[:3]:  # Show first 3 files
                        print(f"{subindent}{f}")
                    if len(files) > 3:
                        print(f"{subindent}... and {len(files)-3} more files")

    else:
        print("\n❌ Failed to create memory directory")
        print("   Check the errors above and try again.")


if __name__ == "__main__":
    main()
