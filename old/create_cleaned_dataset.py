#!/usr/bin/env python3
"""
Script to create a new dataset directory with cleaned trajectory files
(removes Agent actions, thoughts, and observations while keeping metadata)
"""

import os
import shutil
from pathlib import Path


def clean_trace_content(content):
    """Clean trajectory content from file content"""
    lines = content.split("\n")
    cleaned_lines = []

    for line in lines:
        # Stop at the first trajectory line
        if (
            line.startswith("Thought:")
            or line.startswith("Agent_0_Action:")
            or line.startswith("Agent_1_Action:")
            or line.startswith("Agent_0_observation:")
            or line.startswith("Agent_1_observation:")
        ):
            break
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).rstrip() + "\n"


def create_cleaned_dataset():
    """Create new dataset directory with cleaned files"""
    source_dir = Path(
        "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_with_skills"
    )
    target_dir = Path(
        "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_cleaned"
    )

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created target directory: {target_dir}")

    # Copy non-trajectory files directly
    for item in source_dir.iterdir():
        if item.name == "react_trajectories":
            continue  # Handle this separately
        elif item.is_file():
            shutil.copy2(item, target_dir / item.name)
            print(f"Copied file: {item.name}")

    # Handle the react_trajectories directory
    source_traces = source_dir / "react_trajectories"
    target_traces = target_dir / "react_trajectories"

    if source_traces.exists():
        # Recreate directory structure
        for root, _dirs, files in os.walk(source_traces):
            root_path = Path(root)
            relative_path = root_path.relative_to(source_traces)
            target_root = target_traces / relative_path
            target_root.mkdir(parents=True, exist_ok=True)

            # Process .txt files, copy others as-is
            for file in files:
                source_file = root_path / file
                target_file = target_root / file

                if file.endswith(".txt"):
                    # Clean trajectory content from .txt files
                    try:
                        with open(source_file, "r", encoding="utf-8") as f:
                            content = f.read()

                        cleaned_content = clean_trace_content(content)

                        with open(target_file, "w", encoding="utf-8") as f:
                            f.write(cleaned_content)

                    except Exception as e:
                        print(f"Error processing {source_file}: {e}")
                else:
                    # Copy other files as-is
                    shutil.copy2(source_file, target_file)

    # Count processed files
    txt_files = list(target_traces.rglob("*.txt")) if target_traces.exists() else []
    print("\nDataset creation completed!")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Processed {len(txt_files)} trace files")


if __name__ == "__main__":
    create_cleaned_dataset()
