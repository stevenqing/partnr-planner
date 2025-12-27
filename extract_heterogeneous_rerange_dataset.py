#!/usr/bin/env python3
"""
Extract heterogeneous+rerange tasks with fully compatible subtasks.
"""

import argparse
import gzip
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def clean_instruction(instruction):
    """Clean instruction text"""
    return instruction.strip().strip('"').strip()


def extract_subtasks(instruction):
    """Extract subtasks from instruction"""
    instruction.lower()

    # Split instruction into subtasks
    subtasks = []

    # Split by periods
    sentences = re.split(r"[.!?]+", instruction)
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            subtasks.append(sentence)

    # Split by commas (if contains multiple actions)
    if len(subtasks) == 1:
        comma_parts = re.split(
            r",\s*(?=(?:and|or|then|also|first|second|finally))", subtasks[0]
        )
        if len(comma_parts) > 1:
            subtasks = [part.strip() for part in comma_parts if part.strip()]

    return subtasks


def is_exact_match(task1, task2, threshold=0.9):
    """Check if two tasks match exactly"""
    # Clean task text
    clean1 = clean_instruction(task1).lower()
    clean2 = clean_instruction(task2).lower()

    # Calculate similarity
    similarity = SequenceMatcher(None, clean1, clean2).ratio()

    # Check keyword matching
    key_words1 = set(re.findall(r"\b\w+\b", clean1))
    key_words2 = set(re.findall(r"\b\w+\b", clean2))

    # Calculate keyword overlap
    if key_words1 and key_words2:
        overlap = len(key_words1.intersection(key_words2)) / len(
            key_words1.union(key_words2)
        )
    else:
        overlap = 0

    # Combined judgment
    return similarity >= threshold or overlap >= 0.7


def load_reference_datasets():
    """Load reference datasets for comparison"""
    reference_instructions = {"rerange": [], "heterogeneous": []}

    # Try to load reference datasets
    try:
        with gzip.open(
            "task_classification_datasets/rerange_only.json.gz", "rt", encoding="utf-8"
        ) as f:
            rerange_data = json.load(f)
            for episode in rerange_data["episodes"]:
                if "instruction" in episode:
                    reference_instructions["rerange"].append(episode["instruction"])
    except FileNotFoundError:
        print("Warning: rerange_only.json.gz not found")

    try:
        with gzip.open(
            "task_classification_datasets/heterogeneous_only.json.gz",
            "rt",
            encoding="utf-8",
        ) as f:
            heterogeneous_data = json.load(f)
            for episode in heterogeneous_data["episodes"]:
                if "instruction" in episode:
                    reference_instructions["heterogeneous"].append(
                        episode["instruction"]
                    )
    except FileNotFoundError:
        print("Warning: heterogeneous_only.json.gz not found")

    return reference_instructions


def extract_dataset(
    input_file: str, output_file: str = None, format_type: str = "json"
):
    """
    Extract heterogeneous+rerange tasks with fully compatible subtasks.

    Args:
        input_file: Path to the compressed JSON dataset file
        output_file: Output file path (optional)
        format_type: Output format ('json', 'csv', 'txt')
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load the compressed dataset
    print(f"Loading dataset from {input_file}...")
    with gzip.open(input_path, "rt") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    print(f"Found {len(episodes)} episodes in the dataset")

    # Load reference datasets
    reference_instructions = load_reference_datasets()
    print("Loaded reference instructions:")
    print(f"  Rerange: {len(reference_instructions['rerange'])}")
    print(f"  Heterogeneous: {len(reference_instructions['heterogeneous'])}")

    # Find tasks with fully compatible subtasks
    matched_task_indices = []

    print("\nSearching for tasks with fully compatible subtasks...")

    for i, episode in enumerate(episodes):
        instruction = episode.get("instruction", "")
        subtasks = extract_subtasks(instruction)

        has_match = False

        # Check each subtask
        for subtask in subtasks:
            # Check if matches rerange tasks
            for ref_instruction in reference_instructions["rerange"]:
                if is_exact_match(subtask, ref_instruction):
                    has_match = True
                    break

            if has_match:
                break

            # Check if matches heterogeneous tasks
            for ref_instruction in reference_instructions["heterogeneous"]:
                if is_exact_match(subtask, ref_instruction):
                    has_match = True
                    break

            if has_match:
                break

        if has_match:
            matched_task_indices.append(i)

    print(f"Found {len(matched_task_indices)} tasks with fully compatible subtasks")

    # Extract matched tasks
    matched_episodes = []
    for idx in matched_task_indices:
        matched_episodes.append(episodes[idx])

    # Create new data structure
    matched_data = {"config": data.get("config"), "episodes": matched_episodes}

    print(f"Extracted {len(matched_episodes)} episodes with compatible subtasks")

    # Generate output filename if not provided
    if output_file is None:
        if format_type == "json":
            output_file = "task_classification_datasets/rerange+heterogeneous.json.gz"
        else:
            output_file = f"extracted_heterogeneous_rerange.{format_type}"

    # Save extracted data
    output_path = Path(output_file)
    print(f"Saving extracted data to {output_file}...")

    if format_type == "json":
        # Save matched data in compressed format
        if output_file.endswith(".gz"):
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                json.dump(matched_data, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(matched_data, f, indent=2, ensure_ascii=False)

    elif format_type == "csv":
        import csv

        with open(output_path, "w", newline="") as f:
            if matched_episodes:
                # Extract first episode keys for CSV header
                first_episode = matched_episodes[0]
                writer = csv.DictWriter(f, fieldnames=first_episode.keys())
                writer.writeheader()
                for episode in matched_episodes:
                    # Convert complex data structures to strings for CSV
                    csv_episode = {}
                    for key, value in episode.items():
                        if isinstance(value, (list, dict)):
                            csv_episode[key] = json.dumps(value)
                        else:
                            csv_episode[key] = value
                    writer.writerow(csv_episode)

    elif format_type == "txt":
        with open(output_path, "w") as f:
            f.write("Heterogeneous+Rerange Dataset with Compatible Subtasks\n")
            f.write(f"Total Episodes: {len(matched_episodes)}\n")
            f.write(f"Original Episodes: {len(episodes)}\n")
            f.write(
                f"Extraction Ratio: {len(matched_episodes)/len(episodes)*100:.1f}%\n\n"
            )

            for i, episode in enumerate(matched_episodes):
                f.write(f"Episode {i+1}:\n")
                f.write(f"  ID: {episode.get('episode_id')}\n")
                f.write(f"  Scene: {episode.get('scene_id')}\n")
                f.write(f"  Instruction: {episode.get('instruction')}\n")
                f.write(f"  Targets: {episode.get('targets')}\n")
                f.write(f"  Goal Receptacles: {episode.get('goal_receptacles')}\n")
                f.write("  ---\n")

    # Display statistics
    print("\nStatistics:")
    print(f"  Original episodes: {len(episodes)}")
    print(f"  Episodes with compatible subtasks: {len(matched_episodes)}")
    print(f"  Extraction ratio: {len(matched_episodes)/len(episodes)*100:.1f}%")

    # Show first few matched tasks as examples
    if matched_episodes:
        print("\nFirst 3 matched task examples:")
        for i, episode in enumerate(matched_episodes[:3]):
            print(f"  {i+1}. {episode.get('instruction')}")

    print(
        f"Extraction complete! Saved {len(matched_episodes)} episodes to {output_file}"
    )
    return matched_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract heterogeneous+rerange dataset"
    )
    parser.add_argument("input_file", help="Input compressed JSON file path")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "csv", "txt"],
        default="json",
        help="Output format",
    )

    args = parser.parse_args()

    try:
        extract_dataset(args.input_file, args.output, args.format)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
