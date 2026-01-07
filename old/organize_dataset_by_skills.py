#!/usr/bin/env python3
import gzip
import json
import os
from collections import defaultdict


def extract_skill_categories(skill_text):
    """Extract skill categories from skill text based on keywords and patterns"""
    categories = set()
    skill_lower = skill_text.lower()

    # Object manipulation keywords
    if any(word in skill_lower for word in ["pick up", "grab", "hold", "lift", "take"]):
        categories.add("Object Manipulation")

    # Object placement keywords
    if any(
        word in skill_lower for word in ["place", "put", "drop", "set down", "position"]
    ):
        categories.add("Object Placement")

    # Navigation keywords
    if any(
        word in skill_lower
        for word in ["navigate", "move to", "go to", "walk to", "travel to"]
    ):
        categories.add("Navigation")

    # Container management keywords
    if any(
        word in skill_lower
        for word in ["container", "box", "drawer", "cabinet", "shelf"]
    ):
        categories.add("Container Management")

    # Multi-agent coordination keywords
    if any(
        word in skill_lower
        for word in ["coordinate", "cooperate", "wait for", "avoid collision"]
    ):
        categories.add("Multi-Agent Coordination")

    # Task planning keywords
    if any(
        word in skill_lower
        for word in ["first", "then", "after", "before", "sequence", "plan"]
    ):
        categories.add("Task Planning")

    # Room-specific organization - check for room names
    rooms = ["bedroom", "kitchen", "living room", "dining room", "bathroom", "hallway"]
    if any(room in skill_lower for room in rooms):
        categories.add("Room-Specific Organization")

    return categories if categories else {"General Task"}


def organize_dataset_by_skills():
    input_file = "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_with_skills/react_trajectories.json.gz"
    output_dir = "/home/shuqing/partnr-planner/data/rag_datasets/rerange_only_organized_by_skills"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    with gzip.open(input_file, "rt") as f:
        data = json.load(f)

    # Organize episodes by skill categories
    skill_based_datasets = defaultdict(list)
    task_type_datasets = defaultdict(list)
    complexity_datasets = defaultdict(list)

    print(f"Processing {len(data['episodes'])} episodes...")

    for episode in data["episodes"]:
        # Extract all skill categories from this episode
        episode_skill_categories = set()

        if "skills" in episode:
            for _agent_id, skill_text in episode["skills"].items():
                skill_cats = extract_skill_categories(skill_text)
                episode_skill_categories.update(skill_cats)

        # Add episode to each relevant skill category dataset
        for skill_cat in episode_skill_categories:
            skill_based_datasets[skill_cat].append(episode)

        # Also organize by task type and complexity
        task_type_datasets[episode["task_type"]].append(episode)
        complexity_datasets[episode["complexity"]].append(episode)

    # Create skill-based datasets
    print("\nCreating skill-based datasets...")
    for skill_category, episodes in skill_based_datasets.items():
        output_file = os.path.join(
            output_dir,
            f'skill_{skill_category.lower().replace(" ", "_").replace("-", "_")}.json.gz',
        )

        skill_dataset = {
            "metadata": {
                "skill_category": skill_category,
                "total_episodes": len(episodes),
                "source": "rerange_only_with_skills",
                "created_by": "organize_dataset_by_skills.py",
                "organization_type": "skill_based",
                "description": f"Episodes focused on {skill_category} skills",
            },
            "episodes": episodes,
        }

        with gzip.open(output_file, "wt") as f:
            json.dump(skill_dataset, f, indent=2)

        print(f"  {skill_category}: {len(episodes)} episodes -> {output_file}")

    # Create task-type-based datasets
    print("\nCreating task-type-based datasets...")
    task_type_dir = os.path.join(output_dir, "by_task_type")
    os.makedirs(task_type_dir, exist_ok=True)

    for task_type, episodes in task_type_datasets.items():
        output_file = os.path.join(
            task_type_dir, f'{task_type.lower().replace(" ", "_")}.json.gz'
        )

        task_dataset = {
            "metadata": {
                "task_type": task_type,
                "total_episodes": len(episodes),
                "source": "rerange_only_with_skills",
                "created_by": "organize_dataset_by_skills.py",
                "organization_type": "task_type_based",
                "description": f"Episodes for {task_type} tasks",
            },
            "episodes": episodes,
        }

        with gzip.open(output_file, "wt") as f:
            json.dump(task_dataset, f, indent=2)

        print(f"  {task_type}: {len(episodes)} episodes -> {output_file}")

    # Create complexity-based datasets
    print("\nCreating complexity-based datasets...")
    complexity_dir = os.path.join(output_dir, "by_complexity")
    os.makedirs(complexity_dir, exist_ok=True)

    for complexity, episodes in complexity_datasets.items():
        output_file = os.path.join(
            complexity_dir, f"{complexity.lower()}_complexity.json.gz"
        )

        complexity_dataset = {
            "metadata": {
                "complexity": complexity,
                "total_episodes": len(episodes),
                "source": "rerange_only_with_skills",
                "created_by": "organize_dataset_by_skills.py",
                "organization_type": "complexity_based",
                "description": f"{complexity} complexity episodes",
            },
            "episodes": episodes,
        }

        with gzip.open(output_file, "wt") as f:
            json.dump(complexity_dataset, f, indent=2)

        print(f"  {complexity} Complexity: {len(episodes)} episodes -> {output_file}")

    # Create summary report
    summary_file = os.path.join(output_dir, "organization_summary.json")
    summary = {
        "original_dataset": {
            "total_episodes": len(data["episodes"]),
            "source": input_file,
        },
        "skill_based_organization": {
            category: len(episodes)
            for category, episodes in skill_based_datasets.items()
        },
        "task_type_organization": {
            task_type: len(episodes)
            for task_type, episodes in task_type_datasets.items()
        },
        "complexity_organization": {
            complexity: len(episodes)
            for complexity, episodes in complexity_datasets.items()
        },
        "total_organized_files": len(skill_based_datasets)
        + len(task_type_datasets)
        + len(complexity_datasets),
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nOrganization complete!")
    print(f"Created {len(skill_based_datasets)} skill-based datasets")
    print(f"Created {len(task_type_datasets)} task-type datasets")
    print(f"Created {len(complexity_datasets)} complexity-based datasets")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    organize_dataset_by_skills()
