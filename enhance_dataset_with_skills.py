#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
import re
import shutil
import subprocess

from tqdm import tqdm


def enhance_dataset_with_skills():
    """Enhance the rerange_only_converted dataset by adding skills using Llama 3.1 8B"""

    # Input and output paths
    input_dir = "data/rag_datasets/rerange_only_converted"
    output_dir = "data/rag_datasets/rerange_only_with_skills"

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/0", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/1", exist_ok=True)

    print("Enhancing rerange_only_converted dataset with skills using Llama 3.1 8B...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load existing metadata
    with gzip.open(
        f"{input_dir}/react_trajectories.json.gz", "rt", encoding="utf-8"
    ) as f:
        original_data = json.load(f)

    print(f"Found {len(original_data['episodes'])} episodes in original dataset")

    # Copy episode_result_log.csv unchanged
    shutil.copy2(
        f"{input_dir}/episode_result_log.csv", f"{output_dir}/episode_result_log.csv"
    )

    # Process episodes and add skills
    enhanced_episodes = []
    processed_count = 0

    # Create progress bar
    pbar = tqdm(original_data["episodes"], desc="Processing episodes", unit="episode")

    for episode in pbar:
        episode_id = episode["episode_id"]
        instruction = episode["instruction"]

        try:
            # Process trace files for both agents
            agent_skills = {}
            for agent_id in ["0", "1"]:
                trace_file = f"trace-episode_{episode_id}_0-{agent_id}.txt"
                input_trace_path = (
                    f"{input_dir}/react_trajectories/traces/{agent_id}/{trace_file}"
                )
                output_trace_path = (
                    f"{output_dir}/react_trajectories/traces/{agent_id}/{trace_file}"
                )

                if os.path.exists(input_trace_path):
                    # Read original trace
                    with open(input_trace_path, "r", encoding="utf-8") as f:
                        trace_content = f.read()

                    # Generate skills using Llama 3.1 8B
                    skill_summary = generate_skills_with_llama(
                        trace_content, agent_id, instruction
                    )
                    agent_skills[agent_id] = skill_summary

                    # Create enhanced content
                    enhanced_content = enhance_trace_with_skills(
                        trace_content, skill_summary, agent_id, instruction
                    )

                    # Save enhanced trace
                    with open(output_trace_path, "w", encoding="utf-8") as f:
                        f.write(enhanced_content)

            # Create enhanced episode info
            enhanced_episode = episode.copy()  # Preserve all original information

            # Add skill information
            enhanced_episode.update(
                {
                    "task_type": analyze_task_type(instruction),
                    "complexity": analyze_complexity(instruction),
                    "skills": agent_skills,
                    "skill_categories": extract_skill_categories(agent_skills),
                    "coordination_required": len(agent_skills) > 1,
                }
            )

            enhanced_episodes.append(enhanced_episode)
            processed_count += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "Processed": processed_count,
                    "Episode": episode_id,
                    "Success": f"{processed_count}/{len(original_data['episodes'])}",
                }
            )

        except Exception as e:
            pbar.write(f"Error processing episode {episode_id}: {e}")
            # Keep original episode if processing fails
            enhanced_episodes.append(episode)
            processed_count += 1

    # Create enhanced metadata
    enhanced_metadata = original_data["metadata"].copy()  # Preserve original metadata
    enhanced_metadata.update(
        {
            "enhanced_with_skills": True,
            "enhancement_date": "2025-08-28",
            "enhanced_by": "enhance_dataset_with_skills.py",
            "skill_model": "Llama 3.1 8B",
            "description": enhanced_metadata["description"]
            + " - Enhanced with Llama 3.1 8B skill extraction",
            "skill_statistics": analyze_skill_statistics(enhanced_episodes),
        }
    )

    # Create enhanced dataset
    enhanced_data = {"metadata": enhanced_metadata, "episodes": enhanced_episodes}

    # Save enhanced dataset
    output_json_path = f"{output_dir}/react_trajectories.json.gz"
    with gzip.open(output_json_path, "wt", encoding="utf-8") as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    print("\nDataset enhancement completed!")
    print(f"Successfully processed: {processed_count} episodes")
    print(f"Output directory: {output_dir}")
    print(f"Enhanced dataset file: {output_json_path}")

    return output_dir


def generate_skills_with_llama(trace_content, agent_id, instruction):
    """Generate specific, actionable skill summary using Llama 3.1 8B model"""

    # Extract agent actions and observations
    lines = trace_content.split("\n")
    agent_actions = []
    observations = []
    for line in lines:
        if line.startswith(f"Agent_{agent_id}_Action:"):
            action = line.replace(f"Agent_{agent_id}_Action:", "").strip()
            agent_actions.append(action)
        elif line.startswith(f"Agent_{agent_id}_Observation:"):
            obs = line.replace(f"Agent_{agent_id}_Observation:", "").strip()
            observations.append(obs)

    if not agent_actions:
        return f"Agent {agent_id} did not perform any recorded actions for this task."

    # Create enhanced prompt for more specific skills
    prompt = f"""You are analyzing a robotic agent's behavior in a household environment. Generate very specific, actionable skills that describe EXACTLY where the agent should go and what they should do.

Task: {instruction}
Agent ID: {agent_id}
Actions performed:
{chr(10).join([f"- {action}" for action in agent_actions])}

Observations made:
{chr(10).join([f"- {obs}" for obs in observations[-5:]])}

Generate a skill description that follows this format:
"Agent should [specific action] at [specific location] to [specific purpose], then [next specific action] at [next location]."

Focus on:
1. SPECIFIC locations (room names, furniture, containers)
2. SPECIFIC actions (navigate to X, pick up Y from Z, place A on B)
3. SPECIFIC objects mentioned in the task
4. LOGICAL sequence of where to go and what to do
5. Coordination points with other agents if applicable

Example: "Agent should navigate to the kitchen counter, pick up the apple from the fruit bowl, then move to the dining table and place the apple in the center, while coordinating with Agent 1 who handles the plates."

Specific Skill Description:"""

    try:
        # Use ollama to call Llama 3.1 8B
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            skill_summary = result.stdout.strip()
            # Clean up the response
            if skill_summary.startswith("Specific Skill Description:"):
                skill_summary = skill_summary[28:].strip()
            elif skill_summary.startswith("Skill Summary:"):
                skill_summary = skill_summary[14:].strip()
            # Ensure the response follows the specific format
            if not skill_summary.startswith("Agent should"):
                skill_summary = f"Agent {agent_id} should {skill_summary.lower()}"
            return skill_summary
        else:
            # Don't print error for each call, will be handled at episode level
            return generate_fallback_skills(agent_actions, agent_id, instruction)

    except subprocess.TimeoutExpired:
        return generate_fallback_skills(agent_actions, agent_id, instruction)
    except FileNotFoundError:
        # Only print this once
        if not hasattr(generate_skills_with_llama, "ollama_warning_shown"):
            print("Warning: Ollama not found, using fallback skill generation")
            generate_skills_with_llama.ollama_warning_shown = True
        return generate_fallback_skills(agent_actions, agent_id, instruction)
    except Exception:
        return generate_fallback_skills(agent_actions, agent_id, instruction)


def generate_fallback_skills(actions, agent_id, instruction):
    """Generate specific, actionable fallback skills if Llama is not available"""
    # Extract specific objects and locations from actions
    locations = set()
    objects = set()
    action_sequence = []

    for action in actions:
        action_lower = action.lower()

        # Extract locations
        for room in [
            "kitchen",
            "bedroom",
            "living room",
            "dining room",
            "bathroom",
            "hallway",
        ]:
            if room in action_lower:
                locations.add(room)

        # Extract furniture/containers
        for furniture in [
            "counter",
            "table",
            "bed",
            "couch",
            "chair",
            "cabinet",
            "drawer",
            "shelf",
        ]:
            if furniture in action_lower:
                locations.add(furniture)

        # Extract objects
        import re

        # Look for objects after common action words
        obj_patterns = [
            r"pick up (\w+)",
            r"place (\w+)",
            r"move (\w+)",
            r"get (\w+)",
            r"take (\w+)",
        ]
        for pattern in obj_patterns:
            matches = re.findall(pattern, action_lower)
            objects.update(matches)

        # Categorize actions more specifically
        if "navigate" in action_lower or "move to" in action_lower:
            # Extract destination if possible
            dest_match = re.search(
                r"(?:navigate to|move to) (\w+(?:\s+\w+)?)", action_lower
            )
            if dest_match:
                dest = dest_match.group(1)
                action_sequence.append(f"navigate to the {dest}")
            else:
                action_sequence.append("navigate to target location")
        elif "pick" in action_lower or "grasp" in action_lower:
            obj_match = re.search(r"(?:pick up|pick|grasp) (\w+)", action_lower)
            if obj_match:
                obj = obj_match.group(1)
                action_sequence.append(f"pick up the {obj}")
            else:
                action_sequence.append("pick up target object")
        elif "place" in action_lower or "put" in action_lower:
            # Try to extract both object and destination
            place_match = re.search(
                r"(?:place|put) (\w+) (?:on|in|at) (\w+(?:\s+\w+)?)", action_lower
            )
            if place_match:
                obj, dest = place_match.groups()
                action_sequence.append(f"place the {obj} on the {dest}")
            else:
                action_sequence.append("place object at target location")
        elif "open" in action_lower:
            container_match = re.search(r"open (\w+(?:\s+\w+)?)", action_lower)
            if container_match:
                container = container_match.group(1)
                action_sequence.append(f"open the {container}")
            else:
                action_sequence.append("open container")
        elif "close" in action_lower:
            container_match = re.search(r"close (\w+(?:\s+\w+)?)", action_lower)
            if container_match:
                container = container_match.group(1)
                action_sequence.append(f"close the {container}")
            else:
                action_sequence.append("close container")
        elif "wait" in action_lower:
            action_sequence.append("wait for coordination with other agent")

    # Generate specific skill description
    if action_sequence:
        # Create a logical sequence
        if len(action_sequence) <= 2:
            sequence_desc = " and then ".join(action_sequence[:2])
        else:
            sequence_desc = ", then ".join(action_sequence[:3])
            if len(action_sequence) > 3:
                sequence_desc += ", and continue with remaining tasks"

        # Add location context if available
        location_context = ""
        if locations:
            main_location = (
                list(locations)[0] if len(locations) == 1 else "various locations"
            )
            location_context = f" in the {main_location}"

        # Add object context if available
        object_context = ""
        if objects:
            if len(objects) == 1:
                object_context = f" focusing on the {list(objects)[0]}"
            elif len(objects) <= 3:
                object_context = f" handling {', '.join(list(objects)[:3])}"
            else:
                object_context = " managing multiple objects"

        return f"Agent {agent_id} should {sequence_desc}{location_context}{object_context} to complete the task: {instruction}"
    else:
        # Fallback to basic description
        return f"Agent {agent_id} should analyze the task requirements and execute appropriate actions to complete: {instruction}"


def enhance_trace_with_skills(content, skill_summary, agent_id, instruction):
    """Enhance trace content by adding skill information while preserving original content"""
    lines = content.split("\n")

    task_type = analyze_task_type(instruction)
    complexity = analyze_complexity(instruction)

    # Create enhanced content by adding skill information at the top
    enhanced_lines = []
    enhanced_lines.append("=== SKILL ANALYSIS ===")
    enhanced_lines.append(f"Specific Actions Required: {skill_summary}")
    enhanced_lines.append(f"Task Category: {task_type}")
    enhanced_lines.append(f"Task Complexity: {complexity}")
    enhanced_lines.append(f"Agent ID: Agent_{agent_id}")
    enhanced_lines.append(f"Task Instruction: {instruction}")
    enhanced_lines.append("=== ORIGINAL TRACE ===")
    enhanced_lines.append("")
    enhanced_lines.extend(lines)  # Preserve all original content

    return "\n".join(enhanced_lines)


def analyze_task_type(instruction):
    """Analyze and categorize the task type"""
    instruction_lower = instruction.lower()

    if any(word in instruction_lower for word in ["bedroom", "bed"]):
        return "Bedroom Organization"
    elif any(word in instruction_lower for word in ["kitchen", "cook", "food"]):
        return "Kitchen Organization"
    elif any(word in instruction_lower for word in ["dining", "table", "dinner"]):
        return "Dining Room Setup"
    elif any(word in instruction_lower for word in ["living room", "couch", "sofa"]):
        return "Living Room Organization"
    elif any(word in instruction_lower for word in ["bathroom", "toilet", "shower"]):
        return "Bathroom Organization"
    elif any(word in instruction_lower for word in ["hallway", "corridor"]):
        return "Hallway Organization"
    elif instruction_lower.count("move") > 2 or instruction_lower.count("and") > 2:
        return "Complex Multi-Object Movement"
    elif "move" in instruction_lower and (
        "from" in instruction_lower or "to" in instruction_lower
    ):
        return "Multi-Object Movement"
    elif any(word in instruction_lower for word in ["clean", "organize", "tidy"]):
        return "General Cleaning/Organization"
    else:
        return "General Task"


def analyze_complexity(instruction):
    """Analyze task complexity based on instruction"""
    instruction_lower = instruction.lower()

    # Count objects mentioned (approximate)
    object_mentions = len(
        re.findall(r"\b(?:move|place|put|take|get)\s+\w+", instruction_lower)
    )
    conjunction_count = instruction_lower.count("and") + instruction_lower.count("or")
    word_count = len(instruction.split())

    if object_mentions > 3 or conjunction_count > 2 or word_count > 20:
        return "High"
    elif object_mentions > 1 or conjunction_count > 1 or word_count > 12:
        return "Medium"
    else:
        return "Low"


def extract_skill_categories(agent_skills):
    """Extract high-level skill categories from agent skills"""
    categories = set()

    for _agent_id, skill_text in agent_skills.items():
        if not skill_text:
            continue

        skill_lower = skill_text.lower()

        if (
            "navigation" in skill_lower
            or "pathfinding" in skill_lower
            or "spatial" in skill_lower
        ):
            categories.add("Navigation")

        if (
            "manipulation" in skill_lower
            or "grasping" in skill_lower
            or "pick" in skill_lower
        ):
            categories.add("Object Manipulation")

        if (
            "placement" in skill_lower
            or "place" in skill_lower
            or "arrangement" in skill_lower
        ):
            categories.add("Object Placement")

        if (
            "coordination" in skill_lower
            or "timing" in skill_lower
            or "multi-agent" in skill_lower
        ):
            categories.add("Multi-Agent Coordination")

        if (
            "container" in skill_lower
            or "storage" in skill_lower
            or "open" in skill_lower
        ):
            categories.add("Container Management")

        if any(
            room in skill_lower
            for room in ["bedroom", "kitchen", "living", "bathroom", "dining"]
        ):
            categories.add("Room-Specific Organization")

        if (
            "planning" in skill_lower
            or "execution" in skill_lower
            or "task" in skill_lower
        ):
            categories.add("Task Planning")

    return list(categories)


def analyze_skill_statistics(episodes):
    """Analyze skill distribution across the dataset"""
    task_type_counts = {}
    complexity_counts = {}
    skill_category_counts = {}

    for episode in episodes:
        # Task types
        task_type = episode.get("task_type", "Unknown")
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        # Complexity
        complexity = episode.get("complexity", "Unknown")
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # Skill categories
        skill_categories = episode.get("skill_categories", [])
        for category in skill_categories:
            skill_category_counts[category] = skill_category_counts.get(category, 0) + 1

    return {
        "task_type_distribution": task_type_counts,
        "complexity_distribution": complexity_counts,
        "skill_category_distribution": skill_category_counts,
        "total_episodes_with_skills": len([e for e in episodes if e.get("skills")]),
    }


if __name__ == "__main__":
    output_dir = enhance_dataset_with_skills()
    print(f"\nEnhancement completed! Enhanced dataset available at: {output_dir}")
