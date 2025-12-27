#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import gzip
import json
import os
import re
import shutil
from datetime import datetime

from habitat_llm.llm import instantiate_llm


def convert_spatial_to_rag_format():
    """将spatial_only结果转换成RAG数据集格式"""

    # 初始化Llama 3.1 8B模型（必须成功，不能fallback）
    print("正在初始化Llama 3.1 8B模型...")
    # 尝试从环境变量获取模型路径，如果没有则使用Hugging Face模型ID
    model_path = os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")

    llm = instantiate_llm(
        "llama",
        generation_params={
            "engine": model_path,
            "max_tokens": 250,
            "temperature": 0.0,
            "stop": "\n\n",  # 使用双换行作为stop，避免过早截断
        },
    )
    print(f"模型初始化成功: {model_path}")

    # 输入路径
    input_base_dir = "outputs/habitat_llm/2025-11-14_13-53-27-spatial_only.json/results"
    input_traces_dir = f"{input_base_dir}/spatial_only.json.gz/traces"
    episode_log_file = f"{input_base_dir}/episode_result_log.csv"

    # 输出路径
    output_dir = "data/rag_datasets/spatial_only"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/0", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/1", exist_ok=True)

    print("开始转换spatial_only轨迹文件...")
    print(f"输入目录: {input_traces_dir}")
    print(f"输出目录: {output_dir}")

    # 读取episode结果日志，找出成功的episodes
    successful_episodes = {}
    if os.path.exists(episode_log_file):
        with open(episode_log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episode_id = row.get("episode_id", "")
                task_state_success = float(row.get("task_state_success", 0))
                if task_state_success >= 1.0:  # 成功的episode
                    successful_episodes[episode_id] = {
                        "instruction": row.get("instruction", ""),
                        "success": task_state_success,
                    }
        print(f"从日志中找到 {len(successful_episodes)} 个成功的episodes")
    else:
        print("警告: 未找到episode_result_log.csv文件")

    # 收集所有trace文件
    trace_files = []
    for agent_id in ["0", "1"]:
        trace_dir = f"{input_traces_dir}/{agent_id}"
        if os.path.exists(trace_dir):
            for file in os.listdir(trace_dir):
                if file.startswith("trace-episode_") and file.endswith(".txt"):
                    # 提取episode_id: trace-episode_260_0-0.txt -> 260
                    parts = (
                        file.replace("trace-episode_", "")
                        .replace(".txt", "")
                        .split("_")
                    )
                    episode_id = parts[0] if parts else file
                    trace_files.append((agent_id, file, episode_id))

    print(f"找到 {len(trace_files)} 个trace文件")

    # 按episode_id组织trace文件
    episode_traces = {}
    for agent_id, filename, episode_id in trace_files:
        if episode_id not in episode_traces:
            episode_traces[episode_id] = {}
        episode_traces[episode_id][agent_id] = (
            filename,
            f"{input_traces_dir}/{agent_id}/{filename}",
        )

    print(f"找到 {len(episode_traces)} 个唯一的episodes")

    # 转换每个episode
    converted_episodes = []
    success_count = 0

    for episode_id, traces in episode_traces.items():
        try:
            # 读取所有agent的trace文件
            agent_contents = {}
            task_instruction = ""

            for agent_id, (filename, input_path) in traces.items():
                if os.path.exists(input_path):
                    with open(input_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    agent_contents[agent_id] = content

                    # 提取任务指令（从任意agent的trace中提取）
                    if not task_instruction:
                        lines = content.split("\n")
                        for line in lines:
                            if line.startswith("Task: "):
                                task_instruction = line.replace("Task: ", "").strip()
                                break

            # 如果没有从trace中提取到指令，从successful_episodes中获取
            if not task_instruction:
                task_instruction = successful_episodes.get(episode_id, {}).get(
                    "instruction", ""
                )

            # 如果episode是成功的或有指令，则处理
            if episode_id in successful_episodes or task_instruction:
                # 为每个agent生成技能（使用Llama 3.1 8B）
                agent_skills = {}
                for agent_id, content in agent_contents.items():
                    skill = generate_skills_with_llama(
                        content, agent_id, task_instruction, llm
                    )
                    agent_skills[agent_id] = skill

                    # 复制trace文件
                    filename = traces[agent_id][0]
                    output_path = (
                        f"{output_dir}/react_trajectories/traces/{agent_id}/{filename}"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy2(traces[agent_id][1], output_path)

                # 创建增强的episode信息
                episode_info = {
                    "episode_id": episode_id,
                    "instruction": task_instruction,
                    "success_rate": successful_episodes.get(episode_id, {}).get(
                        "success", 1.0
                    ),
                    "format": "react",
                    "task_type": analyze_task_type(task_instruction),
                    "complexity": analyze_complexity(task_instruction),
                    "skills": agent_skills,
                    "skill_categories": extract_skill_categories(agent_skills),
                    "coordination_required": len(agent_skills) > 1,
                }
                converted_episodes.append(episode_info)
                success_count += 1

                print(f"转换成功: episode {episode_id}")

        except Exception as e:
            print(f"转换失败 episode {episode_id}: {e}")
            import traceback

            traceback.print_exc()

    # 创建增强的metadata（匹配rerange_only_cleaned格式）
    metadata = {
        "total_episodes": len(converted_episodes),
        "source": "habitat_llm run results (spatial_only)",
        "created_by": "convert_spatial_to_rag.py",
        "example_type": "react",
        "description": "从spatial_only结果转换的React格式轨迹 - Enhanced with skill extraction",
        "enhanced_with_skills": True,
        "enhancement_date": datetime.now().strftime("%Y-%m-%d"),
        "enhanced_by": "convert_spatial_to_rag.py",
        "skill_model": "Llama 3.1 8B",
        "skill_statistics": analyze_skill_statistics(converted_episodes),
    }

    # 创建主JSON文件
    main_data = {"metadata": metadata, "episodes": converted_episodes}

    # 保存主JSON文件
    output_json_path = f"{output_dir}/react_trajectories.json.gz"
    with gzip.open(output_json_path, "wt", encoding="utf-8") as f:
        json.dump(main_data, f, indent=2, ensure_ascii=False)

    # 创建episode_result_log.csv
    csv_path = f"{output_dir}/episode_result_log.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["episode_id", "instruction", "success"])
        for ep in converted_episodes:
            writer.writerow([ep["episode_id"], ep["instruction"], ep["success_rate"]])

    print("\n转换完成!")
    print(f"成功转换: {success_count} 个episodes")
    print(f"输出目录: {output_dir}")
    print(f"主文件: {output_json_path}")
    print(f"CSV文件: {csv_path}")

    return output_dir


def analyze_task_type(instruction):
    """分析并分类任务类型"""
    if not instruction:
        return "General Task"

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
        return "Multi-Object Movement"
    elif "move" in instruction_lower and (
        "from" in instruction_lower or "to" in instruction_lower
    ):
        return "Multi-Object Movement"
    elif any(word in instruction_lower for word in ["clean", "organize", "tidy"]):
        return "General Task"
    else:
        return "General Task"


def analyze_complexity(instruction):
    """分析任务复杂度"""
    if not instruction:
        return "Low"

    instruction_lower = instruction.lower()

    # 计算对象提及次数（近似）
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


def generate_skills_with_llama(trace_content, agent_id, instruction, llm):
    """使用Llama 3.1 8B生成具体的、可操作的技能描述（必须使用LLM，不能fallback）"""

    # 提取agent的动作和观察
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

    # 创建增强的prompt以生成更具体的技能
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

    # 必须使用LLM生成技能，不能fallback
    # 使用双换行作为stop，让模型生成完整句子
    skill_summary = llm.generate(prompt, max_length=250, stop="\n\n")
    skill_summary = skill_summary.strip()

    # 调试：打印原始输出
    if len(skill_summary) < 50:
        print(
            f"Warning: Generated skill for agent {agent_id} is very short ({len(skill_summary)} chars): {repr(skill_summary[:100])}"
        )

    # 清理响应
    if skill_summary.startswith("Specific Skill Description:"):
        skill_summary = skill_summary[28:].strip()
    elif skill_summary.startswith("Skill Summary:"):
        skill_summary = skill_summary[14:].strip()
    # 确保响应遵循特定格式
    if not skill_summary.startswith("Agent should"):
        # 如果内容为空或太短，可能是模型输出问题
        if len(skill_summary) < 20:
            print(
                f"Error: Model generated empty or very short response for agent {agent_id}"
            )
            # 这里不应该fallback，但我们需要确保模型正常工作
        skill_summary = (
            f"Agent {agent_id} should {skill_summary.lower()}"
            if skill_summary
            else f"Agent {agent_id} should complete the task: {instruction}"
        )

    return skill_summary


def generate_fallback_skills(actions, agent_id, instruction):
    """如果Llama不可用，使用fallback方法生成技能"""
    if not actions:
        return f"Agent {agent_id} should analyze the task requirements and execute appropriate actions to complete: {instruction}"

    # 提取位置和对象
    locations = set()
    objects = set()
    action_sequence = []

    for action in actions:
        action_lower = action.lower()

        # 提取房间位置
        for room in [
            "kitchen",
            "bedroom",
            "living room",
            "dining room",
            "bathroom",
            "hallway",
            "office",
        ]:
            if room in action_lower:
                locations.add(room)

        # 提取家具/容器
        for furniture in [
            "counter",
            "table",
            "bed",
            "couch",
            "chair",
            "cabinet",
            "drawer",
            "shelf",
            "bench",
            "desk",
        ]:
            if furniture in action_lower:
                locations.add(furniture)

        # 提取对象
        obj_patterns = [
            r"pick up (\w+)",
            r"place (\w+)",
            r"move (\w+)",
            r"get (\w+)",
            r"take (\w+)",
            r"put (\w+)",
        ]
        for pattern in obj_patterns:
            matches = re.findall(pattern, action_lower)
            objects.update(matches)

        # 分类动作
        if "navigate" in action_lower or "move to" in action_lower:
            dest_match = re.search(
                r"(?:navigate to|move to) (?:the )?(\w+(?:\s+\w+)?)", action_lower
            )
            if dest_match:
                dest = dest_match.group(1)
                action_sequence.append(f"navigate to the {dest}")
            else:
                action_sequence.append("navigate to target location")
        elif "pick" in action_lower or "grasp" in action_lower:
            obj_match = re.search(
                r"(?:pick up|pick|grasp) (?:the )?(\w+)", action_lower
            )
            if obj_match:
                obj = obj_match.group(1)
                action_sequence.append(f"pick up the {obj}")
            else:
                action_sequence.append("pick up target object")
        elif "place" in action_lower or "put" in action_lower:
            place_match = re.search(
                r"(?:place|put) (?:the )?(\w+) (?:on|in|at|next to) (?:the )?(\w+(?:\s+\w+)?)",
                action_lower,
            )
            if place_match:
                obj, dest = place_match.groups()
                action_sequence.append(f"place the {obj} on the {dest}")
            else:
                action_sequence.append("place object at target location")

    # 生成技能描述
    if action_sequence:
        if len(action_sequence) <= 2:
            sequence_desc = " and then ".join(action_sequence[:2])
        else:
            sequence_desc = ", then ".join(action_sequence[:3])
            if len(action_sequence) > 3:
                sequence_desc += ", and continue with remaining tasks"

        location_context = ""
        if locations:
            main_location = (
                list(locations)[0] if len(locations) == 1 else "various locations"
            )
            location_context = f" in the {main_location}"

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
        return f"Agent {agent_id} should analyze the task requirements and execute appropriate actions to complete: {instruction}"


def extract_skill_categories(agent_skills):
    """从agent技能中提取高级技能类别"""
    categories = set()

    for _agent_id, skill_text in agent_skills.items():
        if not skill_text:
            continue

        skill_lower = skill_text.lower()

        if (
            "navigation" in skill_lower
            or "pathfinding" in skill_lower
            or "navigate" in skill_lower
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
            or "close" in skill_lower
        ):
            categories.add("Container Management")

        if any(
            room in skill_lower
            for room in [
                "bedroom",
                "kitchen",
                "living",
                "bathroom",
                "dining",
                "hallway",
                "office",
            ]
        ):
            categories.add("Room-Specific Organization")

        if (
            "planning" in skill_lower
            or "execution" in skill_lower
            or "task" in skill_lower
        ):
            categories.add("Task Planning")

    return list(categories) if categories else ["General Task"]


def analyze_skill_statistics(episodes):
    """分析数据集的技能分布统计"""
    task_type_counts = {}
    complexity_counts = {}
    skill_category_counts = {}

    for episode in episodes:
        # 任务类型
        task_type = episode.get("task_type", "Unknown")
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        # 复杂度
        complexity = episode.get("complexity", "Unknown")
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # 技能类别
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
    output_dir = convert_spatial_to_rag_format()
    print(f"\n转换完成！输出目录: {output_dir}")
