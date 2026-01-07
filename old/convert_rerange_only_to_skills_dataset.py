#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
import re


def convert_rerange_only_to_skills_dataset():
    """将rerange_only_converted数据集转换为skills_dataset格式，分别处理agent0和agent1"""

    # 输入路径
    input_dir = "data/rag_datasets/rerange_only_converted"

    # 输出路径
    output_dir = "data/rag_datasets/rerange_only_skills_dataset"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/0", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/1", exist_ok=True)

    print("开始转换rerange_only_converted数据集...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 收集所有trace文件
    trace_files = []
    for agent_id in ["0", "1"]:
        trace_dir = f"{input_dir}/react_trajectories/traces/{agent_id}"
        if os.path.exists(trace_dir):
            for file in os.listdir(trace_dir):
                if file.startswith("trace-episode_") and file.endswith(".txt"):
                    trace_files.append((agent_id, file))

    print(f"找到 {len(trace_files)} 个trace文件")

    # 转换每个trace文件
    converted_episodes = []
    success_count = 0

    for agent_id, filename in trace_files:
        input_path = f"{input_dir}/react_trajectories/traces/{agent_id}/{filename}"
        output_path = f"{output_dir}/react_trajectories/traces/{agent_id}/{filename}"

        try:
            # 读取原始trace文件
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 解析episode_id
            episode_id = filename.replace("trace-episode_", "").replace(
                f"_0-{agent_id}.txt", ""
            )

            # 提取任务指令和转换格式
            (
                converted_content,
                task_instruction,
                skill_summary,
            ) = convert_to_skills_format(content, agent_id)

            # 保存转换后的trace文件
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(converted_content)

            # 添加到episodes列表
            episode_info = {
                "episode_id": episode_id,
                "instruction": task_instruction,
                "success_rate": 1.0,
                "format": "react",
                "agent_id": agent_id,
                "skill_summary": skill_summary,
            }
            converted_episodes.append(episode_info)
            success_count += 1

            print(f"转换成功: {filename} (Agent {agent_id})")

        except Exception as e:
            print(f"转换失败 {filename}: {e}")

    # 创建episode_result_log.csv
    create_episode_result_log(converted_episodes, output_dir)

    # 创建metadata
    metadata = {
        "total_episodes": len(converted_episodes),
        "source": "rerange_only_converted dataset with skill extraction",
        "created_by": "convert_rerange_only_to_skills_dataset.py",
        "example_type": "react",
        "description": "Skills extracted from rerange_only trajectories with English summaries",
        "statistics": {
            "task_types": analyze_task_types(converted_episodes),
            "complexity": analyze_complexity(converted_episodes),
        },
    }

    # 创建主JSON文件
    main_data = {"metadata": metadata, "episodes": converted_episodes}

    # 保存主JSON文件
    output_json_path = f"{output_dir}/react_trajectories.json.gz"
    with gzip.open(output_json_path, "wt", encoding="utf-8") as f:
        json.dump(main_data, f, indent=2, ensure_ascii=False)

    print("\n转换完成!")
    print(f"成功转换: {success_count} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"主文件: {output_json_path}")

    return output_dir


def convert_to_skills_format(content, agent_id):
    """将原始轨迹转换为skills格式"""
    lines = content.split("\n")

    # 提取任务指令
    task_instruction = ""
    for line in lines:
        if line.startswith("Task: "):
            task_instruction = line.replace("Task: ", "").strip()
            break

    # 分析任务类型和复杂度
    task_type, complexity = analyze_task(task_instruction)

    # 生成技能总结
    skill_summary = generate_skill_summary(content, task_type, agent_id)

    # 转换格式
    converted_lines = []
    converted_lines.append(f"Task: {task_instruction}")
    converted_lines.append(f"Skill Summary: {skill_summary}")
    converted_lines.append(f"Task Type: {task_type}")
    converted_lines.append(f"Complexity: {complexity}")
    converted_lines.append("")

    # 提取该agent的动作序列
    agent_actions = extract_agent_actions(lines, agent_id)

    # 转换为React格式
    for action in agent_actions:
        converted_lines.append(
            "Thought: I need to execute this action to complete the task."
        )
        converted_lines.append(f"Agent_{agent_id}_Action: {action}")
        converted_lines.append(f"Agent_{agent_id}_observation: Successful execution!")
        converted_lines.append("")

    converted_lines.append(
        "Final Thought: Task completed successfully using the learned skill strategy."
    )

    return "\n".join(converted_lines), task_instruction, skill_summary


def extract_agent_actions(lines, agent_id):
    """提取指定agent的动作序列"""
    actions = []
    for line in lines:
        if line.startswith(f"Agent_{agent_id}_Action:"):
            action = line.replace(f"Agent_{agent_id}_Action:", "").strip()
            actions.append(action)
    return actions


def analyze_task(instruction):
    """分析任务类型和复杂度"""
    instruction_lower = instruction.lower()

    # 任务类型分析
    if any(word in instruction_lower for word in ["bedroom", "bed"]):
        task_type = "Bedroom Organization"
    elif any(word in instruction_lower for word in ["kitchen", "cook", "food"]):
        task_type = "Kitchen Organization"
    elif any(word in instruction_lower for word in ["dining", "table", "dinner"]):
        task_type = "Dining Room Setup"
    elif instruction_lower.count("move") > 2 or instruction_lower.count("and") > 1:
        task_type = "Complex Multi-Object Movement"
    elif (
        "move" in instruction_lower
        and "from" in instruction_lower
        and "to" in instruction_lower
    ):
        task_type = "Multi-Object Movement"
    else:
        task_type = "General Organization"

    # 复杂度分析
    object_count = len(re.findall(r"\b\w+\b", instruction))
    if object_count > 8 or instruction_lower.count("and") > 2:
        complexity = "Complex"
    elif object_count > 5 or instruction_lower.count("and") > 1:
        complexity = "Medium"
    else:
        complexity = "Simple"

    return task_type, complexity


def generate_skill_summary(content, task_type, agent_id):
    """根据agent的实际轨迹生成个性化的技能总结"""
    lines = content.split("\n")

    # 提取该agent的动作序列
    agent_actions = extract_agent_actions(lines, agent_id)

    # 分析动作模式
    action_patterns = analyze_action_patterns(agent_actions)

    base_summary = f"I have successfully solved this {task_type} task. Next time when I encounter similar tasks, I know that I should"

    # 根据实际动作序列生成技能总结
    skills = []

    # 基础技能
    if "Navigate" in action_patterns:
        skills.append("navigate to target objects efficiently")
    if "Pick" in action_patterns:
        skills.append("pick up objects carefully")
    if "Place" in action_patterns:
        skills.append("place objects in designated locations")
    if "Open" in action_patterns:
        skills.append("open containers when needed")
    if "Wait" in action_patterns:
        skills.append("coordinate timing with other agents")

    # 根据动作数量判断复杂度
    if len(agent_actions) > 10:
        skills.append("handle complex multi-step sequences")
    elif len(agent_actions) > 5:
        skills.append("execute moderate complexity tasks")
    else:
        skills.append("complete simple tasks efficiently")

    # 根据任务类型添加特定技能
    if task_type == "Bedroom Organization":
        skills.append("pay attention to proper placement of items on the bed")
    elif task_type == "Kitchen Organization":
        skills.append("pay attention to proper placement of items on kitchen surfaces")
    elif task_type == "Dining Room Setup":
        skills.append("ensure proper table arrangement")

    # 多智能体协作技能
    if len(agent_actions) > 0 and any("Wait" in action for action in agent_actions):
        skills.append("coordinate with other agents for efficient task completion")

    # 去重并组合
    unique_skills = list(dict.fromkeys(skills))  # 保持顺序的去重
    skills_text = ", ".join(unique_skills)

    return f"{base_summary} {skills_text}."


def analyze_action_patterns(actions):
    """分析动作模式"""
    patterns = set()
    for action in actions:
        if "Navigate" in action:
            patterns.add("Navigate")
        if "Pick" in action:
            patterns.add("Pick")
        if "Place" in action:
            patterns.add("Place")
        if "Open" in action:
            patterns.add("Open")
        if "Wait" in action:
            patterns.add("Wait")
        if "Close" in action:
            patterns.add("Close")
    return patterns


def create_episode_result_log(episodes, output_dir):
    """创建episode_result_log.csv文件"""
    csv_path = f"{output_dir}/episode_result_log.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("episode_id,instruction,success\n")
        for episode in episodes:
            # 转义指令中的逗号
            instruction = episode["instruction"].replace('"', '""')
            f.write(f"{episode['episode_id']}, \"{instruction}\", 1.0\n")

    print(f"创建episode_result_log.csv: {csv_path}")


def analyze_task_types(episodes):
    """分析任务类型分布"""
    task_types = {}
    for episode in episodes:
        task_type = episode.get("task_type", "Unknown")
        task_types[task_type] = task_types.get(task_type, 0) + 1
    return task_types


def analyze_complexity(episodes):
    """分析复杂度分布"""
    complexity = {}
    for episode in episodes:
        comp = episode.get("complexity", "Unknown")
        complexity[comp] = complexity.get(comp, 0) + 1
    return complexity


if __name__ == "__main__":
    output_dir = convert_rerange_only_to_skills_dataset()
    print(f"\n转换完成！输出目录: {output_dir}")
