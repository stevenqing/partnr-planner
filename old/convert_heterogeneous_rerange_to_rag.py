#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import gzip
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime


def convert_heterogeneous_rerange_to_rag_format():
    """将heterogeneous+rerange_heurstic结果转换成RAG数据集格式"""

    # 输入路径
    input_base_dir = "/home/shuqing/partnr-planner/outputs/habitat_llm/2025-12-22_10-26-56-heterogeneous+rerange_heurstic.json/results"
    input_traces_dir = f"{input_base_dir}/heterogeneous+rerange.json.gz/traces"
    episode_log_file = f"{input_base_dir}/episode_result_log.csv"

    # 输出路径
    output_dir = (
        "/home/shuqing/partnr-planner/data/rag_datasets/heterogeneous_rerange_heuristic"
    )

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/0", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/1", exist_ok=True)

    print("=" * 80)
    print("转换heterogeneous+rerange_heurstic结果到RAG数据集格式")
    print("=" * 80)
    print(f"输入目录: {input_traces_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 读取episode结果日志
    successful_episodes = {}
    all_episodes = {}
    if os.path.exists(episode_log_file):
        print(f"📖 读取episode结果日志: {episode_log_file}")
        with open(episode_log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episode_id = row.get("episode_id", "").strip()
                task_state_success = float(row.get("task_state_success", 0))
                instruction = row.get("instruction", "").strip()

                all_episodes[episode_id] = {
                    "instruction": instruction,
                    "success": task_state_success,
                    "task_percent_complete": float(row.get("task_percent_complete", 0)),
                }

                if task_state_success >= 1.0:  # 成功的episode
                    successful_episodes[episode_id] = {
                        "instruction": instruction,
                        "success": task_state_success,
                    }
        print(f"  ✓ 读取 {len(all_episodes)} 个episodes")
        print(f"  ✓ 其中 {len(successful_episodes)} 个成功的episodes")
    else:
        print("  ⚠️  警告: 未找到episode_result_log.csv文件")

    # 收集所有trace文件
    print("\n📁 收集trace文件...")
    trace_files = []
    for agent_id in ["0", "1"]:
        trace_dir = f"{input_traces_dir}/{agent_id}"
        if os.path.exists(trace_dir):
            for file in os.listdir(trace_dir):
                if file.startswith("trace-episode_") and file.endswith(".txt"):
                    # 提取episode_id: trace-episode_1621_0-0.txt -> 1621
                    match = re.search(r"trace-episode_(\d+)_", file)
                    if match:
                        episode_id = match.group(1)
                        trace_files.append((agent_id, file, episode_id))

    print(f"  ✓ 找到 {len(trace_files)} 个trace文件")

    # 按episode_id组织trace文件
    episode_traces = {}
    for agent_id, filename, episode_id in trace_files:
        if episode_id not in episode_traces:
            episode_traces[episode_id] = {}
        episode_traces[episode_id][agent_id] = (
            filename,
            f"{input_traces_dir}/{agent_id}/{filename}",
        )

    print(f"  ✓ 找到 {len(episode_traces)} 个唯一的episodes")

    # 转换每个episode
    print("\n🔄 转换episodes...")
    converted_episodes = []
    success_count = 0
    skipped_count = 0

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

            # 如果没有从trace中提取到指令，从episode日志中获取
            if not task_instruction:
                task_instruction = all_episodes.get(episode_id, {}).get(
                    "instruction", ""
                )

            # 获取episode的成功率
            episode_info = all_episodes.get(episode_id, {})
            success_rate = episode_info.get("success", 0.0)

            # 如果episode有指令，则处理（包括成功和失败的）
            if task_instruction:
                # 复制trace文件
                for agent_id, (filename, input_path) in traces.items():
                    output_path = (
                        f"{output_dir}/react_trajectories/traces/{agent_id}/{filename}"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy2(input_path, output_path)

                # 创建episode信息
                episode_data = {
                    "episode_id": episode_id,
                    "instruction": task_instruction,
                    "success_rate": success_rate,
                    "format": "react",
                }

                # 分析任务类型和复杂度（可选）
                task_type = analyze_task_type(task_instruction)
                complexity = analyze_complexity(task_instruction)
                episode_data["task_type"] = task_type
                episode_data["complexity"] = complexity

                converted_episodes.append(episode_data)
                success_count += 1

                if success_count % 50 == 0:
                    print(f"  已处理 {success_count} 个episodes...")
            else:
                skipped_count += 1

        except Exception as e:
            print(f"  ⚠️  转换失败 episode {episode_id}: {e}")
            skipped_count += 1

    print(f"  ✓ 成功转换 {success_count} 个episodes")
    if skipped_count > 0:
        print(f"  ⚠️  跳过 {skipped_count} 个episodes")

    # 分析统计信息
    print("\n📊 分析统计信息...")
    task_type_dist = Counter()
    complexity_dist = Counter()

    for ep in converted_episodes:
        task_type_dist[ep.get("task_type", "Unknown")] += 1
        complexity_dist[ep.get("complexity", "Unknown")] += 1

    # 创建metadata
    metadata = {
        "total_episodes": len(converted_episodes),
        "source": "habitat_llm run results (heterogeneous+rerange_heurstic)",
        "created_by": "convert_heterogeneous_rerange_to_rag.py",
        "example_type": "react",
        "description": "从heterogeneous+rerange_heurstic结果转换的React格式轨迹",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "skill_statistics": {
            "task_type_distribution": dict(task_type_dist),
            "complexity_distribution": dict(complexity_dist),
        },
    }

    print(f"  ✓ 任务类型分布: {len(task_type_dist)} 种")
    print(f"  ✓ 复杂度分布: {len(complexity_dist)} 种")

    # 创建主JSON文件
    print("\n💾 保存数据集...")
    main_data = {"metadata": metadata, "episodes": converted_episodes}

    output_json_path = f"{output_dir}/react_trajectories.json.gz"
    with gzip.open(output_json_path, "wt", encoding="utf-8") as f:
        json.dump(main_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 保存JSON文件: {output_json_path}")

    # 创建episode_result_log.csv（使用空格分隔符，与RAG加载器兼容）
    csv_path = f"{output_dir}/episode_result_log.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        # RAG加载器期望格式: episode_id instruction success
        # 使用空格分隔符，instruction用引号包围
        f.write("episode_id instruction success\n")
        valid_count = 0

        for ep in converted_episodes:
            episode_id = str(ep.get("episode_id", ""))
            instruction = ep.get("instruction", "").replace('"', '""')  # 转义引号
            success = ep.get("success_rate", 0)
            # 格式: episode_id "instruction" success
            f.write(f'{episode_id} "{instruction}" {success}\n')
            valid_count += 1

        print(f"  ✓ 保存CSV文件: {csv_path} ({valid_count} 个episodes)")

    # 打印总结
    print("\n" + "=" * 80)
    print("✅ 转换完成!")
    print("=" * 80)
    print(f"总episodes数: {len(converted_episodes)}")
    print(f"输出目录: {output_dir}")
    print(f"主文件: {output_json_path}")
    print(f"CSV文件: {csv_path}")
    print("=" * 80)

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


if __name__ == "__main__":
    output_dir = convert_heterogeneous_rerange_to_rag_format()
    print(f"\n转换完成！输出目录: {output_dir}")
