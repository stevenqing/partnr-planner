#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os


def convert_trajs_to_rag_format():
    """将trajs文件转换成RAG数据集格式"""

    # 输入路径
    input_dir = "outputs/habitat_llm/2025-08-18_19-10-09-rerange_only.json/results/rerange_only.json.gz"

    # 输出路径
    output_dir = "data/rag_datasets/rerange_only_converted"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/0", exist_ok=True)
    os.makedirs(f"{output_dir}/react_trajectories/traces/1", exist_ok=True)

    print("开始转换trajs文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 收集所有trace文件
    trace_files = []
    for agent_id in ["0", "1"]:
        trace_dir = f"{input_dir}/traces/{agent_id}"
        if os.path.exists(trace_dir):
            for file in os.listdir(trace_dir):
                if file.startswith("trace-episode_") and file.endswith(".txt"):
                    trace_files.append((agent_id, file))

    print(f"找到 {len(trace_files)} 个trace文件")

    # 转换每个trace文件
    converted_episodes = []
    success_count = 0

    for agent_id, filename in trace_files:
        input_path = f"{input_dir}/traces/{agent_id}/{filename}"
        output_path = f"{output_dir}/react_trajectories/traces/{agent_id}/{filename}"

        try:
            # 读取原始trace文件
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 解析episode_id
            episode_id = filename.replace("trace-episode_", "").replace("_0-0.txt", "")

            # 提取任务指令
            lines = content.split("\n")
            task_instruction = ""
            for line in lines:
                if line.startswith("Task: "):
                    task_instruction = line.replace("Task: ", "").strip()
                    break

            # 转换格式：添加Thought字段
            converted_content = convert_trace_format(content)

            # 保存转换后的trace文件
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(converted_content)

            # 添加到episodes列表
            episode_info = {
                "episode_id": episode_id,
                "instruction": task_instruction,
                "success_rate": 1.0,  # 假设所有trajs都是成功的
                "format": "react",
            }
            converted_episodes.append(episode_info)
            success_count += 1

            print(f"转换成功: {filename}")

        except Exception as e:
            print(f"转换失败 {filename}: {e}")

    # 创建metadata
    metadata = {
        "total_episodes": len(converted_episodes),
        "source": "habitat_llm run results (rerange_only)",
        "created_by": "convert_trajs_to_rag_format.py",
        "example_type": "react",
        "description": "从rerange_only trajs转换的React格式轨迹",
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


def convert_trace_format(content):
    """转换trace格式，添加Thought字段"""
    lines = content.split("\n")
    converted_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if (
            line.startswith("Task:")
            or line.startswith("Furniture:")
            or line.startswith("Objects:")
        ):
            # 保持原样
            converted_lines.append(line)
        elif line.startswith("Agent_") and "_Action:" in line:
            # 在Action前添加Thought
            thought_line = (
                "Thought: I need to execute this action to complete the task."
            )
            converted_lines.append(thought_line)
            converted_lines.append(line)
        elif line.startswith("Agent_") and "_observation:" in line:
            # 保持原样
            converted_lines.append(line)
        else:
            # 其他行保持原样
            if line:  # 只添加非空行
                converted_lines.append(line)

        i += 1

    return "\n".join(converted_lines)


if __name__ == "__main__":
    output_dir = convert_trajs_to_rag_format()
    print(f"\n转换完成！输出目录: {output_dir}")
