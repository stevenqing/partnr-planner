#!/usr/bin/env python3
"""
从 heterogeneous+rerange_heurstic 目录中提取 cooperation skills
基于 extract_cooperation_skills.py 的逻辑
"""

import csv
import gzip
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# 导入 enhanced_skill_extractor
try:
    from enhanced_skill_extractor import EnhancedSkillExtractor
except ImportError:
    print(
        "Error: enhanced_skill_extractor.py not found. Please ensure it's in the same directory."
    )
    exit(1)

# 导入 extract_tom_cooperation_skills 函数
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_cooperation_skills import extract_tom_cooperation_skills


def extract_cooperation_skills_to_rag_format(
    input_dir: str,
    output_dir: str = "data/rag_datasets/cooperation_skills_heterogeneous_rerange",
    filter_by_cooperation: bool = True,
):
    """
    从指定目录中提取 cooperation skills 并整理成 RAG 数据集格式

    Args:
        input_dir: 输入目录路径
        output_dir: 输出 RAG 数据集目录路径
        filter_by_cooperation: 是否只保留有 cooperation skills 的 episodes
    """
    # 初始化 skill extractor
    extractor = EnhancedSkillExtractor(use_llm=False, cache_results=False)

    # 确定路径
    base_dir = Path(input_dir)
    traces_dir = base_dir / "results" / "heterogeneous+rerange.json.gz" / "traces"
    episode_log = base_dir / "results" / "episode_result_log.csv"

    if not traces_dir.exists():
        print(f"Error: Traces directory not found: {traces_dir}")
        return None

    # 创建输出目录结构
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "react_trajectories" / "traces" / "0").mkdir(
        parents=True, exist_ok=True
    )
    (output_path / "react_trajectories" / "traces" / "1").mkdir(
        parents=True, exist_ok=True
    )

    print("=" * 80)
    print("提取cooperation skills到RAG数据集格式")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 读取 episode 信息
    episodes_info = {}
    if episode_log.exists():
        print("📖 读取episode结果日志...")
        with open(episode_log, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episode_id = row["episode_id"].strip()
                episodes_info[episode_id] = {
                    "instruction": row["instruction"],
                    "task_state_success": row.get("task_state_success", "0"),
                    "task_percent_complete": row.get("task_percent_complete", "0"),
                }
        print(f"  ✓ 读取 {len(episodes_info)} 个episodes")
    else:
        print("  ⚠️  警告: 未找到episode_result_log.csv文件")

    # 收集所有 trace 文件
    print("\n📁 收集trace文件...")
    trace_files_0 = list((traces_dir / "0").glob("trace-episode_*_0-0.txt"))
    trace_files_1 = list((traces_dir / "1").glob("trace-episode_*_0-1.txt"))

    # 提取 episode IDs
    episode_ids = set()
    for trace_file in trace_files_0 + trace_files_1:
        # 从文件名提取 episode_id: trace-episode_{id}_0-{agent_id}.txt
        import re

        match = re.search(r"trace-episode_(\d+)_", trace_file.name)
        if match:
            episode_id = match.group(1)
            episode_ids.add(episode_id)

    print(f"  ✓ 找到 {len(episode_ids)} 个唯一的episodes")
    print(f"  ✓ Agent 0: {len(trace_files_0)} 个trace文件")
    print(f"  ✓ Agent 1: {len(trace_files_1)} 个trace文件")

    # 处理每个 episode
    cooperation_skills_data = []
    episodes_for_rag = []
    episode_result_log_rows = []

    pbar = tqdm(sorted(episode_ids), desc="提取cooperation skills")

    for episode_id in pbar:
        episode_data = {
            "episode_id": episode_id,
            "instruction": episodes_info.get(episode_id, {}).get("instruction", ""),
            "task_state_success": episodes_info.get(episode_id, {}).get(
                "task_state_success", "0"
            ),
            "task_percent_complete": episodes_info.get(episode_id, {}).get(
                "task_percent_complete", "0"
            ),
            "agents": {},
        }

        # 处理两个 agent 的 traces
        for agent_id in ["0", "1"]:
            trace_file = (
                traces_dir / agent_id / f"trace-episode_{episode_id}_0-{agent_id}.txt"
            )

            if trace_file.exists():
                try:
                    # 读取 trace 内容
                    with open(trace_file, "r", encoding="utf-8") as f:
                        trace_content = f.read()

                    # 提取 enhanced skills
                    instruction = episode_data["instruction"]
                    enhanced_skill_data = extractor.extract_enhanced_skills(
                        trace_content, agent_id, instruction
                    )

                    # 提取 ToM 形式的 cooperation skills
                    tom_cooperation_data = extract_tom_cooperation_skills(
                        trace_content, agent_id, instruction
                    )

                    # 提取 cooperation/coordination 相关信息
                    coordination_data = enhanced_skill_data.get(
                        "coordination_requirements", {}
                    )
                    skill_patterns = enhanced_skill_data.get("skill_patterns", [])

                    # 筛选出 coordination 相关的 skill patterns
                    coordination_patterns = [
                        pattern
                        for pattern in skill_patterns
                        if pattern.get("skill_type") == "coordination"
                        or "coordination" in pattern.get("skill_name", "").lower()
                        or "cooperation" in pattern.get("skill_name", "").lower()
                    ]

                    # 提取 coordination decision points
                    decision_points = enhanced_skill_data.get("decision_points", [])
                    coordination_decisions = [
                        dp
                        for dp in decision_points
                        if dp.get("decision_type") == "coordination"
                    ]

                    episode_data["agents"][agent_id] = {
                        "coordination_requirements": coordination_data,
                        "coordination_patterns": [
                            p.__dict__ if hasattr(p, "__dict__") else p
                            for p in coordination_patterns
                        ],
                        "coordination_decisions": [
                            dp.__dict__ if hasattr(dp, "__dict__") else dp
                            for dp in coordination_decisions
                        ],
                        "requires_coordination": coordination_data.get(
                            "requires_coordination", False
                        ),
                        "coordination_actions_count": coordination_data.get(
                            "coordination_actions_count", 0
                        ),
                        "coordination_effectiveness": coordination_data.get(
                            "coordination_effectiveness", False
                        ),
                        "tom_cooperation_skills": tom_cooperation_data,  # ToM 形式的 cooperation skills
                    }

                except Exception as e:
                    print(f"\n⚠️  警告: 处理episode {episode_id}, agent {agent_id}时出错: {e}")
                    import traceback

                    traceback.print_exc()
                    episode_data["agents"][agent_id] = {"error": str(e)}
            else:
                episode_data["agents"][agent_id] = {"error": "Trace file not found"}

        # 计算 episode 级别的 cooperation metrics（包括 ToM）
        agent_0_coord = episode_data["agents"].get("0", {})
        agent_1_coord = episode_data["agents"].get("1", {})

        agent_0_tom = agent_0_coord.get("tom_cooperation_skills", {})
        agent_1_tom = agent_1_coord.get("tom_cooperation_skills", {})

        episode_data["episode_cooperation"] = {
            "both_require_coordination": (
                agent_0_coord.get("requires_coordination", False)
                and agent_1_coord.get("requires_coordination", False)
            ),
            "total_coordination_actions": (
                agent_0_coord.get("coordination_actions_count", 0)
                + agent_1_coord.get("coordination_actions_count", 0)
            ),
            "coordination_effective": (
                agent_0_coord.get("coordination_effectiveness", False)
                or agent_1_coord.get("coordination_effectiveness", False)
            ),
            "has_coordination_patterns": (
                len(agent_0_coord.get("coordination_patterns", [])) > 0
                or len(agent_1_coord.get("coordination_patterns", [])) > 0
            ),
            # ToM 相关的 cooperation metrics
            "has_tom_reasoning": (
                agent_0_tom.get("has_tom_reasoning", False)
                or agent_1_tom.get("has_tom_reasoning", False)
            ),
            "tom_cooperation_quality": {
                "avg_tom_completeness": (
                    (
                        agent_0_tom.get("tom_quality_metrics", {}).get(
                            "tom_completeness", 0
                        )
                        + agent_1_tom.get("tom_quality_metrics", {}).get(
                            "tom_completeness", 0
                        )
                    )
                    / 2.0
                    if (
                        agent_0_tom.get("has_tom_reasoning", False)
                        or agent_1_tom.get("has_tom_reasoning", False)
                    )
                    else 0.0
                ),
                "avg_cooperation_focus": (
                    (
                        agent_0_tom.get("tom_quality_metrics", {}).get(
                            "cooperation_focus", 0
                        )
                        + agent_1_tom.get("tom_quality_metrics", {}).get(
                            "cooperation_focus", 0
                        )
                    )
                    / 2.0
                    if (
                        agent_0_tom.get("has_tom_reasoning", False)
                        or agent_1_tom.get("has_tom_reasoning", False)
                    )
                    else 0.0
                ),
                "total_tom_cooperation_patterns": (
                    len(agent_0_tom.get("tom_cooperation_patterns", []))
                    + len(agent_1_tom.get("tom_cooperation_patterns", []))
                ),
                "both_have_complete_tom": (
                    agent_0_tom.get("tom_quality_metrics", {}).get(
                        "has_all_three_steps", False
                    )
                    and agent_1_tom.get("tom_quality_metrics", {}).get(
                        "has_all_three_steps", False
                    )
                ),
            },
        }

        # 更新 has_coordination_patterns 以包含 ToM patterns
        episode_data["episode_cooperation"]["has_coordination_patterns"] = (
            episode_data["episode_cooperation"]["has_coordination_patterns"]
            or episode_data["episode_cooperation"]["has_tom_reasoning"]
            or episode_data["episode_cooperation"]["tom_cooperation_quality"][
                "total_tom_cooperation_patterns"
            ]
            > 0
        )

        cooperation_skills_data.append(episode_data)

        # 决定是否包含在 RAG 数据集中
        include_in_rag = True
        if filter_by_cooperation:
            # 只保留有 cooperation patterns 或 ToM reasoning 的 episodes
            include_in_rag = (
                episode_data["episode_cooperation"]["has_coordination_patterns"]
                or episode_data["episode_cooperation"]["has_tom_reasoning"]
                or episode_data["episode_cooperation"]["tom_cooperation_quality"][
                    "total_tom_cooperation_patterns"
                ]
                > 0
            )

        if include_in_rag:
            # 复制 trace 文件到输出目录
            for agent_id in ["0", "1"]:
                source_trace = (
                    traces_dir
                    / agent_id
                    / f"trace-episode_{episode_id}_0-{agent_id}.txt"
                )
                if source_trace.exists():
                    dest_trace = (
                        output_path
                        / "react_trajectories"
                        / "traces"
                        / agent_id
                        / f"trace-episode_{episode_id}_0-{agent_id}.txt"
                    )
                    shutil.copy2(source_trace, dest_trace)

            # 添加到 RAG episodes 列表
            episode_rag_data = {
                "episode_id": int(episode_id),
                "instruction": episode_data["instruction"],
                "success_rate": float(episode_data.get("task_state_success", 0)),
                "task_percent_complete": float(
                    episode_data.get("task_percent_complete", 0)
                ),
                "cooperation_skills": {
                    "requires_coordination": episode_data["episode_cooperation"][
                        "both_require_coordination"
                    ],
                    "total_coordination_actions": episode_data["episode_cooperation"][
                        "total_coordination_actions"
                    ],
                    "coordination_effective": episode_data["episode_cooperation"][
                        "coordination_effective"
                    ],
                    "has_coordination_patterns": episode_data["episode_cooperation"][
                        "has_coordination_patterns"
                    ],
                    "agent_0_coordination": episode_data["agents"]
                    .get("0", {})
                    .get("coordination_requirements", {}),
                    "agent_1_coordination": episode_data["agents"]
                    .get("1", {})
                    .get("coordination_requirements", {}),
                    # ToM 形式的 cooperation skills
                    "tom_cooperation": {
                        "has_tom_reasoning": episode_data["episode_cooperation"][
                            "has_tom_reasoning"
                        ],
                        "tom_quality": episode_data["episode_cooperation"][
                            "tom_cooperation_quality"
                        ],
                        "agent_0_tom": episode_data["agents"]
                        .get("0", {})
                        .get("tom_cooperation_skills", {}),
                        "agent_1_tom": episode_data["agents"]
                        .get("1", {})
                        .get("tom_cooperation_skills", {}),
                    },
                },
            }
            episodes_for_rag.append(episode_rag_data)

            # 添加到 CSV 日志
            episode_result_log_rows.append(
                {
                    "episode_id": episode_id,
                    "instruction": episode_data["instruction"],
                    "task_state_success": episode_data.get("task_state_success", "0"),
                    "task_percent_complete": episode_data.get(
                        "task_percent_complete", "0"
                    ),
                    "requires_coordination": str(
                        episode_data["episode_cooperation"]["both_require_coordination"]
                    ),
                    "coordination_actions": str(
                        episode_data["episode_cooperation"][
                            "total_coordination_actions"
                        ]
                    ),
                    "coordination_effective": str(
                        episode_data["episode_cooperation"]["coordination_effective"]
                    ),
                    "has_tom_reasoning": str(
                        episode_data["episode_cooperation"]["has_tom_reasoning"]
                    ),
                    "tom_cooperation_patterns": str(
                        episode_data["episode_cooperation"]["tom_cooperation_quality"][
                            "total_tom_cooperation_patterns"
                        ]
                    ),
                    "tom_completeness": f"{episode_data['episode_cooperation']['tom_cooperation_quality']['avg_tom_completeness']:.2f}",
                }
            )

        pbar.set_postfix(
            {
                "Processed": len(cooperation_skills_data),
                "In RAG": len(episodes_for_rag),
                "Episode": episode_id,
            }
        )

    # 保存 cooperation skills JSON（详细数据）
    print("\n💾 保存数据集...")
    skills_json_path = output_path / "cooperation_skills_detailed.json"
    with open(skills_json_path, "w", encoding="utf-8") as f:
        json.dump(cooperation_skills_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 保存详细skills数据: {skills_json_path}")

    # 创建 episode_result_log.csv
    csv_path = output_path / "episode_result_log.csv"
    if episode_result_log_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "episode_id",
                "instruction",
                "task_state_success",
                "task_percent_complete",
                "requires_coordination",
                "coordination_actions",
                "coordination_effective",
                "has_tom_reasoning",
                "tom_cooperation_patterns",
                "tom_completeness",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_result_log_rows)
        print(f"  ✓ 保存CSV文件: {csv_path} ({len(episode_result_log_rows)} 个episodes)")

    # 创建 react_trajectories.json.gz
    metadata = {
        "total_episodes": len(episodes_for_rag),
        "source": "heterogeneous+rerange_heurstic cooperation skills extraction",
        "created_by": "extract_cooperation_skills_heterogeneous_rerange.py",
        "example_type": "react",
        "description": f"Cooperation skills dataset extracted from heterogeneous+rerange_heurstic with Theory of Mind (ToM) analysis ({len(episodes_for_rag)} episodes)",
        "enhanced_with_cooperation_skills": True,
        "enhanced_with_tom_analysis": True,
        "enhancement_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filter_by_cooperation": filter_by_cooperation,
        "cooperation_statistics": {
            "total_episodes_processed": len(cooperation_skills_data),
            "episodes_with_coordination_patterns": sum(
                1
                for ep in cooperation_skills_data
                if ep["episode_cooperation"]["has_coordination_patterns"]
            ),
            "episodes_requiring_coordination": sum(
                1
                for ep in cooperation_skills_data
                if ep["episode_cooperation"]["both_require_coordination"]
            ),
            "episodes_in_rag_dataset": len(episodes_for_rag),
            "episodes_with_tom_reasoning": sum(
                1
                for ep in cooperation_skills_data
                if ep["episode_cooperation"]["has_tom_reasoning"]
            ),
            "episodes_with_complete_tom": sum(
                1
                for ep in cooperation_skills_data
                if ep["episode_cooperation"]["tom_cooperation_quality"][
                    "both_have_complete_tom"
                ]
            ),
            "avg_tom_completeness": sum(
                ep["episode_cooperation"]["tom_cooperation_quality"][
                    "avg_tom_completeness"
                ]
                for ep in cooperation_skills_data
                if ep["episode_cooperation"]["has_tom_reasoning"]
            )
            / max(
                1,
                sum(
                    1
                    for ep in cooperation_skills_data
                    if ep["episode_cooperation"]["has_tom_reasoning"]
                ),
            ),
        },
    }

    rag_data = {"episodes": episodes_for_rag, "metadata": metadata}

    json_gz_path = output_path / "react_trajectories.json.gz"
    with gzip.open(json_gz_path, "wt", encoding="utf-8") as f:
        json.dump(rag_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 保存RAG JSON文件: {json_gz_path}")

    # 打印统计信息
    total_episodes = len(cooperation_skills_data)
    episodes_with_coordination = sum(
        1
        for ep in cooperation_skills_data
        if ep["episode_cooperation"]["has_coordination_patterns"]
    )
    episodes_requiring_coordination = sum(
        1
        for ep in cooperation_skills_data
        if ep["episode_cooperation"]["both_require_coordination"]
    )

    episodes_with_tom = sum(
        1
        for ep in cooperation_skills_data
        if ep["episode_cooperation"]["has_tom_reasoning"]
    )
    episodes_with_complete_tom = sum(
        1
        for ep in cooperation_skills_data
        if ep["episode_cooperation"]["tom_cooperation_quality"][
            "both_have_complete_tom"
        ]
    )

    print("\n" + "=" * 80)
    print("✅ 提取完成!")
    print("=" * 80)
    print(f"总episodes处理数: {total_episodes}")
    print(
        f"有coordination patterns的episodes: {episodes_with_coordination} ({episodes_with_coordination/total_episodes*100:.1f}%)"
    )
    print(
        f"需要coordination的episodes: {episodes_requiring_coordination} ({episodes_requiring_coordination/total_episodes*100:.1f}%)"
    )
    print(
        f"有ToM reasoning的episodes: {episodes_with_tom} ({episodes_with_tom/total_episodes*100:.1f}%)"
    )
    print(
        f"有完整ToM（所有3步）的episodes: {episodes_with_complete_tom} ({episodes_with_complete_tom/total_episodes*100:.1f}%)"
    )
    print(f"RAG数据集中的episodes: {len(episodes_for_rag)}")
    print(f"输出目录: {output_path.absolute()}")
    print("=" * 80)

    return output_path


if __name__ == "__main__":
    import sys

    input_dir = "/home/shuqing/partnr-planner/outputs/habitat_llm/2025-12-22_10-26-56-heterogeneous+rerange_heurstic.json"
    output_dir = "data/rag_datasets/cooperation_skills_heterogeneous_rerange"
    filter_by_cooperation = True

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        filter_by_cooperation = sys.argv[3].lower() == "true"

    extract_cooperation_skills_to_rag_format(
        input_dir, output_dir, filter_by_cooperation
    )
