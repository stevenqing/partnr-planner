#!/usr/bin/env python3
"""
从 org_re_sp 目录中提取 cooperation skills
"""

import csv
import gzip
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

from tqdm import tqdm

# 导入 enhanced_skill_extractor
try:
    from enhanced_skill_extractor import EnhancedSkillExtractor
except ImportError:
    print(
        "Error: enhanced_skill_extractor.py not found. Please ensure it's in the same directory."
    )
    exit(1)


def extract_tom_cooperation_skills(
    trace_content: str, agent_id: str, instruction: str
) -> Dict:
    """
    从 trace 内容中提取基于 Theory of Mind 的 cooperation skills

    支持两种形式：
    1. 完整的 ToM 三步推理（Belief Formation, Hypothesis Generation, Prediction & Planning）
    2. 简化的 ToM 推理（从 Thought 中推断 cooperation patterns）
    """
    import re

    tom_data = {
        "has_tom_reasoning": False,
        "tom_type": "none",  # 'full', 'simplified', 'none'
        "tom_steps": {
            "belief_formation": [],
            "hypothesis_generation": [],
            "prediction_planning": [],
        },
        "simplified_tom_reasoning": [],
        "tom_cooperation_patterns": [],
        "tom_quality_metrics": {},
    }

    # 查找 Thought 部分
    thought_pattern = (
        r"Thought:?\s*(.*?)(?=\n\s*(?:Agent_\d+_Action|Done|Exit|Result:|$))"
    )
    thoughts = re.findall(thought_pattern, trace_content, re.DOTALL | re.IGNORECASE)

    if not thoughts:
        return tom_data

    tom_data["has_tom_reasoning"] = True

    # 检查是否有完整的 ToM 结构
    has_full_tom = False

    # 提取 ToM 三步推理
    for thought in thoughts:
        thought_lower = thought.lower()

        # Step 1: Belief Formation
        belief_patterns = [
            r"belief formation[:\-]?\s*(.*?)(?=\n\s*(?:hypothesis|prediction|planning|$))",
            r"what do they know[:\-]?\s*(.*?)(?=\n\s*(?:hypothesis|prediction|planning|$))",
            r"understanding.*?knowledge[:\-]?\s*(.*?)(?=\n\s*(?:hypothesis|prediction|planning|$))",
            r"spatial knowledge[:\-]?\s*(.*?)(?=\n\s*(?:hypothesis|prediction|planning|$))",
            r"behavioral patterns[:\-]?\s*(.*?)(?=\n\s*(?:hypothesis|prediction|planning|$))",
        ]

        for pattern in belief_patterns:
            matches = re.findall(pattern, thought, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if match.strip():
                    tom_data["tom_steps"]["belief_formation"].append(
                        {
                            "content": match.strip(),
                            "mentions_cooperation": any(
                                word in match.lower()
                                for word in [
                                    "coordinate",
                                    "cooperate",
                                    "other agent",
                                    "collaborate",
                                    "together",
                                ]
                            ),
                        }
                    )

        # Step 2: Hypothesis Generation
        hypothesis_patterns = [
            r"hypothesis[:\-]?\s*(.*?)(?=\n\s*(?:prediction|planning|$))",
            r"what is their.*?plan[:\-]?\s*(.*?)(?=\n\s*(?:prediction|planning|$))",
            r"inferring.*?intentions[:\-]?\s*(.*?)(?=\n\s*(?:prediction|planning|$))",
            r"primary hypothesis[:\-]?\s*(.*?)(?=\n\s*(?:prediction|planning|$))",
            r"strategic hypothesis[:\-]?\s*(.*?)(?=\n\s*(?:prediction|planning|$))",
        ]

        for pattern in hypothesis_patterns:
            matches = re.findall(pattern, thought, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if match.strip():
                    tom_data["tom_steps"]["hypothesis_generation"].append(
                        {
                            "content": match.strip(),
                            "mentions_cooperation": any(
                                word in match.lower()
                                for word in [
                                    "coordinate",
                                    "cooperate",
                                    "other agent",
                                    "collaborate",
                                    "together",
                                    "complement",
                                ]
                            ),
                        }
                    )

        # Step 3: Prediction & Planning
        prediction_patterns = [
            r"prediction.*?planning[:\-]?\s*(.*?)(?=\n\s*(?:Agent_\d+_Action|Done|Exit|$))",
            r"what will they do[:\-]?\s*(.*?)(?=\n\s*(?:Agent_\d+_Action|Done|Exit|$))",
            r"coordinating.*?actions[:\-]?\s*(.*?)(?=\n\s*(?:Agent_\d+_Action|Done|Exit|$))",
            r"coordination strategy[:\-]?\s*(.*?)(?=\n\s*(?:Agent_\d+_Action|Done|Exit|$))",
            r"complementary[:\-]?\s*(.*?)(?=\n\s*(?:Agent_\d+_Action|Done|Exit|$))",
        ]

        for pattern in prediction_patterns:
            matches = re.findall(pattern, thought, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if match.strip():
                    # 识别 coordination strategy 类型
                    strategy_type = "unknown"
                    if "complementary" in match.lower():
                        strategy_type = "complementary"
                    elif "sequential" in match.lower() or "handoff" in match.lower():
                        strategy_type = "sequential_handoff"
                    elif "parallel" in match.lower():
                        strategy_type = "parallel_preparation"
                    elif "exploratory" in match.lower() or "explore" in match.lower():
                        strategy_type = "exploratory"

                    tom_data["tom_steps"]["prediction_planning"].append(
                        {
                            "content": match.strip(),
                            "strategy_type": strategy_type,
                            "mentions_cooperation": any(
                                word in match.lower()
                                for word in [
                                    "coordinate",
                                    "cooperate",
                                    "other agent",
                                    "collaborate",
                                    "together",
                                    "complement",
                                    "handoff",
                                    "avoid conflict",
                                ]
                            ),
                        }
                    )

    # 检查是否有完整的 ToM 结构
    has_full_tom = (
        len(tom_data["tom_steps"]["belief_formation"]) > 0
        or len(tom_data["tom_steps"]["hypothesis_generation"]) > 0
        or len(tom_data["tom_steps"]["prediction_planning"]) > 0
    )

    if has_full_tom:
        tom_data["tom_type"] = "full"
    else:
        # 如果没有完整的 ToM 结构，尝试从简化的 Thought 中提取
        tom_data["tom_type"] = "simplified"

        # 提取简化的 ToM 推理（从 Thought 中推断）
        for _i, thought in enumerate(thoughts):
            thought_lower = thought.lower()

            # 检查是否提到其他 agent
            if "other agent" in thought_lower or "another agent" in thought_lower:
                # 尝试推断简化的 ToM 步骤
                # Step 1: 推断 Belief Formation（观察其他 agent 的行为）
                if any(
                    word in thought_lower
                    for word in ["holding", "picked", "placed", "moved", "found"]
                ):
                    tom_data["simplified_tom_reasoning"].append(
                        {
                            "step": "belief_formation",
                            "content": thought.strip(),
                            "inferred_from": "observation_of_other_agent_action",
                        }
                    )

                # Step 2: 推断 Hypothesis Generation（推断其他 agent 的意图）
                if any(
                    word in thought_lower
                    for word in [
                        "trying",
                        "will",
                        "should",
                        "likely",
                        "probably",
                        "seems",
                    ]
                ):
                    tom_data["simplified_tom_reasoning"].append(
                        {
                            "step": "hypothesis_generation",
                            "content": thought.strip(),
                            "inferred_from": "inference_about_other_agent_intent",
                        }
                    )

                # Step 3: 推断 Prediction & Planning（规划协调行动）
                if any(
                    word in thought_lower
                    for word in [
                        "wait",
                        "move on",
                        "different",
                        "instead",
                        "complement",
                    ]
                ):
                    tom_data["simplified_tom_reasoning"].append(
                        {
                            "step": "prediction_planning",
                            "content": thought.strip(),
                            "inferred_from": "coordination_planning",
                        }
                    )

    # 提取 ToM cooperation patterns
    cooperation_keywords = [
        "coordinate",
        "cooperate",
        "collaborate",
        "complement",
        "other agent",
        "another agent",
        "together",
        "handoff",
        "avoid conflict",
        "wait",
        "different part",
        "move on",
    ]

    for thought in thoughts:
        thought_lower = thought.lower()
        if any(keyword in thought_lower for keyword in cooperation_keywords):
            # 提取包含 cooperation 的完整句子
            sentences = re.split(r"[.!?]\s+", thought)
            cooperation_sentences = [
                s.strip()
                for s in sentences
                if any(keyword in s.lower() for keyword in cooperation_keywords)
            ]

            for sentence in cooperation_sentences:
                if sentence:
                    # 推断这个 pattern 属于哪个 ToM 步骤
                    inferred_step = "unknown"
                    if any(
                        word in sentence.lower()
                        for word in ["holding", "picked", "placed", "found"]
                    ):
                        inferred_step = "belief_formation"
                    elif any(
                        word in sentence.lower()
                        for word in ["trying", "will", "likely", "probably"]
                    ):
                        inferred_step = "hypothesis_generation"
                    elif any(
                        word in sentence.lower()
                        for word in ["wait", "move on", "different", "instead"]
                    ):
                        inferred_step = "prediction_planning"

                    tom_data["tom_cooperation_patterns"].append(
                        {
                            "pattern": sentence,
                            "inferred_tom_step": inferred_step,
                            "contains_belief_formation": any(
                                bf["mentions_cooperation"]
                                for bf in tom_data["tom_steps"]["belief_formation"]
                            )
                            or any(
                                s["step"] == "belief_formation"
                                for s in tom_data["simplified_tom_reasoning"]
                            ),
                            "contains_hypothesis": any(
                                hg["mentions_cooperation"]
                                for hg in tom_data["tom_steps"]["hypothesis_generation"]
                            )
                            or any(
                                s["step"] == "hypothesis_generation"
                                for s in tom_data["simplified_tom_reasoning"]
                            ),
                            "contains_prediction": any(
                                pp["mentions_cooperation"]
                                for pp in tom_data["tom_steps"]["prediction_planning"]
                            )
                            or any(
                                s["step"] == "prediction_planning"
                                for s in tom_data["simplified_tom_reasoning"]
                            ),
                        }
                    )

    # 计算 ToM 质量指标
    # 完整的 ToM 步骤
    full_tom_steps = (
        len(tom_data["tom_steps"]["belief_formation"])
        + len(tom_data["tom_steps"]["hypothesis_generation"])
        + len(tom_data["tom_steps"]["prediction_planning"])
    )

    # 简化的 ToM 步骤
    simplified_tom_steps = len(tom_data["simplified_tom_reasoning"])

    # 总 ToM 步骤
    total_tom_steps = full_tom_steps + simplified_tom_steps

    # Cooperation 相关的步骤
    cooperation_related_steps = (
        sum(
            1
            for bf in tom_data["tom_steps"]["belief_formation"]
            if bf["mentions_cooperation"]
        )
        + sum(
            1
            for hg in tom_data["tom_steps"]["hypothesis_generation"]
            if hg["mentions_cooperation"]
        )
        + sum(
            1
            for pp in tom_data["tom_steps"]["prediction_planning"]
            if pp["mentions_cooperation"]
        )
        + len(tom_data["simplified_tom_reasoning"])  # 简化的推理都涉及其他 agent
    )

    tom_data["tom_quality_metrics"] = {
        "tom_type": tom_data["tom_type"],
        "full_tom_steps": full_tom_steps,
        "simplified_tom_steps": simplified_tom_steps,
        "total_tom_steps": total_tom_steps,
        "cooperation_related_steps": cooperation_related_steps,
        "tom_completeness": 1.0
        if full_tom_steps >= 3
        else (total_tom_steps / 3.0 if total_tom_steps > 0 else 0.0),
        "cooperation_focus": cooperation_related_steps / total_tom_steps
        if total_tom_steps > 0
        else 0.0,
        "has_all_three_steps": (
            len(tom_data["tom_steps"]["belief_formation"]) > 0
            and len(tom_data["tom_steps"]["hypothesis_generation"]) > 0
            and len(tom_data["tom_steps"]["prediction_planning"]) > 0
        ),
        "has_simplified_tom": simplified_tom_steps > 0,
        "cooperation_patterns_count": len(tom_data["tom_cooperation_patterns"]),
        "mentions_other_agent": any(
            "other agent" in thought.lower() or "another agent" in thought.lower()
            for thought in thoughts
        ),
    }

    return tom_data


def extract_cooperation_skills_to_rag_format(
    input_dir: str,
    output_dir: str = "data/rag_datasets/cooperation_skills_org_re_sp",
    filter_by_cooperation: bool = True,
):
    """
    从指定目录中提取 cooperation skills 并整理成 RAG 数据集格式

    Args:
        input_dir: 输入目录路径 (org_re_sp/results/...)
        output_dir: 输出 RAG 数据集目录路径
        filter_by_cooperation: 是否只保留有 cooperation skills 的 episodes
    """
    # 初始化 skill extractor
    extractor = EnhancedSkillExtractor(use_llm=False, cache_results=False)

    # 确定路径
    base_dir = Path(input_dir)
    traces_dir = (
        base_dir / "results" / "rerange+spatial_matched_subtasks.json.gz" / "traces"
    )
    episode_log = base_dir / "results" / "episode_result_log.csv"

    if not traces_dir.exists():
        print(f"Error: Traces directory not found: {traces_dir}")
        return

    # 创建输出目录结构
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "react_trajectories" / "traces" / "0").mkdir(
        parents=True, exist_ok=True
    )
    (output_path / "react_trajectories" / "traces" / "1").mkdir(
        parents=True, exist_ok=True
    )

    print(f"📁 Output directory: {output_path.absolute()}")

    # 读取 episode 信息
    episodes_info = {}
    if episode_log.exists():
        with open(episode_log, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episode_id = row["episode_id"]
                episodes_info[episode_id] = {
                    "instruction": row["instruction"],
                    "task_state_success": row.get("task_state_success", "0"),
                    "task_percent_complete": row.get("task_percent_complete", "0"),
                }

    # 收集所有 trace 文件
    trace_files_0 = list((traces_dir / "0").glob("trace-episode_*_0-0.txt"))
    trace_files_1 = list((traces_dir / "1").glob("trace-episode_*_0-1.txt"))

    # 提取 episode IDs
    episode_ids = set()
    for trace_file in trace_files_0 + trace_files_1:
        # 从文件名提取 episode_id: trace-episode_{id}_0-{agent_id}.txt
        parts = trace_file.stem.split("_")
        if len(parts) >= 2:
            episode_id = parts[1]  # episode_1005 -> 1005
            episode_ids.add(episode_id)

    print(f"Found {len(episode_ids)} episodes")

    # 处理每个 episode
    cooperation_skills_data = []
    episodes_for_rag = []
    episode_result_log_rows = []

    pbar = tqdm(sorted(episode_ids), desc="Extracting cooperation skills")

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
                    print(
                        f"\nWarning: Error processing episode {episode_id}, agent {agent_id}: {e}"
                    )
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
    skills_json_path = output_path / "cooperation_skills_detailed.json"
    with open(skills_json_path, "w", encoding="utf-8") as f:
        json.dump(cooperation_skills_data, f, indent=2, ensure_ascii=False)

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

    # 创建 react_trajectories.json.gz
    metadata = {
        "total_episodes": len(episodes_for_rag),
        "source": "org_re_sp cooperation skills extraction",
        "created_by": "extract_cooperation_skills.py",
        "example_type": "react",
        "description": f"Cooperation skills dataset extracted from org_re_sp with Theory of Mind (ToM) analysis ({len(episodes_for_rag)} episodes)",
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

    print(
        f"\n✅ Extracted cooperation skills from {len(cooperation_skills_data)} episodes"
    )
    print(f"📁 RAG dataset saved to: {output_path.absolute()}")
    print(f"   - {len(episodes_for_rag)} episodes in RAG dataset")
    print("   - Trace files copied to: react_trajectories/traces/")
    print("   - Episode log: episode_result_log.csv")
    print("   - Metadata: react_trajectories.json.gz")
    print("   - Detailed skills: cooperation_skills_detailed.json")

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

    print("\n📊 Statistics:")
    print(f"   Total episodes processed: {total_episodes}")
    print(
        f"   Episodes with coordination patterns: {episodes_with_coordination} ({episodes_with_coordination/total_episodes*100:.1f}%)"
    )
    print(
        f"   Episodes requiring coordination: {episodes_requiring_coordination} ({episodes_requiring_coordination/total_episodes*100:.1f}%)"
    )
    print(
        f"   Episodes with ToM reasoning: {episodes_with_tom} ({episodes_with_tom/total_episodes*100:.1f}%)"
    )
    print(
        f"   Episodes with complete ToM (all 3 steps): {episodes_with_complete_tom} ({episodes_with_complete_tom/total_episodes*100:.1f}%)"
    )
    print(f"   Episodes in RAG dataset: {len(episodes_for_rag)}")


if __name__ == "__main__":
    import sys

    input_dir = "/home/shuqing/partnr-planner/outputs/habitat_llm/org_re_sp"
    output_dir = "data/rag_datasets/cooperation_skills_org_re_sp"
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
