#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
from collections import defaultdict
from datetime import datetime


def organize_cooperation_skills_by_category():
    """将cooperation skills数据集按技能类别组织，格式类似rerange_only_organized_by_skills_enhanced"""

    input_file = "/home/shuqing/partnr-planner/data/rag_datasets/cooperation_skills_heterogeneous_rerange/react_trajectories.json.gz"
    output_dir = "/home/shuqing/partnr-planner/data/rag_datasets/cooperation_skills_heterogeneous_rerange_organized"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/by_complexity", exist_ok=True)
    os.makedirs(f"{output_dir}/by_task_type", exist_ok=True)

    print("=" * 80)
    print("组织Cooperation Skills数据集")
    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print()

    # 读取数据
    print("📖 读取数据...")
    with gzip.open(input_file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"]
    print(f"  ✓ 读取 {len(episodes)} 个episodes")

    # 组织episodes
    skill_based_datasets = defaultdict(list)
    task_type_datasets = defaultdict(list)
    complexity_datasets = defaultdict(list)

    print("\n🔄 处理episodes...")

    for episode in episodes:
        episode["episode_id"]
        cooperation_skills = episode.get("cooperation_skills", {})

        # 提取技能类别
        skill_categories = set()

        # 检查是否有coordination
        if (
            cooperation_skills.get("requires_coordination")
            or cooperation_skills.get("total_coordination_actions", 0) > 0
        ):
            skill_categories.add("Multi-Agent Coordination")

        # 检查是否有ToM reasoning
        tom_cooperation = cooperation_skills.get("tom_cooperation", {})
        if tom_cooperation.get("has_tom_reasoning"):
            skill_categories.add("Multi-Agent Coordination")
            skill_categories.add("Theory of Mind")

        # 从coordination requirements中提取其他技能类别
        agent_0_coord = cooperation_skills.get("agent_0_coordination", {})
        agent_1_coord = cooperation_skills.get("agent_1_coordination", {})

        # 如果有coordination actions，添加相关技能
        if (
            agent_0_coord.get("coordination_actions_count", 0) > 0
            or agent_1_coord.get("coordination_actions_count", 0) > 0
        ):
            skill_categories.add("Multi-Agent Coordination")

        # 如果没有明确的技能类别，至少添加Multi-Agent Coordination（因为这是cooperation skills数据集）
        if not skill_categories:
            skill_categories.add("Multi-Agent Coordination")

        # 转换episode格式，添加enhanced_skills字段
        enhanced_episode = convert_to_enhanced_format(episode)

        # 添加到技能类别数据集
        for skill_cat in skill_categories:
            skill_based_datasets[skill_cat].append(enhanced_episode)

        # 按任务类型组织（需要从instruction推断或使用默认值）
        task_type = infer_task_type(episode.get("instruction", ""))
        task_type_datasets[task_type].append(enhanced_episode)

        # 按复杂度组织（需要推断）
        complexity = infer_complexity(episode.get("instruction", ""))
        complexity_datasets[complexity].append(enhanced_episode)

    # 创建技能类别数据集
    print("\n📁 创建技能类别数据集...")
    for skill_category, episodes_list in skill_based_datasets.items():
        output_file = os.path.join(
            output_dir,
            f'skill_{skill_category.lower().replace(" ", "_").replace("-", "_")}.json.gz',
        )

        skill_dataset = {
            "metadata": {
                "skill_category": skill_category,
                "total_episodes": len(episodes_list),
                "source": "cooperation_skills_heterogeneous_rerange",
                "created_by": "organize_cooperation_skills_by_category.py",
                "organization_type": "skill_based",
                "description": f"Episodes focused on {skill_category} skills from heterogeneous+rerange_heurstic",
                "enhanced_with_cooperation_skills": True,
                "enhancement_date": datetime.now().strftime("%Y-%m-%d"),
            },
            "episodes": episodes_list,
        }

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(skill_dataset, f, indent=2, ensure_ascii=False)

        print(
            f"  ✓ {skill_category}: {len(episodes_list)} episodes -> {os.path.basename(output_file)}"
        )

    # 创建任务类型数据集
    print("\n📁 创建任务类型数据集...")
    for task_type, episodes_list in task_type_datasets.items():
        output_file = os.path.join(
            output_dir, "by_task_type", f'{task_type.lower().replace(" ", "_")}.json.gz'
        )

        task_dataset = {
            "metadata": {
                "task_type": task_type,
                "total_episodes": len(episodes_list),
                "source": "cooperation_skills_heterogeneous_rerange",
                "created_by": "organize_cooperation_skills_by_category.py",
                "organization_type": "task_type_based",
                "description": f"Episodes for {task_type} tasks with cooperation skills",
                "enhanced_with_cooperation_skills": True,
                "enhancement_date": datetime.now().strftime("%Y-%m-%d"),
            },
            "episodes": episodes_list,
        }

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(task_dataset, f, indent=2, ensure_ascii=False)

        print(
            f"  ✓ {task_type}: {len(episodes_list)} episodes -> {os.path.basename(output_file)}"
        )

    # 创建复杂度数据集
    print("\n📁 创建复杂度数据集...")
    for complexity, episodes_list in complexity_datasets.items():
        output_file = os.path.join(
            output_dir, "by_complexity", f"{complexity.lower()}_complexity.json.gz"
        )

        complexity_dataset = {
            "metadata": {
                "complexity": complexity,
                "total_episodes": len(episodes_list),
                "source": "cooperation_skills_heterogeneous_rerange",
                "created_by": "organize_cooperation_skills_by_category.py",
                "organization_type": "complexity_based",
                "description": f"{complexity} complexity episodes with cooperation skills",
                "enhanced_with_cooperation_skills": True,
                "enhancement_date": datetime.now().strftime("%Y-%m-%d"),
            },
            "episodes": episodes_list,
        }

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(complexity_dataset, f, indent=2, ensure_ascii=False)

        print(
            f"  ✓ {complexity} Complexity: {len(episodes_list)} episodes -> {os.path.basename(output_file)}"
        )

    # 创建总结报告
    summary_file = os.path.join(output_dir, "organization_summary.json")
    summary = {
        "original_dataset": {"total_episodes": len(episodes), "source": input_file},
        "skill_based_organization": {
            category: len(episodes_list)
            for category, episodes_list in skill_based_datasets.items()
        },
        "task_type_organization": {
            task_type: len(episodes_list)
            for task_type, episodes_list in task_type_datasets.items()
        },
        "complexity_organization": {
            complexity: len(episodes_list)
            for complexity, episodes_list in complexity_datasets.items()
        },
        "total_organized_files": len(skill_based_datasets)
        + len(task_type_datasets)
        + len(complexity_datasets),
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n✅ 组织完成!")
    print(f"创建了 {len(skill_based_datasets)} 个技能类别数据集")
    print(f"创建了 {len(task_type_datasets)} 个任务类型数据集")
    print(f"创建了 {len(complexity_datasets)} 个复杂度数据集")
    print(f"总结报告: {summary_file}")
    print("=" * 80)

    return output_dir


def convert_to_enhanced_format(episode):
    """将cooperation skills格式的episode转换为enhanced格式"""
    cooperation_skills = episode.get("cooperation_skills", {})

    # 提取coordination信息
    agent_0_coord = cooperation_skills.get("agent_0_coordination", {})
    agent_1_coord = cooperation_skills.get("agent_1_coordination", {})
    tom_cooperation = cooperation_skills.get("tom_cooperation", {})

    # 为每个agent构建enhanced_skills结构（格式：{'0': {...}, '1': {...}}）
    agent_enhanced_skills = {}
    episode_skill_patterns = []
    episode_decision_points = []

    for agent_id in ["0", "1"]:
        agent_coord = agent_0_coord if agent_id == "0" else agent_1_coord
        agent_tom = tom_cooperation.get(f"agent_{agent_id}_tom", {})

        # 构建coordination skill pattern
        skill_patterns = []
        if (
            agent_coord.get("requires_coordination")
            or agent_coord.get("coordination_actions_count", 0) > 0
        ):
            skill_patterns.append(
                {
                    "skill_name": "multi_agent_coordination",
                    "skill_type": "coordination",
                    "description": f"Agent {agent_id} demonstrates coordination with {agent_coord.get('coordination_actions_count', 0)} coordination actions",
                    "action_sequence": ["coordination_wait", "coordination_action"]
                    if agent_coord.get("wait_actions_count", 0) > 0
                    else ["coordination_action"],
                    "success_indicators": ["coordination_effective"]
                    if agent_coord.get("coordination_effectiveness")
                    else [],
                    "failure_indicators": [],
                    "contextual_conditions": {
                        "coordination_points": agent_coord.get(
                            "coordination_points", []
                        ),
                        "requires_coordination": agent_coord.get(
                            "requires_coordination", False
                        ),
                    },
                }
            )

        # 构建decision points
        decision_points = []
        coord_points = agent_coord.get("coordination_points", [])
        for point in coord_points[:5]:  # 限制数量
            decision_points.append(
                {
                    "step_number": point,
                    "decision_type": "coordination",
                    "context": f"Coordination point at step {point}",
                    "alternatives": ["wait", "proceed"],
                    "chosen_action": "Wait"
                    if agent_coord.get("wait_actions_count", 0) > 0
                    else "Proceed",
                    "outcome": "coordination"
                    if agent_coord.get("coordination_effectiveness")
                    else "unknown",
                    "reasoning": f"Agent {agent_id} coordinates with other agent",
                }
            )

        # 构建agent的enhanced_skills
        agent_enhanced_skills[agent_id] = {
            "enhanced_skill_description": f"Agent {agent_id} demonstrates multi-agent coordination with {agent_coord.get('coordination_actions_count', 0)} coordination actions",
            "skill_patterns": skill_patterns,
            "decision_points": decision_points,
            "coordination_requirements": agent_coord,
            "action_efficiency": {
                "efficiency_score": 0.8
                if agent_coord.get("coordination_effectiveness")
                else 0.5,
                "metrics": {
                    "success_rate": episode.get("success_rate", 0.0),
                    "coordination_actions": agent_coord.get(
                        "coordination_actions_count", 0
                    ),
                },
            },
            "skill_categories": ["Multi-Agent Coordination"],
            "learning_insights": [],
        }

        # 添加ToM信息
        if agent_tom.get("has_tom_reasoning"):
            agent_enhanced_skills[agent_id]["skill_categories"].append("Theory of Mind")
            agent_enhanced_skills[agent_id]["tom_reasoning"] = agent_tom

        # 累积episode级别的数据
        episode_skill_patterns.extend(skill_patterns)
        episode_decision_points.extend(decision_points)

    # 创建enhanced格式的episode
    enhanced_episode = {
        "episode_id": str(episode.get("episode_id", "")),
        "instruction": episode.get("instruction", ""),
        "success_rate": episode.get("success_rate", 0.0),
        "task_percent_complete": episode.get("task_percent_complete", 0.0),
        "format": "react",
        "task_type": infer_task_type(episode.get("instruction", "")),
        "complexity": infer_complexity(episode.get("instruction", "")),
        "cooperation_skills": cooperation_skills,
        "enhanced_skills": agent_enhanced_skills,  # 格式：{'0': {...}, '1': {...}}
        "skill_patterns": episode_skill_patterns,
        "decision_points": episode_decision_points,
        "coordination_analysis": {"0": agent_0_coord, "1": agent_1_coord},
        "enhanced_skill_categories": ["Multi-Agent Coordination"],
        "coordination_required": cooperation_skills.get("requires_coordination", False),
        "episode_quality_metrics": {
            "overall_quality": episode.get("success_rate", 0.0),
            "metrics": {
                "total_coordination_actions": cooperation_skills.get(
                    "total_coordination_actions", 0
                ),
                "coordination_effective": cooperation_skills.get(
                    "coordination_effective", False
                ),
            },
        },
        "episode_insights": [
            f"Requires {cooperation_skills.get('total_coordination_actions', 0)} coordination actions",
            "Demonstrates multi-agent coordination skills",
        ]
        if cooperation_skills.get("total_coordination_actions", 0) > 0
        else [],
        "enhancement_metadata": {
            "enhanced_with_cooperation_skills": True,
            "enhancement_date": datetime.now().strftime("%Y-%m-%d"),
            "source": "cooperation_skills_heterogeneous_rerange",
            "agents_processed": ["0", "1"],
            "total_patterns": len(episode_skill_patterns),
            "total_decisions": len(episode_decision_points),
        },
    }

    # 添加ToM相关的skill categories
    if tom_cooperation.get("has_tom_reasoning"):
        enhanced_episode["enhanced_skill_categories"].append("Theory of Mind")

    return enhanced_episode


def infer_task_type(instruction):
    """从instruction推断任务类型"""
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
    else:
        return "General Task"


def infer_complexity(instruction):
    """从instruction推断复杂度"""
    if not instruction:
        return "Low"

    import re

    instruction_lower = instruction.lower()

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
    output_dir = organize_cooperation_skills_by_category()
    print(f"\n组织完成！输出目录: {output_dir}")
