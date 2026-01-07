#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os

# 导入LLM
import sys
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from habitat_llm.llm import instantiate_llm
except ImportError:
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "habitat_llm",
            os.path.join(os.path.dirname(__file__), "habitat_llm", "llm.py"),
        )
        if spec and spec.loader:
            habitat_llm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(habitat_llm)
            instantiate_llm = habitat_llm.instantiate_llm
        else:
            raise ImportError("Cannot find habitat_llm")
    except Exception as e:
        print(f"Warning: Cannot import LLM: {e}")
        print("Will use rule-based classification instead")
        instantiate_llm = None

# 定义5种cooperation skills类型
COOPERATION_SKILL_TYPES = {
    "yield_path": {
        "name": "Yield Path",
        "description": "Agent moves to clear position when partner is navigating toward shared space",
        "trigger": "Partner navigating toward shared space",
        "precondition": "Self occupying shared space",
        "response": "Move to clear position",
        "joint_effect": "Collision avoidance",
    },
    "receive_handoff": {
        "name": "Receive Handoff",
        "description": "Agent positions for transfer when partner is holding object and approaching",
        "trigger": "Partner holding object, approaching",
        "precondition": "Self hands free, nearby",
        "response": "Position for transfer",
        "joint_effect": "Object transfer",
    },
    "assist_placement": {
        "name": "Assist Placement",
        "description": "Agent clears surface when partner is placing object at location",
        "trigger": "Partner placing object at ℓ",
        "precondition": "Self near ℓ, hands free",
        "response": "Clear surface at ℓ",
        "joint_effect": "Successful placement",
    },
    "coordinate_order": {
        "name": "Coordinate Order",
        "description": "Agent proceeds to location when partner is waiting and task requires that location first",
        "trigger": "Partner waiting at location ℓ",
        "precondition": "Self task requires ℓ first",
        "response": "Proceed to ℓ",
        "joint_effect": "Temporal coordination",
    },
    "divide_task": {
        "name": "Divide Task",
        "description": "Agent claims different subtask when partner claims a subtask",
        "trigger": "Partner claiming subtask t₁",
        "precondition": "Self capable of subtask t₂",
        "response": "Claim subtask t₂",
        "joint_effect": "Parallel execution",
    },
}


def classify_cooperation_skills_with_llm():
    """使用LLM分析cooperation skills的类型"""

    input_file = "/home/shuqing/partnr-planner/data/rag_datasets/cooperation_skills_heterogeneous_rerange/react_trajectories.json.gz"
    output_file = "/home/shuqing/partnr-planner/data/rag_datasets/cooperation_skills_heterogeneous_rerange/react_trajectories_classified.json.gz"

    print("=" * 80)
    print("使用LLM分类Cooperation Skills")
    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()

    # 初始化LLM
    print("🤖 初始化LLM...")
    llm = None
    if instantiate_llm:
        try:
            llm = instantiate_llm(
                "llama",
                generation_params={
                    "engine": "meta-llama/Llama-3.1-8B-Instruct",
                    "max_tokens": 300,
                    "temperature": 0.0,
                },
            )
            print("  ✓ LLM初始化成功")
        except Exception as e:
            print(f"  ⚠️  LLM初始化失败: {e}")
            print("  使用fallback方法（基于规则）")
            llm = None
    else:
        print("  ⚠️  LLM不可用，使用基于规则的分类")

    # 读取数据
    print("\n📖 读取数据...")
    with gzip.open(input_file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"]
    print(f"  ✓ 读取 {len(episodes)} 个episodes")

    # 处理每个episode
    print("\n🔄 分析cooperation skills类型...")
    classified_episodes = []

    for episode in tqdm(episodes, desc="分类cooperation skills"):
        classified_episode = classify_episode_cooperation_skills(episode, llm)
        classified_episodes.append(classified_episode)

    # 更新metadata
    metadata = data["metadata"].copy()
    metadata.update(
        {
            "classified_with_llm": True,
            "classification_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cooperation_skill_types": list(COOPERATION_SKILL_TYPES.keys()),
            "classification_method": "llm" if llm else "rule_based",
        }
    )

    # 统计信息
    skill_type_counts = {}
    for skill_type in COOPERATION_SKILL_TYPES:
        count = sum(
            1
            for ep in classified_episodes
            if ep.get("cooperation_skills", {})
            .get("classified_skill_types", {})
            .get(skill_type, False)
        )
        skill_type_counts[skill_type] = count

    metadata["skill_type_statistics"] = skill_type_counts

    # 保存结果
    print("\n💾 保存分类结果...")
    output_data = {"metadata": metadata, "episodes": classified_episodes}

    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 保存到: {output_file}")

    # 打印统计信息
    print("\n📊 分类统计:")
    for skill_type, count in skill_type_counts.items():
        percentage = (
            count / len(classified_episodes) * 100 if classified_episodes else 0
        )
        print(
            f"  - {COOPERATION_SKILL_TYPES[skill_type]['name']}: {count} ({percentage:.1f}%)"
        )

    print("\n" + "=" * 80)
    print("✅ 分类完成!")
    print("=" * 80)

    return output_file


def classify_episode_cooperation_skills(episode, llm):
    """分类单个episode的cooperation skills"""
    cooperation_skills = episode.get("cooperation_skills", {})
    instruction = episode.get("instruction", "")

    # 提取coordination信息
    agent_0_coord = cooperation_skills.get("agent_0_coordination", {})
    agent_1_coord = cooperation_skills.get("agent_1_coordination", {})
    tom_cooperation = cooperation_skills.get("tom_cooperation", {})

    # 构建分析文本
    analysis_text = build_cooperation_analysis_text(
        instruction, agent_0_coord, agent_1_coord, tom_cooperation
    )

    # 使用LLM分类
    if llm:
        classified_types = classify_with_llm(analysis_text, instruction, llm)
    else:
        classified_types = classify_with_rules(
            agent_0_coord, agent_1_coord, tom_cooperation
        )

    # 更新episode
    classified_episode = episode.copy()
    cooperation_skills["classified_skill_types"] = classified_types
    cooperation_skills["cooperation_analysis"] = analysis_text
    classified_episode["cooperation_skills"] = cooperation_skills

    return classified_episode


def build_cooperation_analysis_text(
    instruction, agent_0_coord, agent_1_coord, tom_cooperation
):
    """构建cooperation分析文本"""
    analysis_parts = []

    analysis_parts.append(f"Task: {instruction}")
    analysis_parts.append("\nAgent 0 Coordination:")
    analysis_parts.append(
        f"  - Requires coordination: {agent_0_coord.get('requires_coordination', False)}"
    )
    analysis_parts.append(
        f"  - Coordination actions: {agent_0_coord.get('coordination_actions_count', 0)}"
    )
    analysis_parts.append(
        f"  - Wait actions: {agent_0_coord.get('wait_actions_count', 0)}"
    )
    analysis_parts.append(
        f"  - Coordination points: {len(agent_0_coord.get('coordination_points', []))}"
    )

    analysis_parts.append("\nAgent 1 Coordination:")
    analysis_parts.append(
        f"  - Requires coordination: {agent_1_coord.get('requires_coordination', False)}"
    )
    analysis_parts.append(
        f"  - Coordination actions: {agent_1_coord.get('coordination_actions_count', 0)}"
    )
    analysis_parts.append(
        f"  - Wait actions: {agent_1_coord.get('wait_actions_count', 0)}"
    )
    analysis_parts.append(
        f"  - Coordination points: {len(agent_1_coord.get('coordination_points', []))}"
    )

    if tom_cooperation.get("has_tom_reasoning"):
        analysis_parts.append("\nToM Reasoning: Yes")
        analysis_parts.append(
            f"  - ToM patterns: {tom_cooperation.get('tom_quality', {}).get('total_tom_cooperation_patterns', 0)}"
        )

    return "\n".join(analysis_parts)


def classify_with_llm(analysis_text, instruction, llm):
    """使用LLM分类cooperation skills类型"""

    # 构建prompt
    skill_types_description = "\n".join(
        [
            f"{i+1}. {skill_type}: {info['description']}"
            for i, (skill_type, info) in enumerate(COOPERATION_SKILL_TYPES.items())
        ]
    )

    prompt = f"""Analyze the following multi-agent cooperation scenario and classify which cooperation skill types are present.

Cooperation Skill Types:
{skill_types_description}

Scenario:
{analysis_text}

Task: {instruction}

Based on the coordination patterns, identify which cooperation skill types are demonstrated.
Respond with a JSON object containing boolean values for each skill type:
{{
  "yield_path": true/false,
  "receive_handoff": true/false,
  "assist_placement": true/false,
  "coordinate_order": true/false,
  "divide_task": true/false
}}

Only include skill types that are clearly present. Be conservative - only mark as true if there is clear evidence.

JSON Response:"""

    try:
        response = llm.generate(prompt, max_length=300)

        # 解析JSON响应
        import re

        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            classified = json.loads(json_str)
            return classified
        else:
            # Fallback to rule-based
            return classify_with_rules_from_text(analysis_text)
    except Exception as e:
        print(f"  ⚠️  LLM分类失败: {e}")
        return classify_with_rules_from_text(analysis_text)


def classify_with_rules(agent_0_coord, agent_1_coord, tom_cooperation):
    """基于规则的分类（fallback方法）"""
    classified = {
        "yield_path": False,
        "receive_handoff": False,
        "assist_placement": False,
        "coordinate_order": False,
        "divide_task": False,
    }

    # 检查wait actions（可能表示yield_path或coordinate_order）
    total_wait = agent_0_coord.get("wait_actions_count", 0) + agent_1_coord.get(
        "wait_actions_count", 0
    )
    if total_wait > 0:
        # 如果有wait actions，可能是yield_path或coordinate_order
        classified["yield_path"] = True
        classified["coordinate_order"] = True

    # 检查coordination actions（可能表示各种cooperation）
    total_coord = agent_0_coord.get(
        "coordination_actions_count", 0
    ) + agent_1_coord.get("coordination_actions_count", 0)
    if total_coord > 5:
        # 如果有大量coordination actions，可能涉及多种类型
        classified["receive_handoff"] = True
        classified["assist_placement"] = True

    # 如果两个agent都需要coordination，可能是divide_task
    if agent_0_coord.get("requires_coordination") and agent_1_coord.get(
        "requires_coordination"
    ):
        classified["divide_task"] = True

    return classified


def classify_with_rules_from_text(analysis_text):
    """从分析文本中基于规则分类"""
    text_lower = analysis_text.lower()

    classified = {
        "yield_path": False,
        "receive_handoff": False,
        "assist_placement": False,
        "coordinate_order": False,
        "divide_task": False,
    }

    # 检查关键词
    if "wait" in text_lower and "coordination" in text_lower:
        classified["yield_path"] = True
        classified["coordinate_order"] = True

    if "transfer" in text_lower or "handoff" in text_lower:
        classified["receive_handoff"] = True

    if "place" in text_lower and "clear" in text_lower:
        classified["assist_placement"] = True

    if "coordination actions" in text_lower:
        coord_count = 0
        import re

        match = re.search(r"coordination actions:\s*(\d+)", text_lower)
        if match:
            coord_count = int(match.group(1))
            if coord_count > 5:
                classified["divide_task"] = True

    return classified


if __name__ == "__main__":
    output_file = classify_cooperation_skills_with_llm()
    print(f"\n分类完成！输出文件: {output_file}")
