#!/usr/bin/env python3
"""
检查 trace 文件中是否包含 ToM 形式的推理
"""

import re
from collections import Counter
from pathlib import Path


def check_tom_in_trace(trace_content: str) -> dict:
    """检查 trace 内容是否包含 ToM 推理"""

    tom_indicators = {
        "has_thought": False,
        "has_belief_formation": False,
        "has_hypothesis": False,
        "has_prediction_planning": False,
        "has_tom_structure": False,
        "mentions_other_agent": False,
        "mentions_coordination": False,
    }

    # 检查是否有 Thought 部分
    if re.search(r"Thought:", trace_content, re.IGNORECASE):
        tom_indicators["has_thought"] = True

    # 检查 ToM 三步推理的关键词
    belief_patterns = [
        r"belief formation",
        r"what do they know",
        r"understanding.*?knowledge",
        r"spatial knowledge",
        r"behavioral patterns",
        r"knowledge gaps",
        r"what have they.*?observed",
    ]

    hypothesis_patterns = [
        r"hypothesis.*?generation",
        r"what is their.*?plan",
        r"inferring.*?intentions",
        r"primary hypothesis",
        r"strategic hypothesis",
        r"what are they.*?trying",
    ]

    prediction_patterns = [
        r"prediction.*?planning",
        r"what will they do",
        r"coordinating.*?actions",
        r"coordination strategy",
        r"complementary",
        r"sequential handoff",
        r"parallel preparation",
    ]

    for pattern in belief_patterns:
        if re.search(pattern, trace_content, re.IGNORECASE):
            tom_indicators["has_belief_formation"] = True
            break

    for pattern in hypothesis_patterns:
        if re.search(pattern, trace_content, re.IGNORECASE):
            tom_indicators["has_hypothesis"] = True
            break

    for pattern in prediction_patterns:
        if re.search(pattern, trace_content, re.IGNORECASE):
            tom_indicators["has_prediction_planning"] = True
            break

    # 检查是否提到其他 agent
    if re.search(
        r"other agent|another agent|the other agent", trace_content, re.IGNORECASE
    ):
        tom_indicators["mentions_other_agent"] = True

    # 检查是否提到 coordination
    if re.search(
        r"coordinate|cooperate|collaborate|coordination", trace_content, re.IGNORECASE
    ):
        tom_indicators["mentions_coordination"] = True

    # 检查是否有完整的 ToM 结构（至少包含两步）
    tom_steps = sum(
        [
            tom_indicators["has_belief_formation"],
            tom_indicators["has_hypothesis"],
            tom_indicators["has_prediction_planning"],
        ]
    )

    tom_indicators["has_tom_structure"] = tom_steps >= 2
    tom_indicators["tom_steps_count"] = tom_steps

    return tom_indicators


def check_tom_in_directory(input_dir: str):
    """检查目录中所有 trace 文件的 ToM 推理情况"""

    base_dir = Path(input_dir)
    traces_dir = (
        base_dir / "results" / "rerange+spatial_matched_subtasks.json.gz" / "traces"
    )

    if not traces_dir.exists():
        print(f"Error: Traces directory not found: {traces_dir}")
        return

    # 收集所有 trace 文件
    trace_files_0 = list((traces_dir / "0").glob("trace-episode_*_0-0.txt"))
    trace_files_1 = list((traces_dir / "1").glob("trace-episode_*_0-1.txt"))

    print(
        f"Found {len(trace_files_0)} agent_0 traces and {len(trace_files_1)} agent_1 traces"
    )
    print("=" * 80)

    # 统计信息
    stats = {
        "total_traces": 0,
        "has_thought": 0,
        "has_belief_formation": 0,
        "has_hypothesis": 0,
        "has_prediction_planning": 0,
        "has_tom_structure": 0,
        "mentions_other_agent": 0,
        "mentions_coordination": 0,
        "tom_steps_distribution": Counter(),
    }

    # 检查每个 trace 文件
    for trace_file in trace_files_0 + trace_files_1:
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace_content = f.read()

            tom_indicators = check_tom_in_trace(trace_content)

            stats["total_traces"] += 1
            if tom_indicators["has_thought"]:
                stats["has_thought"] += 1
            if tom_indicators["has_belief_formation"]:
                stats["has_belief_formation"] += 1
            if tom_indicators["has_hypothesis"]:
                stats["has_hypothesis"] += 1
            if tom_indicators["has_prediction_planning"]:
                stats["has_prediction_planning"] += 1
            if tom_indicators["has_tom_structure"]:
                stats["has_tom_structure"] += 1
            if tom_indicators["mentions_other_agent"]:
                stats["mentions_other_agent"] += 1
            if tom_indicators["mentions_coordination"]:
                stats["mentions_coordination"] += 1

            stats["tom_steps_distribution"][tom_indicators["tom_steps_count"]] += 1

        except Exception as e:
            print(f"Error reading {trace_file}: {e}")

    # 打印统计结果
    print("\n📊 ToM 推理检查结果:")
    print(f"   Total traces checked: {stats['total_traces']}")
    print("\n   Basic indicators:")
    print(
        f"   - Has Thought section: {stats['has_thought']} ({stats['has_thought']/stats['total_traces']*100:.1f}%)"
    )
    print(
        f"   - Mentions other agent: {stats['mentions_other_agent']} ({stats['mentions_other_agent']/stats['total_traces']*100:.1f}%)"
    )
    print(
        f"   - Mentions coordination: {stats['mentions_coordination']} ({stats['mentions_coordination']/stats['total_traces']*100:.1f}%)"
    )

    print("\n   ToM Structure indicators:")
    print(
        f"   - Has Belief Formation: {stats['has_belief_formation']} ({stats['has_belief_formation']/stats['total_traces']*100:.1f}%)"
    )
    print(
        f"   - Has Hypothesis Generation: {stats['has_hypothesis']} ({stats['has_hypothesis']/stats['total_traces']*100:.1f}%)"
    )
    print(
        f"   - Has Prediction & Planning: {stats['has_prediction_planning']} ({stats['has_prediction_planning']/stats['total_traces']*100:.1f}%)"
    )
    print(
        f"   - Has ToM Structure (≥2 steps): {stats['has_tom_structure']} ({stats['has_tom_structure']/stats['total_traces']*100:.1f}%)"
    )

    print("\n   ToM Steps Distribution:")
    for steps in sorted(stats["tom_steps_distribution"].keys()):
        count = stats["tom_steps_distribution"][steps]
        print(
            f"   - {steps} ToM steps: {count} traces ({count/stats['total_traces']*100:.1f}%)"
        )

    # 检查是否有完整的 ToM 推理
    complete_tom = sum(
        1 for count in stats["tom_steps_distribution"].items() if count[0] == 3
    )

    if complete_tom == 0:
        print(
            "\n⚠️  Warning: No traces found with complete ToM structure (all 3 steps)"
        )
        print(
            "   This suggests the traces may not use ToM prompts, or use a simplified version."
        )
    else:
        print(
            f"\n✅ Found {complete_tom} traces with complete ToM structure (all 3 steps)"
        )


if __name__ == "__main__":
    import sys

    input_dir = "/home/shuqing/partnr-planner/outputs/habitat_llm/org_re_sp"

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]

    check_tom_in_directory(input_dir)
