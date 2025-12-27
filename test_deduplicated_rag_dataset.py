#!/usr/bin/env python3
"""
测试去重后的RAG数据集质量

该脚本会：
1. 验证数据集格式正确性
2. 测试RAG检索功能
3. 分析轨迹质量和多样性
4. 提供质量报告

使用方法:
python test_deduplicated_rag_dataset.py \
    --rag-dir data/rag_datasets/react_rag_dataset_dedup \
    --data-source-name react_trajectories
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict


def analyze_trajectory_quality(content: str) -> Dict[str, Any]:
    """分析单个轨迹的质量"""
    lines = content.split("\n")

    # 统计基本信息
    total_lines = len(lines)
    action_lines = [line for line in lines if "Agent_0_Action:" in line]
    observation_lines = [line for line in lines if "Agent_0_Observation:" in line]
    failure_lines = [line for line in lines if "Unexpected failure!" in line]

    # 提取动作类型
    actions = []
    for line in action_lines:
        if "Agent_0_Action:" in line:
            action = line.split("Agent_0_Action:")[1].strip()
            action_type = action.split("[")[0] if "[" in action else action
            actions.append(action_type)

    # 检查完成状态
    has_completion = any(
        keyword in content for keyword in ["Done[]", "Exit!", "Done[", "Exit["]
    )

    # 计算质量指标
    efficiency = len(action_lines)  # 动作数量（越少越好）
    reliability = len(failure_lines)  # 失败次数（越少越好）
    completeness = 1.0 if has_completion else 0.0

    return {
        "total_lines": total_lines,
        "action_count": len(action_lines),
        "observation_count": len(observation_lines),
        "failure_count": len(failure_lines),
        "actions": actions,
        "action_types": list(set(actions)),
        "has_completion": has_completion,
        "efficiency_score": max(0, 1.0 - (efficiency - 5) / 50.0),
        "reliability_score": max(0, 1.0 - reliability / 10.0),
        "completeness_score": completeness,
    }


def test_rag_dataset_structure(rag_dir: str, data_source_name: str) -> Dict[str, Any]:
    """测试RAG数据集结构"""
    print("🔍 测试RAG数据集结构...")

    results = {
        "structure_valid": True,
        "files_found": [],
        "missing_files": [],
        "errors": [],
    }

    # 检查必要文件
    required_files = [
        "episode_result_log.csv",
        f"{data_source_name}/traces/0",
        f"{data_source_name}/traces/1",
    ]

    for file_path in required_files:
        full_path = os.path.join(rag_dir, file_path)
        if os.path.exists(full_path):
            results["files_found"].append(file_path)
        else:
            results["missing_files"].append(file_path)
            results["structure_valid"] = False

    # 检查轨迹文件
    for agent_id in [0, 1]:
        traces_dir = os.path.join(rag_dir, data_source_name, "traces", str(agent_id))
        if os.path.exists(traces_dir):
            trace_files = [f for f in os.listdir(traces_dir) if f.endswith(".txt")]
            results[f"agent_{agent_id}_traces"] = len(trace_files)
        else:
            results[f"agent_{agent_id}_traces"] = 0

    return results


def test_rag_functionality(rag_dir: str, data_source_name: str) -> Dict[str, Any]:
    """测试RAG检索功能"""
    print("🔍 测试RAG检索功能...")

    try:
        # 导入habitat_llm的RAG类
        from types import SimpleNamespace

        from habitat_llm.planner.rag import RAG

        # 创建虚拟LLM配置
        llm_config = SimpleNamespace(
            system_tag="<|start_header_id|>system<|end_header_id|>\n",
            user_tag="<|start_header_id|>user<|end_header_id|>\n",
            assistant_tag="<|start_header_id|>assistant<|end_header_id|>\n",
            eot_tag="<|eot_id|>\n",
        )

        # 初始化RAG
        rag = RAG(
            example_type="react",
            data_dir=[rag_dir],
            data_source_name=[data_source_name],
            llm_config=llm_config,
        )

        # 测试检索
        test_queries = [
            "Move the apple to the table",
            "Put the toy on the shelf",
            "Help me organize the living room",
            "Place the lamp on the table",
        ]

        retrieval_results = []
        for query in test_queries:
            try:
                scores, indices = rag.retrieve_top_k_given_query(
                    query, top_k=3, agent_id=0
                )

                retrieval_results.append(
                    {
                        "query": query,
                        "scores": scores.tolist(),
                        "indices": indices.tolist(),
                        "success": True,
                    }
                )
            except Exception as e:
                retrieval_results.append(
                    {"query": query, "error": str(e), "success": False}
                )

        return {
            "rag_functional": True,
            "dataset_size": len(rag.data_dict),
            "retrieval_results": retrieval_results,
        }

    except ImportError as e:
        return {"rag_functional": False, "error": f"无法导入RAG模块: {e}"}
    except Exception as e:
        return {"rag_functional": False, "error": f"RAG测试失败: {e}"}


def analyze_dataset_quality(rag_dir: str, data_source_name: str) -> Dict[str, Any]:
    """分析数据集质量"""
    print("🔍 分析数据集质量...")

    # 加载轨迹文件
    trajectories = []
    agent_0_dir = os.path.join(rag_dir, data_source_name, "traces", "0")

    if os.path.exists(agent_0_dir):
        for trace_file in os.listdir(agent_0_dir):
            if trace_file.endswith(".txt"):
                trace_path = os.path.join(agent_0_dir, trace_file)
                with open(trace_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 提取episode_id
                episode_id = trace_file.split("_")[1]

                # 分析质量
                quality = analyze_trajectory_quality(content)
                quality["episode_id"] = episode_id
                quality["file_name"] = trace_file

                trajectories.append(quality)

    # 统计分析
    if not trajectories:
        return {"error": "没有找到轨迹文件"}

    # 基本统计
    total_trajectories = len(trajectories)
    avg_actions = sum(t["action_count"] for t in trajectories) / total_trajectories
    avg_failures = sum(t["failure_count"] for t in trajectories) / total_trajectories
    completion_rate = (
        sum(1 for t in trajectories if t["has_completion"]) / total_trajectories
    )

    # 动作类型多样性
    all_action_types = []
    for t in trajectories:
        all_action_types.extend(t["action_types"])
    action_type_counts = Counter(all_action_types)

    # 质量评分
    quality_scores = []
    for t in trajectories:
        score = (
            t["efficiency_score"] * 0.4
            + t["reliability_score"] * 0.4
            + t["completeness_score"] * 0.2
        )
        quality_scores.append(score)

    avg_quality = sum(quality_scores) / len(quality_scores)

    # 识别最高质量和最低质量的轨迹
    best_trajectory = max(
        trajectories, key=lambda t: quality_scores[trajectories.index(t)]
    )
    worst_trajectory = min(
        trajectories, key=lambda t: quality_scores[trajectories.index(t)]
    )

    return {
        "total_trajectories": total_trajectories,
        "avg_actions_per_trajectory": round(avg_actions, 2),
        "avg_failures_per_trajectory": round(avg_failures, 2),
        "completion_rate": round(completion_rate * 100, 1),
        "avg_quality_score": round(avg_quality, 3),
        "action_type_diversity": len(action_type_counts),
        "action_type_distribution": dict(action_type_counts),
        "best_trajectory": {
            "episode_id": best_trajectory["episode_id"],
            "quality_score": round(
                quality_scores[trajectories.index(best_trajectory)], 3
            ),
            "action_count": best_trajectory["action_count"],
            "failure_count": best_trajectory["failure_count"],
        },
        "worst_trajectory": {
            "episode_id": worst_trajectory["episode_id"],
            "quality_score": round(
                quality_scores[trajectories.index(worst_trajectory)], 3
            ),
            "action_count": worst_trajectory["action_count"],
            "failure_count": worst_trajectory["failure_count"],
        },
        "quality_distribution": {
            "high_quality (>0.8)": sum(1 for s in quality_scores if s > 0.8),
            "medium_quality (0.5-0.8)": sum(
                1 for s in quality_scores if 0.5 <= s <= 0.8
            ),
            "low_quality (<0.5)": sum(1 for s in quality_scores if s < 0.5),
        },
    }


def generate_quality_report(rag_dir: str, data_source_name: str):
    """生成质量报告"""
    print("📊 生成去重RAG数据集质量报告...")

    # 运行所有测试
    structure_test = test_rag_dataset_structure(rag_dir, data_source_name)
    rag_test = test_rag_functionality(rag_dir, data_source_name)
    quality_analysis = analyze_dataset_quality(rag_dir, data_source_name)

    # 加载去重统计
    stats_file = os.path.join(rag_dir, "deduplication_stats.json")
    dedup_stats = {}
    if os.path.exists(stats_file):
        with open(stats_file, "r", encoding="utf-8") as f:
            dedup_stats = json.load(f)

    # 生成报告
    report = {
        "dataset_info": {
            "path": rag_dir,
            "data_source": data_source_name,
            "deduplication_stats": dedup_stats,
        },
        "structure_test": structure_test,
        "rag_functionality_test": rag_test,
        "quality_analysis": quality_analysis,
    }

    # 保存报告
    report_file = os.path.join(rag_dir, "quality_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "=" * 60)
    print("📋 去重RAG数据集质量报告摘要")
    print("=" * 60)

    if dedup_stats:
        print("📈 去重统计:")
        print(f"   原始轨迹数量: {dedup_stats.get('original_count', 'N/A')}")
        print(f"   去重后数量: {dedup_stats.get('deduplicated_count', 'N/A')}")
        print(f"   减少比例: {dedup_stats.get('reduction_percentage', 'N/A'):.1f}%")

    print("\n🏗️  数据集结构:")
    print(f"   结构有效: {'✅' if structure_test['structure_valid'] else '❌'}")
    print(f"   Agent 0轨迹: {structure_test.get('agent_0_traces', 0)}")
    print(f"   Agent 1轨迹: {structure_test.get('agent_1_traces', 0)}")

    print("\n🔧 RAG功能:")
    if rag_test.get("rag_functional"):
        print("   RAG检索: ✅ 正常工作")
        print(f"   数据集大小: {rag_test['dataset_size']}")
        successful_retrievals = sum(
            1 for r in rag_test["retrieval_results"] if r["success"]
        )
        print(
            f"   检索测试: {successful_retrievals}/{len(rag_test['retrieval_results'])} 成功"
        )
    else:
        print(f"   RAG检索: ❌ {rag_test.get('error', '未知错误')}")

    if "error" not in quality_analysis:
        print("\n📊 质量分析:")
        print(f"   轨迹数量: {quality_analysis['total_trajectories']}")
        print(f"   平均动作数: {quality_analysis['avg_actions_per_trajectory']}")
        print(f"   平均失败数: {quality_analysis['avg_failures_per_trajectory']}")
        print(f"   完成率: {quality_analysis['completion_rate']}%")
        print(f"   平均质量分数: {quality_analysis['avg_quality_score']}")
        print(f"   动作类型多样性: {quality_analysis['action_type_diversity']} 种")

        print("\n🏆 质量分布:")
        for category, count in quality_analysis["quality_distribution"].items():
            print(f"   {category}: {count}")

        print(
            f"\n⭐ 最佳轨迹: Episode {quality_analysis['best_trajectory']['episode_id']} (质量分数: {quality_analysis['best_trajectory']['quality_score']})"
        )
        print(
            f"⚠️  最差轨迹: Episode {quality_analysis['worst_trajectory']['episode_id']} (质量分数: {quality_analysis['worst_trajectory']['quality_score']})"
        )

    print(f"\n📄 详细报告已保存: {report_file}")
    print("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="测试去重后的RAG数据集质量")
    parser.add_argument(
        "--rag-dir",
        default="data/rag_datasets/react_rag_dataset_dedup",
        help="RAG数据集目录路径",
    )
    parser.add_argument(
        "--data-source-name", default="react_trajectories", help="数据源名称"
    )

    args = parser.parse_args()

    if not os.path.exists(args.rag_dir):
        print(f"错误: RAG数据集目录不存在: {args.rag_dir}")
        return 1

    # 生成质量报告
    generate_quality_report(args.rag_dir, args.data_source_name)

    return 0


if __name__ == "__main__":
    exit(main())
