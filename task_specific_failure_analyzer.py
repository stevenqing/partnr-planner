#!/usr/bin/env python3
"""
基于任务类型的失败分析工具

这个脚本将失败分析结果按照任务类型和描述进行分类和保存，
提供更细粒度的任务特定失败分析。

使用方法:
    python task_specific_failure_analyzer.py
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

from simple_trace_failure_analyzer import (
    SimpleTraceFailureAnalyzer,
    find_trace_directories,
)


class TaskSpecificFailureAnalyzer:
    def __init__(self):
        self.task_classifications = {}
        self.episode_to_task_map = {}
        self.task_failure_stats = defaultdict(
            lambda: {
                "episodes": [],
                "total_actions": 0,
                "total_failures": 0,
                "failure_by_category": defaultdict(int),
                "failure_by_type": defaultdict(int),
                "action_failure_rates": defaultdict(lambda: {"failed": 0, "total": 0}),
                "instructions": [],
                "failure_examples": [],
            }
        )

    def load_task_classifications(self):
        """加载任务分类数据"""
        classification_file = "task_classification_results/episode_classifications.json"
        if os.path.exists(classification_file):
            with open(classification_file, "r") as f:
                classifications = json.load(f)

            for item in classifications:
                episode_id = item["episode_id"]
                categories = item["categories"]
                instruction = item["instruction"]

                # 将多个类别组合成一个key
                task_key = "+".join(sorted(categories)) if categories else "unknown"

                self.episode_to_task_map[episode_id] = {
                    "task_type": task_key,
                    "instruction": instruction,
                    "categories": categories,
                }

                if task_key not in self.task_classifications:
                    self.task_classifications[task_key] = {
                        "episodes": [],
                        "instructions": [],
                        "categories": categories,
                    }

                self.task_classifications[task_key]["episodes"].append(episode_id)
                self.task_classifications[task_key]["instructions"].append(instruction)

        print(f"加载了 {len(self.episode_to_task_map)} 个episode的任务分类")
        print(
            f"发现 {len(self.task_classifications)} 种任务类型: {list(self.task_classifications.keys())}"
        )

    def extract_episode_id_from_trace(self, trace_file: str) -> str:
        """从trace文件路径中提取episode ID"""
        filename = os.path.basename(trace_file)
        match = re.search(r"episode_(\d+)", filename)
        return match.group(1) if match else "unknown"

    def analyze_traces_by_task(self) -> Dict[str, Any]:
        """按任务类型分析traces"""
        trace_dirs = find_trace_directories("outputs")
        print(f"开始分析 {len(trace_dirs)} 个traces目录...")

        all_results = []

        for i, trace_dir in enumerate(trace_dirs, 1):
            print(f"[{i:2d}/{len(trace_dirs)}] 分析: {os.path.basename(trace_dir)}")

            analyzer = SimpleTraceFailureAnalyzer()
            results = analyzer.analyze_directory(trace_dir)

            if not results["episode_results"]:
                continue

            # 处理每个episode的结果
            for episode_result in results["episode_results"]:
                episode_id = episode_result["episode_id"]

                # 获取任务类型信息
                task_info = self.episode_to_task_map.get(episode_id)
                if not task_info:
                    task_type = "unknown"
                    instruction = "Unknown instruction"
                else:
                    task_type = task_info["task_type"]
                    instruction = task_info["instruction"]

                # 累积任务特定的统计
                task_stats = self.task_failure_stats[task_type]
                task_stats["episodes"].append(episode_id)
                task_stats["total_actions"] += episode_result["action_count"]
                task_stats["total_failures"] += episode_result["failure_count"]
                task_stats["instructions"].append(instruction)

                # 统计失败类型
                for failure in episode_result["failures"]:
                    task_stats["failure_by_category"][failure["category"]] += 1
                    task_stats["failure_by_type"][failure["type"]] += 1
                    task_stats["failure_examples"].append(
                        {
                            "episode_id": episode_id,
                            "instruction": instruction,
                            "failure": failure,
                        }
                    )

                # 统计动作失败率
                for action in episode_result["actions"]:
                    action_type = action["type"]
                    task_stats["action_failure_rates"][action_type]["total"] += 1
                    if not action["success"]:
                        task_stats["action_failure_rates"][action_type]["failed"] += 1

            all_results.append(results)

        return self._generate_task_specific_summary()

    def _generate_task_specific_summary(self) -> Dict[str, Any]:
        """生成按任务类型的详细摘要"""
        summary = {}

        for task_type, stats in self.task_failure_stats.items():
            if not stats["episodes"]:
                continue

            # 计算失败率
            failure_rate = (
                stats["total_failures"] / stats["total_actions"]
                if stats["total_actions"] > 0
                else 0
            )

            # 计算动作失败率
            action_rates = {}
            for action_type, action_stats in stats["action_failure_rates"].items():
                if action_stats["total"] > 0:
                    action_rates[action_type] = (
                        action_stats["failed"] / action_stats["total"]
                    )

            # 分析指令模式
            instruction_keywords = self._extract_instruction_patterns(
                stats["instructions"]
            )

            summary[task_type] = {
                "overview": {
                    "episode_count": len(set(stats["episodes"])),
                    "total_actions": stats["total_actions"],
                    "total_failures": stats["total_failures"],
                    "failure_rate": failure_rate,
                },
                "failure_by_category": dict(stats["failure_by_category"]),
                "failure_by_type": dict(stats["failure_by_type"]),
                "action_failure_rates": action_rates,
                "instruction_patterns": instruction_keywords,
                "sample_instructions": list(set(stats["instructions"]))[
                    :5
                ],  # 取前5个不重复的指令作为样例
                "failure_examples": stats["failure_examples"][:10],  # 取前10个失败样例
            }

        return summary

    def _extract_instruction_patterns(self, instructions: List[str]) -> Dict[str, int]:
        """从指令中提取常见模式和关键词"""
        # 常见的动作关键词
        action_keywords = [
            "move",
            "place",
            "bring",
            "help",
            "set up",
            "clean",
            "tidy",
            "organize",
        ]
        object_keywords = [
            "toy",
            "plant",
            "kettle",
            "lamp",
            "table",
            "chair",
            "bedroom",
            "kitchen",
            "living room",
        ]

        keyword_counts = defaultdict(int)

        for instruction in instructions:
            instruction_lower = instruction.lower()

            # 统计动作关键词
            for keyword in action_keywords:
                if keyword in instruction_lower:
                    keyword_counts[f"action_{keyword}"] += 1

            # 统计物体关键词
            for keyword in object_keywords:
                if keyword in instruction_lower:
                    keyword_counts[f"object_{keyword}"] += 1

            # 统计房间相关
            rooms = ["bedroom", "kitchen", "living room", "dining room"]
            for room in rooms:
                if room in instruction_lower:
                    keyword_counts[f"room_{room}"] += 1

        return dict(keyword_counts)

    def save_task_specific_reports(
        self, summary: Dict[str, Any], output_dir: str = "task_specific_reports"
    ):
        """保存按任务类型分类的报告"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存总体摘要
        overall_summary = {
            "task_types": list(summary.keys()),
            "task_count": len(summary),
            "total_task_instances": sum(
                data["overview"]["episode_count"] for data in summary.values()
            ),
            "task_performance_ranking": [],
        }

        # 按失败率排序任务类型
        sorted_tasks = sorted(
            summary.items(),
            key=lambda x: x[1]["overview"]["failure_rate"],
            reverse=True,
        )

        for rank, (task_type, data) in enumerate(sorted_tasks, 1):
            overall_summary["task_performance_ranking"].append(
                {
                    "rank": rank,
                    "task_type": task_type,
                    "failure_rate": data["overview"]["failure_rate"],
                    "episode_count": data["overview"]["episode_count"],
                    "total_failures": data["overview"]["total_failures"],
                }
            )

        # 保存总体摘要
        with open(
            os.path.join(output_dir, "task_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(overall_summary, f, indent=2, ensure_ascii=False)

        # 2. 为每个任务类型保存详细报告
        for task_type, data in summary.items():
            safe_task_name = task_type.replace("+", "_").replace("/", "_")
            task_dir = os.path.join(output_dir, safe_task_name)
            os.makedirs(task_dir, exist_ok=True)

            # 保存详细数据
            with open(
                os.path.join(task_dir, "detailed_analysis.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # 保存CSV格式的摘要
            self._save_task_csv_summary(
                data, os.path.join(task_dir, "summary.csv"), task_type
            )

            # 保存失败示例
            self._save_failure_examples(
                data["failure_examples"], os.path.join(task_dir, "failure_examples.txt")
            )

        print(f"\n📁 任务特定报告已保存到: {output_dir}")
        return output_dir

    def _save_task_csv_summary(
        self, data: Dict[str, Any], file_path: str, task_type: str
    ):
        """保存单个任务类型的CSV摘要"""
        lines = [
            f"任务类型: {task_type}",
            "",
            "总体统计,数值,百分比",
            f"Episodes数,{data['overview']['episode_count']},100.0%",
            f"总动作数,{data['overview']['total_actions']},100.0%",
            f"总失败数,{data['overview']['total_failures']},{data['overview']['failure_rate']*100:.1f}%",
            "",
            "动作类型失败率",
            "动作类型,失败率",
        ]

        for action_type, rate in data["action_failure_rates"].items():
            lines.append(f"{action_type},{rate*100:.1f}%")

        lines.extend(["", "失败类别统计", "失败类别,次数"])

        for category, count in data["failure_by_category"].items():
            lines.append(f"{category},{count}")

        lines.extend(["", "指令模式统计", "模式,出现次数"])

        for pattern, count in data["instruction_patterns"].items():
            lines.append(f"{pattern},{count}")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_failure_examples(self, examples: List[Dict], file_path: str):
        """保存失败示例到文本文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("失败示例详情\n")
            f.write("=" * 50 + "\n\n")

            for i, example in enumerate(examples, 1):
                f.write(f"示例 {i}:\n")
                f.write(f"Episode ID: {example['episode_id']}\n")
                f.write(f"指令: {example['instruction']}\n")
                f.write(f"失败类别: {example['failure']['category']}\n")
                f.write(f"失败类型: {example['failure']['type']}\n")
                f.write(f"错误信息: {example['failure']['message']}\n")
                if "action" in example["failure"]:
                    action = example["failure"]["action"]
                    f.write(f"相关动作: {action['type']}[{action['params']}]\n")
                f.write("-" * 50 + "\n\n")

    def print_task_summary(self, summary: Dict[str, Any]):
        """打印任务特定的摘要报告"""
        print("\n" + "=" * 80)
        print(" " * 25 + "基于任务类型的失败分析报告")
        print("=" * 80)

        # 按失败率排序
        sorted_tasks = sorted(
            summary.items(),
            key=lambda x: x[1]["overview"]["failure_rate"],
            reverse=True,
        )

        print("\n📊 任务类型性能排名 (按失败率排序):")
        print(f"{'排名':<4} {'任务类型':<30} {'Episodes':<10} {'失败率':<10} {'失败数/总动作数':<15}")
        print("-" * 75)

        for rank, (task_type, data) in enumerate(sorted_tasks, 1):
            overview = data["overview"]
            print(
                f"{rank:<4} {task_type:<30} {overview['episode_count']:<10} "
                f"{overview['failure_rate']:<10.2%} {overview['total_failures']}/{overview['total_actions']}"
            )

        # 详细分析每个任务类型
        print("\n📋 详细任务分析:")
        for task_type, data in sorted_tasks[:5]:  # 只显示前5个最有问题的任务类型
            print(f"\n🔍 任务类型: {task_type}")
            print(f"  Episodes数: {data['overview']['episode_count']}")
            print(f"  失败率: {data['overview']['failure_rate']:.2%}")

            # 显示主要失败类别
            top_failures = sorted(
                data["failure_by_category"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            print("  主要失败类别:")
            for category, count in top_failures:
                percentage = (
                    count / data["overview"]["total_failures"] * 100
                    if data["overview"]["total_failures"] > 0
                    else 0
                )
                print(f"    - {category}: {count} ({percentage:.1f}%)")

            # 显示最有问题的动作
            problematic_actions = sorted(
                data["action_failure_rates"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            print("  最有问题的动作:")
            for action, rate in problematic_actions:
                print(f"    - {action}: {rate:.2%}")

            # 显示样例指令
            print("  样例指令:")
            for instruction in data["sample_instructions"][:2]:
                print(
                    f"    - {instruction[:80]}{'...' if len(instruction) > 80 else ''}"
                )

        print("\n" + "=" * 80)


def main():
    print("开始基于任务类型的失败分析...")

    analyzer = TaskSpecificFailureAnalyzer()

    # 加载任务分类
    analyzer.load_task_classifications()

    # 分析traces
    summary = analyzer.analyze_traces_by_task()

    # 打印摘要
    analyzer.print_task_summary(summary)

    # 保存报告
    output_dir = analyzer.save_task_specific_reports(summary)

    print("\n✅ 基于任务类型的分析完成！")
    print(f"📁 报告保存在: {output_dir}")


if __name__ == "__main__":
    main()
