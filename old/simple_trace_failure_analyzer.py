#!/usr/bin/env python3
"""
简化版Traces失败分析工具

这个脚本用于分析habitat_llm输出中的traces文件，提取和统计失败信息，
包括失败动作类型、失败原因、失败占比等统计信息。

使用方法:
    python simple_trace_failure_analyzer.py --output_dir outputs/habitat_llm/2025-06-24_08-35-55-heterogeneous+rerange_first5.json
    python simple_trace_failure_analyzer.py --traces_dir path/to/traces
    python simple_trace_failure_analyzer.py --all  # 分析所有输出目录
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List


class SimpleTraceFailureAnalyzer:
    def __init__(self):
        self.failure_patterns = {
            # 动作失败模式
            "pick_failures": [
                r"Failed to pick! Not close enough to the object",
                r"Failed to pick! Object is already held",
                r"Failed to pick!.*occluded",
                r"Failed to pick!.*not reachable",
            ],
            "place_failures": [
                r"Failed to place! Not close enough to.*or occluded",
                r"Failed to place! The agent is not holding any object",
                r"Failed to place!.*invalid location",
                r"Failed to place!.*no space",
            ],
            "navigate_failures": [
                r"Failed to navigate!.*not reachable",
                r"Failed to navigate!.*path blocked",
                r"Failed to navigate!.*invalid destination",
                r"Navigation timeout",
            ],
            "rearrange_failures": [
                r"Failed! This object is already held by this agent",
                r"Failed! Object not found",
                r"Failed! Target location invalid",
                r"Failed! Cannot rearrange",
            ],
            "unexpected_failures": [
                r"Unexpected failure!.*",
                r"Runtime error.*",
                r"Exception.*",
            ],
        }

        self.action_patterns = {
            "Pick": r"Pick\[([^\]]+)\]",
            "Place": r"Place\[([^\]]+)\]",
            "Navigate": r"Navigate\[([^\]]+)\]",
            "Rearrange": r"Rearrange\[([^\]]+)\]",
            "Explore": r"Explore\[([^\]]+)\]",
        }

        self.failure_stats = defaultdict(lambda: defaultdict(int))
        self.episode_failures = defaultdict(list)
        self.total_actions = defaultdict(int)
        self.successful_actions = defaultdict(int)

    def analyze_trace_file(self, trace_file: str) -> Dict[str, Any]:
        """分析单个trace文件"""
        episode_id = self.extract_episode_id(trace_file)
        failures = []
        actions = []

        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 按行分析trace内容
            lines = content.split("\n")
            current_action = None

            for i, line in enumerate(lines):
                line = line.strip()

                # 检测动作
                for action_type, pattern in self.action_patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        current_action = {
                            "type": action_type,
                            "params": match.group(1),
                            "line_num": i + 1,
                            "success": True,  # 默认成功，如果找到失败信息会更新
                        }
                        actions.append(current_action)
                        self.total_actions[action_type] += 1
                        break

                # 检测失败信息
                if (
                    "Failed" in line
                    or "Unexpected failure" in line
                    or "error" in line.lower()
                ):
                    failure_info = self.classify_failure(line)
                    if failure_info and current_action:
                        current_action["success"] = False
                        failure_info.update(
                            {
                                "episode_id": episode_id,
                                "action": current_action,
                                "line_num": i + 1,
                                "context": self.get_context(lines, i),
                            }
                        )
                        failures.append(failure_info)
                        self.failure_stats[failure_info["category"]][
                            failure_info["type"]
                        ] += 1

                # 检测成功信息
                elif (
                    "Successful execution!" in line
                    and current_action
                    and current_action["success"]
                ):
                    self.successful_actions[current_action["type"]] += 1

            self.episode_failures[episode_id] = failures

            return {
                "episode_id": episode_id,
                "failures": failures,
                "actions": actions,
                "failure_count": len(failures),
                "action_count": len(actions),
            }

        except Exception as e:
            print(f"分析文件 {trace_file} 时出错: {e}")
            return {
                "episode_id": episode_id,
                "failures": [],
                "actions": [],
                "error": str(e),
            }

    def classify_failure(self, failure_line: str) -> Dict[str, str]:
        """对失败信息进行分类"""
        for category, patterns in self.failure_patterns.items():
            for pattern in patterns:
                if re.search(pattern, failure_line, re.IGNORECASE):
                    return {
                        "category": category.replace("_failures", ""),
                        "type": self.extract_failure_type(failure_line),
                        "message": failure_line.strip(),
                        "severity": self.assess_severity(failure_line),
                    }

        # 如果没有匹配到已知模式，归类为未知失败
        return {
            "category": "unknown",
            "type": "unknown_failure",
            "message": failure_line.strip(),
            "severity": "medium",
        }

    def extract_failure_type(self, failure_line: str) -> str:
        """从失败信息中提取失败类型"""
        # 提取 "Failed to xxx!" 或 "Unexpected failure! - xxx" 格式的失败类型
        if "Failed to" in failure_line:
            match = re.search(r"Failed to ([^!]+)!", failure_line)
            if match:
                return f"Failed to {match.group(1)}"
        elif "Unexpected failure!" in failure_line:
            parts = failure_line.split(" - ")
            if len(parts) > 1:
                return parts[1].split(".")[0]  # 取第一个句子
        elif "Failed!" in failure_line:
            parts = failure_line.split("Failed! ")
            if len(parts) > 1:
                return parts[1].split(".")[0]  # 取第一个句子

        return "general_failure"

    def assess_severity(self, failure_message: str) -> str:
        """评估失败严重程度"""
        high_severity_keywords = ["exception", "error", "crash", "timeout"]
        medium_severity_keywords = ["unexpected", "failed"]

        message_lower = failure_message.lower()

        if any(keyword in message_lower for keyword in high_severity_keywords):
            return "high"
        elif any(keyword in message_lower for keyword in medium_severity_keywords):
            return "medium"
        else:
            return "low"

    def get_context(
        self, lines: List[str], failure_line_idx: int, context_size: int = 2
    ) -> List[str]:
        """获取失败信息的上下文"""
        start = max(0, failure_line_idx - context_size)
        end = min(len(lines), failure_line_idx + context_size + 1)
        return lines[start:end]

    def extract_episode_id(self, trace_file: str) -> str:
        """从文件路径提取episode ID"""
        filename = os.path.basename(trace_file)
        match = re.search(r"episode_(\d+)", filename)
        return match.group(1) if match else "unknown"

    def analyze_directory(self, traces_dir: str) -> Dict[str, Any]:
        """分析整个traces目录"""
        trace_files = []

        # 递归查找所有trace文件
        for root, _dirs, files in os.walk(traces_dir):
            for file in files:
                if file.startswith("trace-") and file.endswith(".txt"):
                    trace_files.append(os.path.join(root, file))

        print(f"找到 {len(trace_files)} 个trace文件")

        results = []
        for trace_file in trace_files:
            result = self.analyze_trace_file(trace_file)
            results.append(result)

        return {
            "total_episodes": len(trace_files),
            "episode_results": results,
            "summary": self.generate_summary(),
        }

    def generate_summary(self) -> Dict[str, Any]:
        """生成统计摘要"""
        total_failures = sum(
            sum(types.values()) for types in self.failure_stats.values()
        )
        total_actions_count = sum(self.total_actions.values())
        total_successful_count = sum(self.successful_actions.values())

        # 计算失败率
        failure_rates = {}
        for action_type in self.total_actions:
            if self.total_actions[action_type] > 0:
                failed_count = (
                    self.total_actions[action_type]
                    - self.successful_actions[action_type]
                )
                failure_rates[action_type] = (
                    failed_count / self.total_actions[action_type]
                )

        # 失败类别统计
        category_stats = {}
        for category, types in self.failure_stats.items():
            category_stats[category] = {
                "total_count": sum(types.values()),
                "types": dict(types),
                "percentage": sum(types.values()) / total_failures * 100
                if total_failures > 0
                else 0,
            }

        return {
            "total_failures": total_failures,
            "total_actions": total_actions_count,
            "total_successful": total_successful_count,
            "overall_failure_rate": (total_actions_count - total_successful_count)
            / total_actions_count
            if total_actions_count > 0
            else 0,
            "failure_rates_by_action": failure_rates,
            "failure_categories": category_stats,
            "episodes_with_failures": len(
                [ep for ep, failures in self.episode_failures.items() if failures]
            ),
        }

    def save_detailed_report(self, results: Dict[str, Any], output_file: str):
        """保存详细报告"""
        report = {"analysis_summary": results["summary"], "episode_details": {}}

        for episode_result in results["episode_results"]:
            episode_id = episode_result["episode_id"]
            report["episode_details"][episode_id] = {
                "failure_count": episode_result["failure_count"],
                "action_count": episode_result["action_count"],
                "failures": episode_result["failures"],
            }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"详细报告已保存到: {output_file}")

    def save_csv_report(self, results: Dict[str, Any], output_file: str):
        """保存CSV格式的简化报告"""
        summary = results["summary"]

        lines = []
        lines.append("分析项,数值,百分比")
        lines.append(f"总episodes数,{results['total_episodes']},100.0%")
        lines.append(
            f"有失败的episodes数,{summary['episodes_with_failures']},{summary['episodes_with_failures']/results['total_episodes']*100:.1f}%"
        )
        lines.append(f"总动作数,{summary['total_actions']},100.0%")
        lines.append(
            f"总失败数,{summary['total_failures']},{summary['overall_failure_rate']*100:.1f}%"
        )
        lines.append("")

        lines.append("动作类型失败率统计")
        lines.append("动作类型,失败次数,总次数,失败率")
        for action, rate in summary["failure_rates_by_action"].items():
            success_count = self.successful_actions[action]
            total_count = self.total_actions[action]
            failed_count = total_count - success_count
            lines.append(f"{action},{failed_count},{total_count},{rate*100:.1f}%")

        lines.append("")
        lines.append("失败类别统计")
        lines.append("失败类别,失败次数,占总失败的比例")
        for category, data in summary["failure_categories"].items():
            lines.append(f"{category},{data['total_count']},{data['percentage']:.1f}%")
            for fail_type, count in data["types"].items():
                lines.append(
                    f"  {fail_type},{count},{count/summary['total_failures']*100:.1f}%"
                )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"CSV报告已保存到: {output_file}")

    def print_summary(self, results: Dict[str, Any]):
        """打印统计摘要"""
        summary = results["summary"]

        print("\n" + "=" * 70)
        print(" " * 25 + "Traces失败分析报告")
        print("=" * 70)

        print("\n📊 总体统计:")
        print(f"  总episodes数: {results['total_episodes']}")
        print(
            f"  有失败的episodes数: {summary['episodes_with_failures']} ({summary['episodes_with_failures']/results['total_episodes']*100:.1f}%)"
        )
        print(f"  总动作数: {summary['total_actions']}")
        print(f"  总失败数: {summary['total_failures']}")
        print(f"  总体失败率: {summary['overall_failure_rate']:.2%}")

        print("\n🎯 各动作类型失败率:")
        for action, rate in summary["failure_rates_by_action"].items():
            success_count = self.successful_actions[action]
            total_count = self.total_actions[action]
            failed_count = total_count - success_count
            print(f"  {action:12}: {rate:.2%} ({failed_count:3d}/{total_count:3d})")

        print("\n❌ 失败类别统计:")
        sorted_categories = sorted(
            summary["failure_categories"].items(),
            key=lambda x: x[1]["total_count"],
            reverse=True,
        )

        for category, data in sorted_categories:
            print(
                f"  {category:15}: {data['total_count']:3d} ({data['percentage']:.1f}%)"
            )

            # 显示该类别中最常见的失败类型
            sorted_types = sorted(
                data["types"].items(), key=lambda x: x[1], reverse=True
            )
            for i, (fail_type, count) in enumerate(sorted_types[:3]):  # 只显示前3个
                prefix = "    ├─ " if i < len(sorted_types[:3]) - 1 else "    └─ "
                print(f"{prefix}{fail_type[:45]:45}: {count:3d}")

        print("\n" + "=" * 70)

    def print_detailed_failures(self, results: Dict[str, Any], max_examples: int = 5):
        """打印详细的失败示例"""
        print(f"\n📋 失败示例详情 (每类最多显示 {max_examples} 个):")
        print("-" * 70)

        # 按类别组织失败示例
        examples_by_category = defaultdict(list)
        for episode_result in results["episode_results"]:
            for failure in episode_result["failures"]:
                examples_by_category[failure["category"]].append(failure)

        for category, failures in examples_by_category.items():
            print(f"\n{category.upper()} 失败示例:")

            # 按失败类型分组
            failures_by_type = defaultdict(list)
            for failure in failures:
                failures_by_type[failure["type"]].append(failure)

            for fail_type, type_failures in list(failures_by_type.items())[
                :max_examples
            ]:
                example = type_failures[0]  # 取第一个示例
                print(f"  类型: {fail_type}")
                print(f"  Episode: {example['episode_id']}")
                print(
                    f"  动作: {example['action']['type']}[{example['action']['params']}]"
                )
                print(f"  错误: {example['message']}")
                print(f"  严重程度: {example['severity']}")
                print()


def find_trace_directories(base_dir: str) -> List[str]:
    """查找所有traces目录"""
    trace_dirs = []
    for root, dirs, _files in os.walk(base_dir):
        if "traces" in dirs:
            trace_dirs.append(os.path.join(root, "traces"))
    return trace_dirs


def main():
    parser = argparse.ArgumentParser(description="分析habitat_llm traces中的失败信息")
    parser.add_argument("--output_dir", type=str, help="输出目录路径")
    parser.add_argument("--traces_dir", type=str, help="traces目录路径")
    parser.add_argument("--all", action="store_true", help="分析outputs目录下的所有traces")
    parser.add_argument(
        "--save_report",
        type=str,
        default="failure_analysis_report.json",
        help="保存详细报告的文件名",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="failure_analysis_summary.csv",
        help="保存CSV报告的文件名",
    )
    parser.add_argument("--detailed", action="store_true", help="显示详细的失败示例")

    args = parser.parse_args()

    analyzer = SimpleTraceFailureAnalyzer()

    if args.all:
        # 分析所有输出目录
        trace_dirs = find_trace_directories("outputs")
        print(f"找到 {len(trace_dirs)} 个traces目录")

        for trace_dir in trace_dirs:
            print(f"\n{'='*50}")
            print(f"分析目录: {trace_dir}")
            print(f"{'='*50}")
            results = analyzer.analyze_directory(trace_dir)
            analyzer.print_summary(results)
            if args.detailed:
                analyzer.print_detailed_failures(results)

    elif args.traces_dir:
        # 分析指定的traces目录
        results = analyzer.analyze_directory(args.traces_dir)
        analyzer.print_summary(results)
        if args.detailed:
            analyzer.print_detailed_failures(results)
        analyzer.save_detailed_report(results, args.save_report)
        analyzer.save_csv_report(results, args.save_csv)

    elif args.output_dir:
        # 查找指定输出目录下的traces
        trace_dirs = find_trace_directories(args.output_dir)
        if trace_dirs:
            results = analyzer.analyze_directory(trace_dirs[0])
            analyzer.print_summary(results)
            if args.detailed:
                analyzer.print_detailed_failures(results)
            analyzer.save_detailed_report(results, args.save_report)
            analyzer.save_csv_report(results, args.save_csv)
        else:
            print(f"在 {args.output_dir} 中未找到traces目录")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
