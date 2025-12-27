#!/usr/bin/env python3
"""
RAG使用情况监控脚本

这个脚本帮助你确认RAG功能是否被正确调用并使用了examples。

使用方法:
1. 在一个终端运行: python monitor_rag_usage.py
2. 在另一个终端运行: ./run_planner_demo_with_rag.sh

或者直接使用这个脚本的分析函数来检查日志文件。
"""

import argparse
import os
import re
import time
from typing import Any, Dict


def check_rag_in_log_file(log_file_path: str) -> Dict[str, Any]:
    """检查日志文件中的RAG使用情况"""
    rag_info = {
        "rag_initialized": False,
        "dataset_loaded": False,
        "embeddings_built": False,
        "rag_queries": [],
        "rag_examples_added": 0,
        "similar_traces_found": [],
        "log_lines_with_rag": [],
    }

    if not os.path.exists(log_file_path):
        print(f"❌ 日志文件不存在: {log_file_path}")
        return rag_info

    print(f"📖 分析日志文件: {log_file_path}")

    with open(log_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # 检查RAG初始化
        if (
            "loading rag dataset" in line_lower
            or "building rag embeddings" in line_lower
        ):
            rag_info["rag_initialized"] = True
            rag_info["log_lines_with_rag"].append((i + 1, line.strip()))

        # 检查数据集加载
        if "rag dataset loaded" in line_lower or "data_dict" in line_lower:
            rag_info["dataset_loaded"] = True
            rag_info["log_lines_with_rag"].append((i + 1, line.strip()))

        # 检查嵌入构建
        if "embedding" in line_lower and (
            "built" in line_lower or "model" in line_lower
        ):
            rag_info["embeddings_built"] = True
            rag_info["log_lines_with_rag"].append((i + 1, line.strip()))

        # 检查RAG查询
        if "retrieve_top_k_given_query" in line or "retrieving rag" in line_lower:
            rag_info["rag_queries"].append(line.strip())
            rag_info["log_lines_with_rag"].append((i + 1, line.strip()))

        # 检查RAG examples添加到prompt
        if (
            "below are some example solutions" in line_lower
            or "example 1:" in line_lower
        ):
            rag_info["rag_examples_added"] += 1
            rag_info["log_lines_with_rag"].append((i + 1, line.strip()))

        # 检查相似轨迹
        if (
            "similarity score" in line_lower
            or "similar" in line_lower
            and "trace" in line_lower
        ):
            rag_info["similar_traces_found"].append(line.strip())
            rag_info["log_lines_with_rag"].append((i + 1, line.strip()))

    return rag_info


def analyze_prompt_for_rag(prompt_text: str) -> Dict[str, Any]:
    """分析prompt文本中的RAG内容"""
    rag_analysis = {
        "has_rag_examples": False,
        "rag_example_count": 0,
        "rag_sections": [],
        "example_instructions": [],
        "example_actions": [],
    }

    # 检查是否包含RAG examples
    if "below are some example solutions" in prompt_text.lower():
        rag_analysis["has_rag_examples"] = True

        # 计算example数量
        example_matches = re.findall(r"Example \d+:", prompt_text, re.IGNORECASE)
        rag_analysis["rag_example_count"] = len(example_matches)

        # 提取RAG部分
        rag_sections = re.findall(
            r"Below are some example solutions.*?(?=Task:|$)",
            prompt_text,
            re.DOTALL | re.IGNORECASE,
        )
        rag_analysis["rag_sections"] = rag_sections

        # 提取示例中的指令
        instruction_matches = re.findall(r"Task: ([^\n]+)", prompt_text)
        rag_analysis["example_instructions"] = (
            instruction_matches[:-1] if instruction_matches else []
        )

        # 提取示例中的动作
        action_matches = re.findall(r"Agent_\d+_Action: ([^\n]+)", prompt_text)
        rag_analysis["example_actions"] = action_matches

    return rag_analysis


def monitor_live_log(log_file_path: str, interval: int = 2):
    """实时监控日志文件中的RAG活动"""
    print(f"🔍 开始实时监控RAG活动: {log_file_path}")
    print(f"监控间隔: {interval}秒")
    print("按 Ctrl+C 停止监控")
    print("-" * 60)

    last_size = 0
    rag_events_count = 0

    try:
        while True:
            if os.path.exists(log_file_path):
                current_size = os.path.getsize(log_file_path)

                if current_size > last_size:
                    # 读取新添加的内容
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        f.seek(last_size)
                        new_lines = f.readlines()

                    # 检查新行中的RAG活动
                    for line in new_lines:
                        line_lower = line.lower()

                        if any(
                            keyword in line_lower
                            for keyword in [
                                "rag",
                                "embedding",
                                "retrieve",
                                "example solutions",
                                "similarity",
                            ]
                        ):
                            rag_events_count += 1
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"🎯 [{timestamp}] RAG事件 #{rag_events_count}:")
                            print(f"    {line.strip()}")

                    last_size = current_size

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n✅ 监控结束。总共发现 {rag_events_count} 个RAG相关事件。")


def print_rag_analysis_report(rag_info: Dict[str, Any]):
    """打印RAG分析报告"""
    print("\n" + "=" * 60)
    print("📊 RAG使用情况分析报告")
    print("=" * 60)

    # 基本状态
    print(f"🔧 RAG初始化: {'✅ 是' if rag_info['rag_initialized'] else '❌ 否'}")
    print(f"📦 数据集加载: {'✅ 是' if rag_info['dataset_loaded'] else '❌ 否'}")
    print(f"🧠 嵌入构建: {'✅ 是' if rag_info['embeddings_built'] else '❌ 否'}")

    # 使用统计
    print(f"🔍 RAG查询次数: {len(rag_info['rag_queries'])}")
    print(f"📝 RAG examples添加次数: {rag_info['rag_examples_added']}")
    print(f"🎯 相似轨迹发现: {len(rag_info['similar_traces_found'])}")

    # 详细信息
    if rag_info["rag_queries"]:
        print("\n📋 RAG查询详情:")
        for i, query in enumerate(rag_info["rag_queries"][:3]):  # 只显示前3个
            print(f"  {i+1}. {query}")
        if len(rag_info["rag_queries"]) > 3:
            print(f"  ... 还有 {len(rag_info['rag_queries'])-3} 个查询")

    if rag_info["log_lines_with_rag"]:
        print("\n🔍 RAG相关日志行 (前5行):")
        for line_num, line_content in rag_info["log_lines_with_rag"][:5]:
            print(f"  第{line_num}行: {line_content}")
        if len(rag_info["log_lines_with_rag"]) > 5:
            print(f"  ... 还有 {len(rag_info['log_lines_with_rag'])-5} 行相关日志")


def main():
    parser = argparse.ArgumentParser(description="监控RAG使用情况")
    parser.add_argument("--log-file", help="要分析的日志文件路径")
    parser.add_argument("--live", action="store_true", help="实时监控模式")
    parser.add_argument("--interval", type=int, default=2, help="实时监控的间隔秒数")
    parser.add_argument(
        "--find-latest", action="store_true", help="自动查找最新的habitat_llm.log文件"
    )

    args = parser.parse_args()

    # 确定日志文件路径
    log_file_path = args.log_file

    if args.find_latest:
        # 查找最新的日志文件
        outputs_dir = "outputs/habitat_llm"
        if os.path.exists(outputs_dir):
            subdirs = [
                d
                for d in os.listdir(outputs_dir)
                if os.path.isdir(os.path.join(outputs_dir, d))
            ]
            if subdirs:
                latest_dir = sorted(subdirs)[-1]
                log_file_path = os.path.join(outputs_dir, latest_dir, "habitat_llm.log")
                print(f"🔍 自动发现最新日志: {log_file_path}")
            else:
                print("❌ 没有找到outputs/habitat_llm目录下的运行结果")
                return
        else:
            print("❌ outputs/habitat_llm目录不存在")
            return

    if not log_file_path:
        # 默认使用我们知道的日志文件
        log_file_path = "outputs/habitat_llm/2025-07-17_13-39-22-rerange+spatial_first5.json/habitat_llm.log"
        print(f"🔍 使用默认日志文件: {log_file_path}")

    if args.live:
        monitor_live_log(log_file_path, args.interval)
    else:
        # 静态分析
        rag_info = check_rag_in_log_file(log_file_path)
        print_rag_analysis_report(rag_info)

        # 提供使用建议
        print("\n💡 使用建议:")
        if not rag_info["rag_initialized"]:
            print("- RAG没有被初始化，检查enable_rag配置是否为True")
        if not rag_info["dataset_loaded"]:
            print("- RAG数据集没有加载，检查rag_dataset_dir和rag_data_source_name配置")
        if rag_info["rag_examples_added"] == 0:
            print("- RAG examples没有被添加到prompt，检查指令模板是否包含{rag_examples}占位符")
        if rag_info["rag_examples_added"] > 0:
            print("✅ RAG功能正常工作！成功添加了examples到prompt中")


if __name__ == "__main__":
    main()
