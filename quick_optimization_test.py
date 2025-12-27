#!/usr/bin/env python3
"""
快速优化测试脚本
用于快速测试单个优化配置
"""

import os
import subprocess
import time
from datetime import datetime


def run_quick_test(config_name: str, config_overrides: dict, description: str = ""):
    """运行快速测试"""
    print(f"\n开始快速测试: {config_name}")
    print(f"描述: {description}")

    # 构建命令
    cmd = [
        "python",
        "-m",
        "habitat_llm.examples.planner_demo",
        "--config-name=baselines/decentralized_zero_shot_react_summary.yaml",
        "habitat.dataset.data_path=/home/shuqing/partnr-planner/task_classification_datasets_first100/rerange+spatial_first100.json.gz",
        "evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf",
        "evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf",
        "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct",
        "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct",
        "evaluation.agents.agent_0.planner.plan_config.enable_rag=True",
        "evaluation.agents.agent_1.planner.plan_config.enable_rag=True",
    ]

    # 添加配置覆盖
    for key, value in config_overrides.items():
        cmd.append(f"{key}={value}")

    # 设置输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/habitat_llm/{timestamp}-{config_name}"
    cmd.append(f"evaluation.output_dir={output_dir}")

    print(f"输出目录: {output_dir}")
    print(f"命令: {' '.join(cmd)}")

    # 运行测试
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )  # 1小时超时
        end_time = time.time()

        print("\n测试完成:")
        print(f"  成功: {result.returncode == 0}")
        print(f"  运行时间: {end_time - start_time:.2f}秒")

        if result.returncode == 0:
            print(f"  输出目录: {output_dir}")
            # 检查结果文件
            results_dir = os.path.join(output_dir, "results")
            if os.path.exists(results_dir):
                end_result_path = os.path.join(results_dir, "end_result_log.csv")
                if os.path.exists(end_result_path):
                    with open(end_result_path, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last_line = lines[-1].strip().split(",")
                            if len(last_line) >= 4:
                                print(f"  平均重规划次数: {float(last_line[0]):.2f}")
                                print(f"  平均运行时间: {float(last_line[1]):.2f}")
                                print(f"  平均任务完成度: {float(last_line[2]):.3f}")
                                print(f"  平均任务成功率: {float(last_line[3]):.3f}")
                                print(f"  总任务数: {len(lines) - 1}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("测试超时")
        return False
    except Exception as e:
        print(f"测试出错: {e}")
        return False


def main():
    """主函数"""
    print("PartNR Planner 快速优化测试")
    print("=" * 50)

    # 测试1: LLM参数优化
    print("\n测试1: LLM参数优化")
    llm_optimized_config = {
        "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.temperature": 0.1,
        "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.temperature": 0.1,
        "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens": 300,
        "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens": 300,
    }

    success1 = run_quick_test(
        "llm_optimized", llm_optimized_config, "优化LLM参数：temperature=0.1, max_tokens=300"
    )

    # 测试2: 重规划阈值优化
    print("\n测试2: 重规划阈值优化")
    replanning_config = {
        "evaluation.agents.agent_0.planner.plan_config.replanning_threshold": 75,
        "evaluation.agents.agent_1.planner.plan_config.replanning_threshold": 75,
    }

    success2 = run_quick_test("replanning_optimized", replanning_config, "降低重规划阈值到75次")

    # 测试3: RAG示例数量优化
    print("\n测试3: RAG示例数量优化")
    rag_config = {
        "evaluation.agents.agent_0.planner.plan_config.max_number_of_rag_example_added": 3,
        "evaluation.agents.agent_1.planner.plan_config.max_number_of_rag_example_added": 3,
    }

    success3 = run_quick_test("rag_optimized", rag_config, "限制RAG示例数量为3个")

    # 测试4: 组合优化
    print("\n测试4: 组合优化")
    combined_config = {
        # LLM参数优化
        "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.temperature": 0.1,
        "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.temperature": 0.1,
        "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens": 300,
        "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens": 300,
        # 重规划优化
        "evaluation.agents.agent_0.planner.plan_config.replanning_threshold": 75,
        "evaluation.agents.agent_1.planner.plan_config.replanning_threshold": 75,
        # RAG优化
        "evaluation.agents.agent_0.planner.plan_config.max_number_of_rag_example_added": 3,
        "evaluation.agents.agent_1.planner.plan_config.max_number_of_rag_example_added": 3,
    }

    success4 = run_quick_test("combined_optimized", combined_config, "组合所有优化")

    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"LLM参数优化: {'成功' if success1 else '失败'}")
    print(f"重规划阈值优化: {'成功' if success2 else '失败'}")
    print(f"RAG示例数量优化: {'成功' if success3 else '失败'}")
    print(f"组合优化: {'成功' if success4 else '失败'}")


if __name__ == "__main__":
    main()
