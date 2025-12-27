#!/usr/bin/env python3
"""
PartNR Planner 优化实验脚本
用于测试不同的优化配置和参数设置
"""

import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict


class OptimizationExperiments:
    def __init__(
        self, base_config: str = "baselines/decentralized_zero_shot_react_summary.yaml"
    ):
        self.base_config = base_config
        self.experiments = []
        self.results = {}

    def add_experiment(
        self, name: str, config_overrides: Dict[str, Any], description: str = ""
    ):
        """添加一个实验配置"""
        self.experiments.append(
            {
                "name": name,
                "config_overrides": config_overrides,
                "description": description,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            }
        )

    def setup_optimization_experiments(self):
        """设置优化实验"""

        # 实验1: RAG数据集质量过滤
        self.add_experiment(
            name="rag_quality_filtered",
            config_overrides={
                "evaluation.agents.agent_0.planner.plan_config.rag_dataset_dir": "[data/rag_datasets/react_rag_dataset_quality_filtered/]",
                "evaluation.agents.agent_1.planner.plan_config.rag_dataset_dir": "[data/rag_datasets/react_rag_dataset_quality_filtered/]",
                "evaluation.agents.agent_0.planner.plan_config.max_number_of_rag_example_added": 3,
                "evaluation.agents.agent_1.planner.plan_config.max_number_of_rag_example_added": 3,
            },
            description="使用质量过滤的RAG数据集，限制示例数量为3个",
        )

        # 实验2: LLM参数优化
        self.add_experiment(
            name="llm_params_optimized",
            config_overrides={
                "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.temperature": 0.1,
                "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.temperature": 0.1,
                "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens": 300,
                "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens": 300,
                "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.top_k": 30,
                "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.top_k": 30,
                "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.repetition_penalty": 1.1,
                "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.repetition_penalty": 1.1,
            },
            description="优化LLM生成参数：temperature=0.1, max_tokens=300, top_k=30, repetition_penalty=1.1",
        )

        # 实验3: 动态重规划阈值
        self.add_experiment(
            name="dynamic_replanning",
            config_overrides={
                "evaluation.agents.agent_0.planner.plan_config.replanning_threshold": 75,
                "evaluation.agents.agent_1.planner.plan_config.replanning_threshold": 75,
            },
            description="降低重规划阈值到75次，提高效率",
        )

        # 实验4: 组合优化
        self.add_experiment(
            name="combined_optimization",
            config_overrides={
                # RAG优化
                "evaluation.agents.agent_0.planner.plan_config.max_number_of_rag_example_added": 3,
                "evaluation.agents.agent_1.planner.plan_config.max_number_of_rag_example_added": 3,
                # LLM参数优化
                "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.temperature": 0.1,
                "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.temperature": 0.1,
                "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens": 300,
                "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens": 300,
                # 重规划优化
                "evaluation.agents.agent_0.planner.plan_config.replanning_threshold": 75,
                "evaluation.agents.agent_1.planner.plan_config.replanning_threshold": 75,
            },
            description="组合所有优化：RAG质量过滤 + LLM参数优化 + 动态重规划",
        )

        # 实验5: 高重规划阈值（对比实验）
        self.add_experiment(
            name="high_replanning_threshold",
            config_overrides={
                "evaluation.agents.agent_0.planner.plan_config.replanning_threshold": 150,
                "evaluation.agents.agent_1.planner.plan_config.replanning_threshold": 150,
            },
            description="提高重规划阈值到150次，测试极限性能",
        )

    def create_rag_quality_filtered_dataset(self):
        """创建质量过滤的RAG数据集"""
        print("创建质量过滤的RAG数据集...")

        # 这里应该实现质量过滤逻辑
        # 暂时复制现有数据集
        source_dir = "data/rag_datasets/react_rag_dataset_dedup_v2_minified"
        target_dir = "data/rag_datasets/react_rag_dataset_quality_filtered"

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 简单的质量过滤：只保留包含"Successful execution!"的示例
        if os.path.exists(source_dir):
            for filename in os.listdir(source_dir):
                if filename.endswith(".txt"):
                    source_path = os.path.join(source_dir, filename)
                    target_path = os.path.join(target_dir, filename)

                    with open(source_path, "r") as f:
                        content = f.read()

                    # 简单的质量检查：包含成功执行
                    if "Successful execution!" in content:
                        with open(target_path, "w") as f:
                            f.write(content)

        print(f"质量过滤数据集已创建: {target_dir}")

    def run_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个实验"""
        print(f"\n开始运行实验: {experiment['name']}")
        print(f"描述: {experiment['description']}")

        # 构建命令
        cmd = [
            "python",
            "-m",
            "habitat_llm.examples.planner_demo",
            f"--config-name={self.base_config}",
            "habitat.dataset.data_path=/home/shuqing/partnr-planner/task_classification_datasets_first100/rerange+spatial_first100.json.gz",
            "evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf",
            "evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf",
            "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct",
            "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct",
            "evaluation.agents.agent_0.planner.plan_config.enable_rag=True",
            "evaluation.agents.agent_1.planner.plan_config.enable_rag=True",
        ]

        # 添加配置覆盖
        for key, value in experiment["config_overrides"].items():
            cmd.append(f"{key}={value}")

        # 设置输出目录
        output_dir = (
            f"outputs/habitat_llm/{experiment['timestamp']}-{experiment['name']}"
        )
        cmd.append(f"evaluation.output_dir={output_dir}")

        print(f"命令: {' '.join(cmd)}")

        # 运行实验
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200
            )  # 2小时超时
            end_time = time.time()

            return {
                "name": experiment["name"],
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "output_dir": output_dir,
                "config_overrides": experiment["config_overrides"],
            }

        except subprocess.TimeoutExpired:
            return {
                "name": experiment["name"],
                "success": False,
                "error": "Timeout",
                "duration": 7200,
                "output_dir": output_dir,
                "config_overrides": experiment["config_overrides"],
            }
        except Exception as e:
            return {
                "name": experiment["name"],
                "success": False,
                "error": str(e),
                "output_dir": output_dir,
                "config_overrides": experiment["config_overrides"],
            }

    def analyze_results(self, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验结果"""
        if not experiment_result["success"]:
            return {"error": experiment_result.get("error", "Unknown error")}

        output_dir = experiment_result["output_dir"]
        results_dir = os.path.join(output_dir, "results")

        if not os.path.exists(results_dir):
            return {"error": "Results directory not found"}

        # 读取end_result_log.csv
        end_result_path = os.path.join(results_dir, "end_result_log.csv")
        if os.path.exists(end_result_path):
            with open(end_result_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # 解析最后一行（汇总结果）
                    last_line = lines[-1].strip().split(",")
                    if len(last_line) >= 4:
                        return {
                            "avg_replanning_count": float(last_line[0]),
                            "avg_runtime": float(last_line[1]),
                            "avg_task_percent_complete": float(last_line[2]),
                            "avg_task_state_success": float(last_line[3]),
                            "total_episodes": len(lines) - 1,
                        }

        return {"error": "Could not parse results"}

    def run_all_experiments(self):
        """运行所有实验"""
        print("开始运行优化实验...")

        # 创建质量过滤的RAG数据集
        self.create_rag_quality_filtered_dataset()

        # 设置实验
        self.setup_optimization_experiments()

        # 运行实验
        for experiment in self.experiments:
            result = self.run_experiment(experiment)
            analysis = self.analyze_results(result)

            self.results[experiment["name"]] = {
                "experiment": experiment,
                "result": result,
                "analysis": analysis,
            }

            print(f"\n实验 {experiment['name']} 完成:")
            if "error" in analysis:
                print(f"  错误: {analysis['error']}")
            else:
                print(f"  平均重规划次数: {analysis['avg_replanning_count']:.2f}")
                print(f"  平均运行时间: {analysis['avg_runtime']:.2f}")
                print(f"  平均任务完成度: {analysis['avg_task_percent_complete']:.3f}")
                print(f"  平均任务成功率: {analysis['avg_task_state_success']:.3f}")
                print(f"  总任务数: {analysis['total_episodes']}")

        # 保存结果
        self.save_results()

    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = f"optimization_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n实验结果已保存到: {results_file}")

        # 生成对比报告
        self.generate_comparison_report()

    def generate_comparison_report(self):
        """生成对比报告"""
        print("\n" + "=" * 80)
        print("优化实验对比报告")
        print("=" * 80)

        # 基准结果（实验4的结果）
        baseline = self.results.get("combined_optimization", {})
        baseline_analysis = baseline.get("analysis", {})

        if "error" not in baseline_analysis:
            print("\n基准结果 (组合优化):")
            print(f"  平均重规划次数: {baseline_analysis['avg_replanning_count']:.2f}")
            print(f"  平均任务成功率: {baseline_analysis['avg_task_state_success']:.3f}")
            print(f"  平均任务完成度: {baseline_analysis['avg_task_percent_complete']:.3f}")

        print("\n各实验对比:")
        for name, data in self.results.items():
            analysis = data.get("analysis", {})
            if "error" not in analysis:
                print(f"\n{name}:")
                print(f"  重规划次数: {analysis['avg_replanning_count']:.2f}")
                print(f"  成功率: {analysis['avg_task_state_success']:.3f}")
                print(f"  完成度: {analysis['avg_task_percent_complete']:.3f}")
                print(f"  运行时间: {analysis['avg_runtime']:.2f}")
            else:
                print(f"\n{name}: 错误 - {analysis['error']}")


def main():
    """主函数"""
    print("PartNR Planner 优化实验")
    print("=" * 50)

    # 创建实验管理器
    experiments = OptimizationExperiments()

    # 运行所有实验
    experiments.run_all_experiments()


if __name__ == "__main__":
    main()
