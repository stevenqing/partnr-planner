#!/usr/bin/env python3
"""
运行demo并记录日志
"""

import subprocess


def run_demo_with_logging():
    """运行demo并记录日志"""

    print("开始运行demo...")

    # 运行demo命令
    cmd = [
        "conda",
        "run",
        "-n",
        "habitat-llm",
        "python",
        "-m",
        "habitat_llm.examples.planner_demo",
        "--config-name",
        "baselines/decentralized_zero_shot_react_summary.yaml",
        "habitat.dataset.data_path=/home/shuqing/partnr-planner/task_classification_datasets/rerange+spatial_matched_subtasks.json.gz",
        "llm@evaluation.agents.agent_0.planner.plan_config.llm=qwen",
        "llm@evaluation.agents.agent_1.planner.plan_config.llm=qwen",
        "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.max_tokens=800",
        "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.max_tokens=800",
        "evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf",
        "evaluation.agents.agent_1.planner.plan_config.llm.inference_mode=hf",
        "evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=Qwen/Qwen2.5-7B-Instruct",
        "evaluation.agents.agent_1.planner.plan_config.llm.generation_params.engine=Qwen/Qwen2.5-7B-Instruct",
        "instruct@evaluation.agents.agent_0.planner.plan_config.instruct=zero_shot_prompt_qwen",
        "instruct@evaluation.agents.agent_1.planner.plan_config.instruct=zero_shot_prompt_qwen",
    ]

    try:
        # 启动进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        syntax_error_count = 0
        line_count = 0

        # 读取输出
        for line in process.stdout:
            line_count += 1
            print(line.rstrip())

            # 检查语法错误
            if "SyntaxError" in line:
                syntax_error_count += 1
                print(f"🚨 发现语法错误 #{syntax_error_count}")

            # 每100行检查一次
            if line_count % 100 == 0:
                print(f"📊 已处理 {line_count} 行，发现 {syntax_error_count} 个语法错误")

            # 如果发现太多错误，提前退出
            if syntax_error_count >= 5:
                print("🛑 发现太多语法错误，提前退出")
                process.terminate()
                break

        # 等待进程结束
        process.wait()

        print("\n📊 最终统计:")
        print(f"  总行数: {line_count}")
        print(f"  语法错误数: {syntax_error_count}")

        if syntax_error_count == 0:
            print("✅ 没有发现语法错误！")
        else:
            print(f"❌ 发现 {syntax_error_count} 个语法错误")

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
        process.terminate()
    except Exception as e:
        print(f"❌ 运行出错: {e}")


if __name__ == "__main__":
    run_demo_with_logging()
