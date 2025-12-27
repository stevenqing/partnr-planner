#!/usr/bin/env python3
"""
简单测试配置
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf


def test_simple_config():
    """简单测试配置"""

    print("测试Qwen配置...")

    try:
        # 加载Qwen prompt配置
        prompt_cfg = OmegaConf.load(
            "habitat_llm/conf/instruct/zero_shot_prompt_qwen.yaml"
        )

        print("配置加载成功")
        print(f"配置内容: {prompt_cfg}")

        # 检查actions_parser
        if hasattr(prompt_cfg, "actions_parser"):
            print(f"actions_parser存在: {prompt_cfg.actions_parser}")
        else:
            print("actions_parser不存在")

    except Exception as e:
        print(f"配置加载失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_simple_config()
