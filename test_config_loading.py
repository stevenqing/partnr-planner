#!/usr/bin/env python3
"""
测试配置加载是否正确
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf


def test_config_loading():
    """测试配置加载"""

    print("测试Qwen配置加载...")

    # 加载Qwen prompt配置
    prompt_cfg = OmegaConf.load("habitat_llm/conf/instruct/zero_shot_prompt_qwen.yaml")

    print("actions_parser配置:")
    print(f"  _target: {prompt_cfg.actions_parser._target}")
    print(f"  _partial: {prompt_cfg.actions_parser._partial}")

    # 检查是否正确指向了Qwen专用解析器
    if "zero_shot_action_parser_qwen" in prompt_cfg.actions_parser._target:
        print("✅ 配置正确，指向Qwen专用解析器")
    else:
        print("❌ 配置错误，没有指向Qwen专用解析器")

    # 测试解析器导入
    try:
        from habitat_llm.llm.instruct.utils import zero_shot_action_parser_qwen

        print("✅ Qwen专用解析器导入成功")
    except ImportError as e:
        print(f"❌ Qwen专用解析器导入失败: {e}")


if __name__ == "__main__":
    test_config_loading()
