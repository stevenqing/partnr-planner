#!/usr/bin/env python3
"""
最终验证测试 - 确认Qwen模型语法错误已修复
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.instruct.utils import zero_shot_action_parser_qwen
from habitat_llm.llm.qwen import Qwen


class MockAgent:
    def __init__(self, uid):
        self.uid = uid


def test_final_verification():
    """最终验证测试"""

    print("=" * 60)
    print("最终验证测试 - Qwen模型语法错误修复")
    print("=" * 60)

    # 加载配置
    qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")
    prompt_cfg = OmegaConf.load("habitat_llm/conf/instruct/zero_shot_prompt_qwen.yaml")

    # 初始化Qwen模型
    print("加载Qwen模型...")
    qwen = Qwen(qwen_cfg)

    # 创建模拟agent
    agents = [MockAgent(0)]

    # 测试1: 使用实际demo的提示词格式
    print("\n测试1: 实际demo提示词格式")
    print("-" * 40)

    # 模拟实际demo中的提示词
    world_desc = "Furniture:\nliving_room_1: table_1, sofa_2\nbedroom_1: bed_1, nightstand_2\n\nObjects:\nNo objects found yet"
    tool_desc = "- Explore: Search a specific room. Example (Explore[living_room_1])\n- Navigate: Used for navigating to an entity. Example (Navigate[counter_22])\n- Pick: Used for picking up an object. Example (Pick[cup_1])\n- Place: Used for placing an object on a target location. Example (Place[cup_1, on, table_1])\n- Rearrange: Used for moving an object from its current location to the target location. Example (Rearrange[cup_1, on, table_1])\n- Done: Used to indicate that the agent has finished the task. Example (Done[])"

    demo_prompt = prompt_cfg.prompt.format(
        system_tag=qwen_cfg.system_tag,
        eot_tag=qwen_cfg.eot_tag,
        user_tag=qwen_cfg.user_tag,
        assistant_tag=qwen_cfg.assistant_tag,
        input="Move the laptop stand and monitor stand from the living room table to the bedroom bed. Place them next to each other on the bed.",
        world_description=world_desc,
        tool_descriptions=tool_desc,
        agent_role_description="You are Agent 0 (Robot). You can navigate, pick, place, and rearrange objects.",
        rag_examples="",
    )

    print("生成响应...")
    qwen.generate_hf_llm(demo_prompt, max_length=250, stop="Assigned!")

    print(f"响应长度: {len(qwen.response)} 字符")
    print(f"响应内容: {repr(qwen.response)}")
    print("格式化响应:")
    print(qwen.response)

    # 检查格式
    has_brackets = "[" in qwen.response and "]" in qwen.response
    has_colons = ":" in qwen.response and "[" not in qwen.response
    has_stop = "Assigned" in qwen.response

    print("\n格式检查:")
    print(f"  包含方括号: {'✓' if has_brackets else '✗'}")
    print(f"  包含错误冒号: {'✗' if has_colons else '✓'}")
    print(f"  包含停止标记: {'✓' if has_stop else '✗'}")

    # 测试2: 动作解析器测试
    print("\n测试2: 动作解析器测试")
    print("-" * 40)

    # 构造完整的响应格式（包含Assigned!）
    full_response = (
        qwen.response + "\nAssigned!"
        if "Assigned" not in qwen.response
        else qwen.response
    )

    print(f"完整响应: {repr(full_response)}")

    try:
        result = zero_shot_action_parser_qwen(agents, full_response)
        print(f"解析结果: {result}")

        # 检查是否包含语法错误
        has_syntax_error = any(
            error_msg and "SyntaxError" in error_msg
            for _, _, error_msg in result.values()
        )

        if has_syntax_error:
            print("❌ 仍然有语法错误")
            return False
        else:
            print("✅ 没有语法错误，解析成功")

    except Exception as e:
        print(f"❌ 解析异常: {e}")
        return False

    # 测试3: 多次生成测试
    print("\n测试3: 多次生成测试")
    print("-" * 40)

    success_count = 0
    total_tests = 3

    for i in range(total_tests):
        print(f"\n第 {i+1} 次测试:")
        qwen.generate_hf_llm(demo_prompt, max_length=250, stop="Assigned!")

        has_brackets = "[" in qwen.response and "]" in qwen.response
        has_colons = ":" in qwen.response and "[" not in qwen.response

        if has_brackets and not has_colons:
            print("  ✅ 格式正确")
            success_count += 1
        else:
            print("  ❌ 格式错误")
            print(f"    响应: {repr(qwen.response)}")

    print(
        f"\n成功率: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)"
    )

    return success_count == total_tests


if __name__ == "__main__":
    success = test_final_verification()
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！Qwen模型语法错误已修复！")
    else:
        print("❌ 仍有问题需要解决")
    print("=" * 60)
    exit(0 if success else 1)
