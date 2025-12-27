#!/usr/bin/env python3
"""
测试Qwen专用修复方案
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from habitat_llm.llm.instruct.utils import (
    zero_shot_action_parser,
    zero_shot_action_parser_qwen,
)


class MockAgent:
    def __init__(self, uid):
        self.uid = uid


def test_qwen_specific_parser():
    """测试Qwen专用解析器"""

    agents = [MockAgent(0)]

    # 测试案例1: 包含Assigned!的响应
    response1 = """Thought: I need to explore the living room
Explore[living_room_1]
Assigned!"""

    print("测试案例1: 包含Assigned!")
    print(f"响应: {repr(response1)}")

    # 测试原始解析器
    result1_original = zero_shot_action_parser(agents, response1)
    print(f"原始解析器结果: {result1_original}")

    # 测试Qwen专用解析器
    result1_qwen = zero_shot_action_parser_qwen(agents, response1)
    print(f"Qwen专用解析器结果: {result1_qwen}")

    # 测试案例2: 包含Assigned的响应（没有感叹号）
    response2 = """Thought: I need to explore the living room
Explore[living_room_1]
Assigned"""

    print("\n测试案例2: 包含Assigned（无感叹号）")
    print(f"响应: {repr(response2)}")

    # 测试原始解析器
    result2_original = zero_shot_action_parser(agents, response2)
    print(f"原始解析器结果: {result2_original}")

    # 测试Qwen专用解析器
    result2_qwen = zero_shot_action_parser_qwen(agents, response2)
    print(f"Qwen专用解析器结果: {result2_qwen}")

    # 测试案例3: 没有停止标记的响应
    response3 = """Thought: I need to explore the living room
Explore[living_room_1]"""

    print("\n测试案例3: 没有停止标记")
    print(f"响应: {repr(response3)}")

    # 测试原始解析器
    result3_original = zero_shot_action_parser(agents, response3)
    print(f"原始解析器结果: {result3_original}")

    # 测试Qwen专用解析器
    result3_qwen = zero_shot_action_parser_qwen(agents, response3)
    print(f"Qwen专用解析器结果: {result3_qwen}")

    # 检查结果
    print("\n" + "=" * 50)
    print("结果分析:")

    # 检查是否有语法错误
    def has_syntax_error(result):
        return any(
            error_msg and "SyntaxError" in error_msg
            for _, _, error_msg in result.values()
        )

    for i, (original, qwen) in enumerate(
        [
            (result1_original, result1_qwen),
            (result2_original, result2_qwen),
            (result3_original, result3_qwen),
        ],
        1,
    ):
        print(f"案例{i}:")
        print(f"  原始解析器语法错误: {'是' if has_syntax_error(original) else '否'}")
        print(f"  Qwen解析器语法错误: {'是' if has_syntax_error(qwen) else '否'}")

        if not has_syntax_error(qwen):
            print("  ✅ Qwen解析器成功解析")
        else:
            print("  ❌ Qwen解析器仍有问题")


if __name__ == "__main__":
    test_qwen_specific_parser()
