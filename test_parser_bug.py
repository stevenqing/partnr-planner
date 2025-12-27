#!/usr/bin/env python3
"""
测试动作解析器的bug
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from habitat_llm.llm.instruct.utils import zero_shot_action_parser


class MockAgent:
    def __init__(self, uid):
        self.uid = uid


def test_parser_bug():
    """测试解析器的bug"""

    agents = [MockAgent(0)]

    # 测试案例1: 正确格式的响应
    correct_response = """Thought: I need to explore the living room
Explore[living_room_1]
Assigned!"""

    print("测试案例1: 正确格式")
    print("响应:")
    print(correct_response)
    print()

    # 检查zero_shot_action_parser如何分割响应
    lines = correct_response.strip().split("\n")
    print(f"分割后的行数: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"行 {i}: '{line}'")

    print(f"\n最后一行 (会被用作动作): '{lines[-1]}'")
    print(f"倒数第二行 (实际的动作): '{lines[-2]}'")

    # 测试解析
    result = zero_shot_action_parser(agents, correct_response)
    print(f"\n解析结果: {result}")

    # 测试案例2: 没有Assigned!的响应
    response_without_assigned = """Thought: I need to explore the living room
Explore[living_room_1]"""

    print("\n" + "=" * 50)
    print("测试案例2: 没有Assigned!")
    print("响应:")
    print(response_without_assigned)
    print()

    lines2 = response_without_assigned.strip().split("\n")
    print(f"分割后的行数: {len(lines2)}")
    for i, line in enumerate(lines2):
        print(f"行 {i}: '{line}'")

    print(f"\n最后一行 (会被用作动作): '{lines2[-1]}'")

    # 测试解析
    result2 = zero_shot_action_parser(agents, response_without_assigned)
    print(f"\n解析结果: {result2}")

    # 测试案例3: 只有动作行
    action_only = "Explore[living_room_1]"

    print("\n" + "=" * 50)
    print("测试案例3: 只有动作行")
    print("响应:")
    print(action_only)
    print()

    result3 = zero_shot_action_parser(agents, action_only)
    print(f"解析结果: {result3}")


if __name__ == "__main__":
    test_parser_bug()
