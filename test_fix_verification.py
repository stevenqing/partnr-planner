#!/usr/bin/env python3
"""
验证修复的简单测试
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from habitat_llm.llm.instruct.utils import zero_shot_action_parser


class MockAgent:
    def __init__(self, uid):
        self.uid = uid


def test_fix():
    """测试修复后的解析器"""

    agents = [MockAgent(0)]

    # 测试案例：包含Assigned!的正确格式响应
    response = """Thought: I need to explore the living room
Explore[living_room_1]
Assigned!"""

    print("测试响应:")
    print(response)
    print()

    result = zero_shot_action_parser(agents, response)
    print(f"解析结果: {result}")

    # 检查是否包含语法错误
    has_syntax_error = any(
        error_msg and "SyntaxError" in error_msg for _, _, error_msg in result.values()
    )

    if has_syntax_error:
        print("❌ 仍然有语法错误")
        return False
    else:
        print("✅ 修复成功，没有语法错误")
        return True


if __name__ == "__main__":
    success = test_fix()
    exit(0 if success else 1)
