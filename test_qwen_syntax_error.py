#!/usr/bin/env python3
"""
详细的单元测试来复现Qwen模型的语法错误
"""

import sys
import traceback
import unittest

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.instruct.utils import zero_shot_action_parser
from habitat_llm.llm.qwen import Qwen


class MockAgent:
    def __init__(self, uid):
        self.uid = uid


class TestQwenSyntaxError(unittest.TestCase):
    """测试Qwen模型的语法错误复现"""

    def setUp(self):
        """设置测试环境"""
        print("\n" + "=" * 60)
        print("设置测试环境...")

        # 加载配置
        self.qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")
        self.prompt_cfg = OmegaConf.load(
            "habitat_llm/conf/instruct/zero_shot_prompt_qwen.yaml"
        )

        # 初始化Qwen模型
        print("加载Qwen模型...")
        self.qwen = Qwen(self.qwen_cfg)

        # 创建模拟agent
        self.agents = [MockAgent(0)]

        print("测试环境设置完成")
        print("=" * 60)

    def test_1_simple_prompt_generation(self):
        """测试1: 简单提示词生成"""
        print("\n测试1: 简单提示词生成")
        print("-" * 40)

        # 简单的测试提示词
        test_prompt = """<|im_start|>system
You are an agent that solves planning problems.
<|im_end|><|im_start|>user
Task: Explore the living room.

What is the next action?
Return your response in this format:

Thought: <reasoning>
<action>
Assigned!
<|im_end|><|im_start|>assistant
"""

        print("输入提示词:")
        print(test_prompt)
        print("\n生成响应...")

        self.qwen.generate_hf_llm(test_prompt, max_length=100, stop="Assigned!")

        print("原始响应:")
        print(repr(self.qwen.response))
        print("\n格式化响应:")
        print(self.qwen.response)

        # 检查响应格式
        self._check_response_format(self.qwen.response, "简单提示词")

    def test_2_actual_demo_prompt(self):
        """测试2: 实际demo中使用的提示词"""
        print("\n测试2: 实际demo中使用的提示词")
        print("-" * 40)

        # 模拟实际demo中的提示词
        world_desc = "Furniture:\nliving_room_1: table_1, sofa_2\nbedroom_1: bed_1, nightstand_2\n\nObjects:\nNo objects found yet"
        tool_desc = "- Explore: Search a specific room. Example (Explore[living_room_1])\n- Navigate: Used for navigating to an entity. Example (Navigate[counter_22])\n- Pick: Used for picking up an object. Example (Pick[cup_1])\n- Place: Used for placing an object on a target location. Example (Place[cup_1, on, table_1])\n- Rearrange: Used for moving an object from its current location to the target location. Example (Rearrange[cup_1, on, table_1])\n- Done: Used to indicate that the agent has finished the task. Example (Done[])"

        demo_prompt = self.prompt_cfg.prompt.format(
            system_tag=self.qwen_cfg.system_tag,
            eot_tag=self.qwen_cfg.eot_tag,
            user_tag=self.qwen_cfg.user_tag,
            assistant_tag=self.qwen_cfg.assistant_tag,
            input="Move the laptop stand and monitor stand from the living room table to the bedroom bed. Place them next to each other on the bed.",
            world_description=world_desc,
            tool_descriptions=tool_desc,
            agent_role_description="You are Agent 0 (Robot). You can navigate, pick, place, and rearrange objects.",
            rag_examples="",
        )

        print("输入提示词长度:", len(demo_prompt))
        print("提示词片段:")
        print(demo_prompt[:500] + "...")
        print("\n生成响应...")

        self.qwen.generate_hf_llm(demo_prompt, max_length=250, stop="Assigned!")

        print("原始响应:")
        print(repr(self.qwen.response))
        print("\n格式化响应:")
        print(self.qwen.response)

        # 检查响应格式
        self._check_response_format(self.qwen.response, "实际demo提示词")

    def test_3_action_parser_integration(self):
        """测试3: 动作解析器集成测试"""
        print("\n测试3: 动作解析器集成测试")
        print("-" * 40)

        # 测试不同的响应格式
        test_cases = [
            {
                "name": "正确格式",
                "response": "Thought: I need to explore the living room\nExplore[living_room_1]\nAssigned!",
                "should_pass": True,
            },
            {
                "name": "错误格式 - 使用冒号",
                "response": "Thought: I need to explore the living room\nExplore:living_room_1\nAssigned!",
                "should_pass": False,
            },
            {
                "name": "错误格式 - 缺少括号",
                "response": "Thought: I need to explore the living room\nExplore living_room_1\nAssigned!",
                "should_pass": False,
            },
            {
                "name": "错误格式 - 自然语言",
                "response": "Thought: I need to explore the living room\nI will enter the living room and look around\nAssigned!",
                "should_pass": False,
            },
        ]

        for test_case in test_cases:
            print(f"\n测试案例: {test_case['name']}")
            print(f"响应: {test_case['response']}")

            try:
                # 使用zero_shot_action_parser
                result = zero_shot_action_parser(self.agents, test_case["response"])
                print(f"解析结果: {result}")

                # 检查是否包含语法错误
                has_syntax_error = any(
                    error_msg and "SyntaxError" in error_msg
                    for _, _, error_msg in result.values()
                )

                if test_case["should_pass"]:
                    self.assertFalse(has_syntax_error, f"应该通过但出现了语法错误: {result}")
                    print("✓ 测试通过")
                else:
                    self.assertTrue(has_syntax_error, f"应该失败但没有语法错误: {result}")
                    print("✓ 正确检测到语法错误")

            except Exception as e:
                print(f"解析异常: {e}")
                traceback.print_exc()

    def test_4_repetitive_output_issue(self):
        """测试4: 重复输出问题"""
        print("\n测试4: 重复输出问题")
        print("-" * 40)

        # 使用可能导致重复输出的提示词
        repetitive_prompt = """<|im_start|>system
You are an agent that solves planning problems. You must follow the exact format specified.
<|im_end|><|im_start|>user
Task: Move the laptop stand and monitor stand from the living room table to the bedroom bed.

Furniture:
living_room_1: table_1, sofa_2
bedroom_1: bed_1, nightstand_2

Objects:
No objects found yet

Possible Actions:
- Explore: Search a specific room. Example (Explore[living_room_1])
- Navigate: Used for navigating to an entity. Example (Navigate[counter_22])
- Pick: Used for picking up an object. Example (Pick[cup_1])
- Place: Used for placing an object on a target location. Example (Place[cup_1, on, table_1])
- Rearrange: Used for moving an object from its current location to the target location. Example (Rearrange[cup_1, on, table_1])
- Done: Used to indicate that the agent has finished the task. Example (Done[])

What is the next action to make progress towards completing the task?
Return your response in the following format

Thought: <reasoning for why you are taking the next action>
<next action call>
Assigned!

Here are examples of CORRECT responses:
Thought: Since there are no objects found I should explore a room I have not explored yet
Explore[living_room_1]
Assigned!

Thought: I need to navigate to the table
Navigate[table_1]
Assigned!

Here are examples of WRONG responses (DO NOT DO THIS):
Thought: I need to explore
Action: Enter the living room
Assigned!

Thought: I need to explore
Explore:living_room_1
Assigned!

CRITICAL FORMATTING RULES:
1. ALWAYS use square brackets [] around parameters, NEVER use colons :
2. Correct format: Explore[living_room_1]
3. WRONG format: Explore:living_room_1 or Explore(living_room_1)
4. Keep your response concise and avoid repetition
5. Stop generating after the "Assigned!" marker
<|im_end|><|im_start|>assistant
"""

        print("输入提示词长度:", len(repetitive_prompt))
        print("生成响应...")

        self.qwen.generate_hf_llm(repetitive_prompt, max_length=250, stop="Assigned!")

        print("原始响应:")
        print(repr(self.qwen.response))
        print("\n格式化响应:")
        print(self.qwen.response)

        # 检查是否有重复内容
        response_text = self.qwen.response
        words = response_text.split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            print(f"重复率: {1 - repetition_ratio:.2%}")

            if repetition_ratio < 0.5:
                print("⚠️  检测到高重复率")
            else:
                print("✓ 重复率正常")

        # 检查响应格式
        self._check_response_format(self.qwen.response, "重复输出测试")

    def test_5_tokenizer_behavior(self):
        """测试5: 分词器行为"""
        print("\n测试5: 分词器行为")
        print("-" * 40)

        # 测试关键字符串的分词
        test_strings = [
            "Explore[living_room_1]",
            "Explore:living_room_1",
            "Assigned!",
            "Thought: I need to explore",
            "<|im_start|>assistant",
            "<|im_end|>",
        ]

        for test_string in test_strings:
            tokens = self.qwen.tokenizer.tokenize(test_string)
            encoded = self.qwen.tokenizer.encode(test_string)
            decoded = self.qwen.tokenizer.decode(encoded)

            print(f"字符串: '{test_string}'")
            print(f"分词: {tokens}")
            print(f"编码/解码: '{decoded}'")
            print(f"是否一致: {test_string == decoded}")
            print()

    def _check_response_format(self, response, test_name):
        """检查响应格式的辅助方法"""
        print(f"\n{test_name} - 格式检查:")

        # 检查是否包含方括号
        has_brackets = "[" in response and "]" in response
        print(f"包含方括号: {'✓' if has_brackets else '✗'}")

        # 检查是否包含冒号（错误格式）
        has_colons = ":" in response and "[" not in response
        print(f"包含错误冒号: {'✗' if has_colons else '✓'}")

        # 检查是否包含停止标记
        has_stop = "Assigned!" in response
        print(f"包含停止标记: {'✓' if has_stop else '✗'}")

        # 检查响应长度
        print(f"响应长度: {len(response)} 字符")

        # 检查是否包含重复内容
        words = response.split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            print(f"重复率: {1 - repetition_ratio:.2%}")

        return {
            "has_brackets": has_brackets,
            "has_colons": has_colons,
            "has_stop": has_stop,
            "length": len(response),
            "repetition_ratio": 1 - repetition_ratio if len(words) > 0 else 0,
        }


def run_comprehensive_test():
    """运行综合测试"""
    print("开始Qwen语法错误复现测试")
    print("=" * 60)

    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQwenSyntaxError)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("测试完成")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    return result


if __name__ == "__main__":
    run_comprehensive_test()
