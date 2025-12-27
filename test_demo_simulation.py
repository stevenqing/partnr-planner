#!/usr/bin/env python3
"""
模拟demo运行过程
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.instruct.utils import zero_shot_action_parser_qwen
from habitat_llm.llm.qwen import Qwen


class MockAgent:
    def __init__(self, uid):
        self.uid = uid


def test_demo_simulation():
    """模拟demo运行过程"""

    print("模拟demo运行过程...")

    # 加载配置
    qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")
    prompt_cfg = OmegaConf.load("habitat_llm/conf/instruct/zero_shot_prompt_qwen.yaml")

    # 初始化Qwen模型
    print("加载Qwen模型...")
    qwen = Qwen(qwen_cfg)

    # 创建模拟agent
    agents = [MockAgent(0)]

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

    print(f"响应: {repr(qwen.response)}")
    print("格式化响应:")
    print(qwen.response)

    # 使用Qwen专用解析器
    print("\n使用Qwen专用解析器...")
    result = zero_shot_action_parser_qwen(agents, qwen.response)
    print(f"解析结果: {result}")

    # 检查是否有语法错误
    has_syntax_error = any(
        error_msg and "SyntaxError" in error_msg for _, _, error_msg in result.values()
    )

    if has_syntax_error:
        print("❌ 仍然有语法错误")
        return False
    else:
        print("✅ 没有语法错误，解析成功")
        return True


if __name__ == "__main__":
    success = test_demo_simulation()
    print(f"\n测试结果: {'成功' if success else '失败'}")
