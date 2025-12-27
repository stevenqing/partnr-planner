#!/usr/bin/env python3
"""
Debug script to test Qwen LLM output format
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.qwen import Qwen


def test_qwen_output():
    """Test Qwen LLM output to debug formatting issues"""

    # Load configuration
    qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")
    OmegaConf.load("habitat_llm/conf/instruct/zero_shot_prompt_qwen.yaml")

    # Initialize Qwen model
    print("Loading Qwen model...")
    qwen = Qwen(qwen_cfg)

    # Create a simple test prompt
    test_prompt = """<|im_start|>system
You are an agent that solves multi-agent planning problems. You strictly follow any format specifications.

Many calls to the same action in a row are a sign that something has gone wrong and you should try a different action.<|im_end|><|im_start|>user
Task: Move the laptop stand and monitor stand from the living room table to the bedroom bed. Place them next to each other on the bed.

Furniture:
living_room_1: table_1, sofa_2
bedroom_1: bed_1, nightstand_2

Objects:
No objects found yet

Possible Actions:
- Explore: Search a specific room. Example (Explore[living_room_1])
- Done: Used to indicate that the agent has finished the task. Example (Done[])

What is the next action to make progress towards completing the task?
Return your response in the following format

Thought: <reasoning for why you are taking the next action>
<next action call>
Assigned!

Here is an example:
Thought: Since there are no objects found I should explore a room I have not explored yet
Explore[living_room_1]
Assigned!
<|im_end|><|im_start|>assistant
"""

    print("Testing Qwen generation...")
    print("Input prompt length:", len(test_prompt))

    # Generate response
    qwen.generate_hf_llm(test_prompt, max_length=150)

    print("\n" + "=" * 50)
    print("QWEN OUTPUT:")
    print("=" * 50)
    print(repr(qwen.response))
    print("\n" + "=" * 50)
    print("FORMATTED OUTPUT:")
    print("=" * 50)
    print(qwen.response)
    print("=" * 50)

    # Test action parser
    from habitat_llm.llm.instruct.utils import zero_shot_action_parser

    class MockAgent:
        def __init__(self, uid):
            self.uid = uid

    agents = [MockAgent(0)]

    try:
        parsed = zero_shot_action_parser(agents, qwen.response)
        print("PARSED ACTION:", parsed)
    except Exception as e:
        print("PARSING ERROR:", e)
        print("Last line:", qwen.response.strip().split("\n")[-1])

        # Debug the action parser step by step
        from habitat_llm.llm.instruct.utils import actions_parser

        action_line = qwen.response.strip().split("\n")[-1]
        agent_id = agents[0].uid
        formatted_line = f"Agent_{agent_id}_Action: {action_line}"
        print("Formatted for actions_parser:", formatted_line)

        try:
            parsed2 = actions_parser(agents, formatted_line)
            print("ACTIONS_PARSER RESULT:", parsed2)
        except Exception as e2:
            print("ACTIONS_PARSER ERROR:", e2)


if __name__ == "__main__":
    test_qwen_output()
