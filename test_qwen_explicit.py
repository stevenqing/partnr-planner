#!/usr/bin/env python3
"""
Test script with very explicit format instructions
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.qwen import Qwen


def test_explicit_format():
    """Test with very explicit format instructions"""

    # Load configuration
    qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")

    # Initialize Qwen model
    print("Loading Qwen model...")
    qwen = Qwen(qwen_cfg)

    # Very explicit prompt with multiple examples
    test_prompt = """<|im_start|>system
You are an agent that solves planning problems. You MUST follow the exact format specified.
<|im_end|><|im_start|>user
Task: Explore the living room.

Available actions:
- Explore[room_name]: Search a room
- Navigate[object_name]: Go to an object
- Pick[object_name]: Pick up an object
- Place[object_name, on, furniture]: Place an object

You MUST respond in this EXACT format:
Thought: <your reasoning>
<action with square brackets>
Assigned!

Examples of CORRECT responses:
Thought: I need to explore the living room to find objects
Explore[living_room_1]
Assigned!

Thought: I need to navigate to the table
Navigate[table_1]
Assigned!

Examples of WRONG responses:
Thought: I need to explore
Action: Enter the living room
Assigned!

Thought: I need to explore
Explore:living_room_1
Assigned!

Now respond to the task:
<|im_end|><|im_start|>assistant
"""

    print("Testing with explicit format instructions...")
    print("Input prompt:")
    print(test_prompt)
    print("\nGenerating...")

    qwen.generate_hf_llm(test_prompt, max_length=100, stop="Assigned!")

    print("Response:")
    print(repr(qwen.response))
    print("\nFormatted response:")
    print(qwen.response)

    # Check format
    if "[" in qwen.response and "]" in qwen.response:
        print("✓ Response contains square brackets")
    else:
        print("✗ Response missing square brackets")

    if ":" in qwen.response and "[" not in qwen.response:
        print("✗ Response contains colons instead of brackets")
    else:
        print("✓ No problematic colons found")

    if "Assigned!" in qwen.response:
        print("✓ Response contains stop token")
    else:
        print("✗ Response missing stop token")


if __name__ == "__main__":
    test_explicit_format()
