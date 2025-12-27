#!/usr/bin/env python3
"""
Test script to debug Qwen format issues
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.qwen import Qwen


def test_qwen_format():
    """Test Qwen with a very simple, explicit prompt"""

    # Load configuration
    qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")

    # Initialize Qwen model
    print("Loading Qwen model...")
    qwen = Qwen(qwen_cfg)

    # Create a very simple, explicit test prompt
    test_prompt = """<|im_start|>system
You are an agent that solves planning problems. You must follow the exact format specified.
<|im_end|><|im_start|>user
Task: Explore the living room.

Possible Actions:
- Explore: Search a room. Example (Explore[living_room_1])

What is the next action?
Return your response in this EXACT format:

Thought: <your reasoning>
<action with square brackets>
Assigned!

Example:
Thought: I need to explore the living room
Explore[living_room_1]
Assigned!
<|im_end|><|im_start|>assistant
"""

    print("Testing Qwen generation with explicit format...")
    print("Input prompt:")
    print(test_prompt)
    print("\n" + "=" * 50)

    # Generate response
    qwen.generate_hf_llm(test_prompt, max_length=50)

    print("QWEN OUTPUT:")
    print("=" * 50)
    print(repr(qwen.response))
    print("\nFORMATTED OUTPUT:")
    print("=" * 50)
    print(qwen.response)
    print("=" * 50)

    # Check if output contains proper brackets
    if "[" in qwen.response and "]" in qwen.response:
        print("✓ Output contains square brackets")
    else:
        print("✗ Output missing square brackets")

    if ":" in qwen.response and "[" not in qwen.response:
        print("✗ Output contains colons instead of brackets")
    else:
        print("✓ No problematic colons found")


if __name__ == "__main__":
    test_qwen_format()
