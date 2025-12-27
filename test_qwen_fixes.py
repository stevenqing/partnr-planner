#!/usr/bin/env python3
"""
Test script to verify Qwen model fixes
Tests:
1. Stop token handling with "Assigned!"
2. Special token cleaning
3. Action parsing compatibility
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.instruct.utils import zero_shot_action_parser_qwen
from habitat_llm.llm.qwen import Qwen


def test_qwen_fixes():
    """Test Qwen model with the implemented fixes"""

    print("=" * 70)
    print("TESTING QWEN MODEL FIXES")
    print("=" * 70)

    # Load configuration
    qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")

    print("\n[1] Loading Qwen model...")
    print(f"    Model: {qwen_cfg.generation_params.engine}")
    print(f"    Stop token: '{qwen_cfg.generation_params.stop}'")
    print(f"    Max tokens: {qwen_cfg.generation_params.max_tokens}")

    # Initialize Qwen model
    qwen = Qwen(qwen_cfg)

    # Test prompt
    test_prompt = """<|im_start|>system
You are an agent that solves multi-agent planning problems. You strictly follow any format specifications.

Many calls to the same action in a row are a sign that something has gone wrong and you should try a different action.<|im_end|><|im_start|>user
Task: Move the laptop stand and monitor stand from the living room table to the bedroom bed.

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

    print("\n[2] Testing generation with stop token handling...")
    print(f"    Generating with max_tokens={qwen_cfg.generation_params.max_tokens}")

    # Generate response
    qwen.generate_hf_llm(test_prompt, max_length=150)

    print("\n" + "=" * 70)
    print("QWEN OUTPUT:")
    print("=" * 70)
    print(f"Raw response repr: {repr(qwen.response)}")
    print("\nFormatted output:")
    print(qwen.response)
    print("=" * 70)

    # Check for issues
    print("\n[3] Checking for common issues...")

    issues = []

    # Check if Assigned! is present (should NOT be in cleaned response)
    if "Assigned!" in qwen.response:
        issues.append("❌ Stop token 'Assigned!' still in output (should be removed)")
    else:
        print("✓ Stop token properly removed from output")

    # Check for special tokens
    special_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
    found_special = [t for t in special_tokens if t in qwen.response]
    if found_special:
        issues.append(f"❌ Special tokens found: {found_special}")
    else:
        print("✓ No special tokens in output")

    # Check length (shouldn't be excessively long)
    if len(qwen.response) > 500:
        issues.append(
            f"❌ Response too long ({len(qwen.response)} chars) - may indicate failed stopping"
        )
    else:
        print(f"✓ Response length reasonable: {len(qwen.response)} chars")

    # Check for square brackets (correct format)
    has_brackets = "[" in qwen.response and "]" in qwen.response
    if has_brackets:
        print("✓ Action uses square brackets (correct format)")
    else:
        issues.append("❌ No square brackets found in action")

    # Check for repetition
    lines = qwen.response.strip().split("\n")
    if len(lines) > 10:
        issues.append(f"❌ Too many lines ({len(lines)}) - possible repetition")
    else:
        print(f"✓ Reasonable number of lines: {len(lines)}")

    print("\n[4] Testing action parser...")

    # Create mock agent for parser
    class MockAgent:
        def __init__(self, uid):
            self.uid = uid

    agents = [MockAgent(0)]

    try:
        parsed = zero_shot_action_parser_qwen(agents, qwen.response)
        print("✓ Action parsed successfully:")
        for agent_id, (skill_name, args, _params) in parsed.items():
            print(f"    Agent {agent_id}: {skill_name}[{args}]")
    except Exception as e:
        issues.append(f"❌ Parser failed: {e}")
        print(f"❌ Action parsing failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    if not issues:
        print("✅ ALL TESTS PASSED! Qwen model fixes are working correctly.")
    else:
        print("❌ SOME TESTS FAILED:")
        for issue in issues:
            print(f"   {issue}")

    print("\n" + "=" * 70)

    return len(issues) == 0


if __name__ == "__main__":
    try:
        success = test_qwen_fixes()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
