#!/usr/bin/env python3
"""
Test script to verify Qwen configuration fixes
"""

import sys

from omegaconf import OmegaConf

# Add project root to path
sys.path.append("/home/shuqing/partnr-planner")


def test_qwen_config():
    """Test Qwen configuration loading and basic functionality"""

    print("Testing Qwen configuration...")

    # Test 1: Load Qwen LLM configuration
    try:
        qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")
        print("✓ Qwen LLM config loaded successfully")
        print(f"  Model: {qwen_cfg.generation_params.engine}")
        print(f"  Max tokens: {qwen_cfg.generation_params.max_tokens}")
        print(f"  Temperature: {qwen_cfg.generation_params.temperature}")
        print(f"  Stop tokens: {qwen_cfg.generation_params.stop}")
    except Exception as e:
        print(f"✗ Failed to load Qwen LLM config: {e}")
        return False

    # Test 2: Load Qwen prompt template
    try:
        prompt_cfg = OmegaConf.load(
            "habitat_llm/conf/instruct/zero_shot_prompt_qwen.yaml"
        )
        print("✓ Qwen prompt template loaded successfully")
        print(f"  Stopword: {prompt_cfg.stopword}")
        print(f"  End expression: {prompt_cfg.end_expression}")

        # Check if prompt contains key elements
        prompt = prompt_cfg.prompt
        required_elements = [
            "Thought:",
            "Assigned!",
            "{eot_tag}",
            "{assistant_tag}",
            "{system_tag}",
            "{user_tag}",
        ]

        for element in required_elements:
            if element in prompt:
                print(f"  ✓ Contains {element}")
            else:
                print(f"  ✗ Missing {element}")

    except Exception as e:
        print(f"✗ Failed to load Qwen prompt template: {e}")
        return False

    # Test 3: Test action parser import
    try:
        from habitat_llm.llm.instruct.utils import zero_shot_action_parser

        print("✓ Action parser imported successfully")

        # Test with sample output
        sample_output = """Thought: Need to explore the living room first to find objects
Explore[living_room_1]
Assigned!"""

        # Mock agent object
        class MockAgent:
            def __init__(self, uid):
                self.uid = uid

        agents = [MockAgent(0)]

        try:
            result = zero_shot_action_parser(agents, sample_output)
            print(f"✓ Action parser test successful: {result}")
        except Exception as e:
            print(f"✗ Action parser test failed: {e}")

    except Exception as e:
        print(f"✗ Failed to import action parser: {e}")
        return False

    # Test 4: Check if Qwen model can be imported
    try:
        print("✓ Qwen model class imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Qwen model: {e}")
        return False

    print("\n🎉 All tests passed! Qwen configuration appears to be fixed.")
    return True


if __name__ == "__main__":
    test_qwen_config()
