#!/usr/bin/env python3
"""
Test script to debug Qwen generation issues
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.qwen import Qwen


def test_qwen_generation():
    """Test Qwen generation with different configurations"""

    # Load configuration
    qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")

    print("Testing Qwen generation...")
    print(
        f"Config: max_tokens={qwen_cfg.generation_params.max_tokens}, stop='{qwen_cfg.generation_params.stop}'"
    )

    # Initialize Qwen model
    print("Loading Qwen model...")
    qwen = Qwen(qwen_cfg)

    # Test 1: Simple prompt with explicit stop
    print("\n=== Test 1: Simple prompt with explicit stop ===")
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

    print("Input prompt:")
    print(test_prompt)
    print("\nGenerating...")

    qwen.generate_hf_llm(test_prompt, max_length=50, stop="Assigned!")

    print("Response:")
    print(repr(qwen.response))
    print("\nFormatted response:")
    print(qwen.response)

    # Check if response contains proper format
    if "[" in qwen.response and "]" in qwen.response:
        print("✓ Response contains square brackets")
    else:
        print("✗ Response missing square brackets")

    if "Assigned!" in qwen.response:
        print("✓ Response contains stop token")
    else:
        print("✗ Response missing stop token")

    # Test 2: Test with different stop tokens
    print("\n=== Test 2: Test with different stop tokens ===")

    # Test with newline as stop
    qwen.generate_hf_llm(test_prompt, max_length=50, stop="\n")
    print("Response with newline stop:")
    print(repr(qwen.response))

    # Test 3: Test tokenizer behavior
    print("\n=== Test 3: Tokenizer behavior ===")

    # Test tokenization of the stop token
    stop_token = "Assigned!"
    tokens = qwen.tokenizer.tokenize(stop_token)
    print(f"Stop token '{stop_token}' tokens: {tokens}")

    # Test if stop token is properly recognized
    test_text = "Thought: I need to explore\nExplore[living_room_1]\nAssigned!"
    tokens = qwen.tokenizer.tokenize(test_text)
    print(f"Full text tokens: {tokens}")

    # Test 4: Test generation config
    print("\n=== Test 4: Generation config ===")
    print(f"Model device: {qwen.model.device}")
    print(
        f"Tokenizer pad token: {qwen.tokenizer.pad_token} (id: {qwen.tokenizer.pad_token_id})"
    )
    print(
        f"Tokenizer eos token: {qwen.tokenizer.eos_token} (id: {qwen.tokenizer.eos_token_id})"
    )


if __name__ == "__main__":
    test_qwen_generation()
