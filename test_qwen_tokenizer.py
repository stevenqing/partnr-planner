#!/usr/bin/env python3
"""
Test script to debug Qwen tokenizer issues
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from transformers import AutoTokenizer


def test_qwen_tokenizer():
    """Test Qwen tokenizer behavior with different configurations"""

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print("Testing Qwen tokenizer configurations...")

    # Test 1: Fast tokenizer (default)
    print("\n=== Test 1: Fast tokenizer ===")
    try:
        tokenizer_fast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print("✓ Fast tokenizer loaded successfully")

        # Test tokenization of square brackets
        test_text = "Explore[living_room_1]"
        tokens_fast = tokenizer_fast.tokenize(test_text)
        print(f"Fast tokenizer tokens for '{test_text}': {tokens_fast}")

        # Test encoding/decoding
        encoded = tokenizer_fast.encode(test_text)
        decoded = tokenizer_fast.decode(encoded)
        print(f"Fast tokenizer encode/decode: '{test_text}' -> '{decoded}'")

    except Exception as e:
        print(f"✗ Fast tokenizer failed: {e}")

    # Test 2: Slow tokenizer (current config)
    print("\n=== Test 2: Slow tokenizer (current config) ===")
    try:
        tokenizer_slow = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("✓ Slow tokenizer loaded successfully")

        # Test tokenization of square brackets
        test_text = "Explore[living_room_1]"
        tokens_slow = tokenizer_slow.tokenize(test_text)
        print(f"Slow tokenizer tokens for '{test_text}': {tokens_slow}")

        # Test encoding/decoding
        encoded = tokenizer_slow.encode(test_text)
        decoded = tokenizer_slow.decode(encoded)
        print(f"Slow tokenizer encode/decode: '{test_text}' -> '{decoded}'")

    except Exception as e:
        print(f"✗ Slow tokenizer failed: {e}")

    # Test 3: Check special tokens
    print("\n=== Test 3: Special tokens ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        print(f"UNK token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

        # Check if pad token is set
        if tokenizer.pad_token is None:
            print("⚠️  PAD token is None - this might cause issues!")
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set PAD token to EOS token: {tokenizer.pad_token}")

    except Exception as e:
        print(f"✗ Special tokens test failed: {e}")

    # Test 4: Test chat format tokens
    print("\n=== Test 4: Chat format tokens ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # Test the chat format tokens used in the prompt
        system_tag = "<|im_start|>system\n"
        user_tag = "<|im_start|>user\n"
        assistant_tag = "<|im_start|>assistant\n"
        eot_tag = "<|im_end|>"

        print(f"System tag: '{system_tag}' -> {tokenizer.tokenize(system_tag)}")
        print(f"User tag: '{user_tag}' -> {tokenizer.tokenize(user_tag)}")
        print(
            f"Assistant tag: '{assistant_tag}' -> {tokenizer.tokenize(assistant_tag)}"
        )
        print(f"EOT tag: '{eot_tag}' -> {tokenizer.tokenize(eot_tag)}")

        # Test full prompt tokenization
        test_prompt = f"{system_tag}You are an agent.{eot_tag}{user_tag}Task: Explore.{eot_tag}{assistant_tag}"
        tokens = tokenizer.tokenize(test_prompt)
        print(f"Full prompt tokens: {tokens[:20]}...")  # Show first 20 tokens

    except Exception as e:
        print(f"✗ Chat format test failed: {e}")


if __name__ == "__main__":
    test_qwen_tokenizer()
