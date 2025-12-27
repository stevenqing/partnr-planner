#!/usr/bin/env python3
"""
Test tokenizer fix for Qwen model
Verifies that the fast tokenizer is being used
"""

from transformers import AutoTokenizer

print("=" * 70)
print("TESTING QWEN TOKENIZER CONFIGURATION")
print("=" * 70)

model_name = "Qwen/Qwen2.5-7B-Instruct"

print("\n[1] Testing tokenizer loading...")
print(f"    Model: {model_name}")

# Test with use_fast=True (what we changed to)
print("\n    Loading with use_fast=True...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
print(f"    ✓ Tokenizer type: {type(tokenizer).__name__}")
print(f"    ✓ pad_token: {tokenizer.pad_token}")
print(f"    ✓ eos_token: {tokenizer.eos_token}")

# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("    ✓ Set pad_token to eos_token")

# Test encoding and decoding
test_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Task: Move the laptop stand from the living room to the bedroom.

What is the next action?

Thought: <reasoning>
<action>
Assigned!<|im_end|>
<|im_start|>assistant
"""

print("\n[2] Testing encoding/decoding...")
print(f"    Prompt length: {len(test_prompt)} chars")

# Encode
model_inputs = tokenizer(test_prompt, return_tensors="pt")
print(f"    ✓ Encoded to {model_inputs.input_ids.shape[1]} tokens")

# Test decoding with different parameters
test_response_tokens = tokenizer.encode(
    "Thought: I should explore the living room first\nExplore[living_room_1]\nAssigned!",
    add_special_tokens=False,
)
print("\n[3] Testing decoding variations...")
print(f"    Response tokens: {len(test_response_tokens)}")

# Default decode
decoded_default = tokenizer.decode(test_response_tokens, skip_special_tokens=False)
print("    Default decode:")
print(f"    {repr(decoded_default)}")

# With clean_up_tokenization_spaces=True
decoded_clean = tokenizer.decode(
    test_response_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True
)
print("\n    With clean_up_tokenization_spaces=True:")
print(f"    {repr(decoded_clean)}")

# With clean_up_tokenization_spaces=False
decoded_no_clean = tokenizer.decode(
    test_response_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
print("\n    With clean_up_tokenization_spaces=False:")
print(f"    {repr(decoded_no_clean)}")

# Batch decode (what we use in the code)
batch_decode_result = tokenizer.batch_decode(
    [test_response_tokens], skip_special_tokens=False, clean_up_tokenization_spaces=True
)
print("\n    Batch decode (clean_up=True):")
print(f"    {repr(batch_decode_result[0])}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

issues = []
if type(tokenizer).__name__ != "Qwen2TokenizerFast":
    issues.append(f"❌ Wrong tokenizer type: {type(tokenizer).__name__}")
else:
    print("✓ Using Qwen2TokenizerFast (correct)")

if " " not in decoded_clean or "_" in decoded_clean:
    issues.append("❌ Spacing issue detected in decoded text")
else:
    print("✓ Spacing looks correct in decoded text")

if tokenizer.pad_token is None:
    issues.append("❌ pad_token not set")
else:
    print("✓ pad_token is set")

if not issues:
    print("\n✅ ALL CHECKS PASSED!")
else:
    print("\n❌ ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")

print("=" * 70)
