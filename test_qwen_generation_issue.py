#!/usr/bin/env python3
"""
专门测试Qwen模型生成问题的脚本
"""

import sys

sys.path.append("/home/shuqing/partnr-planner")

from omegaconf import OmegaConf

from habitat_llm.llm.qwen import Qwen


def test_qwen_generation_issues():
    """测试Qwen模型生成问题"""

    print("=" * 60)
    print("测试Qwen模型生成问题")
    print("=" * 60)

    # 加载配置
    qwen_cfg = OmegaConf.load("habitat_llm/conf/llm/qwen.yaml")
    print("当前配置:")
    print(f"  max_tokens: {qwen_cfg.generation_params.max_tokens}")
    print(f"  temperature: {qwen_cfg.generation_params.temperature}")
    print(f"  do_sample: {qwen_cfg.generation_params.do_sample}")
    print(f"  stop: '{qwen_cfg.generation_params.stop}'")
    print(f"  repetition_penalty: {qwen_cfg.generation_params.repetition_penalty}")

    # 初始化Qwen模型
    print("\n加载Qwen模型...")
    qwen = Qwen(qwen_cfg)

    # 测试1: 检查tokenizer的停止标记处理
    print("\n" + "=" * 40)
    print("测试1: Tokenizer停止标记处理")
    print("=" * 40)

    stop_token = "Assigned!"
    print(f"停止标记: '{stop_token}'")

    # 检查停止标记的tokenization
    stop_tokens = qwen.tokenizer.tokenize(stop_token)
    stop_token_ids = qwen.tokenizer.encode(stop_token, add_special_tokens=False)
    print(f"停止标记分词: {stop_tokens}")
    print(f"停止标记token IDs: {stop_token_ids}")

    # 测试2: 简单生成测试
    print("\n" + "=" * 40)
    print("测试2: 简单生成测试")
    print("=" * 40)

    simple_prompt = """<|im_start|>system
You are an agent. Follow the format exactly.
<|im_end|><|im_start|>user
Task: Explore the living room.

Format:
Thought: <reasoning>
<action>
Assigned!

Example:
Thought: I need to explore
Explore[living_room_1]
Assigned!
<|im_end|><|im_start|>assistant
"""

    print("简单提示词:")
    print(simple_prompt)
    print("\n生成响应...")

    # 使用不同的max_tokens值测试
    for max_tokens in [50, 100, 200]:
        print(f"\n--- max_tokens = {max_tokens} ---")
        qwen.generate_hf_llm(simple_prompt, max_length=max_tokens, stop="Assigned!")

        print(f"响应长度: {len(qwen.response)} 字符")
        print(f"响应内容: {repr(qwen.response)}")

        # 检查是否包含停止标记
        if "Assigned!" in qwen.response:
            print("✓ 包含停止标记")
        else:
            print("✗ 不包含停止标记")

        # 检查是否有重复
        words = qwen.response.split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            print(f"重复率: {repetition_ratio:.2%}")

    # 测试3: 检查生成配置
    print("\n" + "=" * 40)
    print("测试3: 生成配置检查")
    print("=" * 40)

    # 检查模型的生成配置
    print(f"模型设备: {qwen.model.device}")
    print(f"模型dtype: {qwen.model.dtype}")

    # 检查tokenizer配置
    print(
        f"Tokenizer pad token: {qwen.tokenizer.pad_token} (id: {qwen.tokenizer.pad_token_id})"
    )
    print(
        f"Tokenizer eos token: {qwen.tokenizer.eos_token} (id: {qwen.tokenizer.eos_token_id})"
    )

    # 测试4: 检查停止标记在生成过程中的处理
    print("\n" + "=" * 40)
    print("测试4: 停止标记处理检查")
    print("=" * 40)

    # 创建一个包含停止标记的测试文本
    test_text = "Thought: I need to explore\nExplore[living_room_1]\nAssigned!"

    # 检查tokenization
    tokens = qwen.tokenizer.tokenize(test_text)
    token_ids = qwen.tokenizer.encode(test_text, add_special_tokens=False)

    print(f"测试文本: {repr(test_text)}")
    print(f"分词结果: {tokens}")
    print(f"Token IDs: {token_ids}")

    # 找到停止标记的位置
    stop_token_ids = qwen.tokenizer.encode("Assigned!", add_special_tokens=False)
    print(f"停止标记Token IDs: {stop_token_ids}")

    # 检查停止标记是否在token序列中
    stop_found = False
    for i in range(len(token_ids) - len(stop_token_ids) + 1):
        if token_ids[i : i + len(stop_token_ids)] == stop_token_ids:
            print(f"停止标记在位置 {i}-{i+len(stop_token_ids)-1}")
            stop_found = True
            break

    if not stop_found:
        print("✗ 停止标记未在token序列中找到")

    # 测试5: 检查模型是否理解格式
    print("\n" + "=" * 40)
    print("测试5: 格式理解检查")
    print("=" * 40)

    # 使用更明确的格式指令
    explicit_prompt = """<|im_start|>system
You are an agent. You MUST follow this EXACT format:

Thought: <your reasoning>
<action with square brackets>
Assigned!

DO NOT use colons in actions. Use square brackets.
<|im_end|><|im_start|>user
Task: Explore the living room.

What is the next action?
<|im_end|><|im_start|>assistant
"""

    print("明确格式提示词:")
    print(explicit_prompt)
    print("\n生成响应...")

    qwen.generate_hf_llm(explicit_prompt, max_length=100, stop="Assigned!")

    print(f"响应: {repr(qwen.response)}")
    print("格式化响应:")
    print(qwen.response)

    # 检查格式
    has_brackets = "[" in qwen.response and "]" in qwen.response
    has_colons = ":" in qwen.response and "[" not in qwen.response

    print(f"包含方括号: {'✓' if has_brackets else '✗'}")
    print(f"包含错误冒号: {'✗' if has_colons else '✓'}")


if __name__ == "__main__":
    test_qwen_generation_issues()
