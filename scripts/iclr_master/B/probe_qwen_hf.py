"""Probe: does Qwen2.5-7B under the B repo's HF settings (fp16, greedy, repetition_penalty 1.05, stop 'Assigned!')
produce sane text on a real first-turn prompt? Compares fp16 and bf16. Usage: probe_qwen_hf.py <trace file>"""
import sys, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
t = open(sys.argv[1]).read()
p = t[: t.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")]
m = glob.glob("/mnt/pfs/devs/pn5wp/shishuqing/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/")[0]
tok = AutoTokenizer.from_pretrained(m, use_fast=False)
for dt in (torch.float16, torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(m, device_map="auto", torch_dtype=dt, low_cpu_mem_usage=True)
    x = tok(p, return_tensors="pt").to(model.device)
    g = GenerationConfig.from_model_config(model.config)
    g.max_new_tokens, g.do_sample, g.temperature, g.repetition_penalty = 120, False, 0.1, 1.05
    out = model.generate(**x, generation_config=g, pad_token_id=tok.eos_token_id)
    print(dt, repr(tok.decode(out[0, x.input_ids.shape[1]:])))
    del model; torch.cuda.empty_cache()
