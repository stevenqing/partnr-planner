"""Add an opt-in inference mode `vllm_openai` to the Isambard pipeline's HF model class (habitat_llm/llm/hf_model.py).

With `llm.inference_mode=vllm_openai` the planner's generate() sends the same prompt string to a vLLM
OpenAI-compatible /v1/completions endpoint instead of running HF generate in-process:
  - prompt: the exact string the HF path tokenizes (the server adds BOS like the HF tokenizer does);
  - greedy (temperature 0), max_tokens = the HF path's max_new_tokens, stop = the planner stopword;
  - the grammar the planner builds per call (generation_args["grammar_definition"], transformers-CFG EBNF) is passed
    as a structured-output grammar (xgrammar, GBNF), so decoding is constrained by the same rules;
  - the response is cut at the stopword and right-stripped, as in the HF path.
Endpoint from env ICLR_VLLM_URL (e.g. http://127.0.0.1:8201/v1), model name from ICLR_VLLM_MODEL.
Each request/response pair is appended to $ICLR_VLLM_LOG_DIR/<pid>.jsonl when that env var is set.
Any other inference_mode is untouched (the `hf` path runs the original code).
Usage: python apply_vllm_backend.py <repo_root>   (refuses to patch twice)
"""
import sys

p = sys.argv[1] + "/habitat_llm/llm/hf_model.py"
s = open(p).read()
MARK = "# iclr_two_exp vllm_openai"
assert MARK not in s, "already patched"

old_init = """        if self.inference_mode == "hf":
            self.init_local_model()
        elif self.inference_mode == "rlm":
            self.init_remote_model()
"""
new_init = """        if self.inference_mode == "hf":
            self.init_local_model()
        elif self.inference_mode == "vllm_openai":  # iclr_two_exp vllm_openai
            self._iclr_init_vllm()
        elif self.inference_mode == "rlm":
            self.init_remote_model()
"""
old_gen = """        if self.inference_mode == "hf":
            self.generate_hf(prompt, stop, max_length, generation_args=generation_args)
        elif self.inference_mode == "rlm":"""
new_gen = """        if self.inference_mode == "hf":
            self.generate_hf(prompt, stop, max_length, generation_args=generation_args)
        elif self.inference_mode == "vllm_openai":  # iclr_two_exp vllm_openai
            self._iclr_generate_vllm(prompt, stop, max_length, generation_args=generation_args)
        elif self.inference_mode == "rlm":"""
old_remote = """    def init_remote_model(self):"""
new_remote = '''    def _iclr_init_vllm(self):  # iclr_two_exp vllm_openai
        import os as _os
        self._iclr_url = _os.environ["ICLR_VLLM_URL"].rstrip("/")
        self._iclr_model = _os.environ.get("ICLR_VLLM_MODEL", "llama31-8b")
        self._iclr_logdir = _os.environ.get("ICLR_VLLM_LOG_DIR")
        self.model = None
        self.tokenizer = None

    def _iclr_generate_vllm(self, prompt, stop, max_length, generation_args=None):  # iclr_two_exp vllm_openai
        import json as _json
        import os as _os
        import time as _time
        import requests as _requests
        if self.generation_params.batch_response:
            raise NotImplementedError("vllm_openai: batch_response not supported")
        stops = [stop] if isinstance(stop, str) else list(stop)
        body = {"model": self._iclr_model, "prompt": prompt, "max_tokens": int(max_length),
                "temperature": 0.0, "top_p": 1.0, "n": 1, "stop": stops,
                "skip_special_tokens": True, "include_stop_str_in_output": False}
        g = (generation_args or {}).get("grammar_definition")
        if g is not None:
            body["structured_outputs"] = {"grammar": g}
        t0 = _time.time()
        err = None
        for attempt in range(60):  # endpoint blips: retry for up to ~30 min, then raise
            try:
                r = _requests.post(self._iclr_url + "/completions", json=body, timeout=900)
                if r.status_code == 200:
                    out = r.json()["choices"][0]
                    text = out["text"]
                    break
                err = f"HTTP {r.status_code}: {r.text[:500]}"
                if r.status_code == 400:
                    raise RuntimeError("vllm_openai request rejected: " + err)
            except (_requests.ConnectionError, _requests.Timeout) as e:
                err = repr(e)
            _time.sleep(30)
        else:
            raise RuntimeError("vllm_openai endpoint unavailable: " + str(err))
        response = text
        for s_ in stops:
            if s_ in response:
                response = response.split(s_)[0]
                break
        self.response = response.rstrip()
        if self._iclr_logdir:
            _os.makedirs(self._iclr_logdir, exist_ok=True)
            with open(_os.path.join(self._iclr_logdir, f"{_os.getpid()}.jsonl"), "a") as f:
                f.write(_json.dumps({"t": t0, "dt": round(_time.time() - t0, 3), "grammar": g is not None,
                                     "finish_reason": out.get("finish_reason"), "stop_reason": out.get("stop_reason"),
                                     "prompt_chars": len(prompt), "response": self.response}) + "\\n")

    def init_remote_model(self):'''
for old, new in ((old_init, new_init), (old_gen, new_gen), (old_remote, new_remote)):
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
