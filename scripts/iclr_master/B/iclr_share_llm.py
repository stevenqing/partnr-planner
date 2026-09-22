"""Opt-in switch `+iclr_share_llm=True` for planner_demo_seeded.py: one HF model instance per process.

In the unchanged pipeline each agent's planner builds its own LLM object, and HFModel.init_local_model
loads the weights and tokenizer again (two copies per process, ~38-48 GB peak). With the switch on,
init_local_model is memoized per (LLM class, engine path) inside the worker process: the second agent
gets the same model and tokenizer objects. Weights, dtype, generation settings and the grammar
processor (built per call from the tokenizer) are unchanged; the two agents call the model one after
the other in the same thread (sequential execution), so no state is shared across calls.
With the switch off this module is never imported.
"""
_CACHE = {}


def install():
    from habitat_llm.llm.hf_model import HFModel

    if getattr(HFModel, "_iclr_shared", False):
        return
    orig = HFModel.init_local_model

    def init_local_model(self):
        key = (type(self).__name__, str(self.generation_params.engine))
        if key in _CACHE:
            self.model, self.tokenizer = _CACHE[key]
            print(f"ICLR_SHARE_LLM: reusing {key}", flush=True)
            return
        orig(self)
        _CACHE[key] = (self.model, self.tokenizer)
        print(f"ICLR_SHARE_LLM: loaded {key}", flush=True)

    HFModel.init_local_model = init_local_model
    HFModel._iclr_shared = True
