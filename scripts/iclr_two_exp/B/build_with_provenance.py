"""Run the archived H_R builder unchanged, and record which episode produced each saved instance.

The builder (our_method/build_hierarchical_skill_memory.py) only appends instances (instances.extend) inside
add_episode_to_memory and saves them in list order, so the instances added during one add_episode_to_memory call
are exactly the index range [len_before, len_after) of each skill. This wrapper wraps that method, notes the ranges,
and writes provenance.json next to the library: {"L_ind_skills": {skill_key: [episode_id per instance]}, ...}.
It changes no argument, prompt, or call of the builder. Arguments are the builder's own, parsed the same way.
Run from <repo>/our_method (the builder imports llm_skill_extractor from the cwd).
"""
import json, os, sys

sys.path.insert(0, os.getcwd())
import build_hierarchical_skill_memory as B  # noqa: E402

PROV = {"L_ind_skills": {}, "L_coop_skills": {}}
_orig_add = B.HierarchicalSkillMemory.add_episode_to_memory
_orig_save = B.HierarchicalSkillMemory.save_memory


def _add(self, episode_id, trace_paths, episode_info):
    before = {("L_ind_skills", k): len(s.instances) for k, s in self.L_ind.items()}
    before.update({("L_coop_skills", k): len(s.instances) for k, s in self.L_coop.items()})
    r = _orig_add(self, episode_id, trace_paths, episode_info)
    for name, lib in (("L_ind_skills", self.L_ind), ("L_coop_skills", self.L_coop)):
        for k, s in lib.items():
            n0 = before.get((name, k), 0)
            lst = PROV[name].setdefault(k, [])
            assert len(lst) == n0, (name, k, len(lst), n0)
            lst.extend([str(episode_id)] * (len(s.instances) - n0))
    return r


def _save(self, output_path):
    r = _orig_save(self, output_path)
    for name, lib in (("L_ind_skills", self.L_ind), ("L_coop_skills", self.L_coop)):
        for k, s in lib.items():
            assert len(PROV[name].get(k, [])) == len(s.instances), (name, k)
    with open(os.path.join(output_path, "provenance.json"), "w") as f:
        json.dump(PROV, f)
    print("provenance written")
    return r


B.HierarchicalSkillMemory.add_episode_to_memory = _add
B.HierarchicalSkillMemory.save_memory = _save

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--include-failed", action="store_true")
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--use-api", action="store_true")
    p.add_argument("--vllm-host", default="localhost")
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--model-name", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--patch-failed", action="store_true")
    a = p.parse_args()
    B.build_memory_from_heuristic_results(
        results_dir=a.results_dir, output_dir=a.output_dir, filter_successful=not a.include_failed,
        use_llm=a.use_llm, use_local=not a.use_api, vllm_host=a.vllm_host, vllm_port=a.vllm_port,
        model_name=a.model_name, patch_failed=a.patch_failed)
