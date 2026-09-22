"""Add three config switches to a copy of the B repo (Isambard pipeline). All default off.

  +seed_override=<int>                          planner_demo: replaces the hard-coded seed 47668090
  +...plan_config.role_hint="<sentence>"        llm_planner: appended on its own line after the task instruction
  +...plan_config.retrieval_dump_dir=<dir>      llm_planner/rag: per episode and agent, the memory instances whose
                                                text enters the prompt (skill key + index in the bank file)

With none of them set, every edited line is behind a guard that is never entered.
Usage: python apply_switches.py <repo_root>   (refuses to patch twice)
"""
import sys

root = sys.argv[1]
MARK = "# iclr_master C switch"


def patch(rel, pairs):
    p = f"{root}/{rel}"
    s = open(p).read()
    if MARK in s:
        sys.exit(f"already patched: {rel}")
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, (rel, n, old[:80])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", rel)


patch("habitat_llm/examples/planner_demo.py", [(
    """    seed = 47668090
    t0 = time.time()""",
    """    seed = 47668090
    if config.get("seed_override", None) is not None:  # iclr_master C switch
        seed = int(config.seed_override)
    t0 = time.time()""",
)])

patch("habitat_llm/planner/llm_planner.py", [
    (
        """                params["rag_examples"] = example_str
            else:
                params["rag_examples"] = ""
""",
        """                params["rag_examples"] = example_str
            else:
                params["rag_examples"] = ""
            if getattr(self.planner_config, "retrieval_dump_dir", None):  # iclr_master C switch
                self._iclr_dump_retrieval()
        if getattr(self.planner_config, "role_hint", None):  # iclr_master C switch
            params["input"] = input_instruction + "\\n" + self.planner_config.role_hint
""",
    ),
    (
        """    def _format_rag_examples(
        self, indices: List[int], scores: List[float], current_agent_id: int,""",
        """    def _iclr_dump_retrieval(self):  # iclr_master C switch
        import json as _json
        import os as _os

        ep = self.env_interface.env.env.env._env.current_episode
        uid = self.agents[0].uid
        rec = {"episode_id": str(ep.episode_id), "agent": int(uid), "rag_enabled": self.rag is not None}
        if self.rag is not None:
            rec.update(getattr(self.rag, "last_retrieval", None) or {"mode": "unrecorded", "items": []})
        else:
            rec.update({"mode": "no_memory", "items": []})
        d = self.planner_config.retrieval_dump_dir
        _os.makedirs(d, exist_ok=True)
        with open(_os.path.join(d, f"episode_{ep.episode_id}_agent_{uid}.json"), "w") as f:
            _json.dump(rec, f)

    def _format_rag_examples(
        self, indices: List[int], scores: List[float], current_agent_id: int,""",
    ),
    (
        """                ablation_config=ablation_config,
            )
""",
        """                ablation_config=ablation_config,
            )
            if getattr(plan_config, "retrieval_dump_dir", None):  # iclr_master C switch
                self.rag.dump_retrieval = True
""",
    ),
])

patch("habitat_llm/planner/rag.py", [
    (
        """        if not retrieved_skills:
            print("No skills retrieved, falling back to flat retrieval")
            return self._flat_retrieval(query, top_k, agent_id)
""",
        """        if not retrieved_skills:
            print("No skills retrieved, falling back to flat retrieval")
            if getattr(self, "dump_retrieval", False):  # iclr_master C switch
                _s, _ix = self._flat_retrieval(query, top_k, agent_id)
                self.last_retrieval = {"mode": "fallback_flat", "items": [
                    {"entry_skill_key": self.data_dict[int(i)].get("skill_key"),
                     "entry_instance_idx": self.data_dict[int(i)].get("instance_idx"),
                     "trace_skill_key": self.data_dict[int(i)].get("skill_key"),
                     "trace_instance_idx": self.data_dict[int(i)].get("instance_idx")} for i in _ix]}
                return _s, _ix
            return self._flat_retrieval(query, top_k, agent_id)
        if getattr(self, "dump_retrieval", False):  # iclr_master C switch
            self.last_retrieval = {"mode": "hierarchical", "items": []}
""",
    ),
    (
        """            if not found:
                # Create a virtual entry for this skill
                virtual_idx = self._create_virtual_entry(skill, agent_id)
                indices.append(virtual_idx)
                scores.append(skill.abstract_score)
""",
        """            if not found:
                # Create a virtual entry for this skill
                virtual_idx = self._create_virtual_entry(skill, agent_id)
                indices.append(virtual_idx)
                scores.append(skill.abstract_score)
            if getattr(self, "dump_retrieval", False):  # iclr_master C switch
                _e = self.data_dict[indices[-1]]
                _tk, _ti = self._iclr_locate(skill.instances[0]) if skill.instances else (None, None)
                self.last_retrieval["items"].append({
                    "skill_type": skill.skill_type, "found": found,
                    "trace_skill_key": _tk, "trace_instance_idx": _ti,
                    "entry_skill_key": _e.get("skill_key") if found else _tk,
                    "entry_instance_idx": _e.get("instance_idx") if found else _ti})
""",
    ),
    (
        """    def _retrieve_memento_top_k(
        self,""",
        """    def _iclr_locate(self, inst):  # iclr_master C switch
        hr = self.hierarchical_retriever
        for lib in (hr.L_ind, hr.L_coop):
            for k, s in lib.items():
                for i, x in enumerate(s.get("instances", [])):
                    if x is inst:
                        return k, i
        return None, None

    def _retrieve_memento_top_k(
        self,""",
    ),
])
