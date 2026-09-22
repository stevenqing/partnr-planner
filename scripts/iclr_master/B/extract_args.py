"""Extract the hydra command line of each archived Isambard run script into args/<method>_<target>_<mem>.args.

Keeps the archived overrides verbatim except the ones the runner sets itself:
LLM choice / engine / inference mode (backbone), CUDA and data_path only when the target is overridden.
Secrets (API keys) are never copied: only the lines after `planner_demo \\` are read.
Run locally; source = ~/restore_isambard/partnr-planner (B repo, e9abe2b).
"""
import pathlib, re, hashlib, json

B = pathlib.Path.home() / "restore_isambard/partnr-planner"
OUT = pathlib.Path(__file__).parent / "args"
DROP = re.compile(r"generation_params\.engine=|generation_params\.model=|^llm@|plan_config\.llm\.inference_mode=|^num_proc=")
# cell key -> (archived script, dataset override or None, extra edits)
CELLS = {
    "ours_RS_R":        ("all_scripts/bash_files/run_ours_rs_mem_r.sh", None),
    "ours_RS_RS":       ("all_scripts/bash_files/run_ours_rs_mem_rs.sh", None),
    "ours_HRT_HRHT":    ("all_scripts/bash_files/run_ours_hrt_mem_hr_ht.sh", None),
    "ours_HRST_RSHRHT": ("all_scripts/bash_files/run_ours_hrst_mem_hrst.sh", None),
    "rag_RS_R":         ("all_scripts/bash_files/run_traj_rs_mem_r.sh", None),
    "rag_RS_RS":        ("all_scripts/bash_files/run_traj_rs_mem_rs.sh", None),
    "rag_HRT_HRHT":     ("all_scripts/bash_files/run_traj_rht_mem_rht.sh", None),
    "rag_HRST_RSHRHT":  ("all_scripts/bash_files/run_traj_rhst_mem_rhst.sh", None),
    "memento_RS_R":     ("all_scripts/bash_files/run_memento_rs_mem_r.sh", None),
    "memento_RS_RS":    ("all_scripts/bash_files/run_memento_rs_mem_rs.sh", None),
    "memento_HRT_HRHT": ("all_scripts/bash_files/run_memento_rht_mem_rht.sh", "B2"),
    "memento_HRST_RSHRHT": ("all_scripts/bash_files/run_memento_rhst_mem_rhst.sh", None),
    "zs":               ("old/run_planner_demo_openrouter_llama31_70b_org.sh", None),
    "tom":              ("old/run_planner_demo_openrouter_llama33_70b_tom.sh", None),
}

def args_of(path):
    lines = (B / path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if "planner_demo" in l)
    out = []
    for l in lines[i + 1:]:
        s = l.strip().rstrip("\\").strip()
        if not s:
            if not l.rstrip().endswith("\\"):
                break
            continue
        if s.startswith("#"):
            continue
        out.append(s)
        if not l.rstrip().endswith("\\"):
            break
    return out

def main():
    OUT.mkdir(exist_ok=True)
    manifest = {}
    for key, (path, edit) in CELLS.items():
        a = [x for x in args_of(path) if not DROP.search(x)]
        # dataset path: keep only the file name; runner prefixes task_classification_datasets/
        a = [re.sub(r'habitat.dataset.data_path="?[^"]*/([^/"]+)"?', r"habitat.dataset.data_path=task_classification_datasets/\1", x) for x in a]
        if edit == "B2":  # Table 2 states H_R+H_T; B2 (09-21) dropped rerange_only from the archived list
            a = [x.replace("[data/memory_memento_dataset/rerange_only/,", "[") for x in a]
            a = [re.sub(r"\[habitat_trajectories(,habitat_trajectories)*\]", "[habitat_trajectories]", x) for x in a]
        (OUT / f"{key}.args").write_text("\n".join(a) + "\n")
        manifest[key] = dict(source=path, source_sha256=hashlib.sha256((B / path).read_bytes()).hexdigest(), n_args=len(a))
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))

if __name__ == "__main__":
    main()
