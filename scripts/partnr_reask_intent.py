#!/usr/bin/env python3
"""Ask for the goal again after the agent has looked around, and nothing else changed.

The intent arm asks its model once, at step 0, when every archived prompt reads
`Objects: No objects found yet`, and never asks again (`partnr_planner.py:942`). The
model then writes the instruction's words -- `candle`, `plant` -- and 47% of those
requirements are later abandoned because no entity answers to them. This measures how
much of that is the timing of the question.

Each archived prompt is kept byte for byte except the world block before `Task:`, which
is rebuilt with the planner's own `get_world_descr` from an archived world graph:

  own      the graph agent 0 of the archived arm itself held just before the first
           Pick / Place / Rearrange by either agent -- what a re-ask after exploring
           would really have seen, with every object still where it started.
  ceiling  the centralized, fully observed run's graph at the same cut -- every object,
           still in place. An upper bound on what re-asking can buy, not a design.

The cut matters: the last graph of a run shows objects already on their goal furniture,
which would hand the model the answer.

The call is the planner's own: `VLLMChat.generate(prompt, stop="\\n\\n", max_length=256)`
against `conf/llm/vllm.yaml` (temperature 0). Output is `<out>/prompts/0/prompt-*-0.txt`,
prompt plus response, which `partnr_intent_diagnostic.py --intent <out>` grades exactly
as it graded the archive.

  python scripts/partnr_reask_intent.py --graphs own \\
      --source outputs/headtohead/val_mini/v2_intent_7b/results/val_mini.json.gz \\
      --model qwen2.5-vl-7b --out outputs/reask_0913/own_7b
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner")
sys.path.insert(0, str(ROOT))

GRAPHS = {
    "ceiling": ROOT / "outputs/sweep/val_mini/ceiling/results/val_mini.json.gz/detailed_traces",
}
MANIPULATION = {"Pick", "Place", "Rearrange"}


def graph_before_manipulation(trace_path: Path, uid: int = 0):
    """Agent `uid`'s own graph at its last recorded step before anyone moves an object."""
    trace = pickle.load(open(trace_path, "rb"))
    history = trace.get("action_history") or {}
    events = sorted(
        (element.timestamp, agent, index, element)
        for agent, elements in history.items()
        for index, element in enumerate(elements)
    )
    cut = next((t for t, _, _, e in events if (e.action or [None])[0] in MANIPULATION), None)
    mine = [e for t, agent, _, e in events if agent == uid and (cut is None or t < cut)]
    if not mine:
        return None, cut
    graphs = getattr(mine[-1], "world_graph", None) or {}
    return graphs.get(uid), cut


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="archived intent arm results dir")
    ap.add_argument("--graphs", choices=["own", "ceiling"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8061/v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="build prompts and count, call no model")
    args = ap.parse_args()

    import os
    from omegaconf import OmegaConf
    from habitat_llm.llm.instruct.utils import get_world_descr

    trace_dir = args.source / "detailed_traces" if args.graphs == "own" else GRAPHS["ceiling"]
    prompts = sorted((args.source / "prompts" / "0").glob("prompt-episode_*-0.txt"))
    if args.limit:
        prompts = prompts[: args.limit]
    out_dir = args.out / "prompts" / "0"
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = None
    if not args.dry:
        from habitat_llm.llm.vllm_chat import VLLMChat
        os.environ["VLLM_BASE_URL"] = args.base_url
        conf = OmegaConf.load(ROOT / "habitat_llm/conf/llm/vllm.yaml")
        conf.generation_params.model = args.model
        llm = VLLMChat(conf)

    def one(path: Path):
        key = path.name[len("prompt-"):-len("-0.txt")]
        archived = path.read_text(errors="replace")
        head, marker, _ = archived.rpartition("Requirements:\n")
        world_end = head.find("\n\nTask: ")
        if not marker or world_end < 0:
            return {"episode": key, "status": "unparseable archive"}
        trace = trace_dir / f"detailed_trace-{key}.pkl"
        if not trace.exists():
            return {"episode": key, "status": "no trace"}
        graph, cut = graph_before_manipulation(trace)
        if graph is None:
            return {"episode": key, "status": "no graph before the cut"}
        world = get_world_descr(graph, agent_uid=0, include_room_name=True, add_state_info=True)
        objects = world.split("Objects:", 1)[1] if "Objects:" in world else ""
        n_objects = sum(1 for line in objects.splitlines() if ":" in line)
        prompt = world + head[world_end:] + marker
        answer = "" if llm is None else (llm.generate(prompt, stop="\n\n", max_length=256) or "")
        (out_dir / path.name).write_text(prompt + answer)
        return {"episode": key, "status": "ok", "objects": n_objects, "cut": cut,
                "answer_lines": len([l for l in answer.splitlines() if l.strip()])}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, prompts))

    ok = [r for r in rows if r["status"] == "ok"]
    summary = {
        "source": str(args.source), "graphs": args.graphs, "model": args.model, "dry": args.dry,
        "prompts": len(rows), "written": len(ok),
        "status": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
        "episodes_with_zero_objects": sum(1 for r in ok if r["objects"] == 0),
        "mean_objects": sum(r["objects"] for r in ok) / max(len(ok), 1),
        "no_manipulation_in_run": sum(1 for r in ok if r["cut"] is None),
        "rows": rows,
    }
    (args.out / "reask_summary.json").write_text(json.dumps(summary, indent=1))  # disk before print
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
