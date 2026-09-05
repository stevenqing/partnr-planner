"""Score an operator inducer against the reference one, mechanically.

Layers 2 and 3 are taken from the reference artefact so that only Layer 1 -- the operator
library -- varies. The gate is the memory's own self-check: plan held-out training episodes
from their goals with the symbolic planner and score them with the official judge. No model
is called here, and the inducer never scores itself.

  --inducer reference        the shipped `induction.induce`
  --inducer <path.py>        any module exposing induce(episodes, sim, seed, per_family,
                             exclude_family) -> {"operators": [...], ...}
"""
import argparse, importlib.util, json, sys, time, traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from our_method.skill_memory_v2 import induction
from our_method.skill_memory_v2.build import load_episodes
from our_method.skill_memory_v2.memory import FORMAT, SkillMemoryV2
from our_method.skill_memory_v2.simulator import SEED, Simulator

REFERENCE = Path("results/viki_memory_experiments/amendment11/skill_memory_v2.json")

parser = argparse.ArgumentParser()
parser.add_argument("--inducer", default="reference")
parser.add_argument("--train", type=Path,
                    default=Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R/data/VIKI-R/viki/VIKI-L2/train.parquet"))
parser.add_argument("--benchmark-root", type=Path,
                    default=Path("/mnt/pfs/devs/pn5wp/shishuqing/VIKI-R"))
parser.add_argument("--per-family", type=int, default=250)
parser.add_argument("--validate", type=int, default=200)
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()

sim = Simulator(args.benchmark_root)
episodes = load_episodes(args.train)
induction_set = episodes[::2]
reference = json.loads(REFERENCE.read_text())

if args.inducer == "reference":
    induce = induction.induce
    label = "reference"
else:
    path = Path(args.inducer)
    spec = importlib.util.spec_from_file_location("candidate_inducer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    induce = module.induce
    label = str(path)

result = {"inducer": label}
started = time.time()
try:
    layer1 = induce(induction_set, sim, SEED, args.per_family, None)
    result["ran"] = True
except Exception:
    result["ran"] = False
    result["error"] = traceback.format_exc()[-2000:]
    print(result["error"])
    print("\nINDUCER CRASHED")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
    raise SystemExit(1)
result["induce_sec"] = round(time.time() - started, 2)

operators = layer1.get("operators", [])
result["operators"] = len(operators)
result["by_kind"] = dict(Counter(o.get("kind", "achievement") for o in operators))
result["support_sum"] = sum(o.get("support", 0) for o in operators)
result["replay_outcomes"] = layer1.get("replay_outcomes")
result["episodes_replayed"] = layer1.get("episodes_replayed")

# Contract check: what memory.py and planner.py read off an operator.
missing = Counter()
for o in operators:
    for field in ("effect", "preconditions", "cost", "support"):
        if field not in o:
            missing[field] += 1
    if o.get("coordinated"):
        if "roles" not in o:
            missing["roles"] += 1
    elif "body" not in o:
        missing["body"] += 1
    if not isinstance(o.get("effect"), dict) or "key" not in (o.get("effect") or {}):
        missing["effect.key"] += 1
    if not isinstance(o.get("families"), list):
        missing["families"] += 1
result["contract_violations"] = dict(missing)

# Objective properties of the artefact. These are not hints about how the shipped inducer
# works; they are facts about what the candidate produced and how it lines up with what the
# held-out goals ask for. Without them a failing library looks like an unexplained 0.0.
def body_tokens(o):
    if o.get("coordinated"):
        return [t for role in o.get("roles", []) for item in role.get("actions", [])
                for t in (item.get("action") or [])[1:]]
    return [t for action in o.get("body", []) for t in action[1:]]

with_vars = sum(1 for o in operators
                if any(isinstance(t, str) and "?" in t for t in body_tokens(o)))
result["bodies_containing_a_variable"] = "%d/%d" % (with_vars, len(operators))
supports = sorted((o.get("support", 0) for o in operators), reverse=True)
result["support"] = {"max": supports[0] if supports else 0,
                     "mean": round(sum(supports) / len(supports), 2) if supports else 0,
                     "operators_with_support_1": sum(1 for s in supports if s == 1)}
offered = Counter(o.get("effect", {}).get("key") for o in operators)
result["effect_keys_offered"] = dict(offered)

record = {"format": FORMAT, "built_from": "harness", "excluded_family": None,
          "seed": SEED, "per_family": args.per_family, "layer1": layer1,
          "layer2": reference["layer2"], "layer3": reference["layer3"]}
memory = SkillMemoryV2(record)

holdout = [t for t in episodes[1::2] if isinstance(t, dict)][: args.validate]
wanted = Counter()
for truth in holdout:
    for predicate in induction.requirements_of(truth):
        status = induction.predicate_status(predicate) or {}
        wanted["pos.name" if "pos.name" in status
               else "is_activated" if status.get("is_activated") is True
               else "|".join(sorted(status)) or "empty"] += 1
result["effect_keys_the_holdout_goals_need"] = dict(wanted)
started = time.time()
try:
    report = memory.validate(holdout, sim, SEED)
    result["self_check"] = report
except Exception:
    result["self_check"] = None
    result["validate_error"] = traceback.format_exc()[-1500:]
result["validate_sec"] = round(time.time() - started, 2)

ref_ops = reference["layer1"]["operators"]
result["reference"] = {
    "operators": len(ref_ops),
    "by_kind": dict(Counter(o.get("kind", "achievement") for o in ref_ops)),
    "support_sum": sum(o.get("support", 0) for o in ref_ops),
    "self_check": reference.get("self_check"),
}

print(json.dumps(result, indent=1, ensure_ascii=False)[:4000])
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("\n->", args.out)
