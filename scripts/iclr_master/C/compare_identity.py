"""Switch-off identity check: two cells (pristine repo vs patched repo, switches off) on the same episodes.
Compares prompt files byte for byte, trace files byte for byte, and per-step high-level actions from planner-log."""
import filecmp, glob, hashlib, json, os, sys
a, b = sys.argv[1:3]
ra = glob.glob(f"{a}/hydra/results/*.json.gz")[0]; rb = glob.glob(f"{b}/hydra/results/*.json.gz")[0]
out = {}
for sub in ("prompts", "traces"):
    fa = sorted(os.path.relpath(p, ra) for p in glob.glob(f"{ra}/{sub}/*/*"))
    fb = sorted(os.path.relpath(p, rb) for p in glob.glob(f"{rb}/{sub}/*/*"))
    same = [f for f in fa if f in fb and filecmp.cmp(f"{ra}/{f}", f"{rb}/{f}", shallow=False)]
    out[sub] = dict(files_a=len(fa), files_b=len(fb), identical=len(same), differ=sorted(set(fa) ^ set(fb)) + [f for f in fa if f in fb and f not in same])
def acts(p):
    d = json.load(open(p)); steps = d["steps"]
    if isinstance(steps, str):
        import ast; steps = ast.literal_eval(steps)
    return [str(s.get("high_level_actions", s.get("actions", ""))) for s in steps]
pa = sorted(os.path.basename(p) for p in glob.glob(f"{ra}/planner-log/*.json"))
pb = sorted(os.path.basename(p) for p in glob.glob(f"{rb}/planner-log/*.json"))
same = [f for f in pa if f in pb and acts(f"{ra}/planner-log/{f}") == acts(f"{rb}/planner-log/{f}")]
out["planner_log_actions"] = dict(files_a=len(pa), files_b=len(pb), identical=len(same))
for f in pa:
    s1 = json.load(open(f"{ra}/stats/{f.split('episode_')[1].split('_')[0]}.json"))
    s2p = f"{rb}/stats/{f.split('episode_')[1].split('_')[0]}.json"
    out.setdefault("stats", {})[f] = [s1["stats"], json.load(open(s2p))["stats"] if os.path.exists(s2p) else None]
print(json.dumps(out, indent=1))
