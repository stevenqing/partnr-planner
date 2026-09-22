"""C.1.1: H_R 197 episodes sorted by integer id; even positions -> build, odd -> eval."""
import gzip, hashlib, json, sys
src, out = sys.argv[1], sys.argv[2]
d = json.load(gzip.open(src))
ids = sorted({str(e["episode_id"]) for e in d["episodes"]}, key=int)
assert len(ids) == 197, len(ids)
build, ev = ids[0::2], ids[1::2]
for name, lst in (("build", build), ("eval", ev)):
    txt = "\n".join(lst) + "\n"
    open(f"{out}/hr_{name}_ids.txt", "w").write(txt)
    print(name, len(lst), hashlib.sha256(txt.encode()).hexdigest())
