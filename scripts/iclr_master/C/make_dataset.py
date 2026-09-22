"""Filter a PARTNR episode dataset (json.gz) to the ids listed in a file (one per line, order kept as in source)."""
import gzip, json, sys
src, idfile, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else None
ids = open(idfile).read().split()
if n: ids = ids[:n]
keep = set(ids)
d = json.load(gzip.open(src))
d["episodes"] = [e for e in d["episodes"] if str(e["episode_id"]) in keep]
assert len(d["episodes"]) == len(keep), (len(d["episodes"]), len(keep))
with gzip.open(out, "wt") as f: json.dump(d, f)
print(out, len(d["episodes"]))
