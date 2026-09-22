"""Write a subset of a PARTNR dataset (.json.gz) with the given episode ids, same format."""
import gzip, json, sys
src, dst, ids = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
d = json.load(gzip.open(src))
d["episodes"] = [e for e in d["episodes"] if str(e["episode_id"]) in ids]
assert len(d["episodes"]) == len(ids), (len(d["episodes"]), ids)
json.dump(d, gzip.open(dst, "wt"))
print(dst, len(d["episodes"]))
