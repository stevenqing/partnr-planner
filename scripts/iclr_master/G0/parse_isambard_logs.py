"""Parse per-episode metrics from Isambard stdout logs (restore_isambard/partnr-planner/logs).

Each episode block looks like
    Metrics For Run 0 Episode <id>:
    task_state_success: x
    task_percent_complete: y
Writes one CSV row per (log, episode) and one summary row per log.
"""
import csv, hashlib, pathlib, re, sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HDR = re.compile(r"Metrics For Run (\d+) Episode (\S+?):")
MET = re.compile(r"^(task_state_success|task_percent_complete|[a-z_]+): ([-0-9.]+)$")
MEM = re.compile(r"Loaded hierarchical memory from: (\S+)|rag_dataset_dir.*?\[(\S+?)\]|Loading (\d+) MEMENTO memory file")

def parse(path):
    eps, cur, mem = {}, None, set()
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line).strip()
        m = HDR.search(line)
        if m:
            cur = m.group(2); eps.setdefault(cur, {}); continue
        m = MET.match(line)
        if m and cur is not None and m.group(1) not in eps[cur]:
            eps[cur][m.group(1)] = float(m.group(2))
        m = MEM.search(line)
        if m:
            mem.add(next(g for g in m.groups() if g))
    return eps, sorted(mem)

def main(logdir, out_prefix):
    rows, summ = [], []
    for p in sorted(pathlib.Path(logdir).glob("*.log")):
        eps, mem = parse(p)
        if not eps:
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        ss = [e.get("task_state_success") for e in eps.values() if "task_state_success" in e]
        pc = [e.get("task_percent_complete") for e in eps.values() if "task_percent_complete" in e]
        for k, e in eps.items():
            rows.append(dict(log=p.name, episode_id=k, success=e.get("task_state_success"),
                             percent_complete=e.get("task_percent_complete")))
        summ.append(dict(log=p.name, sha256=sha, n_episodes=len(eps),
                         success_sum=sum(ss), success_mean=round(sum(ss)/len(ss), 4) if ss else None,
                         pc_mean=round(sum(pc)/len(pc), 4) if pc else None, memory=";".join(mem)))
    for name, data in (("per_episode", rows), ("per_log", summ)):
        with open(f"{out_prefix}_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

if __name__ == "__main__":
    main(*sys.argv[1:])
