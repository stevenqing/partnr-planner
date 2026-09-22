#!/bin/bash
# Create habitat_llm/examples/planner_demo_seeded.py in the B repo (partnr-isambard) from the
# unchanged planner_demo.py. Two opt-in config keys, both absent => original behaviour:
#   +iclr_seed=N               replaces the hard-coded seed 47668090 (default 47668090)
#   +iclr_env_over_metrics=True  installs scripts/iclr_master/B/iclr_env_over.py (see its docstring)
#   +iclr_share_llm=True       installs scripts/iclr_master/B/iclr_share_llm.py (one HF model per process)
# planner_demo.py itself is never modified.
set -eu
B=/mnt/pfs/devs/pn5wp/shishuqing/partnr-isambard
HERE=/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner/scripts/iclr_master/B
SRC=$B/habitat_llm/examples/planner_demo.py
DST=$B/habitat_llm/examples/planner_demo_seeded.py
echo "b5a0f4a2a27c97828cee19534a6ab17febfe30582ec9c74129d377777ccff44e  $SRC" | sha256sum -c -
/root/venvs/partnr/bin/python - "$SRC" "$DST" "$HERE" <<'PY'
import sys
src, dst, here = sys.argv[1:]
s = open(src).read()
a = "    seed = 47668090\n"
assert s.count(a) == 1
s = s.replace(a, '    seed = int(config.get("iclr_seed", 47668090)); print(f"ICLR_SEED {seed}", flush=True)\n')
b = "    # Print the planner\n"
assert s.count(b) == 1
s = s.replace(b, (
    '    if config.get("iclr_env_over_metrics", False):\n'
    f'        sys.path.insert(0, "{here}")\n'
    '        import iclr_env_over\n'
    '        iclr_env_over.install(eval_runner, env_interface)\n\n') + b)
if "\nimport sys\n" not in s:
    s = "import sys\n" + s
c = "    # Instantiate the agent planner\n"
assert s.count(c) == 1
s = s.replace(c, (
    '    if config.get("iclr_share_llm", False):\n'
    f'        sys.path.insert(0, "{here}")\n'
    '        import iclr_share_llm\n'
    '        iclr_share_llm.install()\n\n') + c)
open(dst, "w").write(s)
PY
diff $SRC $DST || true
sha256sum $DST
