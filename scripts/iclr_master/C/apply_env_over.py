"""Add B's opt-in switch +iclr_env_over_metrics=True (scripts/iclr_master/B/iclr_env_over.py) to the C repo's
planner_demo.py, with the same insertion point as B's make_seeded_demo.sh. Absent => original behaviour."""
import sys
p = sys.argv[1] + "/habitat_llm/examples/planner_demo.py"
here = "/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner/scripts/iclr_master/C"  # frozen copy of B/iclr_env_over.py
s = open(p).read()
assert "iclr_env_over" not in s, "already applied"
b = "    # Print the planner\n"
assert s.count(b) == 1
s = s.replace(b, ('    if config.get("iclr_env_over_metrics", False):  # iclr_master C switch (same as B)\n'
                  f'        sys.path.insert(0, "{here}")\n'
                  '        import iclr_env_over\n'
                  '        iclr_env_over.install(eval_runner, env_interface)\n\n') + b)
if "\nimport sys\n" not in s:
    s = "import sys\n" + s
open(p, "w").write(s)
print("applied")
