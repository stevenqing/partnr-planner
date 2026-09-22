"""Hook part B's +iclr_share_llm=True switch (scripts/iclr_master/B/iclr_share_llm.py, imported from B's directory,
not copied) into the C repo's planner_demo.py at the same point as B's make_seeded_demo.sh. Absent => original."""
import sys
p = sys.argv[1] + "/habitat_llm/examples/planner_demo.py"
bdir = "/mnt/pfs/devs/pn5wp/shishuqing/partnr-planner/scripts/iclr_master/B"
s = open(p).read()
assert "iclr_share_llm" not in s, "already applied"
c = "    # Instantiate the agent planner\n"
assert s.count(c) == 1
s = s.replace(c, ('    if config.get("iclr_share_llm", False):  # iclr_master C switch (part B implementation)\n'
                  f'        sys.path.insert(0, "{bdir}")\n'
                  '        import iclr_share_llm\n'
                  '        iclr_share_llm.install()\n\n') + c)
open(p, "w").write(s)
print("applied")
