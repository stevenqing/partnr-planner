#!/usr/bin/env python3
"""Two more calibration variants, and the reason they are needed.

The first pair of broken operators was rejected for reasons that never reached execution:
a reversed body is refused when the memory loads it, and a key-swapped body is never
offered for a requirement of the original key. Both are correct rejections and neither
tests what the gate is claimed to test -- that an operator which *runs* and is *wrong*
scores no gain. These two do run:

  no_pick        navigates and places without ever picking the object up
  back_in_place  opens the container the object came from and puts it back there
"""
import json
from pathlib import Path

lib = json.load(open("results/partnr_operators.json"))["operators"]
kept = [o for o in lib if o["effect"]["key"] != "is_on_top"]
factory = max([o for o in lib if o["effect"]["key"] == "is_on_top"],
              key=lambda o: o.get("support") or 0)

no_pick = json.loads(json.dumps(factory))
no_pick["body"] = [["Navigate", "?x", ""], ["Navigate", "?y", ""],
                   ["Place", "?x, on, ?y, none, none", ""]]

back = json.loads(json.dumps(factory))
back["body"] = [["Navigate", "?z1", ""], ["Open", "?z1", ""], ["Navigate", "?x", ""],
                ["Pick", "?x", ""], ["Navigate", "?z1", ""],
                ["Place", "?x, on, ?z1, none, none", ""]]

out = Path("results/partnr_calib")
for name, operator in (("is_on_top_no_pick", no_pick), ("is_on_top_back_in_place", back)):
    (out / f"{name}.json").write_text(json.dumps(
        {"operators": kept + [operator], "variant": name}, indent=1))
    print(name, [a[0] for a in operator["body"]])
