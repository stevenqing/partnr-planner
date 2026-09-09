#!/usr/bin/env python3
"""Build the libraries the acceptance gate is calibrated on.

The calibration the trace-matching gate failed is repeated here against the execution
gate, and it has to be built differently. Dropping ONE of twenty `is_on_top` operators
proves nothing -- the other nineteen cover the same requirement, so a correct re-add and a
broken one both change nothing. So the key is emptied first: the library keeps everything
except that effect, and then exactly one operator is put back. That is also the shape the
real question has, since `is_in_room` has no entries at all.

Three things go back in, and the gate is only usable if it tells them apart:
  factory   the operator as induced
  reversed  the same body, backwards -- places before it picks
  keyswap   the same body, claiming a different effect
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, default=ROOT / "results/partnr_operators.json")
    ap.add_argument("--key", default="is_on_top")
    ap.add_argument("--swap-to", default="is_inside")
    ap.add_argument("--pick", default="support", choices=["support", "first"],
                    help="which operator of the emptied key comes back")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results/partnr_calib")
    args = ap.parse_args()

    blob = json.loads(args.library.read_text())
    operators = blob["operators"] if isinstance(blob, dict) else blob
    kept = [o for o in operators if (o.get("effect") or {}).get("key") != args.key]
    dropped = [o for o in operators if (o.get("effect") or {}).get("key") == args.key]
    if not dropped:
        raise SystemExit(f"library holds no {args.key} operator")
    chosen = max(dropped, key=lambda o: o.get("support") or 0) if args.pick == "support" else dropped[0]

    reversed_op = json.loads(json.dumps(chosen))
    reversed_op["body"] = list(reversed(reversed_op["body"]))
    swapped = json.loads(json.dumps(chosen))
    swapped["effect"] = {**swapped["effect"], "key": args.swap_to}

    args.outdir.mkdir(parents=True, exist_ok=True)
    made = {}
    for name, operators_out in (
        (f"no_{args.key}", kept),
        (f"{args.key}_factory", kept + [chosen]),
        (f"{args.key}_reversed", kept + [reversed_op]),
        (f"{args.key}_keyswap", kept + [swapped]),
    ):
        path = args.outdir / f"{name}.json"
        path.write_text(json.dumps({"operators": operators_out,
                                    "built_from": str(args.library.relative_to(ROOT)),
                                    "variant": name}, indent=1))
        made[name] = {"path": str(path.relative_to(ROOT)), "n": len(operators_out)}

    made["chosen_operator"] = {"key": args.key, "support": chosen.get("support"),
                               "body": [a[0] for a in chosen["body"]]}
    (args.outdir / "MANIFEST.json").write_text(json.dumps(made, indent=1))
    print(json.dumps(made, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
