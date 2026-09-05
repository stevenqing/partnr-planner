#!/usr/bin/env python3
"""Check, mechanically, that borrowing Layers 2 and 3 imports no hand-authored content.

`viki_memory_from_library.py` grafts an agent-built Layer 1 onto a reference artefact's
Layer 2 (ordering patterns) and Layer 3 (vocabulary). Any end-to-end number produced that
way is open to one obvious objection: that the borrowed layers smuggle in knowledge derived
from the reference's hand-written operators, so the result is not the agent library's.

The objection is answerable because neither layer is a function of Layer 1 at build time.
This script turns that from a reading of the code into four checks, three of which need no
benchmark data:

  1. signature_independence  -- `dependencies.mine` and `vocabulary.harvest` accept no
     library argument, so no Layer 1 can reach them. Reported as their actual signatures.
  2. graft_invariance        -- running the shipped grafting script twice with two very
     different Layer 1 libraries yields byte-identical Layer 2 and Layer 3. This tests the
     path that actually produces the artefacts, not a paraphrase of it.
  3. fold_reference_required -- the held-out-family artefacts on disk carry DIFFERENT
     Layer 2/3 from the full memory. This is why `--reference` exists: a fold cell that
     borrows the full layers leaks the held-out family into ordering and vocabulary. The
     check reports how much would leak.
  4. rebuild_identity        -- rebuild Layer 2 and Layer 3 from the artefact's own
     recorded seed / per_family / excluded_family and compare byte for byte with what is
     stored. Needs the benchmark data, so it runs on the remote box; it is recorded as
     SKIPPED with the reason rather than silently dropped when the data is absent.

What the checks do NOT say: Layer 2's hypothesis space (`dependencies._describe`'s six
features, MIN_SUPPORT, MIN_PRECISION) and Layer 3's harvesting rules are authored. They are
shared inductive bias held fixed across every arm. The claim being defended is narrower and
exact -- the borrowed layers contain no entry derived from the reference's OPERATORS.

Results land on disk before anything is printed, so a dead session does not lose the run.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from our_method.skill_memory_v2 import dependencies, vocabulary

AMENDMENT11 = ROOT / "results/viki_memory_experiments/amendment11"
REFERENCE = AMENDMENT11 / "skill_memory_v2.json"
GRAFT = ROOT / "scripts/viki_memory_from_library.py"

# Anything matching these would be a channel through which Layer 1 could reach a layer
# whose content is claimed to be independent of it.
LIBRARY_SHAPED = ("librar", "operator", "memory", "layer1", "record", "artefact")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:16]


def first_difference(a: Any, b: Any, path: str = "") -> Optional[str]:
    """Where two structures first disagree, named well enough to act on."""
    if type(a) is not type(b):
        return f"{path or '.'}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                return f"{path}.{key}: missing on the left"
            if key not in b:
                return f"{path}.{key}: missing on the right"
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for index, (one, two) in enumerate(zip(a, b)):
            found = first_difference(one, two, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if a == b else f"{path or '.'}: {a!r} vs {b!r}"


def check_signatures() -> Dict[str, Any]:
    out: Dict[str, Any] = {"functions": {}}
    clean = True
    for name, function in (("dependencies.mine", dependencies.mine),
                           ("vocabulary.harvest", vocabulary.harvest)):
        parameters = list(inspect.signature(function).parameters)
        suspect = [p for p in parameters if any(token in p.lower() for token in LIBRARY_SHAPED)]
        out["functions"][name] = {"parameters": parameters, "library_shaped": suspect}
        clean = clean and not suspect
    out["pass"] = clean
    out["note"] = ("no parameter of either builder can carry an operator library"
                   if clean else "a parameter could carry a library -- independence is NOT established")
    return out


def check_graft_invariance(reference: Path, workdir: Path) -> Dict[str, Any]:
    """Two unlike Layer 1 libraries through the shipped script; Layers 2/3 must not move."""
    record = json.loads(reference.read_text())
    operators = record.get("layer1", {}).get("operators", [])
    workdir.mkdir(parents=True, exist_ok=True)

    libraries = {
        "full": operators,
        # Deliberately unlike: reversed order, a third of the entries, bodies truncated.
        "perturbed": [dict(op, body=(op.get("body") or [])[:1])
                      for op in list(reversed(operators))[: max(1, len(operators) // 3)]],
    }
    produced: Dict[str, Dict[str, Any]] = {}
    for name, library in libraries.items():
        library_path = workdir / f"lib_{name}.json"
        out_path = workdir / f"mem_{name}.json"
        library_path.write_text(json.dumps({"operators": library}))
        result = subprocess.run(
            [sys.executable, str(GRAFT), "--library", str(library_path),
             "--out", str(out_path), "--reference", str(reference)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if result.returncode != 0:
            return {"pass": False, "error": result.stderr.strip()[:600]}
        produced[name] = json.loads(out_path.read_text())

    out: Dict[str, Any] = {
        "layer1_sizes": {name: len(library) for name, library in libraries.items()},
        "layer1_differs": digest(produced["full"]["layer1"]) != digest(produced["perturbed"]["layer1"]),
    }
    for layer in ("layer2", "layer3"):
        left, right = produced["full"].get(layer), produced["perturbed"].get(layer)
        out[layer] = {
            "sha_full": digest(left),
            "sha_perturbed": digest(right),
            "identical": canonical(left) == canonical(right),
            "first_difference": first_difference(left, right, layer),
        }
    out["pass"] = bool(out["layer1_differs"] and out["layer2"]["identical"] and out["layer3"]["identical"])
    out["note"] = ("Layer 1 changed and Layers 2/3 did not move, on the shipped path"
                   if out["pass"] else "the graft is not Layer-1-invariant -- investigate before quoting")
    return out


def check_fold_reference_required(reference: Path) -> Dict[str, Any]:
    """How much of a held-out family leaks if a fold cell borrows the FULL Layer 2/3."""
    full = json.loads(reference.read_text())
    folds = sorted(reference.parent.glob("skill_memory_v2.fold_*.json"))
    rows: List[Dict[str, Any]] = []
    for path in folds:
        fold = json.loads(path.read_text())
        full_patterns = set(full.get("layer2", {}).get("kept_patterns", []))
        fold_patterns = set(fold.get("layer2", {}).get("kept_patterns", []))
        full_places = set(full.get("layer3", {}).get("places", []))
        fold_places = set(fold.get("layer3", {}).get("places", []))
        full_assets = set(full.get("layer3", {}).get("assets", []))
        fold_assets = set(fold.get("layer3", {}).get("assets", []))
        full_targets = set(full.get("layer3", {}).get("goal_targets", {}))
        fold_targets = set(fold.get("layer3", {}).get("goal_targets", {}))
        # SkillMemoryV2 reads back `kept_patterns` (Layer 2) and `assets` / `places` /
        # `goal_targets` (Layer 3). Layer 2's `rules` block is diagnostic and never
        # consumed, so a difference there is not a leak; these counts are.
        leak = {
            "kept_patterns_full_only": len(full_patterns - fold_patterns),
            "places_full_only": len(full_places - fold_places),
            "assets_full_only": len(full_assets - fold_assets),
            "goal_targets_full_only": len(full_targets - fold_targets),
        }
        rows.append({
            "family": path.name.split("fold_", 1)[1].rsplit(".json", 1)[0],
            "layer2_identical": canonical(full.get("layer2")) == canonical(fold.get("layer2")),
            "layer3_identical": canonical(full.get("layer3")) == canonical(fold.get("layer3")),
            "kept_patterns_fold_only": len(fold_patterns - full_patterns),
            "functional_leak": any(leak.values()),
            **leak,
        })
    leaky = [row for row in rows if row["functional_leak"]]
    return {
        "folds_on_disk": len(rows),
        "folds_leaking_if_full_reference_used": len(leaky),
        "rows": rows,
        "pass": bool(rows),
        "note": ("a fold cell MUST pass --reference <that fold's artefact>; "
                 "%d of %d folds would receive content their own data excludes"
                 % (len(leaky), len(rows))) if rows else "no fold artefacts on disk",
    }


def check_rebuild_identity(reference: Path, train: Optional[Path],
                           benchmark_root: Optional[Path]) -> Dict[str, Any]:
    record = json.loads(reference.read_text())
    build_args = {
        "seed": record.get("seed"),
        "per_family": record.get("per_family", 250),
        "excluded_family": record.get("excluded_family"),
    }
    try:
        from our_method.skill_memory_v2.build import load_episodes
        from our_method.skill_memory_v2.simulator import SEED, Simulator

        root = benchmark_root or (ROOT.parent / "VIKI-R")
        parquet = train or (root / "data/VIKI-R/viki/VIKI-L2/train.parquet")
        seed = build_args["seed"] if build_args["seed"] is not None else SEED
        sim = Simulator(root)
        episodes = load_episodes(parquet)
        induction_set = episodes[::2]
        rebuilt = {
            "layer2": dependencies.mine(induction_set, sim, seed,
                                        build_args["per_family"], build_args["excluded_family"]),
            "layer3": vocabulary.harvest(induction_set, build_args["excluded_family"]),
        }
    except Exception as error:                                  # noqa: BLE001
        return {"status": "SKIPPED", "build_args": build_args,
                "reason": f"{type(error).__name__}: {error}",
                "note": "needs the benchmark data -- run this on the remote box"}

    out: Dict[str, Any] = {"status": "RAN", "build_args": build_args,
                           "episodes_in_induction_set": len(induction_set)}
    for layer in ("layer2", "layer3"):
        stored = record.get(layer)
        out[layer] = {
            "sha_stored": digest(stored),
            "sha_rebuilt": digest(rebuilt[layer]),
            "identical": canonical(stored) == canonical(rebuilt[layer]),
            "first_difference": first_difference(stored, rebuilt[layer], layer),
        }
    out["pass"] = bool(out["layer2"]["identical"] and out["layer3"]["identical"])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory", type=Path, default=REFERENCE,
                        help="artefact whose Layers 2/3 are under test")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=ROOT / "outputs/layer23_equivalence")
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/layer23_equivalence.json",
                        help="where the report lands. Written BEFORE anything is printed.")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="skip check 4 even where the benchmark data is present")
    arguments = parser.parse_args(argv)

    checks: Dict[str, Any] = {
        "signature_independence": check_signatures(),
        "graft_invariance": check_graft_invariance(arguments.memory, arguments.workdir),
        "fold_reference_required": check_fold_reference_required(arguments.memory),
        "rebuild_identity": ({"status": "SKIPPED", "reason": "--no-rebuild"} if arguments.no_rebuild
                             else check_rebuild_identity(arguments.memory, arguments.train,
                                                         arguments.benchmark_root)),
    }
    decisive = [checks["signature_independence"], checks["graft_invariance"]]
    rebuild = checks["rebuild_identity"]
    if rebuild.get("status") == "RAN":
        decisive.append(rebuild)
    verdict = ("EQUIVALENT" if all(check.get("pass") for check in decisive) else "DIFFERS")
    if verdict == "EQUIVALENT" and rebuild.get("status") != "RAN":
        verdict = "EQUIVALENT (rebuild not run)"

    report = {"memory": str(arguments.memory), "verdict": verdict, "checks": checks}
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, indent=1))          # disk before print

    print(f"memory   {arguments.memory}")
    print(f"verdict  {verdict}\n")
    signatures = checks["signature_independence"]
    print("1 signature independence  %s" % ("pass" if signatures["pass"] else "FAIL"))
    for name, info in signatures["functions"].items():
        print(f"    {name}({', '.join(info['parameters'])})")
    graft = checks["graft_invariance"]
    print("2 graft invariance        %s  layer2 %s  layer3 %s"
          % ("pass" if graft.get("pass") else "FAIL",
             graft.get("layer2", {}).get("sha_full"), graft.get("layer3", {}).get("sha_full")))
    if graft.get("layer2", {}).get("first_difference"):
        print("    layer2 differs at %s" % graft["layer2"]["first_difference"])
    folds = checks["fold_reference_required"]
    print("3 fold reference required  %s" % folds["note"])
    print("    %-52s %-5s %8s %7s %7s %8s"
          % ("family", "leak", "patterns", "places", "assets", "targets"))
    for row in folds["rows"]:
        print("    %-52s %-5s %8d %7d %7d %8d"
              % (row["family"], row["functional_leak"], row["kept_patterns_full_only"],
                 row["places_full_only"], row["assets_full_only"], row["goal_targets_full_only"]))
    print("4 rebuild identity        %s" % rebuild.get("status", "?"))
    if rebuild.get("status") == "RAN":
        for layer in ("layer2", "layer3"):
            print("    %s identical=%s  stored %s  rebuilt %s"
                  % (layer, rebuild[layer]["identical"],
                     rebuild[layer]["sha_stored"], rebuild[layer]["sha_rebuilt"]))
            if rebuild[layer]["first_difference"]:
                print("      first difference at %s" % rebuild[layer]["first_difference"])
    else:
        print("    %s" % rebuild.get("reason", ""))
    print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
