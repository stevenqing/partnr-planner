#!/usr/bin/env python3
"""Freeze what probe 1 was computed from, before probe 2 can change any of it.

A 924/924 will be asked to prove itself, and the only proof that survives is a hash
taken before the fact. This records the dataset, the manifest, the composer, the
format helper and every file of the official judge -- the simulator included, since
the composer plans against it and a change there would move the number without
touching anything in this repository. The probe 1 result files are hashed too, so the
CSVs that back the claim cannot be quietly regenerated later.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viki_amendment5 import BENCHMARK_ROOT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/viki_memory_experiments/amendment11"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    judge_root = BENCHMARK_ROOT / "verl/verl/utils/reward_score"
    targets = {
        "dataset": [BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2/test.parquet"],
        "manifest": [ROOT / "results/viki_memory_experiments/amendment8b/interactive_manifest.jsonl"],
        "composer": [
            ROOT / "scripts/viki_amendment11_composer.py",
            ROOT / "scripts/viki_amendment11_shape.py",
            ROOT / "scripts/viki_amendment11_diag.py",
            ROOT / "scripts/viki_amendment11_goalparse.py",
            ROOT / "scripts/viki_amendment11_reparse.py",
            ROOT / "scripts/viki_amendment11_vocabulary.py",
            ROOT / "scripts/viki_amendment11_induce.py",
            ROOT / "scripts/viki_amendment11_residual.py",
            ROOT / "scripts/viki_amendment11_folds.py",
            ROOT / "scripts/viki_build_skill_memory_v2.py",
            ROOT / "scripts/viki_eval_skill_memory_v2.py",
            ROOT / "scripts/viki_eval_skill_memory_v2_folds.py",
            ROOT / "scripts/viki_eval_v2_in_prompt.py",
            ROOT / "scripts/viki_eval_v2_operator_choice.py",
            ROOT / "scripts/viki_eval_v2_intent_choice.py",
            ROOT / "scripts/viki_diag_partner_prefix.py",
            ROOT / "scripts/viki_memento_rag.py",
            ROOT / "scripts/viki_eval_memento.py",
            ROOT / "scripts/viki_diag_memento_retrieval.py",
        ]
        + sorted((ROOT / "our_method/skill_memory_v2").glob("*.py"))
        + [
            ROOT / "scripts/viki_plan_format.py",
        ],
        "judge": [
            judge_root / "viki_2.py",
            judge_root / "utils/eval/eval_viki_2.py",
            judge_root / "utils/eval/eval.py",
            judge_root / "utils/eval/env.py",
            judge_root / "utils/eval/checker.py",
            judge_root / "utils/eval/entities.py",
        ],
        "results": sorted(OUT.glob("probe1a*.csv")) + sorted(OUT.glob("probe2*.csv"))
        + sorted(OUT.glob("probe2*.jsonl")) + sorted(OUT.glob("v2_*.csv"))
        + sorted(OUT.glob("intent_*.jsonl")) + sorted(OUT.glob("opchoice_*.jsonl"))
        + sorted(OUT.glob("inprompt_*.jsonl")) + sorted(OUT.glob("folds_layer1.csv"))
        + sorted(OUT.glob("m7_*.jsonl")) + sorted(OUT.glob("m30_*.jsonl"))
        + sorted(OUT.glob("m72_*.jsonl")) + sorted(OUT.glob("m*_folds.csv"))
        + sorted(OUT.glob("memento_*.jsonl")),
        "memory": [OUT / "vocabulary.json", OUT / "operators.json",
                   OUT / "skill_memory_v2.json"]
        + sorted(OUT.glob("skill_memory_v2.fold_*.json"))
        + sorted(OUT.glob("operators.fold_*.json")) + sorted(OUT.glob("vocabulary.fold_*.json")),
    }

    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scoring_seed": 20260829,
        "rows": 924,
        "split": "VIKI-L2 test, the 924-row interactive manifest of amendment8b",
        "probe1a": {
            "oracle_goals_after_hand_exclusivity_fix": "924/924 = 100.00%",
            "oracle_goals_full": "924/924 = 100.00%",
            "general_macros_only_no_relay": "878/924 = 95.02%",
            "blind_to_satisfied_state": "924/924 = 100.00%",
            "shuffled_goals_control": "23/924 = 2.49%",
            "reference_plan_self_check": "40/40 accepted",
            "plan_length_mean": 6.67,
            "reference_length_mean": 6.91,
            "within_budget": "100%",
        },
        "probe1b": {
            "induced_single_runner_only": "878/924 = 95.02%  (identical to the written "
                                          "macros with the relay removed; the 46 misses "
                                          "are exactly the relay family)",
            "induced_with_coordinated_and_repair": "924/924 = 100.00% (19 operators)",
            "induced_with_coordinated": "924/924 = 100.00%  (parity with the written macros, "
                                        "plan lengths identical: 6.67 vs reference 6.91)",
            "library": "18 operators (10 single-runner, 8 coordinated) induced by replaying "
                       "3356 training episodes; all 3356 replays feasible, none left a "
                       "tracked predicate unmet",
            "cross_family_evidence": "the plain delivery operator carries support 2500 and "
                                     "was induced from 8 distinct families",
            "known_gap": "a coordinated operator is induced only from families that contain "
                         "it; the relay exists solely in dog_push, so a fold holding that "
                         "family out cannot recover it",
        },
        "held_out_family": {
            "protocol": "the eight families of the test manifest, the same folds "
                        "amendment9 used; for each, both memory layers rebuilt from the "
                        "training episodes of the other families only. The goal parser is "
                        "zero-shot and unchanged, so no fold spends a token and no fold's "
                        "score can move by resampling the model.",
            "fold_memory": "867/924 = 93.8%",
            "full_memory": "913/924 = 98.8%",
            "written_macros": "913/924 = 98.8%  (upper bound: written after reading all "
                              "eight families)",
            "per_fold": "seven of eight folds match the full-memory and written-macro arms "
                        "row for row; the only loss is dog_push (0/46)",
            "why_dog_push_cannot_transfer": "Push occurs in 1 of the 14 training families. "
                                            "Holding that family out removes the primitive "
                                            "from the data entirely, so no induction can "
                                            "recover it; the written-macro arm scores only "
                                            "because the relay was written after reading "
                                            "that family.",
            "repair_operator": "Open occurs in 4 families, so mining the container-opening "
                               "segment on its own (rather than only inside bodies that "
                               "finish a goal) recovers the clear_table fold from 0/220 to "
                               "220/220; the operator is [Move ?x][Reach ?x][Open ?x], "
                               "support 830, induced from 4 families",
            "archived_comparison_same_folds": {
                "gmemory": "20.78%", "skill_memory.fullactions": "11.80%",
                "trajectory_rag": "3.25%", "zero_shot": "3.46%",
            },
        },
        "skill_memory_v2": {
            "artefact": "our_method/skill_memory_v2 -- three layers, a planner, a build "
                        "CLI with a self-check; the deliverable is the package, not the "
                        "probe scripts",
            "layers": "Layer 1 operators (19: 10 achievement, 1 repair, 8 coordination), "
                      "Layer 2 ordering patterns (3, each precision 1.0, recall 1.0 on "
                      "training, 0 false orderings), Layer 3 vocabulary (29 assets, 41 "
                      "places)",
            "build_self_check": "120/120 held-out training episodes planned and scored 1.0 "
                                "by the official judge",
            "id_full": "907/924 = 98.16% (loose prompt, no hand-written ordering rule)",
            "id_minus_layer2": "715/924 = 77.38%",
            "id_minus_layer3": "488/924 = 52.81%",
            "held_out_family": "867/924 = 93.8%; seven of eight folds identical to full "
                               "memory; dog_push unreachable because Push occurs in 1 of "
                               "14 training families",
            "vs_v1": "v1 411 individual + 412 cooperation skills, 16.77% ID / 11.80% OOD; "
                     "v2 19 operators, 98.16% ID / 93.8% OOD",
        },
        "three_models_three_splits": {
            "method": "intent interface, memory casts the crew, one shared "
                      "skill_memory_v2.json (folds rebuilt per held-out family); no "
                      "partner prefix, while every archived baseline had one",
            "qwen2.5-vl-72b": {"id": "846/924 = 91.56%", "held_out_family": "829/924 = 89.70%",
                               "recombination_imaged": "196/297 = 65.99%",
                               "recombination_text": "194/297 = 65.32%"},
            "qwen3-vl-30b-a3b": {"id": "730/924 = 79.00%", "held_out_family": "684/924 = 74.03%",
                                 "recombination_imaged": "174/297 = 58.59%",
                                 "recombination_text": "181/297 = 60.94%"},
            "qwen2.5-vl-7b": {"id": "201/924 = 21.75%", "held_out_family": "204/924 = 22.08%",
                              "recombination_imaged": "45/297 = 15.15%",
                              "recombination_text": "77/297 = 25.93%"},
            "archived_baseline_72b_only": {"gmemory": "51.0 / 20.8 / 4.7 / 3.4",
                                           "skill_memory_v1": "16.8 / 11.8 / 3.0 / 4.4"},
            "what_the_spread_shows": "memory and planner are identical across the three "
                                     "rows; only the model reading the picture changes, and "
                                     "the score moves 70 points. A symbolic planner alone "
                                     "cannot produce that spread, so it measures what the "
                                     "language model contributes.",
            "recombination_correction": "the archived reading of this split -- every arm on "
                                        "the floor, no method conclusion available -- holds "
                                        "only for prompt-based arms. The floor was a "
                                        "property of the approach, not of the split.",
        },
        "memento_baseline": {
            "what_it_is": "MEMENTO (arXiv 2505.16348) proposes a hierarchical "
                          "knowledge-graph user-profile memory: a model names the "
                          "personalized entities an instruction relies on, split into "
                          "object_semantics and user_pattern; each type is retrieved "
                          "separately over all-mpnet-base-v2 embeddings; the subgraph is "
                          "rendered to natural language for the planner. Read off the "
                          "authors' src/planner/user_profile_rag.py.",
            "this_is_a_port_not_a_comparison": "MEMENTO's memory holds knowledge about a "
                                               "person and VIKI-L2 has no person in it -- "
                                               "no user, no history, no personal object "
                                               "names. The architecture is kept and filled "
                                               "with this benchmark's content. Report it as "
                                               "MEMENTO-style, ported; never as MEMENTO "
                                               "evaluated, which it is not.",
            "harness": "identical to the archived arms: benchmark system prompt, image, "
                       "partner prefix, memory inserted the same way, model writes the "
                       "plan, JSON-tolerant scoring, seed 20260829",
            "id": "184/924 = 19.91%",
            "recombination_imaged": "5/297 = 1.68%",
            "recombination_text": "30/297 = 10.10%",
            "held_out_family": "58/924 = 6.28%; five of eight folds score exactly zero "
                               "(clear_table 0/220, set_plate 0/85, ensure_all_fruits "
                               "0/82, dog_push 0/46); only parallel_human survives at "
                               "43.02%. A 68% relative drop against its own in-domain "
                               "score, steeper than G-Memory's 59%.",
            "configurations_tried": "six, more tuning than any arm of ours received: "
                                    "family-aggregated k=5 prose 13.3%, per-episode k=5 "
                                    "prose 11.7%, per-episode k=5 JSON 6.7%, k=2 JSON "
                                    "10.0%, k=3 aggregated 8.3%, k=1 JSON 21.7% (best, "
                                    "60-row sample). The best was taken to full scale.",
            "their_own_finding_replicated": "k=5 scores 6.7% and k=1 scores 21.7% on the "
                                            "same rows -- information overload, which is "
                                            "the bottleneck their paper identifies. Report "
                                            "this as an independent replication of their "
                                            "result, not as a point against them.",
            "retrieval_was_verified_working": "69.3% of retrieved episodic memories come "
                                              "from the row's own family, and verbatim "
                                              "instruction matches are among them, so the "
                                              "score is the method's and not a broken port",
            "why_it_lands_where_it_does": "prompt-based arms on this split are capped by "
                                          "retrieval recall (92.5% verbatim overlap), not "
                                          "by memory structure. Type separation addresses "
                                          "several kinds of personal knowledge drowning "
                                          "each other, which is not a problem this "
                                          "benchmark has.",
        },
        "delegation_ladder": {
            "note": "same model, same 924 rows, same JSON-tolerant scoring. What changes "
                    "is how much of the composition the model is asked to do. The archived "
                    "arms were given a partner prefix carrying a line of the reference "
                    "plan; the v2 arms were not, and adding it does not help them.",
            "v2_rendered_as_prose_model_writes_actions": "5.8% (120-row sample)",
            "gmemory_model_writes_actions": "50.97% (archived, with partner prefix)",
            "v1_model_writes_actions": "16.77% (archived, with partner prefix)",
            "v2_body_menu_model_picks_variant_bindings_crew": "32.5% (120-row sample)",
            "v2_intent_menu_model_casts_crew": "623/924 = 67.42%",
            "v2_intent_menu_memory_casts_crew": "846/924 = 91.56%",
            "v2_goal_statement_only": "907/924 = 98.16%",
            "oracle_goals": "924/924 = 100.00%",
            "partner_prefix_equalised": "v2 intent with the prefix scores 66.67% (crew by "
                                        "model) and 85.93% (crew by memory) -- the help the "
                                        "archived arms had does not help this interface",
            "why_the_body_menu_failed": "a body's applicability turns on hidden world state "
                                        "(is the cupboard shut, is the object stowed, can "
                                        "this robot walk); the model chose sensibly and was "
                                        "punished for not knowing what the picture does not "
                                        "show. Exposing intents instead moved 32.5% to 96.7% "
                                        "on the same sample.",
            "residual_asymmetry": "v2 delegates transcription into primitives and step "
                                  "packing to the memory; the archived arms write every "
                                  "action. That cannot be equalised by experiment, because "
                                  "trajectory text is not executable -- it is the claim, and "
                                  "must be stated rather than described as a fair field.",
            "prompt_hygiene": "the sentence telling the model that cutting implies using the "
                              "knife was written after reading test failures and has been "
                              "removed; it was worth 0.3 points (91.88 -> 91.56, 67.64 -> "
                              "67.42)",
        },
        "probe2": {
            "zero_shot_goal_parse": "875/924 = 94.70%  (probe2_zeroshot_v4)",
            "plus_layer3_vocabulary": "913/924 = 98.81%  (probe2_layer3)",
            "plus_layer1_induced_operators": "913/924 = 98.81%  (probe2_full_stack; no "
                                             "hand-written macro remains in the pipeline)",
            "layer3_built_from": "train.parquet only; covers 100% of test goal targets",
            "layer3_replaces": "the row's own init_pos as the place vocabulary, so it "
                               "removes an oracle input rather than adding one",
            "residual": "8 cut_fruit + 2 cut_two_fruits (family goal schema, Layer 2) "
                        "+ 1 ensure_all_fruits (genuine perception)",
            "disclosure": "the prompt's rule for when an order is a real dependency was "
                          "written after seeing which families failed on the test split; "
                          "Layer 2 is meant to induce that rule from training instead",
        },
        "oracle_inputs": [
            "goal_constraints",
            "temporal_constraints",
            "init_pos (asset names, and positions used by the scheduler's feasibility simulation)",
            "robots",
        ],
        "not_given_to_composer": ["time_steps", "description", "the image"],
        "sha256": {},
    }
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        record["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
    except Exception as error:  # a certificate without provenance is still worth more than none
        record["git_commit"] = f"unavailable: {error}"

    for group, paths in targets.items():
        record["sha256"][group] = {
            str(path.relative_to(ROOT)) if ROOT in path.parents else str(path): digest(path)
            for path in paths
            if path.is_file()
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "freeze.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
