# AUDIT_REPORT

结局 A

## L0
通过：True

```json
{
 "missing": [],
 "train_rows": 7196,
 "induction_half": 3598,
 "selfcheck_half": 3598,
 "halves_disjoint": true,
 "halves_cover_train": true,
 "test_split": 1800,
 "id_manifest_924": 924,
 "heldout_family_folds": 8,
 "comp_pairing": {
  "rows_per_form": {
   "text": 297,
   "imaged": 297
  },
  "distinct_task_ids_text": 295,
  "distinct_task_ids_imaged": 295,
  "distinct_task_ids_overall": 295,
  "text_ids_repeated": {
   "4307_8-1": 2,
   "1426_9-1": 2
  },
  "imaged_ids_repeated": {
   "4307_8-1": 2,
   "1426_9-1": 2
  },
  "text_only": [],
  "imaged_only": []
 },
 "comp_297": 594,
 "support_pool": 60,
 "unsolved_lists": {
  "files": 10,
  "rungs": 34
 },
 "eight_pools": {
  "induction_half": true,
  "selfcheck_half": true,
  "test_split": true,
  "id_manifest_924": true,
  "heldout_family_folds": true,
  "comp_297": true,
  "support_pool": true,
  "unsolved_lists": true
 },
 "pass": true
}
```

## L1
通过：True

```json
{
 "rule_case": "a",
 "rule": "held out = the co-occurrence of a CUTTING family (cut_fruit_on_board, cut_two_fruits_on_board) with single_move_asset_to_target in one instance",
 "rule_source": "scripts/viki_amendment10_recombine.py:40-41 (CUTTING x DELIVERY), docstring lines 3-14",
 "selfcheck": {
  "decisive_detector": "structural: cut unit (placement ordered before an activation) AND an independent delivery (placement entangled with no activation)",
  "comp_rows": 295,
  "comp_rows_detected": 295,
  "comp_detection_rate": 1.0,
  "donor_family_rows_detected": {
   "cut_fruit_on_board": 0,
   "cut_two_fruits_on_board": 0
  },
  "sensitivity_actor_blind_detector": "cut unit WITHOUT the two-robot requirement -- reported as an upper bound",
  "sensitivity_comp_rows_detected": 295,
  "sensitivity_train_rows_flagged": 1033,
  "sensitivity_families": {
   "toast_bread_and_set_plate": 406,
   "wash_fruit_and_serve": 627
  },
  "superseded_detector": "(effect, solo|coop) multiset -- unfit, kept as the record",
  "superseded_comp_rows_detected": 0,
  "superseded_is_unfit": true,
  "pass": true
 },
 "H_structural": [
  "[true, true]"
 ],
 "H_multiset_size": 4,
 "H_ordered_size": 4,
 "comp_distinct_task_ids": 295,
 "comp_rows_carrying_the_pattern": 295,
 "train_with_heldout_sig_by_multiset": 1424,
 "train_with_heldout_sig_by_ordered": 1033,
 "train_with_heldout_sig_by_pattern": 0,
 "train_with_heldout_sig_decisive": 0,
 "train_with_heldout_sig_actor_blind": 1033,
 "families_of_flagged_decisive": {},
 "families_of_flagged_diagnostic_only": {
  "cut_two_fruits_on_board": 391,
  "toast_bread_and_set_plate": 406,
  "wash_fruit_and_serve": 627
 },
 "pass": true
}
```

## L2
通过：True

```json
{
 "rows": [
  {
   "pool": "rung seed episodes",
   "n_indices": 39,
   "intersect_comp_297_by_task_id": 0,
   "intersect_train_with_heldout_sig": 0,
   "indices_outside_induction_half": 0,
   "families": {
    "sequential_pick_two_and_place": 4,
    "serve_bread_from_counter": 1,
    "dog_push_box_for_two_panda_transport": 4,
    "single_move_asset_to_target": 1,
    "toast_bread_and_set_plate": 2,
    "cut_fruit_on_board": 4,
    "serve_bread_after_checking_cabinet": 4,
    "wash_fruit_and_serve": 3,
    "ensure_all_fruits_on_table": 4,
    "set_plate_and_fork_on_table": 4,
    "cut_two_fruits_on_board": 4,
    "clear_table_with_two_robots_and_put_in_cabinet": 4
   }
  },
  {
   "pool": "unsolved_list (seeds shown)",
   "n_indices": 39,
   "intersect_comp_297_by_task_id": 0,
   "intersect_train_with_heldout_sig": 0,
   "indices_outside_induction_half": 0,
   "families": {
    "sequential_pick_two_and_place": 4,
    "serve_bread_from_counter": 1,
    "dog_push_box_for_two_panda_transport": 4,
    "single_move_asset_to_target": 1,
    "toast_bread_and_set_plate": 2,
    "cut_fruit_on_board": 4,
    "serve_bread_after_checking_cabinet": 4,
    "wash_fruit_and_serve": 3,
    "ensure_all_fruits_on_table": 4,
    "set_plate_and_fork_on_table": 4,
    "cut_two_fruits_on_board": 4,
    "clear_table_with_two_robots_and_put_in_cabinet": 4
   }
  },
  {
   "pool": "unsolved_list (holdout shown)",
   "n_indices": 46,
   "intersect_comp_297_by_task_id": 0,
   "intersect_train_with_heldout_sig": 0,
   "indices_outside_induction_half": 0,
   "families": {
    "dog_check_environment": 4,
    "wash_fruit_and_serve": 1,
    "cut_fruit_on_board": 6,
    "set_plate_and_fork_on_table": 6,
    "parallel_human_dual_asset_to_plate_or_bowl": 1,
    "toast_bread_and_set_plate": 1,
    "serve_bread_after_checking_cabinet": 4,
    "dog_push_box_for_two_panda_transport": 4,
    "cut_two_fruits_on_board": 4,
    "ensure_all_fruits_on_table": 6,
    "clear_table_with_two_robots_and_put_in_cabinet": 5,
    "sequential_pick_two_and_place": 4
   }
  },
  {
   "pool": "support_pool",
   "n_indices": 60,
   "intersect_comp_297_by_task_id": 0,
   "intersect_train_with_heldout_sig": 0,
   "indices_outside_induction_half": 0,
   "families": {
    "sequential_pick_two_and_place": 3,
    "serve_bread_from_counter": 2,
    "dog_push_box_for_two_panda_transport": 4,
    "single_move_asset_to_target": 7,
    "toast_bread_and_set_plate": 3,
    "cut_fruit_on_board": 7,
    "serve_bread_after_checking_cabinet": 1,
    "wash_fruit_and_serve": 7,
    "ensure_all_fruits_on_table": 10,
    "parallel_human_dual_asset_to_plate_or_bowl": 5,
    "set_plate_and_fork_on_table": 2,
    "cut_two_fruits_on_board": 3,
    "clear_table_with_two_robots_and_put_in_cabinet": 3,
    "dog_check_environment": 3
   }
  },
  {
   "pool": "minimality / all tool-addressed episodes",
   "n_indices": 180,
   "intersect_comp_297_by_task_id": 0,
   "intersect_train_with_heldout_sig": 0,
   "indices_outside_induction_half": 0,
   "families": {
    "sequential_pick_two_and_place": 14,
    "serve_bread_from_counter": 10,
    "dog_push_box_for_two_panda_transport": 11,
    "single_move_asset_to_target": 17,
    "toast_bread_and_set_plate": 10,
    "cut_fruit_on_board": 15,
    "serve_bread_after_checking_cabinet": 12,
    "wash_fruit_and_serve": 13,
    "ensure_all_fruits_on_table": 24,
    "parallel_human_dual_asset_to_plate_or_bowl": 8,
    "set_plate_and_fork_on_table": 15,
    "cut_two_fruits_on_board": 11,
    "clear_table_with_two_robots_and_put_in_cabinet": 16,
    "dog_check_environment": 4
   }
  },
  {
   "pool": "tool:run_operator",
   "n_indices": 180,
   "intersect_comp_297_by_task_id": 0,
   "intersect_train_with_heldout_sig": 0,
   "indices_outside_induction_half": 0,
   "families": {
    "sequential_pick_two_and_place": 14,
    "serve_bread_from_counter": 10,
    "dog_push_box_for_two_panda_transport": 11,
    "single_move_asset_to_target": 17,
    "toast_bread_and_set_plate": 10,
    "cut_fruit_on_board": 15,
    "serve_bread_after_checking_cabinet": 12,
    "wash_fruit_and_serve": 13,
    "ensure_all_fruits_on_table": 24,
    "parallel_human_dual_asset_to_plate_or_bowl": 8,
    "set_plate_and_fork_on_table": 15,
    "cut_two_fruits_on_board": 11,
    "clear_table_with_two_robots_and_put_in_cabinet": 16,
    "dog_check_environment": 4
   }
  },
  {
   "pool": "tool:try_bind",
   "n_indices": 180,
   "intersect_comp_297_by_task_id": 0,
   "intersect_train_with_heldout_sig": 0,
   "indices_outside_induction_half": 0,
   "families": {
    "sequential_pick_two_and_place": 14,
    "serve_bread_from_counter": 10,
    "dog_push_box_for_two_panda_transport": 11,
    "single_move_asset_to_target": 17,
    "toast_bread_and_set_plate": 10,
    "cut_fruit_on_board": 15,
    "serve_bread_after_checking_cabinet": 12,
    "wash_fruit_and_serve": 13,
    "ensure_all_fruits_on_table": 24,
    "parallel_human_dual_asset_to_plate_or_bowl": 8,
    "set_plate_and_fork_on_table": 15,
    "cut_two_fruits_on_board": 11,
    "clear_table_with_two_robots_and_put_in_cabinet": 16,
    "dog_check_environment": 4
   }
  },
  {
   "pool": "tool:check_actor",
   "n_indices": 180,
   "intersect_comp_297_by_task_id": 0,
   "intersect_train_with_heldout_sig": 0,
   "indices_outside_induction_half": 0,
   "families": {
    "sequential_pick_two_and_place": 14,
    "serve_bread_from_counter": 10,
    "dog_push_box_for_two_panda_transport": 11,
    "single_move_asset_to_target": 17,
    "toast_bread_and_set_plate": 10,
    "cut_fruit_on_board": 15,
    "serve_bread_after_checking_cabinet": 12,
    "wash_fruit_and_serve": 13,
    "ensure_all_fruits_on_table": 24,
    "parallel_human_dual_asset_to_plate_or_bowl": 8,
    "set_plate_and_fork_on_table": 15,
    "cut_two_fruits_on_board": 11,
    "clear_table_with_two_robots_and_put_in_cabinet": 16,
    "dog_check_environment": 4
   }
  },
  {
   "pool": "tool:contrast_actors",
   "n_indices": 159,
   "intersect_comp_297_by_task_id": 0,
  
```

## L3
通过：True

```json
{
 "transcripts_found": 884,
 "run_dirs": 884,
 "tool_addressed_indices": 180,
 "indices_outside_induction_half": 0,
 "indices_in_comp_297": 0,
 "indices_with_heldout_signature": 0,
 "exact_task_id_text_matches": 0,
 "trigram_plan_matches_ge_0.9": 0,
 "examples": {
  "exact": [],
  "trigram": []
 },
 "pass": true
}
```

## L4
通过：True

```json
{
 "status": "RAN",
 "memory": "outputs/agentic_memory_runner.json",
 "layers_2_3_borrowed_from": "results/viki_memory_experiments/amendment11/skill_memory_v2.json",
 "build_args": {
  "seed": 20260829,
  "per_family": 250,
  "excluded_family": null
 },
 "induction_set": 3598,
 "restricted_pool": 3598,
 "rows_removed": 0,
 "layer2": {
  "identical": true,
  "sha_stored": "5cababcceb0b58d2",
  "sha_rebuilt": "5cababcceb0b58d2"
 },
 "layer3": {
  "identical": true,
  "sha_stored": "0378c3aef2ef7243",
  "sha_rebuilt": "0378c3aef2ef7243"
 },
 "pass": true
}
```

## L5
通过：True

```json
{
 "status": "RAN",
 "library": "outputs/agentic_library_runner.json",
 "operators": 7,
 "provenance_intersections": 0,
 "provenance_rows": [
  {
   "effect_key": "pos.name",
   "seed_episodes": [
    0,
    1,
    2,
    3,
    4,
    7,
    9,
    21,
    27,
    52,
    61,
    69,
    112
   ],
   "verified_on": [
    0,
    1,
    3,
    4,
    5,
    6,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23
   ],
   "intersect_comp_297": 0,
   "intersect_heldout_sig": 0
  },
  {
   "effect_key": "pos.name",
   "seed_episodes": [
    31,
    34
   ],
   "verified_on": [
    31,
    34,
    57
   ],
   "intersect_comp_297": 0,
   "intersect_heldout_sig": 0
  },
  {
   "effect_key": "pos.name",
   "seed_episodes": [
    79
   ],
   "verified_on": [
    1,
    7,
    10,
    21,
    27,
    42,
    51
   ],
   "intersect_comp_297": 0,
   "intersect_heldout_sig": 0
  },
  {
   "effect_key": "pos.name",
   "seed_episodes": [
    90
   ],
   "verified_on": [
    7,
    10,
    51
   ],
   "intersect_comp_297": 0,
   "intersect_heldout_sig": 0
  },
  {
   "effect_key": "pos.name",
   "seed_episodes": [
    21
   ],
   "verified_on": [
    1,
    7,
    10,
    21,
    27,
    42,
    51
   ],
   "intersect_comp_297": 0,
   "intersect_heldout_sig": 0
  },
  {
   "effect_key": "is_activated",
   "seed_episodes": [
    12
   ],
   "verified_on": [
    4,
    6,
    9,
    12,
    13,
    15,
    17,
    18,
    22,
    24,
    26,
    32,
    33,
    37,
    38,
    39,
    48,
    50,
    53,
    58
   ],
   "intersect_comp_297": 0,
   "intersect_heldout_sig": 0
  },
  {
   "effect_key": "is_activated",
   "seed_episodes": [
    18
   ],
   "verified_on": [
    4,
    6,
    9,
    12,
    13,
    15,
    17,
    18,
    22,
    24,
    26,
    32,
    33,
    37,
    38,
    39,
    46,
    47,
    48,
    50
   ],
   "intersect_comp_297": 0,
   "intersect_heldout_sig": 0
  }
 ],
 "body_overlap": [
  {
   "operator": 0,
   "effect": "pos.name",
   "body_len": 5,
   "max_containment_tokens": 0.0,
   "max_containment_verbs_only": 1.0
  },
  {
   "operator": 1,
   "effect": "is_activated",
   "body_len": 2,
   "max_containment_tokens": 0.0,
   "max_containment_verbs_only": 0.0
  },
  {
   "operator": 2,
   "effect": "is_activated",
   "body_len": 3,
   "max_containment_tokens": 0.0,
   "max_containment_verbs_only": 0.0
  },
  {
   "operator": 3,
   "effect": "pos.name",
   "body_len": 8,
   "max_containment_tokens": 0.0,
   "max_containment_verbs_only": 0.5
  },
  {
   "operator": 4,
   "effect": "pos.name",
   "body_len": 8,
   "max_containment_tokens": 0.0,
   "max_containment_verbs_only": 0.5
  },
  {
   "operator": 5,
   "effect": "pos.name",
   "body_len": 7,
   "max_containment_tokens": 0.0,
   "max_containment_verbs_only": 0.6
  },
  {
   "operator": 6,
   "effect": "pos.name",
   "body_len": 8,
   "max_containment_tokens": 0.0,
   "max_containment_verbs_only": 0.5
  }
 ],
 "overlap_equal_1.0_pairs": [],
 "body_overlap_token_level_is_non_discriminating": "operator bodies are variable-abstracted (?x/?y) and comp reference plans are ground, so token trigrams cannot intersect. Max observed: 0.0. This variant establishes nothing either way.",
 "overlap_verbs_only_equal_1.0_pairs": 189,
 "overlap_verbs_only_is_expected_not_evidence": "the primitive set is 5-6 verbs shared by every family by construction, so a verb-sequence match is not evidence of having seen a comp row",
 "max_containment_verbs_only": 1.0,
 "effect_schema": [
  {
   "operator": 0,
   "effect_key": "pos.name",
   "supporting_rows_with_heldout_sig": 0,
   "supporting_rows_without": 2941,
   "only_supported_by_heldout_rows": false
  },
  {
   "operator": 1,
   "effect_key": "is_activated",
   "supporting_rows_with_heldout_sig": 0,
   "supporting_rows_without": 848,
   "only_supported_by_heldout_rows": false
  },
  {
   "operator": 2,
   "effect_key": "is_activated",
   "supporting_rows_with_heldout_sig": 0,
   "supporting_rows_without": 848,
   "only_supported_by_heldout_rows": false
  },
  {
   "operator": 3,
   "effect_key": "pos.name",
   "supporting_rows_with_heldout_sig": 0,
   "supporting_rows_without": 2941,
   "only_supported_by_heldout_rows": false
  },
  {
   "operator": 4,
   "effect_key": "pos.name",
   "supporting_rows_with_heldout_sig": 0,
   "supporting_rows_without": 2941,
   "only_supported_by_heldout_rows": false
  },
  {
   "operator": 5,
   "effect_key": "pos.name",
   "supporting_rows_with_heldout_sig": 0,
   "supporting_rows_without": 2941,
   "only_supported_by_heldout_rows": false
  },
  {
   "operator": 6,
   "effect_key": "pos.name",
   "supporting_rows_with_heldout_sig": 0,
   "supporting_rows_without": 2941,
   "only_supported_by_heldout_rows": false
  }
 ],
 "schemas_only_supported_by_heldout_rows": 0,
 "pass": true
}
```

## L6
未执行，缺 outputs/agentic_*fold*.json

```json
{
 "status": "未执行",
 "missing": "outputs/agentic_*fold*.json",
 "finding": "no per-fold agent-built Layer 1 exists on disk; the agent library has no held-out-family column, so there is nothing to check for reuse",
 "affects": "held-out-family column only; does not bear on comp"
}
```

## 附录：产物

- `audit/comp_leakage_2026-09-05/AUDIT_SUMMARY.json`  sha256 `d0c4018463a3d6e1f1376ab1f966065d2cf0a0f61850b47abe1e727b75f0d4ac`  20217 bytes
- `audit/comp_leakage_2026-09-05/L0_comp_297.json`  sha256 `b3657af294d124954c6fef0081d80a44579d8a838ebe3c272f0880ad3b2e5a45`  110076 bytes
- `audit/comp_leakage_2026-09-05/L0_heldout_family_folds.json`  sha256 `aff973f1f0237354e6f9d7bd43ecf9c14527c21f21818f03d90fadff2c5983df`  2415 bytes
- `audit/comp_leakage_2026-09-05/L0_id_manifest_924.json`  sha256 `f5a310e13eb86193a98cf7535d2f61920028363187d68ab6c43eb0594fbf17d1`  94594 bytes
- `audit/comp_leakage_2026-09-05/L0_induction_half.json`  sha256 `ca07f39af3a1ec697e4130673e9f441a7b41d20f51a9bdad67edb368334529a7`  382450 bytes
- `audit/comp_leakage_2026-09-05/L0_selfcheck_half.json`  sha256 `473530b73d202f768039b2e6bc6776fbf519ad6c9f243f0661a8e8f81f3602f1`  381976 bytes
- `audit/comp_leakage_2026-09-05/L0_support_pool.json`  sha256 `a106095b97bb7cc93dabd3cfff4d79fafa23b1bebefb737af80b2c8da0f279f3`  9318 bytes
- `audit/comp_leakage_2026-09-05/L0_test_split.json`  sha256 `71df410adb65a7e868d3413fbb26637c7a694e3c8a468dc600cccef9a9cc67d8`  190518 bytes
- `audit/comp_leakage_2026-09-05/L0_unsolved_lists.json`  sha256 `c878925649de9d2db25aec10c59097bbaeb156e3e2f01797628c8be2016ee953`  9244 bytes
- `audit/comp_leakage_2026-09-05/L1_H.json`  sha256 `d161abe2a47a43172d46caa6cc5695e691f0ea161b3b8dd698462d7e545619b0`  1950 bytes
- `audit/comp_leakage_2026-09-05/L1_comp_signatures.json`  sha256 `1205856cfd7f386ce43b2bf002d8235cb421f03a6a1f0a74ff606db28e172a94`  172815 bytes
- `audit/comp_leakage_2026-09-05/L1_train_signatures.json`  sha256 `f3494e673be456c42f47c64b74d0f641637efcc84fae2601b55947b2648f497e`  5786062 bytes
- `audit/comp_leakage_2026-09-05/L1_train_with_heldout_sig.json`  sha256 `7fa74567f9924c581f89e9c425dd249d772ace2a46613fca232a93b105fbb078`  8406 bytes
- `audit/comp_leakage_2026-09-05/L2_pool_intersections.json`  sha256 `1291d667e52261cbdf6d9686edd88a82689bbea59b1c5e6b220a24f2da7ace4b`  6687 bytes
- `audit/comp_leakage_2026-09-05/L3_runs.json`  sha256 `a5a911bc0d3059e88d34cb362764eb1fe7e5e88db66d4f2139096e1eb9aa25b1`  467933 bytes
- `audit/comp_leakage_2026-09-05/L3_transcript_audit.json`  sha256 `efd0bd833c4f6a57942591208427d1dc89787f6852235234ddae3c1d01fba5e7`  316 bytes
- `audit/comp_leakage_2026-09-05/L4_mining_pool.json`  sha256 `e0bb793398451fbd10c771b13f5cfdeea1e2066dd28b1887194a4d9f90795122`  559 bytes
- `audit/comp_leakage_2026-09-05/L5_library_covering.json`  sha256 `50d4afa74b20c74e2f9db3348037ed6adaa2b5e7f007e89096b7c0bc097bfdad`  4851 bytes
- `audit/comp_leakage_2026-09-05/L6_fold_reuse.json`  sha256 `b0a47344c763970b711771a2a9f163a9d117b5a952e4c39446c60fc79651d8ea`  286 bytes