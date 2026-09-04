#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from viki_amendment3_f2 import exact_mcnemar_p
from viki_amendment5 import (
    BACKBONES,
    BENCHMARK_ROOT,
    DATA_ROOT,
    H0_PATH,
    M0_INSTANCES_PATH,
    atomic_json,
    file_sha256,
    fingerprint,
    load_bench,
    load_jsonl,
    messages_sha256,
    native,
    token_count,
    validate_local_service,
    write_jsonl_snapshot,
)
from viki_amendment6 import (
    ID_ROWS,
    OOD_ROWS,
    ROOT,
    SEED,
    TOKEN_TOLERANCE,
    GateFailure,
    load_source_rows,
    split_configuration,
    trim_prompt,
    validate_reused_source,
)

from habitat_llm.evaluation.viki_branch_conditions import get_instruction
from habitat_llm.evaluation.viki_gmemory import (
    GMemoryState,
    canonical_json,
    load_author_prompts,
    parse_numbered_list,
    parse_relevance_score,
    parse_rule_operations,
    render_retrieval_prompt,
    reward_retrieved_insights,
    update_rules,
)
from habitat_llm.evaluation.viki_memory_skill import (
    add_memory_to_messages,
    get_prompt_context,
)

OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment7"
PREREGISTRATION_PATH = OUTPUT_DIR / "preregistration.json"
PORT_NOTE_PATH = OUTPUT_DIR / "GMEMORY_PORT_NOTE.md"
SUPERSESSION_PATH = OUTPUT_DIR / "closure_supersession.json"
TRAIN_INTERACTIONS_PATH = OUTPUT_DIR / "train_interactions.jsonl"
TRAIN_EMBEDDINGS_PATH = OUTPUT_DIR / "train_minilm_embeddings.npz"
TRAIN_PREPARATION_SUMMARY_PATH = OUTPUT_DIR / "train_preparation.summary.json"
TRAIN_CONDENSATIONS_PATH = OUTPUT_DIR / "train_condensations.jsonl"
TRAIN_CONDENSATIONS_SUMMARY_PATH = OUTPUT_DIR / "train_condensations.summary.json"
TRAIN_HIERARCHY_CALLS_PATH = OUTPUT_DIR / "train_hierarchy_calls.jsonl"
TRAIN_HIERARCHY_RUN_PATH = OUTPUT_DIR / "train_hierarchy_calls.run.json"
TRAIN_HIERARCHY_ATTEMPTS_PATH = OUTPUT_DIR / "train_hierarchy_attempts.jsonl"
TRAIN_HIERARCHY_PENDING_PATH = OUTPUT_DIR / "train_hierarchy_attempt.pending.json"
TRAIN_HIERARCHY_RECOVERY_PATH = OUTPUT_DIR / "train_hierarchy_recovery.json"
TRAIN_HIERARCHY_INTERRUPTION_RECOVERY_PATH = (
    OUTPUT_DIR / "train_hierarchy_interruption_recovery.json"
)
TRAIN_HIERARCHY_RESOLUTION_PATH = OUTPUT_DIR / "train_hierarchy_resolution.json"
TRAIN_HIERARCHY_PATH = OUTPUT_DIR / "train_hierarchy.json"
TRAIN_HIERARCHY_SUMMARY_PATH = OUTPUT_DIR / "train_hierarchy.summary.json"
GENERATION_CLOSURE_PATH = OUTPUT_DIR / "AMENDMENT7_GENERATION_CLOSED.json"
AMENDMENT6_DIR = ROOT / "results/viki_memory_experiments/amendment6"
AMENDMENT6_CLOSURE_PATH = (
    ROOT
    / "results/viki_memory_experiments/amendment6/AMENDMENT6_GENERATION_CLOSED.json"
)
GMEMORY_ROOT = ROOT / "third_party/GMemory"
GMEMORY_REVISION = "7b581c51d993bd600df14691d101d7e601040cc6"
GMEMORY_REPOSITORY = "https://github.com/bingreeky/GMemory"
GMEMORY_SOURCE_PATHS = (
    GMEMORY_ROOT / "mas/memory/mas_memory/GMemory.py",
    GMEMORY_ROOT / "mas/memory/mas_memory/prompt.py",
    GMEMORY_ROOT / "mas/utils.py",
    GMEMORY_ROOT / "tasks/mas_workflow/format.py",
)
VIKI_REPOSITORY = "https://github.com/MARS-EAI/VIKI-R"
VIKI_REVISION = "a0f13ed1ffe2cc509639fcd34d3f2ecbf4a2e5c5"
TRAIN_ROWS = 6699
TRAIN_SEGMENTS = 19499
SUCCESSFUL_TOPK = 1
FAILED_TOPK = 0
INSIGHTS_TOPK = 3
SIMILARITY_THRESHOLD = 0.0
GRAPH_HOP = 1
START_INSIGHTS_THRESHOLD = 5
ROUNDS_PER_INSIGHTS = 5
INSIGHTS_POINT_NUM = 5
TRAIN_INSIGHT_EVENTS = TRAIN_ROWS // ROUNDS_PER_INSIGHTS
TRAIN_INSIGHT_CALLS = TRAIN_INSIGHT_EVENTS * INSIGHTS_POINT_NUM
TRAIN_MERGE_EVENTS = TRAIN_ROWS // 20
MODE_B_ROWS = OOD_ROWS + ID_ROWS
MODE_B_INSIGHT_EVENTS = (
    (TRAIN_ROWS + OOD_ROWS) // ROUNDS_PER_INSIGHTS
    - TRAIN_ROWS // ROUNDS_PER_INSIGHTS
    + (TRAIN_ROWS + ID_ROWS) // ROUNDS_PER_INSIGHTS
    - TRAIN_ROWS // ROUNDS_PER_INSIGHTS
)
MODE_B_INSIGHT_CALLS = MODE_B_INSIGHT_EVENTS * INSIGHTS_POINT_NUM
MODE_B_MERGE_EVENTS = (
    (TRAIN_ROWS + OOD_ROWS) // 20
    - TRAIN_ROWS // 20
    + (TRAIN_ROWS + ID_ROWS) // 20
    - TRAIN_ROWS // 20
)
REQUIRED_PLAN_CALLS = 2 * (OOD_ROWS + ID_ROWS)
RETRIEVAL_RERANK_CALLS_PER_ROW = 2
MODE_A_RERANK_CALLS = RETRIEVAL_RERANK_CALLS_PER_ROW * (OOD_ROWS + ID_ROWS)
MODE_B_SHADOW_RERANK_CALLS = MODE_A_RERANK_CALLS
MODE_B_LIVE_RERANK_CALLS = MODE_A_RERANK_CALLS
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
MODEL_REVISION = "89c86200743eec961a297729e7990e8f2ddbc4c5"
# Which backbone this process talks to. One switch drives both the gate key and the
# served name, because the two disagreeing is exactly how a cell gets written with the
# wrong model in its metadata: the run would pass the gate against one model and label
# itself with another. Default is the 72B every archived cell was produced on.
from viki_amendment5 import BACKBONE, SERVED_MODEL_FOR_BACKBONE  # noqa: E402

SERVED_MODEL = SERVED_MODEL_FOR_BACKBONE
MEMORY_CONTROL_TEMPERATURE = 0.1
MEMORY_CONTROL_MAX_TOKENS = 512
PLAN_MAX_TOKENS = 2000


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def author_source_hashes() -> Dict[str, str]:
    if any(not path.is_file() for path in GMEMORY_SOURCE_PATHS):
        raise GateFailure("Initialize the pinned G-Memory submodule before freezing")
    revision = subprocess.run(
        ["git", "-C", str(GMEMORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != GMEMORY_REVISION:
        raise GateFailure(f"Unexpected G-Memory revision: {revision}")
    return {
        str(path.relative_to(ROOT)): file_sha256(path) for path in GMEMORY_SOURCE_PATHS
    }


def require_frozen() -> Dict[str, Any]:
    required = [PREREGISTRATION_PATH, PORT_NOTE_PATH, SUPERSESSION_PATH]
    if any(not path.is_file() for path in required):
        raise GateFailure("Freeze Amendment 7 before any build or generation")
    observed = json.loads(PREREGISTRATION_PATH.read_text())
    if observed != preregistration():
        raise GateFailure("Amendment 7 preregistration changed")
    if PORT_NOTE_PATH.read_text() != port_note():
        raise GateFailure("Amendment 7 port note changed")
    return observed


def viki_revision() -> str:
    benchmark_root = DATA_ROOT.parents[3]
    revision = subprocess.run(
        ["git", "-C", str(benchmark_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != VIKI_REVISION:
        raise GateFailure(f"Unexpected VIKI revision: {revision}")
    return revision


def port_note() -> str:
    return f"""# Amendment 7 G-Memory Port Note

Filed before any Amendment 7 model or memory-control generation call.

## Source identity

- Paper: G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems, arXiv:2506.07398v2.
- Official code: `{GMEMORY_REPOSITORY}` at `{GMEMORY_REVISION}`.
- VIKI benchmark: `{VIKI_REPOSITORY}` at `{VIKI_REVISION}`.
- This is a task-interface port, not a claim of byte-identical execution of the authors' ALFWorld/PDDL/FEVER/SciWorld harness.

## Three-tier mapping

- Interaction graph: one VIKI episode is one interaction state because VIKI-L2 emits a complete executable plan in one model turn. The state stores the instruction, activated robot roles and APIs, complete plan, post-generation official success label, and author-prompt condensed key steps.
- Query graph: one node per instruction, inserted in ascending source index order. Edges use the author's 0.7 threshold and one-hop traversal. Duplicate instruction strings collapse as in the author's NetworkX graph while interaction records remain distinct.
- Insight graph: general rules use the released all-success critique prompt, operation parser, score updates, five-point schedule, and 20-memory FINCH merge schedule. The 6,699-row train bank contains only successful ground-truth trajectories, so the failed train tier is empty; no negative example is invented.
- MAS roles: activated VIKI robot IDs, robot morphologies, and available operation APIs map to G-Memory roles. The author default `use_projector=False` is preserved, so insights are shared across activated roles.

## Author defaults preserved

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- `successful_topk={SUCCESSFUL_TOPK}`, `failed_topk={FAILED_TOPK}`, `insights_topk={INSIGHTS_TOPK}`, `threshold={SIMILARITY_THRESHOLD}`, `hop={GRAPH_HOP}`, `use_projector=False`.
- Insight schedule: start at {START_INSIGHTS_THRESHOLD}, update every {ROUNDS_PER_INSIGHTS} memories using {INSIGHTS_POINT_NUM} sampled points, merge every 20 memories.
- Native retrieval expands to two successful candidates, reranks both with the released relevance prompt, and returns one trajectory plus at most three insights.
- Trajectory condensation uses the released extraction prompt, temperature 0.1, and 512 output tokens. Insight, merge, failed-diagnosis, and reranking calls use released prompts. Plan generation remains greedy at temperature 0 with 2,000 output tokens.
- The complete upstream FINCH dependency pair is fixed: `finch-clust==0.2.0` and `finchpy==0.0.1`.

## Required port judgments

- VIKI train ground-truth plans are successful completed interactions. No train failure labels or textual feedback exist in M0.
- The one-shot executable plan replaces the authors' action/observation state chain. Robot-action ownership remains explicit.
- The author Chroma/StateChain wrapper is harness-specific. The port uses structured interaction records and a cached normalized MiniLM index while preserving graph, traversal, candidate, prompt, parser, insight-score, FINCH, and schedule semantics.
- Mode B receives the current official label only after generation and scoring. The current row cannot see its own label or trajectory.
- Mode B OOD and ID are independent forks of the same frozen train hierarchy. Neither split inherits the other's evaluation rows.
- Python and NumPy RNGs use seed {SEED}; sampled source IDs and update order are recorded.
- The authors' successful-example, past-execution, and key-insight sections are inserted into the official VIKI user message without changing the official output contract.

## Token budget and adaptive preflight

- Each row targets the frozen Qwen2.5-VL-72B segment arm's injected-token count.
- Over-budget retrieval uses Amendment 3.1 extend-then-truncate and must finish within {TOKEN_TOLERANCE:.0%} of target.
- Native under-budget retrieval is not padded. Absolute and relative shortfall are recorded per row.
- Mode A receives a full exact preflight before generation.
- Final Mode B prompts depend on earlier generated trajectories and labels. Therefore Mode B receives a full frozen-shadow preflight before each split plus an exact fail-closed retrieval/token/context gate immediately before every sequential generation. This is required by causality, not a post-result change.

## Modes and mutation boundaries

- Mode A is primary and train-only frozen. Its canonical hierarchy hash before and after each split must match.
- Mode B is secondary and labeled test-time adaptation. Each split starts from an independent copy of Mode A, processes ascending dataset indices with one worker, and records hierarchy hashes before and after every assimilation.
- Canonical hashes cover query nodes/edges, interactions, insights, hyperparameters, source identities, author revision, and call-ledger hashes.

## Call accounting

- Required 72B plan calls: {REQUIRED_PLAN_CALLS:,}.
- Train trajectory-condensation calls: {TRAIN_ROWS:,}.
- Train periodic insight calls: {TRAIN_INSIGHT_CALLS:,} across {TRAIN_INSIGHT_EVENTS:,} events.
- Train merge events: {TRAIN_MERGE_EVENTS:,}; exact merge calls are state-dependent and recorded individually.
- Mode A retrieval-rerank calls: {MODE_A_RERANK_CALLS:,}.
- Mode B frozen-shadow rerank calls: {MODE_B_SHADOW_RERANK_CALLS:,}.
- Mode B live rerank calls: {MODE_B_LIVE_RERANK_CALLS:,}.
- Mode B trajectory-condensation calls: {MODE_B_ROWS:,}; failed-diagnosis calls depend on observed failures and are reported.
- Mode B periodic insight calls: {MODE_B_INSIGHT_CALLS:,} across {MODE_B_INSIGHT_EVENTS:,} events; {MODE_B_MERGE_EVENTS:,} merge events have state-dependent call counts.
- Optional GPT-4o is authorized only after all required 72B arms and analyses complete.

## PartNR boundary

+PartNR is the more direct scientific venue because it retains turn-taking, partner-induced state change, and multi-agent trajectories. It remains blocked on original PartNR memory banks and generation logs absent from this workspace. No PartNR generation is authorized here.
""".replace(
        "\n+PartNR", "\nPartNR"
    )


def preregistration() -> Dict[str, Any]:
    note = port_note()
    return {
        "task": "Amendment7_GMemory_baseline",
        "status": "PREREGISTERED",
        "filed_before_generation": True,
        "training": False,
        "gradient_updates": False,
        "fine_tuning": False,
        "completed_results_modified": False,
        "supersedes_amendment6_closure_once": True,
        "seed": SEED,
        "h0_sha256": file_sha256(H0_PATH),
        "source": {
            "paper": "arXiv:2506.07398v2",
            "repository": GMEMORY_REPOSITORY,
            "revision": GMEMORY_REVISION,
            "source_hashes": author_source_hashes(),
            "viki_repository": VIKI_REPOSITORY,
            "viki_revision": VIKI_REVISION,
        },
        "gmemory": {
            "architecture": ["insight", "query", "interaction"],
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "successful_topk": SUCCESSFUL_TOPK,
            "failed_topk": FAILED_TOPK,
            "insights_topk": INSIGHTS_TOPK,
            "threshold": SIMILARITY_THRESHOLD,
            "hop": GRAPH_HOP,
            "use_projector": False,
            "query_edge_threshold": 0.7,
            "start_insights_threshold": START_INSIGHTS_THRESHOLD,
            "rounds_per_insights": ROUNDS_PER_INSIGHTS,
            "insights_point_num": INSIGHTS_POINT_NUM,
            "finch_dependencies": ["finch-clust==0.2.0", "finchpy==0.0.1"],
        },
        "train_build": {
            "rows": TRAIN_ROWS,
            "segments": TRAIN_SEGMENTS,
            "labels": {"successful": TRAIN_ROWS, "failed": 0},
            "order": "ascending source_train_index",
            "trajectory_condensation_calls": TRAIN_ROWS,
            "periodic_insight_events": TRAIN_INSIGHT_EVENTS,
            "periodic_insight_calls": TRAIN_INSIGHT_CALLS,
            "merge_events": TRAIN_MERGE_EVENTS,
            "merge_calls": "state-dependent; record every call",
        },
        "modes": {
            "A": {
                "label": "frozen train-only primary",
                "test_time_adaptation": False,
                "hash_gate": "before equals after for each split",
            },
            "B": {
                "label": "native cross-trial secondary",
                "test_time_adaptation": True,
                "split_isolation": True,
                "order": "ascending dataset index",
                "workers": 1,
                "preflight": "full frozen shadow plus per-row live fail-closed gate",
                "mutation_log": "hierarchy hash before and after every row",
            },
        },
        "required_runs": [
            {"mode": "A", "split": "ood", "rows": OOD_ROWS},
            {"mode": "A", "split": "id", "rows": ID_ROWS},
            {"mode": "B", "split": "ood", "rows": OOD_ROWS},
            {"mode": "B", "split": "id", "rows": ID_ROWS},
        ],
        "required_plan_generation_calls": REQUIRED_PLAN_CALLS,
        "memory_control_calls": {
            "train_condensation": TRAIN_ROWS,
            "train_periodic_insights": TRAIN_INSIGHT_CALLS,
            "train_merge": "state-dependent",
            "mode_A_rerank": MODE_A_RERANK_CALLS,
            "mode_B_shadow_rerank": MODE_B_SHADOW_RERANK_CALLS,
            "mode_B_live_rerank": MODE_B_LIVE_RERANK_CALLS,
            "mode_B_condensation": MODE_B_ROWS,
            "mode_B_failed_diagnosis": "outcome-dependent",
            "mode_B_periodic_insights": MODE_B_INSIGHT_CALLS,
            "mode_B_merge": "state-dependent",
        },
        "optional_backbone": {
            "name": "gpt_4o_optional",
            "condition": "only after all required 72B arms and analyses complete",
            "authorized_plan_generation_calls": REQUIRED_PLAN_CALLS,
        },
        "order": [
            "port_note",
            "train_build",
            "mode_A_ood",
            "mode_A_id",
            "mode_B_ood",
            "mode_B_id",
            "optional_GPT4o",
            "closure",
        ],
        "generation": {
            "backbone": "qwen2_5_vl_72b",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "served_model": SERVED_MODEL,
            "temperature": 0,
            "max_output_tokens": 2000,
            "scorer": "official VIKI-L2",
            "scorer_seed": "original row index",
        },
        "token_budget": {
            "target": "frozen segment-arm injected tokens per row",
            "tolerance": TOKEN_TOLERANCE,
            "over_budget": "Amendment 3.1 extend-then-truncate prefix",
            "under_budget": "do not pad; record shortfall",
        },
        "gates": [
            "port note filed before generation",
            "full Mode A and Mode B shadow token preflight",
            "per-row Mode B live token/context gate",
            "Mode A hierarchy hash unchanged",
            "Mode B fixed order and per-row mutation hashes",
            "reuse hash match on every reused output",
            "format reported with no exemption",
        ],
        "predictions": {
            "OOD_mode_A": "matches trajectory RAG rather than beating it",
            "OOD_mode_B": "exceeds Mode A; gap attributed to test-time adaptation",
            "ID": "no directional prediction",
        },
        "endings": {
            "mode_A_beats_segment_OOD": "G-Memory frozen superiority is the OOD headline",
            "mode_B_beats_all": "adaptation is the headline; adaptation for other arms is future work",
            "otherwise": "report frozen and adaptive channels separately",
        },
        "port_note_sha256": sha256_text(note),
        "partnr": {
            "importance": "primary scientific venue for G-Memory",
            "status": "BLOCKED_MISSING_ORIGINAL_BANKS_AND_LOGS",
        },
    }


def freeze() -> Dict[str, Any]:
    if not AMENDMENT6_CLOSURE_PATH.is_file():
        raise GateFailure("Amendment 7 requires the Amendment 6 closure")
    closure = json.loads(AMENDMENT6_CLOSURE_PATH.read_text())
    if not closure.get("generation_closed"):
        raise GateFailure("Amendment 6 generation closure is not active")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    note = port_note()
    expected = preregistration()
    if (
        PREREGISTRATION_PATH.is_file()
        and json.loads(PREREGISTRATION_PATH.read_text()) != expected
    ):
        raise GateFailure("Amendment 7 preregistration already differs")
    if PORT_NOTE_PATH.is_file() and PORT_NOTE_PATH.read_text() != note:
        raise GateFailure("Amendment 7 port note already differs")
    PORT_NOTE_PATH.write_text(note)
    atomic_json(PREREGISTRATION_PATH, expected)
    supersession = {
        "task": "Amendment7_closure_supersession",
        "status": "ACTIVE_ONCE",
        "scope_complete": True,
        "completed_results_modified": False,
        "supersedes": str(AMENDMENT6_CLOSURE_PATH),
        "superseded_sha256": file_sha256(AMENDMENT6_CLOSURE_PATH),
        "preregistration": str(PREREGISTRATION_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "port_note": str(PORT_NOTE_PATH),
        "port_note_sha256": file_sha256(PORT_NOTE_PATH),
        "authorized_required_plan_calls": REQUIRED_PLAN_CALLS,
        "authorized_optional_plan_calls": REQUIRED_PLAN_CALLS,
        "memory_control_calls": "counted separately by category",
        "after_scope": "reinstate permanent generation closure",
    }
    if (
        SUPERSESSION_PATH.is_file()
        and json.loads(SUPERSESSION_PATH.read_text()) != supersession
    ):
        raise GateFailure("Amendment 7 closure supersession already differs")
    atomic_json(SUPERSESSION_PATH, supersession)
    return expected


def prepare_train() -> Dict[str, Any]:
    require_frozen()
    viki_revision()
    source_ids, source_rows, _ = load_source_rows()
    if len(source_ids) != TRAIN_ROWS:
        raise GateFailure(f"Expected {TRAIN_ROWS} valid train rows")
    train_path = DATA_ROOT / "train.parquet"
    train = pd.read_parquet(train_path)
    if len(train) != 7196:
        raise GateFailure("Frozen VIKI train row count changed")
    raw_instances = [
        json.loads(line) for line in M0_INSTANCES_PATH.read_text().splitlines() if line
    ]
    source_plan_hashes: Dict[int, set[str]] = {}
    for item in raw_instances:
        source = int(item["source_train_index"])
        source_plan_hashes.setdefault(source, set()).add(
            str(item["source_plan_sha256"])
        )
    prompts = load_author_prompts(GMEMORY_ROOT)
    records: Dict[int, Dict[str, Any]] = {}
    task_mains: List[str] = []
    robot_counts: Dict[str, int] = {}
    for source in source_ids:
        sample = native(train.iloc[source].to_dict())
        instruction, robots, available_actions = get_prompt_context(sample)
        plan = native(sample["reward_model"]["ground_truth"]["time_steps"])
        plan_hash = sha256_text(canonical_json(plan))
        if source_plan_hashes.get(source) != {plan_hash}:
            raise GateFailure(f"M0 source-plan hash changed at train:{source}")
        reconstructed = [
            native(step) for segment in source_rows[source] for step in segment.demo
        ]
        if reconstructed != plan:
            raise GateFailure(f"M0 source-plan reconstruction failed at train:{source}")
        roles = [
            {
                "robot_id": robot,
                "robot_type": robots[robot],
                "available_actions": available_actions[robot],
            }
            for robot in sorted(robots)
        ]
        task_description = (
            f"Task instruction: {instruction}\n"
            f"Activated robot roles: {canonical_json(roles)}"
        )
        trajectory = (
            "> Generate one complete multi-robot executable plan\n"
            f"{canonical_json(plan)}\n"
            "Task completed successfully.\n"
        )
        clean_trajectory = re.sub(r"\d+", "", trajectory)
        extraction_messages = [
            {
                "role": "system",
                "content": prompts["extract_true_traj_system_prompt"],
            },
            {
                "role": "user",
                "content": prompts["extract_true_traj_user_prompt"].format(
                    task=task_description,
                    trajectory=clean_trajectory,
                ),
            },
        ]
        task_mains.append(instruction)
        count = str(len(roles))
        robot_counts[count] = robot_counts.get(count, 0) + 1
        records[source] = {
            "index": source,
            "source_train_index": source,
            "memory_id": f"train:{source}",
            "task_main": instruction,
            "task_description": task_description,
            "roles": roles,
            "label": True,
            "interaction_turns": 1,
            "trajectory": trajectory,
            "clean_trajectory": clean_trajectory,
            "source_plan_sha256": plan_hash,
            "source_prompt_sha256": sha256_text(canonical_json(sample["prompt"])),
            "extraction_messages": extraction_messages,
            "extraction_prompt_sha256": sha256_text(
                canonical_json(extraction_messages)
            ),
        }
    write_jsonl_snapshot(TRAIN_INTERACTIONS_PATH, records, source_ids)
    verified = load_jsonl(TRAIN_INTERACTIONS_PATH)
    if set(verified) != set(source_ids):
        raise GateFailure("Train interaction ledger coverage changed after write")
    from sentence_transformers import SentenceTransformer

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(embedding_model, device="cpu")
    embeddings = np.asarray(
        model.encode(
            task_mains,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if embeddings.shape != (TRAIN_ROWS, 384):
        raise GateFailure(f"Unexpected MiniLM embedding shape: {embeddings.shape}")
    cache_key = sha256_text(
        canonical_json(
            {
                "model": embedding_model,
                "source_ids": source_ids,
                "task_mains": task_mains,
            }
        )
    )
    np.savez_compressed(
        TRAIN_EMBEDDINGS_PATH,
        cache_key=np.asarray(cache_key),
        source_ids=np.asarray(source_ids, dtype=np.int64),
        embeddings=embeddings,
    )
    summary = {
        "task": "Amendment7_train_preparation",
        "status": "PASS",
        "model_generation_calls": 0,
        "memory_control_calls": 0,
        "train_rows": len(records),
        "train_segments": sum(len(row) for row in source_rows.values()),
        "interaction_turns_per_episode": 1,
        "source_plan_hashes_verified": len(records),
        "lossless_plan_reconstructions": len(records),
        "robot_count_distribution": robot_counts,
        "embedding_model": embedding_model,
        "embedding_shape": list(embeddings.shape),
        "embedding_cache_key": cache_key,
        "author_revision": GMEMORY_REVISION,
        "author_source_hashes": author_source_hashes(),
        "author_prompt_hashes": {
            name: sha256_text(value) for name, value in prompts.items()
        },
        "viki_revision": VIKI_REVISION,
        "train_parquet_sha256": file_sha256(train_path),
        "results": str(TRAIN_INTERACTIONS_PATH),
        "results_sha256": file_sha256(TRAIN_INTERACTIONS_PATH),
        "embeddings": str(TRAIN_EMBEDDINGS_PATH),
        "embeddings_sha256": file_sha256(TRAIN_EMBEDDINGS_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "port_note_sha256": file_sha256(PORT_NOTE_PATH),
    }
    atomic_json(TRAIN_PREPARATION_SUMMARY_PATH, summary)
    return summary


def validate_train_preparation() -> Dict[int, Dict[str, Any]]:
    if not TRAIN_PREPARATION_SUMMARY_PATH.is_file():
        raise GateFailure("Run Amendment 7 train preparation first")
    summary = json.loads(TRAIN_PREPARATION_SUMMARY_PATH.read_text())
    if (
        summary.get("status") != "PASS"
        or summary.get("train_rows") != TRAIN_ROWS
        or summary.get("results_sha256") != file_sha256(TRAIN_INTERACTIONS_PATH)
        or summary.get("embeddings_sha256") != file_sha256(TRAIN_EMBEDDINGS_PATH)
        or summary.get("author_revision") != GMEMORY_REVISION
    ):
        raise GateFailure("Amendment 7 train preparation certificate is stale")
    records = load_jsonl(TRAIN_INTERACTIONS_PATH)
    if len(records) != TRAIN_ROWS:
        raise GateFailure("Amendment 7 train interaction coverage is incomplete")
    return records


def condense_train(base_url: str, workers: int) -> Dict[str, Any]:
    require_frozen()
    interactions = validate_train_preparation()
    runtime = validate_local_service(BACKBONE, base_url)
    if runtime["models"][0]["id"] != SERVED_MODEL:
        raise GateFailure("Unexpected 72B served model for Amendment 7")
    indices = sorted(interactions)
    prompts = load_author_prompts(GMEMORY_ROOT)
    metadata = {
        "task": "Amendment7_GMemory_train_condensation",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "served_model": SERVED_MODEL,
        "runtime": runtime,
        "rows": TRAIN_ROWS,
        "order": "ascending source_train_index",
        "temperature": MEMORY_CONTROL_TEMPERATURE,
        "max_tokens": MEMORY_CONTROL_MAX_TOKENS,
        "workers": workers,
        "seed_rule": "campaign seed plus source_train_index",
        "author_revision": GMEMORY_REVISION,
        "author_prompt_sha256": sha256_text(
            prompts["extract_true_traj_system_prompt"]
            + prompts["extract_true_traj_user_prompt"]
        ),
        "train_preparation_sha256": file_sha256(TRAIN_PREPARATION_SUMMARY_PATH),
        "train_interactions_sha256": file_sha256(TRAIN_INTERACTIONS_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
    }
    run_hash = fingerprint(metadata)
    run_path = TRAIN_CONDENSATIONS_PATH.with_suffix(".jsonl.run.json")
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure("Cannot resume train condensation: metadata differs")
    atomic_json(run_path, metadata)
    records = load_jsonl(TRAIN_CONDENSATIONS_PATH)
    if any(
        index not in interactions or row.get("run_fingerprint") != run_hash
        for index, row in records.items()
    ):
        raise GateFailure("Invalid train-condensation checkpoint")
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)

    def generate(index: int) -> tuple[int, Dict[str, Any]]:
        messages = interactions[index]["extraction_messages"]
        if (
            sha256_text(canonical_json(messages))
            != interactions[index]["extraction_prompt_sha256"]
        ):
            raise GateFailure(f"Extraction prompt changed at train:{index}")
        completion = client.chat.completions.create(
            model=SERVED_MODEL,
            messages=messages,
            max_tokens=MEMORY_CONTROL_MAX_TOKENS,
            temperature=MEMORY_CONTROL_TEMPERATURE,
            seed=SEED + index,
        )
        response = completion.choices[0].message.content or ""
        if not response.strip():
            raise GateFailure(f"Empty train condensation at train:{index}")
        usage = completion.usage
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "category": "train_trajectory_condensation",
            "prompt_sha256": interactions[index]["extraction_prompt_sha256"],
            "response": response,
            "response_sha256": sha256_text(response),
            "prompt_tokens": None if usage is None else usage.prompt_tokens,
            "completion_tokens": None if usage is None else usage.completion_tokens,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except GateFailure:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except Exception as error:
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "endpoint_error": repr(error),
                }
            records[index] = record
            write_jsonl_snapshot(TRAIN_CONDENSATIONS_PATH, records, indices)
    errors = [row for row in records.values() if row.get("endpoint_error")]
    if set(records) != set(indices) or errors:
        raise GateFailure(
            f"Train condensation incomplete with {len(errors)} endpoint errors"
        )
    summary = {
        "task": "Amendment7_GMemory_train_condensation",
        "status": "PASS",
        "samples": len(records),
        "memory_control_calls": len(records),
        "call_category": "train_trajectory_condensation",
        "temperature": MEMORY_CONTROL_TEMPERATURE,
        "max_tokens": MEMORY_CONTROL_MAX_TOKENS,
        "run_fingerprint": run_hash,
        "runtime": runtime,
        "prompt_tokens": int(
            sum(int(row["prompt_tokens"]) for row in records.values())
        ),
        "completion_tokens": int(
            sum(int(row["completion_tokens"]) for row in records.values())
        ),
        "results": str(TRAIN_CONDENSATIONS_PATH),
        "results_sha256": file_sha256(TRAIN_CONDENSATIONS_PATH),
    }
    atomic_json(TRAIN_CONDENSATIONS_SUMMARY_PATH, summary)
    return summary


def load_append_only_calls(
    path: Path = TRAIN_HIERARCHY_CALLS_PATH,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    calls: List[Dict[str, Any]] = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                call = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateFailure(
                    f"Malformed hierarchy call ledger line {line_number}"
                ) from error
            if call.get("sequence") != len(calls):
                raise GateFailure("Hierarchy call ledger is not contiguous")
            calls.append(call)
    return calls


def append_call(call: Dict[str, Any], path: Path = TRAIN_HIERARCHY_CALLS_PATH) -> None:
    with path.open("a") as destination:
        destination.write(json.dumps(call, ensure_ascii=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def load_hierarchy_attempts() -> List[Dict[str, Any]]:
    if not TRAIN_HIERARCHY_ATTEMPTS_PATH.is_file():
        return []
    attempts: List[Dict[str, Any]] = []
    with TRAIN_HIERARCHY_ATTEMPTS_PATH.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                attempt = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateFailure(
                    f"Malformed hierarchy attempt ledger line {line_number}"
                ) from error
            if attempt.get("attempt_sequence") != len(attempts):
                raise GateFailure("Hierarchy attempt ledger is not contiguous")
            attempts.append(attempt)
    return attempts


def append_hierarchy_attempt(attempt: Dict[str, Any]) -> None:
    with TRAIN_HIERARCHY_ATTEMPTS_PATH.open("a") as destination:
        destination.write(json.dumps(attempt, ensure_ascii=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def hierarchy_success_attempt(
    call: Dict[str, Any], attempt_sequence: int, source: str
) -> Dict[str, Any]:
    return {
        "attempt_sequence": attempt_sequence,
        "logical_sequence": call["sequence"],
        "attempt_number": 1,
        "run_fingerprint": call["run_fingerprint"],
        "category": call["category"],
        "prompt_sha256": call["prompt_sha256"],
        "outcome": "success",
        "response_sha256": call["response_sha256"],
        "prompt_tokens": call.get("prompt_tokens"),
        "completion_tokens": call.get("completion_tokens"),
        "source": source,
    }


def build_train_hierarchy(base_url: str) -> Dict[str, Any]:
    require_frozen()
    interactions = validate_train_preparation()
    if not TRAIN_CONDENSATIONS_SUMMARY_PATH.is_file():
        raise GateFailure("Complete Amendment 7 train condensation first")
    condensation_summary = json.loads(TRAIN_CONDENSATIONS_SUMMARY_PATH.read_text())
    if (
        condensation_summary.get("status") != "PASS"
        or condensation_summary.get("samples") != TRAIN_ROWS
        or condensation_summary.get("results_sha256")
        != file_sha256(TRAIN_CONDENSATIONS_PATH)
    ):
        raise GateFailure("Amendment 7 train condensation certificate is stale")
    condensations = load_jsonl(TRAIN_CONDENSATIONS_PATH)
    if set(condensations) != set(interactions):
        raise GateFailure("Train condensation coverage is incomplete")
    runtime = validate_local_service(BACKBONE, base_url)
    if runtime["models"][0]["id"] != SERVED_MODEL:
        raise GateFailure("Unexpected 72B served model for Amendment 7")
    embedding_cache = np.load(TRAIN_EMBEDDINGS_PATH, allow_pickle=False)
    source_ids = embedding_cache["source_ids"].astype(np.int64).tolist()
    embeddings = np.asarray(embedding_cache["embeddings"], dtype=np.float32)
    if source_ids != sorted(interactions) or embeddings.shape != (TRAIN_ROWS, 384):
        raise GateFailure("Train embedding cache does not match interactions")
    prompts = load_author_prompts(GMEMORY_ROOT)
    metadata = {
        "task": "Amendment7_GMemory_train_hierarchy",
        "served_model": SERVED_MODEL,
        "model_revision": MODEL_REVISION,
        "temperature": MEMORY_CONTROL_TEMPERATURE,
        "max_tokens": MEMORY_CONTROL_MAX_TOKENS,
        "seed": SEED,
        "rows": TRAIN_ROWS,
        "insight_events": TRAIN_INSIGHT_EVENTS,
        "insight_calls": TRAIN_INSIGHT_CALLS,
        "merge_events": TRAIN_MERGE_EVENTS,
        "author_revision": GMEMORY_REVISION,
        "author_prompt_hashes": {
            name: sha256_text(prompts[name])
            for name in (
                "critique_success_rules_system_prompt",
                "critique_success_rules_user_prompt",
                "merge_rules_system_prompt",
                "merge_rules_user_prompt",
            )
        },
        "condensations_sha256": file_sha256(TRAIN_CONDENSATIONS_PATH),
        "embeddings_sha256": file_sha256(TRAIN_EMBEDDINGS_PATH),
        "interactions_sha256": file_sha256(TRAIN_INTERACTIONS_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
    }
    if (
        TRAIN_HIERARCHY_RUN_PATH.is_file()
        and json.loads(TRAIN_HIERARCHY_RUN_PATH.read_text()) != metadata
    ):
        raise GateFailure("Cannot resume train hierarchy: metadata differs")
    atomic_json(TRAIN_HIERARCHY_RUN_PATH, metadata)
    run_hash = fingerprint(metadata)
    calls = load_append_only_calls()
    attempts = load_hierarchy_attempts()
    if not attempts and calls:
        for call in calls:
            attempt = hierarchy_success_attempt(
                call, len(attempts), "migrated_success_ledger"
            )
            append_hierarchy_attempt(attempt)
            attempts.append(attempt)
    interruption_recovery = (
        json.loads(TRAIN_HIERARCHY_INTERRUPTION_RECOVERY_PATH.read_text())
        if TRAIN_HIERARCHY_INTERRUPTION_RECOVERY_PATH.is_file()
        else None
    )
    if interruption_recovery is not None and (
        interruption_recovery.get("status") != "AUTHORIZED_BEFORE_RETRY"
        or interruption_recovery.get("observed_outcome") != "indeterminate_interrupted"
        or interruption_recovery.get("authorized_retry_attempts") != 1
        or interruption_recovery.get("filed_before_retry") is not True
    ):
        raise GateFailure("Train hierarchy interruption recovery is invalid")
    if TRAIN_HIERARCHY_PENDING_PATH.is_file():
        pending = json.loads(TRAIN_HIERARCHY_PENDING_PATH.read_text())
        if pending.get("status") != "complete":
            identity = pending.get("identity")
            if (
                interruption_recovery is None
                or pending.get("status") != "started"
                or not isinstance(identity, dict)
                or identity.get("attempt_sequence") != len(attempts)
                or identity.get("logical_sequence")
                != interruption_recovery.get("logical_sequence")
                or identity.get("attempt_sequence")
                != interruption_recovery.get("interrupted_attempt_sequence")
                or identity.get("prompt_sha256")
                != interruption_recovery.get("prompt_sha256")
                or len(calls)
                != interruption_recovery.get("successful_calls_before_interruption")
            ):
                raise GateFailure("Unresolved train hierarchy remote-call outcome")
            attempt = {
                **identity,
                "outcome": "indeterminate_interrupted",
                "response_sha256": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "source": "interrupted_before_response_commit",
                "recovery_authorization_sha256": file_sha256(
                    TRAIN_HIERARCHY_INTERRUPTION_RECOVERY_PATH
                ),
            }
            append_hierarchy_attempt(attempt)
            attempts.append(attempt)
            TRAIN_HIERARCHY_PENDING_PATH.unlink()
            pending = None
        if pending is None:
            pass
        else:
            attempt = pending.get("attempt")
            if not isinstance(attempt, dict):
                raise GateFailure(
                    "Completed hierarchy pending journal lacks an attempt"
                )
            attempt_sequence = attempt.get("attempt_sequence")
            if attempt_sequence == len(attempts):
                append_hierarchy_attempt(attempt)
                attempts.append(attempt)
            elif (
                not isinstance(attempt_sequence, int)
                or attempt_sequence >= len(attempts)
                or attempts[attempt_sequence] != attempt
            ):
                raise GateFailure("Hierarchy pending attempt differs from its ledger")
            call = pending.get("call")
            if call is not None:
                if not isinstance(call, dict):
                    raise GateFailure("Completed hierarchy pending call is invalid")
                sequence = call.get("sequence")
                if sequence == len(calls):
                    append_call(call)
                    calls.append(call)
                elif (
                    not isinstance(sequence, int)
                    or sequence >= len(calls)
                    or calls[sequence] != call
                ):
                    raise GateFailure("Hierarchy pending call differs from its ledger")
            TRAIN_HIERARCHY_PENDING_PATH.unlink()
    recovery = (
        json.loads(TRAIN_HIERARCHY_RECOVERY_PATH.read_text())
        if TRAIN_HIERARCHY_RECOVERY_PATH.is_file()
        else None
    )
    if recovery is not None and (
        recovery.get("status") != "AUTHORIZED_BEFORE_RETRY"
        or recovery.get("observed_outcome") != "empty_response"
        or recovery.get("authorized_retry_attempts") != 1
        or recovery.get("filed_before_retry") is not True
    ):
        raise GateFailure("Train hierarchy recovery authorization is invalid")
    recovery_paths = {
        recovery_record["logical_sequence"]: recovery_path
        for recovery_record, recovery_path in (
            (recovery, TRAIN_HIERARCHY_RECOVERY_PATH),
            (
                interruption_recovery,
                TRAIN_HIERARCHY_INTERRUPTION_RECOVERY_PATH,
            ),
        )
        if recovery_record is not None
    }
    recoveries = {
        recovery_record["logical_sequence"]: recovery_record
        for recovery_record in (recovery, interruption_recovery)
        if recovery_record is not None
    }
    resolution = (
        json.loads(TRAIN_HIERARCHY_RESOLUTION_PATH.read_text())
        if TRAIN_HIERARCHY_RESOLUTION_PATH.is_file()
        else None
    )
    if resolution is not None and (
        resolution.get("status") != "FILED_BEFORE_RESOLUTION"
        or resolution.get("no_additional_remote_call") is not True
        or resolution.get("accepted_outcome") != "empty_response"
    ):
        raise GateFailure("Train hierarchy zero-output resolution is invalid")
    accepted_attempt_sequence = (
        resolution.get("accepted_attempt_sequence") if resolution is not None else None
    )
    accepted_attempts = [
        attempt
        for attempt in attempts
        if (
            attempt.get("outcome") == "success"
            or attempt.get("attempt_sequence") == accepted_attempt_sequence
        )
        and isinstance(attempt.get("logical_sequence"), int)
        and attempt["logical_sequence"] < len(calls)
    ]
    if len(accepted_attempts) != len(calls) or any(
        attempt.get("logical_sequence") != call.get("sequence")
        or attempt.get("response_sha256") != call.get("response_sha256")
        for attempt, call in zip(accepted_attempts, calls)
    ):
        raise GateFailure("Hierarchy accepted attempts differ from the call ledger")
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)
    state = GMemoryState(query_edge_threshold=0.7, hop=GRAPH_HOP)
    rng = random.Random(SEED)
    cursor = 0
    insight_events = 0
    merge_events = 0

    def complete_call(
        expected: Dict[str, Any],
        messages: List[Dict[str, str]],
        apply_response: Callable[[str], None],
    ) -> None:
        nonlocal cursor
        prompt_sha256 = sha256_text(canonical_json(messages))
        identity = {
            "sequence": cursor,
            "run_fingerprint": run_hash,
            **expected,
            "prompt_sha256": prompt_sha256,
        }
        hierarchy_before = state.hierarchy_sha256()
        zero_output_allowed = (
            expected.get("category") == "train_merge"
            and expected.get("limited_number") == 0
        )
        if cursor < len(calls):
            call = calls[cursor]
            if any(call.get(key) != value for key, value in identity.items()):
                raise GateFailure(f"Hierarchy replay differs at call {cursor}")
            if call.get("hierarchy_before_sha256") != hierarchy_before:
                raise GateFailure(f"Hierarchy pre-call hash differs at call {cursor}")
            response = str(call.get("response", ""))
            if not response and not zero_output_allowed:
                raise GateFailure(f"Empty replayed hierarchy response at call {cursor}")
            apply_response(response)
            if call.get("hierarchy_after_sha256") != state.hierarchy_sha256():
                raise GateFailure(f"Hierarchy post-call hash differs at call {cursor}")
        else:
            if resolution is not None and resolution.get("logical_sequence") == cursor:
                if not zero_output_allowed:
                    raise GateFailure(
                        "Zero-output resolution no longer matches its call"
                    )
                resolved_sequence = resolution.get("accepted_attempt_sequence")
                if not isinstance(resolved_sequence, int) or resolved_sequence >= len(
                    attempts
                ):
                    raise GateFailure("Resolved hierarchy attempt is missing")
                resolved_attempt = attempts[resolved_sequence]
                if (
                    resolved_attempt.get("logical_sequence") != cursor
                    or resolved_attempt.get("prompt_sha256") != prompt_sha256
                    or resolved_attempt.get("outcome") != "empty_response"
                ):
                    raise GateFailure("Resolved hierarchy attempt identity changed")
                response = ""
                apply_response(response)
                call = {
                    **identity,
                    "hierarchy_before_sha256": hierarchy_before,
                    "response": response,
                    "response_sha256": resolved_attempt["response_sha256"],
                    "hierarchy_after_sha256": state.hierarchy_sha256(),
                    "prompt_tokens": resolved_attempt.get("prompt_tokens"),
                    "completion_tokens": resolved_attempt.get("completion_tokens"),
                    "accepted_attempt_sequence": resolved_sequence,
                }
                append_call(call)
                calls.append(call)
                cursor += 1
                return
            failed_attempts = [
                attempt
                for attempt in attempts
                if attempt.get("logical_sequence") == cursor
                and attempt.get("outcome") != "success"
            ]
            active_recovery = recoveries.get(cursor)
            if (
                active_recovery is not None
                and active_recovery.get("observed_outcome") == "empty_response"
                and not failed_attempts
            ):
                if (
                    active_recovery.get("successful_calls_before_failure") != len(calls)
                    or len(calls) != cursor
                ):
                    raise GateFailure("Hierarchy recovery boundary changed")
                failed_attempt = {
                    "attempt_sequence": len(attempts),
                    "logical_sequence": cursor,
                    "attempt_number": 1,
                    "run_fingerprint": run_hash,
                    "category": expected["category"],
                    "prompt_sha256": prompt_sha256,
                    "outcome": "empty_response",
                    "response_sha256": sha256_text(""),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "source": "observed_gate_failure_before_attempt_ledger",
                    "recovery_authorization_sha256": file_sha256(
                        recovery_paths[cursor]
                    ),
                }
                append_hierarchy_attempt(failed_attempt)
                attempts.append(failed_attempt)
                failed_attempts.append(failed_attempt)
            if failed_attempts and (
                active_recovery is None
                or len(failed_attempts)
                > int(active_recovery.get("authorized_retry_attempts", 0))
            ):
                raise GateFailure(
                    f"No retry authorization remains for hierarchy call {cursor}"
                )
            attempt_identity = {
                "attempt_sequence": len(attempts),
                "logical_sequence": cursor,
                "attempt_number": 1
                + sum(
                    attempt.get("logical_sequence") == cursor for attempt in attempts
                ),
                "run_fingerprint": run_hash,
                "category": expected["category"],
                "prompt_sha256": prompt_sha256,
            }
            atomic_json(
                TRAIN_HIERARCHY_PENDING_PATH,
                {"status": "started", "identity": attempt_identity},
            )
            completion = client.chat.completions.create(
                model=SERVED_MODEL,
                messages=messages,
                max_tokens=MEMORY_CONTROL_MAX_TOKENS,
                temperature=MEMORY_CONTROL_TEMPERATURE,
                seed=SEED + cursor,
            )
            response = completion.choices[0].message.content or ""
            usage = completion.usage
            if not response.strip() and not zero_output_allowed:
                attempt = {
                    **attempt_identity,
                    "outcome": "empty_response",
                    "response_sha256": sha256_text(response),
                    "prompt_tokens": None if usage is None else usage.prompt_tokens,
                    "completion_tokens": (
                        None if usage is None else usage.completion_tokens
                    ),
                    "source": "live_remote_call",
                }
                atomic_json(
                    TRAIN_HIERARCHY_PENDING_PATH,
                    {"status": "complete", "attempt": attempt},
                )
                append_hierarchy_attempt(attempt)
                attempts.append(attempt)
                TRAIN_HIERARCHY_PENDING_PATH.unlink()
                raise GateFailure(f"Empty hierarchy response at call {cursor}")
            apply_response(response)
            call = {
                **identity,
                "hierarchy_before_sha256": hierarchy_before,
                "response": response,
                "response_sha256": sha256_text(response),
                "hierarchy_after_sha256": state.hierarchy_sha256(),
                "prompt_tokens": None if usage is None else usage.prompt_tokens,
                "completion_tokens": None if usage is None else usage.completion_tokens,
            }
            attempt = {
                **attempt_identity,
                "outcome": "success",
                "response_sha256": call["response_sha256"],
                "prompt_tokens": call["prompt_tokens"],
                "completion_tokens": call["completion_tokens"],
                "source": "live_remote_call",
            }
            atomic_json(
                TRAIN_HIERARCHY_PENDING_PATH,
                {"status": "complete", "attempt": attempt, "call": call},
            )
            append_hierarchy_attempt(attempt)
            attempts.append(attempt)
            append_call(call)
            calls.append(call)
            TRAIN_HIERARCHY_PENDING_PATH.unlink()
        cursor += 1

    for position, source_id in enumerate(source_ids):
        interaction = interactions[source_id]
        condensation = condensations[source_id]
        record = {
            "memory_id": interaction["memory_id"],
            "task_main": interaction["task_main"],
            "task_description": interaction["task_description"],
            "roles": interaction["roles"],
            "trajectory": interaction["trajectory"],
            "key_steps": condensation["response"],
            "label": True,
            "source_plan_sha256": interaction["source_plan_sha256"],
            "condensation_sha256": condensation["response_sha256"],
        }
        state.add_record(record, embeddings[position])
        memory_size = position + 1
        if (
            memory_size >= START_INSIGHTS_THRESHOLD
            and memory_size % ROUNDS_PER_INSIGHTS == 0
        ):
            insight_events += 1
            for point in range(INSIGHTS_POINT_NUM):
                anchor_index = rng.choice(range(memory_size))
                nearest = state.nearest_record_indices(
                    state.embeddings[anchor_index], count=3, label=True
                )
                selected_indices = nearest + [anchor_index]
                rng.shuffle(selected_indices)
                task_mains = [
                    str(state.records[index]["task_main"]) for index in selected_indices
                ]
                local_ids = state.related_insight_ids(
                    task_mains, threshold=len(task_mains) / 2
                )
                local_rules = [state.insights[index]["rule"] for index in local_ids]
                if not local_rules:
                    local_rules = [""]
                rule_text = "\n".join(
                    f"{index}. {rule}" for index, rule in enumerate(local_rules, 1)
                )
                history = "\n".join(
                    f"task{index}:\n"
                    + str(state.records[record_index]["task_description"])
                    + str(state.records[record_index]["key_steps"])
                    for index, record_index in enumerate(selected_indices)
                )
                user_prompt = prompts["critique_success_rules_user_prompt"].format(
                    success_history=history,
                    existing_rules=rule_text,
                )
                suffix = (
                    "Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule "
                    "unless the new rule is VERY insightful and different from "
                    "EXISTING RULES.\n"
                    if len(state.insights) > 10
                    else ""
                )
                messages = [
                    {
                        "role": "system",
                        "content": prompts["critique_success_rules_system_prompt"]
                        + suffix,
                    },
                    {"role": "user", "content": user_prompt},
                ]

                def apply_insight(
                    response: str,
                    ids: List[int] = local_ids,
                    relative_tasks: List[str] = task_mains,
                ) -> None:
                    update_rules(
                        state.insights,
                        relative_tasks,
                        parse_rule_operations(response),
                        local_insight_ids=ids,
                    )

                complete_call(
                    {
                        "category": "train_periodic_insight",
                        "event_memory_size": memory_size,
                        "point": point,
                        "anchor_memory_id": state.records[anchor_index]["memory_id"],
                        "selected_memory_ids": [
                            state.records[index]["memory_id"]
                            for index in selected_indices
                        ],
                        "local_insight_ids": local_ids,
                    },
                    messages,
                    apply_insight,
                )
        if memory_size % 20 == 0:
            merge_events += 1
            clusters = state.cluster_tasks()
            merged_by_cluster: Dict[int, List[str]] = {}
            for cluster_id, tasks in clusters.items():
                insight_ids = state.related_insight_ids(tasks)
                rules = [state.insights[index]["rule"] for index in insight_ids]
                merged_rules: List[str] = []
                for batch_start in range(0, len(rules), 10):
                    batch = rules[batch_start : batch_start + 10]
                    limited_number = (len(batch) // 3) // 3
                    messages = [
                        {
                            "role": "system",
                            "content": prompts["merge_rules_system_prompt"],
                        },
                        {
                            "role": "user",
                            "content": prompts["merge_rules_user_prompt"].format(
                                current_rules="\n".join(batch),
                                limited_number=limited_number,
                            ),
                        },
                    ]

                    def apply_merge(
                        response: str, destination: List[str] = merged_rules
                    ) -> None:
                        destination.extend(parse_numbered_list(response))

                    complete_call(
                        {
                            "category": "train_merge",
                            "event_memory_size": memory_size,
                            "cluster_id": cluster_id,
                            "batch_start": batch_start,
                            "cluster_tasks_sha256": sha256_text(canonical_json(tasks)),
                            "source_rules_sha256": sha256_text(canonical_json(batch)),
                            "limited_number": limited_number,
                        },
                        messages,
                        apply_merge,
                    )
                merged_by_cluster[cluster_id] = merged_rules
            state.insights.clear()
            for cluster_id, tasks in clusters.items():
                for rule in merged_by_cluster[cluster_id]:
                    state.insights.append(
                        {
                            "rule": rule,
                            "score": 2,
                            "positive_correlation_tasks": list(tasks),
                            "negative_correlation_tasks": [],
                        }
                    )
    if cursor != len(calls):
        raise GateFailure("Hierarchy call ledger has unexpected trailing calls")
    if insight_events != TRAIN_INSIGHT_EVENTS or merge_events != TRAIN_MERGE_EVENTS:
        raise GateFailure("Train hierarchy schedule count changed")
    hierarchy = {
        "task": "Amendment7_GMemory_frozen_train_hierarchy",
        "run_fingerprint": run_hash,
        "state": state.canonical_state(),
        "hierarchy_sha256": state.hierarchy_sha256(),
        "call_ledger_sha256": file_sha256(TRAIN_HIERARCHY_CALLS_PATH),
    }
    atomic_json(TRAIN_HIERARCHY_PATH, hierarchy)
    category_counts: Dict[str, int] = {}
    for call in calls:
        category = str(call["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    attempt_outcome_counts: Dict[str, int] = {}
    for attempt in attempts:
        outcome = str(attempt["outcome"])
        attempt_outcome_counts[outcome] = attempt_outcome_counts.get(outcome, 0) + 1
    accepted_zero_output_attempts = int(accepted_attempt_sequence is not None)
    unaccepted_attempts = len(attempts) - len(calls)
    summary = {
        "task": "Amendment7_GMemory_train_hierarchy",
        "status": "PASS",
        "training": False,
        "gradient_updates": False,
        "rows": len(state.records),
        "insights": len(state.insights),
        "insight_events": insight_events,
        "merge_events": merge_events,
        "memory_control_calls": len(attempts),
        "successful_memory_control_calls": len(calls),
        "accepted_zero_output_attempts": accepted_zero_output_attempts,
        "unaccepted_memory_control_attempts": unaccepted_attempts,
        "call_category_counts": category_counts,
        "attempt_outcome_counts": attempt_outcome_counts,
        "hierarchy_sha256": state.hierarchy_sha256(),
        "hierarchy": str(TRAIN_HIERARCHY_PATH),
        "hierarchy_file_sha256": file_sha256(TRAIN_HIERARCHY_PATH),
        "call_ledger": str(TRAIN_HIERARCHY_CALLS_PATH),
        "call_ledger_sha256": file_sha256(TRAIN_HIERARCHY_CALLS_PATH),
        "attempt_ledger": str(TRAIN_HIERARCHY_ATTEMPTS_PATH),
        "attempt_ledger_sha256": file_sha256(TRAIN_HIERARCHY_ATTEMPTS_PATH),
        "runtime": runtime,
    }
    atomic_json(TRAIN_HIERARCHY_SUMMARY_PATH, summary)
    return summary


def load_frozen_hierarchy() -> GMemoryState:
    if not TRAIN_HIERARCHY_PATH.is_file() or not TRAIN_HIERARCHY_SUMMARY_PATH.is_file():
        raise GateFailure("Build the Amendment 7 train hierarchy first")
    hierarchy = json.loads(TRAIN_HIERARCHY_PATH.read_text())
    summary = json.loads(TRAIN_HIERARCHY_SUMMARY_PATH.read_text())
    if (
        summary.get("status") != "PASS"
        or summary.get("rows") != TRAIN_ROWS
        or summary.get("hierarchy_file_sha256") != file_sha256(TRAIN_HIERARCHY_PATH)
        or hierarchy.get("hierarchy_sha256") != summary.get("hierarchy_sha256")
    ):
        raise GateFailure("Frozen train hierarchy certificate is stale")
    cache = np.load(TRAIN_EMBEDDINGS_PATH, allow_pickle=False)
    embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
    try:
        state = GMemoryState.from_canonical_state(hierarchy["state"], embeddings)
    except (KeyError, TypeError, ValueError) as error:
        raise GateFailure("Frozen train hierarchy restoration failed") from error
    if state.hierarchy_sha256() != hierarchy["hierarchy_sha256"]:
        raise GateFailure("Frozen train hierarchy hash changed after restoration")
    return state


def frozen_preflight(
    mode: str, split: str, base_url: str, workers: int
) -> Dict[str, Any]:
    if mode not in {"A", "B_shadow"}:
        raise GateFailure(f"Unknown frozen preflight mode: {mode}")
    require_frozen()
    if mode == "B_shadow":
        required_prior = (
            OUTPUT_DIR / "mode_A.id.summary.json"
            if split == "ood"
            else OUTPUT_DIR / "mode_B.ood.summary.json"
        )
        require_completed_run(required_prior, ID_ROWS if split == "ood" else OOD_ROWS)
    runtime = validate_local_service(BACKBONE, base_url)
    if runtime["models"][0]["id"] != SERVED_MODEL:
        raise GateFailure("Unexpected 72B served model for Amendment 7")
    state = load_frozen_hierarchy()
    hierarchy_sha256 = state.hierarchy_sha256()
    dataset, indices, _, source_path, source_summary_path = split_configuration(split)
    source_records, source_run_hash, _ = validate_reused_source(
        split, indices, source_path, source_summary_path
    )
    amendment6_preflight_path = (
        AMENDMENT6_DIR / f"qwen2_5_vl_72b.{split}.preflight.jsonl"
    )
    amendment6_preflight_summary_path = amendment6_preflight_path.with_suffix(
        ".summary.json"
    )
    if not amendment6_preflight_summary_path.is_file():
        raise GateFailure(f"Missing certified Amendment 6 {split} preflight")
    amendment6_summary = json.loads(amendment6_preflight_summary_path.read_text())
    amendment6_rows = load_jsonl(amendment6_preflight_path)
    if (
        amendment6_summary.get("status") != "PASS"
        or amendment6_summary.get("results_sha256")
        != file_sha256(amendment6_preflight_path)
        or set(amendment6_rows) != set(indices)
    ):
        raise GateFailure(f"Certified Amendment 6 {split} preflight changed")
    prompts = load_author_prompts(GMEMORY_ROOT)
    mode_label = "A" if mode == "A" else "B"
    artifact_prefix = "mode_A" if mode == "A" else "mode_B.shadow"
    metadata = {
        "task": f"Amendment7_GMemory_Mode_{mode_label}_preflight",
        "mode": mode_label,
        "preflight_kind": "frozen_primary" if mode == "A" else "frozen_shadow",
        "split": split,
        "indices": indices,
        "served_model": SERVED_MODEL,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "workers": workers,
        "successful_candidates": 2,
        "successful_topk": SUCCESSFUL_TOPK,
        "insights_topk": INSIGHTS_TOPK,
        "temperature": MEMORY_CONTROL_TEMPERATURE,
        "max_tokens": MEMORY_CONTROL_MAX_TOKENS,
        "hierarchy_sha256": hierarchy_sha256,
        "hierarchy_file_sha256": file_sha256(TRAIN_HIERARCHY_PATH),
        "author_revision": GMEMORY_REVISION,
        "rerank_prompt_sha256": sha256_text(
            prompts["generative_task_system_prompt"]
            + prompts["generative_task_user_prompt"]
        ),
        "amendment6_preflight_sha256": file_sha256(amendment6_preflight_path),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
    }
    run_hash = fingerprint(metadata)
    output = OUTPUT_DIR / f"{artifact_prefix}.{split}.preflight.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure(
            f"Cannot resume {artifact_prefix} {split} preflight: metadata differs"
        )
    atomic_json(run_path, metadata)
    records = load_jsonl(output)
    if any(
        index not in indices or record.get("run_fingerprint") != run_hash
        for index, record in records.items()
    ):
        raise GateFailure(f"Invalid {artifact_prefix} {split} preflight checkpoint")
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )
    query_embeddings = np.asarray(
        embedding_model.encode(
            [
                get_instruction(native(dataset.iloc[index].to_dict()))
                for index in pending
            ],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    query_by_index = {
        index: query_embeddings[position] for position, index in enumerate(pending)
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    bench = load_bench()
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)

    def preflight_row(index: int) -> tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        instruction = get_instruction(sample)
        query_embedding = query_by_index[index]
        candidate_indices = state.raw_success_candidates(query_embedding, count=2)
        if len(candidate_indices) != 2:
            raise GateFailure(
                f"{artifact_prefix} {split}:{index} did not retrieve two candidates"
            )
        rerank_calls = []
        for candidate_position, candidate_index in enumerate(candidate_indices):
            candidate = state.records[candidate_index]
            user_prompt = prompts["generative_task_user_prompt"].format(
                trajectory=(
                    str(candidate["task_description"])
                    + "\n"
                    + str(candidate["trajectory"])
                ),
                query_scenario=instruction,
            )
            messages = [
                {
                    "role": "system",
                    "content": prompts["generative_task_system_prompt"],
                },
                {"role": "user", "content": user_prompt},
            ]
            completion = client.chat.completions.create(
                model=SERVED_MODEL,
                messages=messages,
                max_tokens=MEMORY_CONTROL_MAX_TOKENS,
                temperature=MEMORY_CONTROL_TEMPERATURE,
                seed=SEED + index * 2 + candidate_position,
            )
            response = completion.choices[0].message.content or ""
            score = parse_relevance_score(response)
            usage = completion.usage
            rerank_calls.append(
                {
                    "candidate_position": candidate_position,
                    "memory_id": candidate["memory_id"],
                    "prompt_sha256": sha256_text(canonical_json(messages)),
                    "response": response,
                    "response_sha256": sha256_text(response),
                    "score": score,
                    "prompt_tokens": None if usage is None else usage.prompt_tokens,
                    "completion_tokens": None
                    if usage is None
                    else usage.completion_tokens,
                }
            )
        selected_position = max(
            range(len(rerank_calls)),
            key=lambda position: rerank_calls[position]["score"],
        )
        selected_index = candidate_indices[selected_position]
        insights = state.related_insights(query_embedding, INSIGHTS_TOPK)
        native_prompt = render_retrieval_prompt(
            [state.records[selected_index]], insights
        )
        zero_messages = bench.get_messages(sample)
        native_messages = add_memory_to_messages(
            bench.get_messages(sample), native_prompt
        )
        zero_tokens = token_count(base_url, SERVED_MODEL, zero_messages)
        target_tokens = int(amendment6_rows[index]["token_counts"]["segment"])
        native_tokens = token_count(base_url, SERVED_MODEL, native_messages)
        final_prompt = native_prompt
        final_tokens = native_tokens
        truncated = False
        if native_tokens > target_tokens:
            final_prompt, final_tokens = trim_prompt(
                sample, base_url, tokenizer, native_prompt, target_tokens
            )
            truncated = final_prompt != native_prompt
        final_messages = add_memory_to_messages(
            bench.get_messages(sample), final_prompt
        )
        if messages_sha256(final_messages) != messages_sha256(
            add_memory_to_messages(bench.get_messages(sample), final_prompt)
        ):
            raise GateFailure(
                f"{artifact_prefix} {split}:{index} prompt is nondeterministic"
            )
        if final_tokens + PLAN_MAX_TOKENS > int(runtime["models"][0]["max_model_len"]):
            raise GateFailure(f"{artifact_prefix} {split}:{index} context gate failed")
        if final_tokens > target_tokens * (1 + TOKEN_TOLERANCE):
            raise GateFailure(
                f"{artifact_prefix} {split}:{index} remains over token budget"
            )
        source_zero_tokens = int(
            source_records[index]["arms"]["zero_shot"]["prompt_tokens"]
        )
        multimodal_surcharge = source_zero_tokens - zero_tokens
        if multimodal_surcharge < 0:
            raise GateFailure(
                f"{artifact_prefix} {split}:{index} invalid multimodal surcharge"
            )
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "mode": mode_label,
            "preflight_kind": metadata["preflight_kind"],
            "split": split,
            "instruction": instruction,
            "candidate_memory_ids": [
                state.records[candidate_index]["memory_id"]
                for candidate_index in candidate_indices
            ],
            "rerank_calls": rerank_calls,
            "selected_memory_id": state.records[selected_index]["memory_id"],
            "insights": insights,
            "native_prompt": native_prompt,
            "native_prompt_sha256": sha256_text(native_prompt),
            "final_prompt": final_prompt,
            "prompt_sha256": messages_sha256(final_messages),
            "zero_shot_input_tokens": zero_tokens,
            "segment_target_input_tokens": target_tokens,
            "native_input_tokens": native_tokens,
            "final_input_tokens": final_tokens,
            "injected_tokens": final_tokens - zero_tokens,
            "under_budget_shortfall": max(0, target_tokens - final_tokens),
            "relative_shortfall": max(0, target_tokens - final_tokens) / target_tokens,
            "truncated": truncated,
            "multimodal_usage_surcharge": multimodal_surcharge,
            "hierarchy_before_sha256": hierarchy_sha256,
            "hierarchy_after_sha256": state.hierarchy_sha256(),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(preflight_row, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except GateFailure:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except Exception as error:
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "endpoint_error": repr(error),
                }
            records[index] = record
            write_jsonl_snapshot(output, records, indices)
    errors = [record for record in records.values() if record.get("endpoint_error")]
    if set(records) != set(indices) or errors:
        raise GateFailure(
            f"{artifact_prefix} {split} preflight incomplete with "
            f"{len(errors)} endpoint errors"
        )
    if any(
        record["hierarchy_before_sha256"] != hierarchy_sha256
        or record["hierarchy_after_sha256"] != hierarchy_sha256
        for record in records.values()
    ):
        raise GateFailure(f"{artifact_prefix} {split} mutated the frozen hierarchy")
    summary = {
        "task": f"Amendment7_GMemory_Mode_{mode_label}_preflight",
        "status": "PASS",
        "mode": mode_label,
        "preflight_kind": metadata["preflight_kind"],
        "split": split,
        "rows": len(records),
        "model_generation_calls": 0,
        "memory_control_calls": 2 * len(records),
        "hierarchy_sha256": hierarchy_sha256,
        "hierarchy_unchanged": True,
        "truncated_rows": sum(record["truncated"] for record in records.values()),
        "under_budget_rows": sum(
            record["under_budget_shortfall"] > 0 for record in records.values()
        ),
        "maximum_relative_shortfall": max(
            record["relative_shortfall"] for record in records.values()
        ),
        "maximum_final_input_tokens": max(
            record["final_input_tokens"] for record in records.values()
        ),
        "runtime": runtime,
        "results": str(output),
        "results_sha256": file_sha256(output),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def mode_a_preflight(split: str, base_url: str, workers: int) -> Dict[str, Any]:
    return frozen_preflight("A", split, base_url, workers)


def mode_b_shadow_preflight(split: str, base_url: str, workers: int) -> Dict[str, Any]:
    return frozen_preflight("B_shadow", split, base_url, workers)


def require_completed_run(summary_path: Path, expected_rows: int) -> Dict[str, Any]:
    result_path = Path(str(summary_path).replace(".summary.json", ".jsonl"))
    if not summary_path.is_file() or not result_path.is_file():
        raise GateFailure(f"Missing prior completed run: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("status") != "PASS"
        or summary.get("samples") != expected_rows
        or summary.get("results_sha256") != file_sha256(result_path)
    ):
        raise GateFailure(f"Invalid prior completed run: {summary_path}")
    return summary


def run_mode_a(split: str, base_url: str, workers: int) -> Dict[str, Any]:
    require_frozen()
    runtime = validate_local_service(BACKBONE, base_url)
    state = load_frozen_hierarchy()
    hierarchy_sha256 = state.hierarchy_sha256()
    if split == "id":
        prior = OUTPUT_DIR / "mode_A.ood.summary.json"
        require_completed_run(prior, OOD_ROWS)
    preflight_path = OUTPUT_DIR / f"mode_A.{split}.preflight.jsonl"
    preflight_summary_path = preflight_path.with_suffix(".summary.json")
    if not preflight_path.is_file() or not preflight_summary_path.is_file():
        raise GateFailure(f"Complete Mode A {split} preflight before generation")
    preflight_summary = json.loads(preflight_summary_path.read_text())
    preflight_rows = load_jsonl(preflight_path)
    dataset, indices, _, source_path, source_summary_path = split_configuration(split)
    source_records, source_run_hash, _ = validate_reused_source(
        split, indices, source_path, source_summary_path
    )
    if (
        preflight_summary.get("status") != "PASS"
        or preflight_summary.get("hierarchy_sha256") != hierarchy_sha256
        or preflight_summary.get("results_sha256") != file_sha256(preflight_path)
        or set(preflight_rows) != set(indices)
    ):
        raise GateFailure(f"Mode A {split} preflight certificate is stale")
    metadata = {
        "task": "Amendment7_GMemory_Mode_A",
        "mode": "A",
        "split": split,
        "indices": indices,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "served_model": SERVED_MODEL,
        "runtime": runtime,
        "temperature": 0,
        "max_tokens": PLAN_MAX_TOKENS,
        "workers": workers,
        "scorer_seed_rule": "original row index",
        "hierarchy_sha256": hierarchy_sha256,
        "preflight_sha256": file_sha256(preflight_path),
        "preflight_summary_sha256": file_sha256(preflight_summary_path),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
    }
    run_hash = fingerprint(metadata)
    output = OUTPUT_DIR / f"mode_A.{split}.jsonl"
    run_path = output.with_suffix(output.suffix + ".run.json")
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure(f"Cannot resume Mode A {split}: metadata differs")
    atomic_json(run_path, metadata)
    records = load_jsonl(output)
    for index, record in records.items():
        if (
            index not in indices
            or record.get("run_fingerprint") != run_hash
            or sha256_text(str(record.get("response", "")))
            != record.get("response_sha256")
        ):
            raise GateFailure(f"Invalid Mode A {split}:{index} generation checkpoint")
    pending = [
        index
        for index in indices
        if index not in records or records[index].get("endpoint_error")
    ]
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)

    def generate(index: int) -> tuple[int, Dict[str, Any]]:
        sample = native(dataset.iloc[index].to_dict())
        row = preflight_rows[index]
        if (
            row["hierarchy_before_sha256"] != hierarchy_sha256
            or row["hierarchy_after_sha256"] != hierarchy_sha256
        ):
            raise GateFailure(f"Mode A {split}:{index} hierarchy gate changed")
        messages = add_memory_to_messages(
            bench.get_messages(sample), row["final_prompt"]
        )
        if messages_sha256(messages) != row["prompt_sha256"]:
            raise GateFailure(f"Mode A {split}:{index} prompt hash changed")
        completion = client.chat.completions.create(
            model=SERVED_MODEL,
            messages=messages,
            max_tokens=PLAN_MAX_TOKENS,
            temperature=0,
        )
        usage = completion.usage
        prompt_tokens = None if usage is None else usage.prompt_tokens
        expected_prompt_tokens = int(row["final_input_tokens"]) + int(
            row["multimodal_usage_surcharge"]
        )
        if prompt_tokens != expected_prompt_tokens:
            raise GateFailure(
                f"Mode A {split}:{index} prompt tokens changed: "
                f"expected={expected_prompt_tokens}, observed={prompt_tokens}"
            )
        response = completion.choices[0].message.content or ""
        metrics = bench.score_response(
            scorer,
            2,
            response,
            bench.get_ground_truth(sample),
            index,
        )
        return index, {
            "index": index,
            "run_fingerprint": run_hash,
            "mode": "A",
            "split": split,
            "prompt_sha256": row["prompt_sha256"],
            "hierarchy_before_sha256": hierarchy_sha256,
            "hierarchy_after_sha256": state.hierarchy_sha256(),
            "selected_memory_id": row["selected_memory_id"],
            "candidate_memory_ids": row["candidate_memory_ids"],
            "insights": row["insights"],
            "response": response,
            "response_sha256": sha256_text(response),
            "prompt_tokens": prompt_tokens,
            "tokenizer_input_tokens": row["final_input_tokens"],
            "multimodal_usage_surcharge": row["multimodal_usage_surcharge"],
            "injected_tokens": row["injected_tokens"],
            "completion_tokens": None if usage is None else usage.completion_tokens,
            **metrics,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate, index): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                _, record = future.result()
            except GateFailure:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except Exception as error:
                record = {
                    "index": index,
                    "run_fingerprint": run_hash,
                    "endpoint_error": repr(error),
                }
            records[index] = record
            write_jsonl_snapshot(output, records, indices)
    errors = [record for record in records.values() if record.get("endpoint_error")]
    if set(records) != set(indices) or errors:
        raise GateFailure(
            f"Mode A {split} generation incomplete with {len(errors)} endpoint errors"
        )
    if state.hierarchy_sha256() != hierarchy_sha256 or any(
        record["hierarchy_before_sha256"] != hierarchy_sha256
        or record["hierarchy_after_sha256"] != hierarchy_sha256
        for record in records.values()
    ):
        raise GateFailure(f"Mode A {split} mutated the frozen hierarchy")
    successes = sum(record["task_score"] == 1 for record in records.values())
    formats = sum(record["format_score"] == 1 for record in records.values())
    summary = {
        "task": "Amendment7_GMemory_Mode_A",
        "status": "PASS",
        "mode": "A",
        "split": split,
        "samples": len(records),
        "successes": successes,
        "accuracy": successes / len(records),
        "format_successes": formats,
        "format_compliance": formats / len(records),
        "plan_generation_calls": len(records),
        "memory_control_calls": 0,
        "hierarchy_sha256": hierarchy_sha256,
        "hierarchy_unchanged": True,
        "prompt_tokens": sum(record["prompt_tokens"] for record in records.values()),
        "completion_tokens": sum(
            record["completion_tokens"] for record in records.values()
        ),
        "runtime": runtime,
        "results": str(output),
        "results_sha256": file_sha256(output),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def run_mode_b(split: str, base_url: str) -> Dict[str, Any]:
    require_frozen()
    required_prior = (
        OUTPUT_DIR / "mode_A.id.summary.json"
        if split == "ood"
        else OUTPUT_DIR / "mode_B.ood.summary.json"
    )
    require_completed_run(required_prior, ID_ROWS if split == "ood" else OOD_ROWS)
    runtime = validate_local_service(BACKBONE, base_url)
    if runtime["models"][0]["id"] != SERVED_MODEL:
        raise GateFailure("Unexpected 72B served model for Amendment 7")
    state = load_frozen_hierarchy()
    frozen_hierarchy_sha256 = state.hierarchy_sha256()
    shadow_path = OUTPUT_DIR / f"mode_B.shadow.{split}.preflight.jsonl"
    shadow_summary_path = shadow_path.with_suffix(".summary.json")
    if not shadow_path.is_file() or not shadow_summary_path.is_file():
        raise GateFailure(f"Complete Mode B {split} shadow preflight first")
    shadow_summary = json.loads(shadow_summary_path.read_text())
    if (
        shadow_summary.get("status") != "PASS"
        or shadow_summary.get("hierarchy_sha256") != frozen_hierarchy_sha256
        or shadow_summary.get("results_sha256") != file_sha256(shadow_path)
    ):
        raise GateFailure(f"Mode B {split} shadow preflight certificate is stale")
    dataset, indices, _, source_path, source_summary_path = split_configuration(split)
    source_records, source_run_hash, _ = validate_reused_source(
        split, indices, source_path, source_summary_path
    )
    amendment6_preflight_path = (
        AMENDMENT6_DIR / f"qwen2_5_vl_72b.{split}.preflight.jsonl"
    )
    amendment6_preflight_summary_path = amendment6_preflight_path.with_suffix(
        ".summary.json"
    )
    if not amendment6_preflight_summary_path.is_file():
        raise GateFailure(f"Missing certified Amendment 6 {split} preflight")
    amendment6_summary = json.loads(amendment6_preflight_summary_path.read_text())
    amendment6_rows = load_jsonl(amendment6_preflight_path)
    if (
        amendment6_summary.get("status") != "PASS"
        or amendment6_summary.get("results_sha256")
        != file_sha256(amendment6_preflight_path)
        or set(amendment6_rows) != set(indices)
    ):
        raise GateFailure(f"Certified Amendment 6 {split} preflight changed")
    prompts = load_author_prompts(GMEMORY_ROOT)
    output = OUTPUT_DIR / f"mode_B.{split}.jsonl"
    calls_path = OUTPUT_DIR / f"mode_B.{split}.calls.jsonl"
    pending_call_path = calls_path.with_suffix(calls_path.suffix + ".pending.json")
    run_path = output.with_suffix(output.suffix + ".run.json")
    metadata = {
        "task": "Amendment7_GMemory_Mode_B",
        "mode": "B",
        "split": split,
        "indices": indices,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "served_model": SERVED_MODEL,
        "runtime": runtime,
        "workers": 1,
        "order": "ascending dataset index",
        "temperature": 0,
        "max_tokens": PLAN_MAX_TOKENS,
        "memory_control_temperature": MEMORY_CONTROL_TEMPERATURE,
        "memory_control_max_tokens": MEMORY_CONTROL_MAX_TOKENS,
        "periodic_insight_policy": "one frozen all-success critique per point",
        "frozen_hierarchy_sha256": frozen_hierarchy_sha256,
        "hierarchy_file_sha256": file_sha256(TRAIN_HIERARCHY_PATH),
        "shadow_preflight_sha256": file_sha256(shadow_path),
        "shadow_preflight_summary_sha256": file_sha256(shadow_summary_path),
        "amendment6_preflight_sha256": file_sha256(amendment6_preflight_path),
        "source_run_fingerprint": source_run_hash,
        "source_results_sha256": file_sha256(source_path),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "seed": SEED,
    }
    if run_path.is_file() and json.loads(run_path.read_text()) != metadata:
        raise GateFailure(f"Cannot resume Mode B {split}: metadata differs")
    atomic_json(run_path, metadata)
    run_hash = fingerprint(metadata)
    calls = load_append_only_calls(calls_path)
    records = load_jsonl(output)
    if any(
        index not in indices or record.get("run_fingerprint") != run_hash
        for index, record in records.items()
    ):
        raise GateFailure(f"Invalid Mode B {split} row checkpoint")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )
    instructions = [
        get_instruction(native(dataset.iloc[index].to_dict())) for index in indices
    ]
    query_embeddings = np.asarray(
        embedding_model.encode(
            instructions,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if query_embeddings.shape != (len(indices), 384):
        raise GateFailure(f"Unexpected Mode B {split} embedding shape")
    query_by_index = {
        index: query_embeddings[position] for position, index in enumerate(indices)
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    bench = load_bench()
    scorer = bench.load_official_scorer(2, BENCHMARK_ROOT)
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=5, timeout=3600)
    rng = random.Random(SEED)
    cursor = 0
    insight_events = 0
    merge_events = 0

    def complete_call(
        expected: Dict[str, Any],
        messages: List[Dict[str, str]],
        apply_response: Callable[[str], None],
        max_tokens: int = MEMORY_CONTROL_MAX_TOKENS,
        temperature: float = MEMORY_CONTROL_TEMPERATURE,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        nonlocal cursor
        identity = {
            "sequence": cursor,
            "run_fingerprint": run_hash,
            **expected,
            "prompt_sha256": sha256_text(canonical_json(messages)),
        }
        hierarchy_before = state.hierarchy_sha256()
        if cursor < len(calls):
            call = calls[cursor]
            if any(call.get(key) != value for key, value in identity.items()):
                raise GateFailure(f"Mode B {split} call replay differs at {cursor}")
            if call.get("hierarchy_before_sha256") != hierarchy_before:
                raise GateFailure(
                    f"Mode B {split} pre-call hierarchy differs at {cursor}"
                )
            response = str(call.get("response", ""))
            if not response:
                raise GateFailure(f"Empty Mode B replay response at call {cursor}")
            if call.get("response_sha256") != sha256_text(response):
                raise GateFailure(f"Mode B response digest differs at call {cursor}")
            apply_response(response)
            if call.get("hierarchy_after_sha256") != state.hierarchy_sha256():
                raise GateFailure(
                    f"Mode B {split} post-call hierarchy differs at {cursor}"
                )
            if pending_call_path.is_file():
                pending = json.loads(pending_call_path.read_text())
                pending_identity = pending.get("identity", {})
                if pending_identity.get("sequence") == cursor:
                    if (
                        pending.get("status") != "complete"
                        or pending.get("call") != call
                    ):
                        raise GateFailure(
                            f"Mode B {split} has an ambiguous pending call at {cursor}"
                        )
                    pending_call_path.unlink()
        else:
            call = None
            if pending_call_path.is_file():
                pending = json.loads(pending_call_path.read_text())
                if pending.get("identity") != identity:
                    raise GateFailure(
                        f"Mode B {split} pending-call identity differs at {cursor}"
                    )
                if pending.get("status") != "complete":
                    raise GateFailure(
                        f"Mode B {split} call {cursor} has unknown remote outcome; "
                        "refusing to duplicate it"
                    )
                call = pending.get("call")
                if not isinstance(call, dict):
                    raise GateFailure(
                        f"Mode B {split} pending call {cursor} is invalid"
                    )
                response = str(call.get("response", ""))
                if call.get("response_sha256") != sha256_text(response):
                    raise GateFailure(
                        f"Mode B pending response digest differs at call {cursor}"
                    )
                apply_response(response)
                if call.get("hierarchy_after_sha256") != state.hierarchy_sha256():
                    raise GateFailure(
                        f"Mode B pending hierarchy differs at call {cursor}"
                    )
            else:
                atomic_json(
                    pending_call_path,
                    {
                        "status": "requesting",
                        "identity": identity,
                        "hierarchy_before_sha256": hierarchy_before,
                    },
                )
                request = {
                    "model": SERVED_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if seed is not None:
                    request["seed"] = seed
                completion = client.chat.completions.create(**request)
                response = completion.choices[0].message.content or ""
                if not response.strip():
                    raise GateFailure(f"Empty Mode B response at call {cursor}")
                apply_response(response)
                usage = completion.usage
                call = {
                    **identity,
                    "hierarchy_before_sha256": hierarchy_before,
                    "response": response,
                    "response_sha256": sha256_text(response),
                    "hierarchy_after_sha256": state.hierarchy_sha256(),
                    "prompt_tokens": None if usage is None else usage.prompt_tokens,
                    "completion_tokens": None
                    if usage is None
                    else usage.completion_tokens,
                }
                atomic_json(
                    pending_call_path,
                    {"status": "complete", "identity": identity, "call": call},
                )
            append_call(call, calls_path)
            calls.append(call)
            pending_call_path.unlink()
        cursor += 1
        return call

    for index in indices:
        row_call_start = cursor
        hierarchy_before = state.hierarchy_sha256()
        sample = native(dataset.iloc[index].to_dict())
        instruction, robots, available_actions = get_prompt_context(sample)
        query_embedding = query_by_index[index]
        candidate_indices = state.raw_success_candidates(query_embedding, count=2)
        if len(candidate_indices) != 2:
            raise GateFailure(f"Mode B {split}:{index} did not retrieve two candidates")
        rerank_calls = []
        for candidate_position, candidate_index in enumerate(candidate_indices):
            candidate = state.records[candidate_index]
            user_prompt = prompts["generative_task_user_prompt"].format(
                trajectory=(
                    str(candidate["task_description"])
                    + "\n"
                    + str(candidate["trajectory"])
                ),
                query_scenario=instruction,
            )
            messages = [
                {
                    "role": "system",
                    "content": prompts["generative_task_system_prompt"],
                },
                {"role": "user", "content": user_prompt},
            ]
            score_holder: Dict[str, Any] = {}

            def apply_rerank(
                response: str, holder: Dict[str, Any] = score_holder
            ) -> None:
                holder["score"] = parse_relevance_score(response)

            call = complete_call(
                {
                    "category": "mode_B_live_rerank",
                    "row_index": index,
                    "candidate_position": candidate_position,
                    "memory_id": candidate["memory_id"],
                },
                messages,
                apply_rerank,
                seed=SEED + index * 2 + candidate_position,
            )
            rerank_calls.append(
                {
                    "candidate_position": candidate_position,
                    "memory_id": candidate["memory_id"],
                    "call_sequence": call["sequence"],
                    "response_sha256": call["response_sha256"],
                    "score": score_holder["score"],
                }
            )
        selected_position = max(
            range(len(rerank_calls)),
            key=lambda position: rerank_calls[position]["score"],
        )
        selected_index = candidate_indices[selected_position]
        retrieved_insights = state.related_insights(query_embedding, INSIGHTS_TOPK)
        native_prompt = render_retrieval_prompt(
            [state.records[selected_index]], retrieved_insights
        )
        zero_messages = bench.get_messages(sample)
        native_messages = add_memory_to_messages(zero_messages, native_prompt)
        zero_tokens = token_count(base_url, SERVED_MODEL, zero_messages)
        target_tokens = int(amendment6_rows[index]["token_counts"]["segment"])
        native_tokens = token_count(base_url, SERVED_MODEL, native_messages)
        final_prompt = native_prompt
        final_tokens = native_tokens
        if native_tokens > target_tokens:
            final_prompt, final_tokens = trim_prompt(
                sample, base_url, tokenizer, native_prompt, target_tokens
            )
        final_messages = add_memory_to_messages(zero_messages, final_prompt)
        if final_tokens + PLAN_MAX_TOKENS > int(runtime["models"][0]["max_model_len"]):
            raise GateFailure(f"Mode B {split}:{index} context gate failed")
        if final_tokens > target_tokens * (1 + TOKEN_TOLERANCE):
            raise GateFailure(f"Mode B {split}:{index} remains over token budget")
        source_zero_tokens = int(
            source_records[index]["arms"]["zero_shot"]["prompt_tokens"]
        )
        multimodal_surcharge = source_zero_tokens - zero_tokens
        if multimodal_surcharge < 0:
            raise GateFailure(f"Mode B {split}:{index} invalid multimodal surcharge")
        plan_holder: Dict[str, str] = {}

        def apply_plan(response: str, holder: Dict[str, str] = plan_holder) -> None:
            holder["response"] = response

        plan_call = complete_call(
            {"category": "mode_B_plan", "row_index": index},
            final_messages,
            apply_plan,
            max_tokens=PLAN_MAX_TOKENS,
            temperature=0,
        )
        expected_prompt_tokens = final_tokens + multimodal_surcharge
        if plan_call.get("prompt_tokens") != expected_prompt_tokens:
            raise GateFailure(
                f"Mode B {split}:{index} prompt tokens changed: "
                f"expected={expected_prompt_tokens}, "
                f"observed={plan_call.get('prompt_tokens')}"
            )
        response = plan_holder["response"]
        metrics = bench.score_response(
            scorer, 2, response, bench.get_ground_truth(sample), index
        )
        success = metrics["task_score"] == 1
        reward_retrieved_insights(state.insights, retrieved_insights, success)
        roles = [
            {
                "robot_id": robot,
                "robot_type": robots[robot],
                "available_actions": available_actions[robot],
            }
            for robot in sorted(robots)
        ]
        task_description = (
            f"Task instruction: {instruction}\n"
            f"Activated robot roles: {canonical_json(roles)}"
        )
        trajectory = (
            "> Generate one complete multi-robot executable plan\n"
            + response
            + ("\nTask completed successfully.\n" if success else "\nTask failed.\n")
        )
        clean_trajectory = re.sub(r"\d+", "", trajectory)
        extraction_messages = [
            {"role": "system", "content": prompts["extract_true_traj_system_prompt"]},
            {
                "role": "user",
                "content": prompts["extract_true_traj_user_prompt"].format(
                    task=task_description, trajectory=clean_trajectory
                ),
            },
        ]
        condensation_holder: Dict[str, str] = {}

        def apply_condensation(
            condensation: str, holder: Dict[str, str] = condensation_holder
        ) -> None:
            holder["response"] = condensation

        condensation_call = complete_call(
            {"category": "mode_B_trajectory_condensation", "row_index": index},
            extraction_messages,
            apply_condensation,
            seed=SEED + cursor,
        )
        failure_reason = None
        failure_call_sequence = None
        if not success:
            failure_messages = [
                {"role": "system", "content": prompts["detect_mistakes_system_prompt"]},
                {
                    "role": "user",
                    "content": prompts["detect_mistakes_user_prompt"].format(
                        task=task_description, trajectory=clean_trajectory
                    ),
                },
            ]
            failure_holder: Dict[str, str] = {}

            def apply_failure(
                reason: str, holder: Dict[str, str] = failure_holder
            ) -> None:
                holder["response"] = reason

            failure_call = complete_call(
                {"category": "mode_B_failed_diagnosis", "row_index": index},
                failure_messages,
                apply_failure,
                seed=SEED + cursor,
            )
            failure_reason = failure_holder["response"]
            failure_call_sequence = failure_call["sequence"]
        memory_record = {
            "memory_id": f"{split}:{index}",
            "task_main": instruction,
            "task_description": task_description,
            "roles": roles,
            "trajectory": trajectory,
            "key_steps": condensation_holder["response"],
            "label": success,
            "response_sha256": sha256_text(response),
            "condensation_sha256": condensation_call["response_sha256"],
            "failure_reason": failure_reason,
        }
        state.add_record(memory_record, query_embedding)
        memory_size = len(state.records)
        if memory_size % ROUNDS_PER_INSIGHTS == 0:
            insight_events += 1
            for point in range(INSIGHTS_POINT_NUM):
                anchor_index = rng.choice(range(memory_size))
                nearest = state.nearest_record_indices(
                    state.embeddings[anchor_index], count=3, label=True
                )
                selected_indices = nearest + (
                    [anchor_index] if state.records[anchor_index]["label"] else []
                )
                rng.shuffle(selected_indices)
                task_mains = [
                    str(state.records[record_index]["task_main"])
                    for record_index in selected_indices
                ]
                local_ids = state.related_insight_ids(
                    task_mains, threshold=len(task_mains) / 2
                )
                local_rules = [state.insights[rule_id]["rule"] for rule_id in local_ids]
                if not local_rules:
                    local_rules = [""]
                rule_text = "\n".join(
                    f"{rule_number}. {rule}"
                    for rule_number, rule in enumerate(local_rules, 1)
                )
                history = "\n".join(
                    f"task{task_number}:\n"
                    + str(state.records[record_index]["task_description"])
                    + str(state.records[record_index]["key_steps"])
                    for task_number, record_index in enumerate(selected_indices)
                )
                user_prompt = prompts["critique_success_rules_user_prompt"].format(
                    success_history=history, existing_rules=rule_text
                )
                suffix = (
                    "Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule "
                    "unless the new rule is VERY insightful and different from "
                    "EXISTING RULES.\n"
                    if len(state.insights) > 10
                    else ""
                )
                messages = [
                    {
                        "role": "system",
                        "content": prompts["critique_success_rules_system_prompt"]
                        + suffix,
                    },
                    {"role": "user", "content": user_prompt},
                ]

                def apply_insight(
                    insight_response: str,
                    ids: List[int] = local_ids,
                    relative_tasks: List[str] = task_mains,
                ) -> None:
                    update_rules(
                        state.insights,
                        relative_tasks,
                        parse_rule_operations(insight_response),
                        local_insight_ids=ids,
                    )

                complete_call(
                    {
                        "category": "mode_B_periodic_insight",
                        "row_index": index,
                        "event_memory_size": memory_size,
                        "point": point,
                        "anchor_memory_id": state.records[anchor_index]["memory_id"],
                        "selected_memory_ids": [
                            state.records[record_index]["memory_id"]
                            for record_index in selected_indices
                        ],
                        "local_insight_ids": local_ids,
                    },
                    messages,
                    apply_insight,
                    seed=SEED + cursor,
                )
        if memory_size % 20 == 0:
            merge_events += 1
            clusters = state.cluster_tasks()
            merged_by_cluster: Dict[int, List[str]] = {}
            for cluster_id, tasks in clusters.items():
                insight_ids = state.related_insight_ids(tasks)
                rules = [state.insights[rule_id]["rule"] for rule_id in insight_ids]
                merged_rules: List[str] = []
                for batch_start in range(0, len(rules), 10):
                    batch = rules[batch_start : batch_start + 10]
                    limited_number = (len(batch) // 3) // 3
                    messages = [
                        {
                            "role": "system",
                            "content": prompts["merge_rules_system_prompt"],
                        },
                        {
                            "role": "user",
                            "content": prompts["merge_rules_user_prompt"].format(
                                current_rules="\n".join(batch),
                                limited_number=limited_number,
                            ),
                        },
                    ]

                    def apply_merge(
                        merge_response: str,
                        destination: List[str] = merged_rules,
                    ) -> None:
                        destination.extend(parse_numbered_list(merge_response))

                    complete_call(
                        {
                            "category": "mode_B_merge",
                            "row_index": index,
                            "event_memory_size": memory_size,
                            "cluster_id": cluster_id,
                            "batch_start": batch_start,
                            "cluster_tasks_sha256": sha256_text(canonical_json(tasks)),
                            "source_rules_sha256": sha256_text(canonical_json(batch)),
                            "limited_number": limited_number,
                        },
                        messages,
                        apply_merge,
                        seed=SEED + cursor,
                    )
                merged_by_cluster[cluster_id] = merged_rules
            state.insights.clear()
            for cluster_id, tasks in clusters.items():
                for rule in merged_by_cluster[cluster_id]:
                    state.insights.append(
                        {
                            "rule": rule,
                            "score": 2,
                            "positive_correlation_tasks": list(tasks),
                            "negative_correlation_tasks": [],
                        }
                    )
        row_record = {
            "index": index,
            "run_fingerprint": run_hash,
            "mode": "B",
            "split": split,
            "instruction": instruction,
            "candidate_memory_ids": [
                state.records[candidate_index]["memory_id"]
                for candidate_index in candidate_indices
            ],
            "rerank_calls": rerank_calls,
            "selected_memory_id": state.records[selected_index]["memory_id"],
            "insights": retrieved_insights,
            "prompt_sha256": messages_sha256(final_messages),
            "native_prompt_sha256": sha256_text(native_prompt),
            "final_prompt": final_prompt,
            "zero_shot_input_tokens": zero_tokens,
            "segment_target_input_tokens": target_tokens,
            "native_input_tokens": native_tokens,
            "tokenizer_input_tokens": final_tokens,
            "injected_tokens": final_tokens - zero_tokens,
            "under_budget_shortfall": max(0, target_tokens - final_tokens),
            "relative_shortfall": max(0, target_tokens - final_tokens) / target_tokens,
            "truncated": final_prompt != native_prompt,
            "multimodal_usage_surcharge": multimodal_surcharge,
            "response": response,
            "response_sha256": sha256_text(response),
            "prompt_tokens": plan_call["prompt_tokens"],
            "completion_tokens": plan_call["completion_tokens"],
            "condensation_call_sequence": condensation_call["sequence"],
            "failure_call_sequence": failure_call_sequence,
            "hierarchy_before_sha256": hierarchy_before,
            "hierarchy_after_sha256": state.hierarchy_sha256(),
            "call_sequence_start": row_call_start,
            "call_sequence_end": cursor,
            **metrics,
        }
        if index in records and records[index] != row_record:
            raise GateFailure(f"Mode B {split}:{index} row replay differs")
        records[index] = row_record
        write_jsonl_snapshot(output, records, indices)
    if cursor != len(calls):
        raise GateFailure(f"Mode B {split} call ledger has trailing calls")
    expected_insight_events = (
        TRAIN_ROWS + len(indices)
    ) // ROUNDS_PER_INSIGHTS - TRAIN_ROWS // ROUNDS_PER_INSIGHTS
    expected_merge_events = (TRAIN_ROWS + len(indices)) // 20 - TRAIN_ROWS // 20
    if (
        insight_events != expected_insight_events
        or merge_events != expected_merge_events
    ):
        raise GateFailure(f"Mode B {split} adaptation schedule count changed")
    category_counts: Dict[str, int] = {}
    for call in calls:
        category = str(call["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    successes = sum(record["task_score"] == 1 for record in records.values())
    formats = sum(record["format_score"] == 1 for record in records.values())
    summary = {
        "task": "Amendment7_GMemory_Mode_B",
        "status": "PASS",
        "mode": "B",
        "split": split,
        "samples": len(records),
        "successes": successes,
        "accuracy": successes / len(records),
        "format_successes": formats,
        "format_compliance": formats / len(records),
        "plan_generation_calls": category_counts.get("mode_B_plan", 0),
        "memory_control_calls": len(calls) - category_counts.get("mode_B_plan", 0),
        "call_category_counts": category_counts,
        "insight_events": insight_events,
        "merge_events": merge_events,
        "frozen_hierarchy_sha256": frozen_hierarchy_sha256,
        "final_hierarchy_sha256": state.hierarchy_sha256(),
        "hierarchy_changed": state.hierarchy_sha256() != frozen_hierarchy_sha256,
        "prompt_tokens": sum(
            int(call["prompt_tokens"])
            for call in calls
            if call.get("prompt_tokens") is not None
        ),
        "completion_tokens": sum(
            int(call["completion_tokens"])
            for call in calls
            if call.get("completion_tokens") is not None
        ),
        "calls_missing_usage": sum(
            call.get("prompt_tokens") is None or call.get("completion_tokens") is None
            for call in calls
        ),
        "runtime": runtime,
        "results": str(output),
        "results_sha256": file_sha256(output),
        "call_ledger": str(calls_path),
        "call_ledger_sha256": file_sha256(calls_path),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary


def analyze_mode_a(split: str) -> Dict[str, Any]:
    require_frozen()
    dataset, indices, _, source_path, source_summary_path = split_configuration(split)
    del dataset
    source_records, source_run_hash, _ = validate_reused_source(
        split, indices, source_path, source_summary_path
    )
    mode_path = OUTPUT_DIR / f"mode_A.{split}.jsonl"
    mode_summary_path = mode_path.with_suffix(".summary.json")
    trajectory_path = AMENDMENT6_DIR / f"qwen2_5_vl_72b.{split}.trajectory_rag.jsonl"
    trajectory_summary_path = trajectory_path.with_suffix(".summary.json")
    required = (
        mode_path,
        mode_summary_path,
        trajectory_path,
        trajectory_summary_path,
    )
    if any(not path.is_file() for path in required):
        raise GateFailure(f"Missing completed Mode A {split} analysis inputs")
    mode_rows = load_jsonl(mode_path)
    trajectory_rows = load_jsonl(trajectory_path)
    mode_summary = json.loads(mode_summary_path.read_text())
    trajectory_summary = json.loads(trajectory_summary_path.read_text())
    if (
        set(mode_rows) != set(indices)
        or set(trajectory_rows) != set(indices)
        or mode_summary.get("status") != "PASS"
        or mode_summary.get("results_sha256") != file_sha256(mode_path)
        or trajectory_summary.get("results_sha256") != file_sha256(trajectory_path)
    ):
        raise GateFailure(f"Completed Mode A {split} analysis inputs changed")
    rows = []
    for index in indices:
        gmemory = mode_rows[index]
        segment = source_records[index]["arms"]["segment"]
        trajectory = trajectory_rows[index]
        rows.append(
            {
                "index": index,
                "gmemory_task_score": int(gmemory["task_score"]),
                "segment_task_score": int(segment["task_score"]),
                "trajectory_rag_task_score": int(trajectory["task_score"]),
                "gmemory_format_score": int(gmemory["format_score"]),
                "segment_format_score": int(segment["format_score"]),
                "trajectory_rag_format_score": int(trajectory["format_score"]),
                "gmemory_prompt_tokens": int(gmemory["prompt_tokens"]),
                "gmemory_completion_tokens": int(gmemory["completion_tokens"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("index")
    row_path = OUTPUT_DIR / f"mode_A.{split}.analysis_rows.parquet"
    frame.to_parquet(row_path, index=False)

    def comparison(baseline: str) -> Dict[str, Any]:
        baseline_values = frame[f"{baseline}_task_score"].to_numpy(dtype=bool)
        gmemory_values = frame["gmemory_task_score"].to_numpy(dtype=bool)
        baseline_only = int((baseline_values & ~gmemory_values).sum())
        gmemory_only = int((~baseline_values & gmemory_values).sum())
        return {
            "baseline": baseline,
            "baseline_successes": int(baseline_values.sum()),
            "gmemory_successes": int(gmemory_values.sum()),
            "baseline_accuracy": float(baseline_values.mean()),
            "gmemory_accuracy": float(gmemory_values.mean()),
            "gmemory_minus_baseline": float(
                gmemory_values.mean() - baseline_values.mean()
            ),
            "baseline_only": baseline_only,
            "gmemory_only": gmemory_only,
            "discordant_pairs": baseline_only + gmemory_only,
            "mcnemar_exact_p": exact_mcnemar_p(baseline_only, gmemory_only),
        }

    summary = {
        "task": "Amendment7_GMemory_Mode_A_analysis",
        "status": "PASS",
        "mode": "A",
        "split": split,
        "samples": len(frame),
        "comparisons": {
            baseline: comparison(baseline) for baseline in ("segment", "trajectory_rag")
        },
        "format_compliance": {
            arm: float(frame[f"{arm}_format_score"].mean())
            for arm in ("gmemory", "segment", "trajectory_rag")
        },
        "model_generation_calls": 0,
        "memory_control_calls": 0,
        "source_run_fingerprint": source_run_hash,
        "hierarchy_sha256": mode_summary["hierarchy_sha256"],
        "row_results": str(row_path),
        "row_results_sha256": file_sha256(row_path),
        "mode_results_sha256": file_sha256(mode_path),
        "trajectory_rag_results_sha256": file_sha256(trajectory_path),
        "segment_source_results_sha256": file_sha256(source_path),
    }
    output = OUTPUT_DIR / f"mode_A.{split}.analysis.json"
    atomic_json(output, summary)
    return summary


def analyze_mode_b(split: str) -> Dict[str, Any]:
    require_frozen()
    dataset, indices, _, source_path, source_summary_path = split_configuration(split)
    del dataset
    source_records, source_run_hash, _ = validate_reused_source(
        split, indices, source_path, source_summary_path
    )
    mode_b_path = OUTPUT_DIR / f"mode_B.{split}.jsonl"
    mode_b_summary_path = mode_b_path.with_suffix(".summary.json")
    mode_a_path = OUTPUT_DIR / f"mode_A.{split}.jsonl"
    mode_a_summary_path = mode_a_path.with_suffix(".summary.json")
    trajectory_path = AMENDMENT6_DIR / f"qwen2_5_vl_72b.{split}.trajectory_rag.jsonl"
    trajectory_summary_path = trajectory_path.with_suffix(".summary.json")
    required = (
        mode_b_path,
        mode_b_summary_path,
        mode_a_path,
        mode_a_summary_path,
        trajectory_path,
        trajectory_summary_path,
    )
    if any(not path.is_file() for path in required):
        raise GateFailure(f"Missing completed Mode B {split} analysis inputs")
    mode_b_rows = load_jsonl(mode_b_path)
    mode_a_rows = load_jsonl(mode_a_path)
    trajectory_rows = load_jsonl(trajectory_path)
    mode_b_summary = json.loads(mode_b_summary_path.read_text())
    mode_a_summary = json.loads(mode_a_summary_path.read_text())
    trajectory_summary = json.loads(trajectory_summary_path.read_text())
    if (
        set(mode_b_rows) != set(indices)
        or set(mode_a_rows) != set(indices)
        or set(trajectory_rows) != set(indices)
        or mode_b_summary.get("status") != "PASS"
        or mode_b_summary.get("results_sha256") != file_sha256(mode_b_path)
        or mode_a_summary.get("status") != "PASS"
        or mode_a_summary.get("results_sha256") != file_sha256(mode_a_path)
        or trajectory_summary.get("results_sha256") != file_sha256(trajectory_path)
    ):
        raise GateFailure(f"Completed Mode B {split} analysis inputs changed")
    rows = []
    for index in indices:
        rows.append(
            {
                "index": index,
                "mode_B_task_score": int(mode_b_rows[index]["task_score"]),
                "mode_A_task_score": int(mode_a_rows[index]["task_score"]),
                "segment_task_score": int(
                    source_records[index]["arms"]["segment"]["task_score"]
                ),
                "trajectory_rag_task_score": int(trajectory_rows[index]["task_score"]),
                "mode_B_format_score": int(mode_b_rows[index]["format_score"]),
                "mode_A_format_score": int(mode_a_rows[index]["format_score"]),
                "segment_format_score": int(
                    source_records[index]["arms"]["segment"]["format_score"]
                ),
                "trajectory_rag_format_score": int(
                    trajectory_rows[index]["format_score"]
                ),
                "mode_B_prompt_tokens": int(mode_b_rows[index]["prompt_tokens"]),
                "mode_B_completion_tokens": int(
                    mode_b_rows[index]["completion_tokens"]
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("index")
    row_path = OUTPUT_DIR / f"mode_B.{split}.analysis_rows.parquet"
    frame.to_parquet(row_path, index=False)

    def comparison(baseline: str) -> Dict[str, Any]:
        baseline_values = frame[f"{baseline}_task_score"].to_numpy(dtype=bool)
        mode_b_values = frame["mode_B_task_score"].to_numpy(dtype=bool)
        baseline_only = int((baseline_values & ~mode_b_values).sum())
        mode_b_only = int((~baseline_values & mode_b_values).sum())
        return {
            "baseline": baseline,
            "baseline_successes": int(baseline_values.sum()),
            "mode_B_successes": int(mode_b_values.sum()),
            "baseline_accuracy": float(baseline_values.mean()),
            "mode_B_accuracy": float(mode_b_values.mean()),
            "mode_B_minus_baseline": float(
                mode_b_values.mean() - baseline_values.mean()
            ),
            "baseline_only": baseline_only,
            "mode_B_only": mode_b_only,
            "discordant_pairs": baseline_only + mode_b_only,
            "mcnemar_exact_p": exact_mcnemar_p(baseline_only, mode_b_only),
        }

    summary = {
        "task": "Amendment7_GMemory_Mode_B_analysis",
        "status": "PASS",
        "mode": "B",
        "split": split,
        "samples": len(frame),
        "comparisons": {
            baseline: comparison(baseline)
            for baseline in ("mode_A", "segment", "trajectory_rag")
        },
        "format_compliance": {
            arm: float(frame[f"{arm}_format_score"].mean())
            for arm in ("mode_B", "mode_A", "segment", "trajectory_rag")
        },
        "model_generation_calls": 0,
        "memory_control_calls": 0,
        "source_run_fingerprint": source_run_hash,
        "frozen_hierarchy_sha256": mode_b_summary["frozen_hierarchy_sha256"],
        "final_hierarchy_sha256": mode_b_summary["final_hierarchy_sha256"],
        "row_results": str(row_path),
        "row_results_sha256": file_sha256(row_path),
        "mode_B_results_sha256": file_sha256(mode_b_path),
        "mode_A_results_sha256": file_sha256(mode_a_path),
        "trajectory_rag_results_sha256": file_sha256(trajectory_path),
        "segment_source_results_sha256": file_sha256(source_path),
    }
    if split == "ood":
        successes = {
            arm: int(frame[f"{arm}_task_score"].sum())
            for arm in ("mode_B", "mode_A", "segment", "trajectory_rag")
        }
        if successes["mode_A"] > successes["segment"]:
            selected = "mode_A_beats_segment_OOD"
        elif successes["mode_B"] > max(
            successes[arm] for arm in ("mode_A", "segment", "trajectory_rag")
        ):
            selected = "mode_B_beats_all"
        else:
            selected = "otherwise"
        summary["ending"] = {
            "selected": selected,
            "text": preregistration()["endings"][selected],
            "successes": successes,
        }
    output = OUTPUT_DIR / f"mode_B.{split}.analysis.json"
    atomic_json(output, summary)
    return summary


def close() -> Dict[str, Any]:
    require_frozen()
    if not TRAIN_HIERARCHY_SUMMARY_PATH.is_file():
        raise GateFailure("Amendment 7 train hierarchy is incomplete")
    train_summary = json.loads(TRAIN_HIERARCHY_SUMMARY_PATH.read_text())
    if (
        train_summary.get("status") != "PASS"
        or train_summary.get("rows") != TRAIN_ROWS
        or train_summary.get("hierarchy_file_sha256")
        != file_sha256(TRAIN_HIERARCHY_PATH)
    ):
        raise GateFailure("Amendment 7 train hierarchy certificate changed")
    generated_artifacts = []
    analysis_artifacts = {}
    accepted_plan_calls = 0
    memory_control_calls = int(train_summary["memory_control_calls"]) + TRAIN_ROWS
    for backbone, prefix in (
        ("qwen2_5_vl_72b", ""),
        ("gpt_4o_optional", "gpt_4o_optional."),
    ):
        for mode in ("A", "B"):
            for split, expected_rows in (("ood", OOD_ROWS), ("id", ID_ROWS)):
                result_path = OUTPUT_DIR / f"{prefix}mode_{mode}.{split}.jsonl"
                summary_path = result_path.with_suffix(".summary.json")
                analysis_path = (
                    OUTPUT_DIR / f"{prefix}mode_{mode}.{split}.analysis.json"
                )
                if any(
                    not path.is_file()
                    for path in (result_path, summary_path, analysis_path)
                ):
                    raise GateFailure(
                        f"Missing completed {backbone} Mode {mode} {split} artifact"
                    )
                summary = json.loads(summary_path.read_text())
                analysis = json.loads(analysis_path.read_text())
                rows = load_jsonl(result_path)
                if (
                    summary.get("status") != "PASS"
                    or summary.get("samples") != expected_rows
                    or summary.get("results_sha256") != file_sha256(result_path)
                    or len(rows) != expected_rows
                    or analysis.get("status") != "PASS"
                    or analysis.get("samples") != expected_rows
                ):
                    raise GateFailure(
                        f"Invalid completed {backbone} Mode {mode} {split} artifact"
                    )
                accepted_plan_calls += int(summary["plan_generation_calls"])
                memory_control_calls += int(summary.get("memory_control_calls", 0))
                call_ledger = summary.get("call_ledger")
                if call_ledger:
                    ledger_path = Path(call_ledger)
                    expected_ledger_path = OUTPUT_DIR / f"mode_B.{split}.calls.jsonl"
                    if (
                        backbone != "qwen2_5_vl_72b"
                        or mode != "B"
                        or (ledger_path != expected_ledger_path)
                    ):
                        raise GateFailure(f"Unexpected call ledger: {ledger_path}")
                    pending_path = ledger_path.with_suffix(
                        ledger_path.suffix + ".pending.json"
                    )
                    if pending_path.exists():
                        raise GateFailure(
                            f"Unresolved pending call journal: {pending_path}"
                        )
                    if summary.get("call_ledger_sha256") != file_sha256(ledger_path):
                        raise GateFailure(f"Call ledger changed: {ledger_path}")
                if backbone == "gpt_4o_optional":
                    if (
                        summary.get("memory_control_calls") != 0
                        or summary.get("native_gpt4o_adaptation_claimed") is not False
                    ):
                        raise GateFailure(
                            f"Invalid optional replay semantics: {summary_path}"
                        )
                    pending_directory = result_path.with_suffix(
                        result_path.suffix + ".pending"
                    )
                    if pending_directory.is_dir() and any(
                        pending_directory.glob("*.json")
                    ):
                        raise GateFailure(
                            f"Unresolved optional pending journals: {pending_directory}"
                        )
                generated_artifacts.append(
                    {
                        "backbone": backbone,
                        "mode": mode,
                        "split": split,
                        "rows": expected_rows,
                        "successes": int(summary["successes"]),
                        "format_successes": int(summary["format_successes"]),
                        "results": str(result_path),
                        "results_sha256": file_sha256(result_path),
                        "summary_sha256": file_sha256(summary_path),
                    }
                )
                analysis_artifacts[f"{backbone}.mode_{mode}.{split}"] = {
                    "path": str(analysis_path),
                    "sha256": file_sha256(analysis_path),
                }
    expected_plan_calls = REQUIRED_PLAN_CALLS * 2
    if accepted_plan_calls != expected_plan_calls:
        raise GateFailure(f"Amendment 7 plan-call count changed: {accepted_plan_calls}")
    ending_path = OUTPUT_DIR / "mode_B.ood.analysis.json"
    optional_protocol_path = OUTPUT_DIR / "gpt_4o_optional.protocol.json"
    if not optional_protocol_path.is_file():
        raise GateFailure("Missing optional GPT-4o protocol certificate")
    optional_protocol = json.loads(optional_protocol_path.read_text())
    if (
        optional_protocol.get("status") != "FILED_BEFORE_A7_PLAN_GENERATION"
        or optional_protocol.get("authorized_plan_generation_calls")
        != REQUIRED_PLAN_CALLS
        or optional_protocol.get("authorized_optional_memory_control_calls") != 0
    ):
        raise GateFailure("Invalid optional GPT-4o protocol certificate")
    ending = json.loads(ending_path.read_text()).get("ending")
    if (
        not isinstance(ending, dict)
        or ending.get("selected") not in preregistration()["endings"]
    ):
        raise GateFailure("Amendment 7 preregistered ending is missing")
    result = {
        "task": "Amendment7_generation_closure",
        "status": "PASS",
        "generation_closed": True,
        "permanent_closure_reinstated": True,
        "completed_results_modified": False,
        "training": False,
        "gradient_updates": False,
        "fine_tuning": False,
        "required_qwen2_5_vl_72b_complete": True,
        "optional_gpt_4o_complete": True,
        "accepted_plan_generation_calls": accepted_plan_calls,
        "accepted_calls_by_backbone": {
            "qwen2_5_vl_72b": REQUIRED_PLAN_CALLS,
            "gpt_4o_optional": REQUIRED_PLAN_CALLS,
        },
        "memory_control_calls": memory_control_calls,
        "generated_artifacts_certified": len(generated_artifacts),
        "generated_artifacts": generated_artifacts,
        "analysis_artifacts": analysis_artifacts,
        "selected_ending": ending,
        "train_hierarchy_sha256": train_summary["hierarchy_sha256"],
        "train_hierarchy_summary_sha256": file_sha256(TRAIN_HIERARCHY_SUMMARY_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "port_note_sha256": file_sha256(PORT_NOTE_PATH),
        "supersession_sha256": file_sha256(SUPERSESSION_PATH),
        "runner_sha256": file_sha256(ROOT / "scripts/viki_amendment7.py"),
        "gmemory_sha256": file_sha256(ROOT / "habitat_llm/evaluation/viki_gmemory.py"),
        "gpt4o_runner_sha256": file_sha256(ROOT / "scripts/viki_amendment7_gpt4o.py"),
        "gpt4o_protocol_sha256": file_sha256(optional_protocol_path),
        "after_scope": "No further VIKI generation is authorized.",
    }
    atomic_json(GENERATION_CLOSURE_PATH, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIKI Amendment 7")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("prepare-train")
    condense_parser = subparsers.add_parser("condense-train")
    condense_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    condense_parser.add_argument("--workers", type=int, default=8)
    hierarchy_parser = subparsers.add_parser("build-train-hierarchy")
    hierarchy_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    mode_a_parser = subparsers.add_parser("preflight-mode-a")
    mode_a_parser.add_argument("--split", choices=("ood", "id"), required=True)
    mode_a_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    mode_a_parser.add_argument("--workers", type=int, default=8)
    run_mode_a_parser = subparsers.add_parser("run-mode-a")
    run_mode_a_parser.add_argument("--split", choices=("ood", "id"), required=True)
    run_mode_a_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    run_mode_a_parser.add_argument("--workers", type=int, default=8)
    mode_b_shadow_parser = subparsers.add_parser("preflight-mode-b-shadow")
    mode_b_shadow_parser.add_argument("--split", choices=("ood", "id"), required=True)
    mode_b_shadow_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    mode_b_shadow_parser.add_argument("--workers", type=int, default=8)
    run_mode_b_parser = subparsers.add_parser("run-mode-b")
    run_mode_b_parser.add_argument("--split", choices=("ood", "id"), required=True)
    run_mode_b_parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    analyze_mode_a_parser = subparsers.add_parser("analyze-mode-a")
    analyze_mode_a_parser.add_argument("--split", choices=("ood", "id"), required=True)
    analyze_mode_b_parser = subparsers.add_parser("analyze-mode-b")
    analyze_mode_b_parser.add_argument("--split", choices=("ood", "id"), required=True)
    subparsers.add_parser("close")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "freeze":
            result = freeze()
        elif args.command == "prepare-train":
            result = prepare_train()
        elif args.command == "condense-train":
            result = condense_train(args.base_url, args.workers)
        elif args.command == "build-train-hierarchy":
            result = build_train_hierarchy(args.base_url)
        elif args.command == "preflight-mode-a":
            result = mode_a_preflight(args.split, args.base_url, args.workers)
        elif args.command == "run-mode-a":
            result = run_mode_a(args.split, args.base_url, args.workers)
        elif args.command == "preflight-mode-b-shadow":
            result = mode_b_shadow_preflight(args.split, args.base_url, args.workers)
        elif args.command == "run-mode-b":
            result = run_mode_b(args.split, args.base_url)
        elif args.command == "analyze-mode-a":
            result = analyze_mode_a(args.split)
        elif args.command == "analyze-mode-b":
            result = analyze_mode_b(args.split)
        else:
            result = close()
    except GateFailure as error:
        print(f"GATE FAILURE: {error}")
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
