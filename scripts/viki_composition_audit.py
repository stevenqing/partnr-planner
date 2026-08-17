#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from habitat_llm.evaluation.viki_composition import parse_composition
from habitat_llm.evaluation.viki_memory_skill import (
    VikiMemorySkillLibrary,
    get_prompt_context,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment1"
MANIFEST_PATH = OUTPUT_DIR / "c1_split_manifest.parquet"
V0_PATH = ROOT / "results/viki_memory_experiments/viki_ood_samples.parquet"
MEMORY_LOG = ROOT / "results/viki_memory_skill_7b_l2_ood.jsonl"
EMBEDDING_CACHE = ROOT / "results/viki_l2_memory_skill_all_mpnet_base_v2.npz"


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if hasattr(value, "tolist"):
        return native(value.tolist())
    return value


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    with path.open() as source:
        return {
            int(record["index"]): record
            for record in (json.loads(line) for line in source if line.strip())
        }


def grounded_plan_tokens(ground_truth: Mapping[str, Any]) -> List[str]:
    tokens = []
    for step in native(ground_truth)["time_steps"]:
        for robot, action in sorted(step["actions"].items()):
            if action is not None:
                tokens.append(":".join([str(robot), *map(str, action)]))
    return tokens


def levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def ngrams(tokens: Sequence[str], size: int = 3) -> Set[Tuple[str, ...]]:
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {
        tuple(tokens[position : position + size])
        for position in range(len(tokens) - size + 1)
    }


def ngram_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_ngrams = ngrams(left)
    right_ngrams = ngrams(right)
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else 1.0


def summarize(frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "samples": len(frame),
        "memory_rows_min": int(frame["memory_pool_size"].min()),
        "memory_rows_max": int(frame["memory_pool_size"].max()),
        "instruction_similarity": {
            "mean": float(frame["max_instruction_similarity"].mean()),
            "median": float(frame["max_instruction_similarity"].median()),
            "p90": float(frame["max_instruction_similarity"].quantile(0.9)),
            "max": float(frame["max_instruction_similarity"].max()),
        },
        "plan_edit_distance": {
            "mean": float(frame["min_plan_edit_distance"].mean()),
            "median": float(frame["min_plan_edit_distance"].median()),
            "min": int(frame["min_plan_edit_distance"].min()),
            "exact_plan_rows": int((frame["min_plan_edit_distance"] == 0).sum()),
        },
        "trigram_jaccard": {
            "mean": float(frame["max_plan_trigram_jaccard"].mean()),
            "median": float(frame["max_plan_trigram_jaccard"].median()),
            "max": float(frame["max_plan_trigram_jaccard"].max()),
            "exact_rows": int((frame["max_plan_trigram_jaccard"] == 1).sum()),
        },
        "full_signature_overlap_rows": int(frame["full_signature_overlap"].sum()),
    }


def audit_pool(
    channel: str,
    test_rows: pd.DataFrame,
    memory_indices: Sequence[int],
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_tokens: Mapping[int, Sequence[str]],
    test_tokens: Mapping[int, Sequence[str]],
    train_signatures: Mapping[int, str],
) -> List[Dict[str, Any]]:
    memory_indices = [int(index) for index in memory_indices]
    memory_embedding_matrix = train_embeddings[memory_indices]
    memory_signature_set = {train_signatures[index] for index in memory_indices}
    rows = []
    for _, manifest_row in test_rows.iterrows():
        index = int(manifest_row["test_index"])
        similarities = test_embeddings[index] @ memory_embedding_matrix.T
        similarity_position = int(np.argmax(similarities))
        best_similarity_id = memory_indices[similarity_position]
        current_tokens = test_tokens[index]
        distances = [
            levenshtein(current_tokens, train_tokens[memory_index])
            for memory_index in memory_indices
        ]
        edit_position = int(np.argmin(distances))
        overlaps = [
            ngram_jaccard(current_tokens, train_tokens[memory_index])
            for memory_index in memory_indices
        ]
        overlap_position = int(np.argmax(overlaps))
        rows.append(
            {
                "audit": "strict_split",
                "channel": channel,
                "test_split": "test",
                "test_index": index,
                "memory_pool_size": len(memory_indices),
                "max_instruction_similarity": float(similarities[similarity_position]),
                "nearest_instruction_memory_index": best_similarity_id,
                "min_plan_edit_distance": int(distances[edit_position]),
                "nearest_edit_memory_index": memory_indices[edit_position],
                "max_plan_trigram_jaccard": float(overlaps[overlap_position]),
                "nearest_ngram_memory_index": memory_indices[overlap_position],
                "full_signature_overlap": (
                    manifest_row["full_signature"] in memory_signature_set
                ),
            }
        )
    return rows


def audit_published_ood(
    val: pd.DataFrame,
    train: pd.DataFrame,
    v0: pd.DataFrame,
    memory_log: Mapping[int, Mapping[str, Any]],
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    train_tokens: Mapping[int, Sequence[str]],
    train_signatures: Mapping[int, str],
) -> List[Dict[str, Any]]:
    rows = []
    for index in range(len(val)):
        demo_ids = [int(value) for value in v0.loc[index, "injected_demo_ids"]]
        current_ground_truth = val.iloc[index]["reward_model"]["ground_truth"]
        current_tokens = grounded_plan_tokens(current_ground_truth)
        current_signature = parse_composition(current_ground_truth).full_signature()
        similarities = val_embeddings[index] @ train_embeddings[demo_ids].T
        distances = [
            levenshtein(current_tokens, train_tokens[demo_id]) for demo_id in demo_ids
        ]
        overlaps = [
            ngram_jaccard(current_tokens, train_tokens[demo_id]) for demo_id in demo_ids
        ]
        rows.append(
            {
                "audit": "published_ood_memory",
                "channel": "published_ood_unconditional",
                "test_split": "val",
                "test_index": index,
                "memory_pool_size": len(demo_ids),
                "max_instruction_similarity": float(np.max(similarities)),
                "nearest_instruction_memory_index": demo_ids[
                    int(np.argmax(similarities))
                ],
                "min_plan_edit_distance": int(min(distances)),
                "nearest_edit_memory_index": demo_ids[int(np.argmin(distances))],
                "max_plan_trigram_jaccard": float(max(overlaps)),
                "nearest_ngram_memory_index": demo_ids[int(np.argmax(overlaps))],
                "full_signature_overlap": any(
                    train_signatures[demo_id] == current_signature
                    for demo_id in demo_ids
                ),
                "published_routed_skill": memory_log[index]["provider_metadata"][
                    "predicted_skill"
                ],
            }
        )
    return rows


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    train = pd.read_parquet(DATA_ROOT / "train.parquet")
    test = pd.read_parquet(DATA_ROOT / "test.parquet")
    val = pd.read_parquet(DATA_ROOT / "val.parquet")
    manifest = pd.read_parquet(args.manifest)
    v0 = pd.read_parquet(V0_PATH).set_index("index")
    memory_log = load_jsonl(MEMORY_LOG)
    library = VikiMemorySkillLibrary(
        BENCHMARK_ROOT,
        "all-mpnet-base-v2",
        "cpu",
        cache_path=EMBEDDING_CACHE,
    )
    train_embeddings = np.asarray(library.embeddings)
    test_instructions = [
        get_prompt_context(row.to_dict())[0] for _, row in test.iterrows()
    ]
    val_instructions = [
        get_prompt_context(row.to_dict())[0] for _, row in val.iterrows()
    ]
    test_embeddings = np.asarray(
        library.embedding_model.encode(
            test_instructions,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    val_embeddings = np.asarray(
        library.embedding_model.encode(
            val_instructions,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    train_tokens = {
        index: grounded_plan_tokens(train.iloc[index]["reward_model"]["ground_truth"])
        for index in range(len(train))
    }
    test_tokens = {
        index: grounded_plan_tokens(test.iloc[index]["reward_model"]["ground_truth"])
        for index in range(len(test))
    }
    train_signatures = {
        index: parse_composition(
            train.iloc[index]["reward_model"]["ground_truth"]
        ).full_signature()
        for index in range(len(train))
    }

    rows: List[Dict[str, Any]] = []
    for channel, group in manifest.groupby("channel"):
        for memory_json, pool_group in group.groupby("allowed_memory_indices"):
            memory_indices = json.loads(memory_json)
            rows.extend(
                audit_pool(
                    channel,
                    pool_group,
                    memory_indices,
                    train,
                    test,
                    train_embeddings,
                    test_embeddings,
                    train_tokens,
                    test_tokens,
                    train_signatures,
                )
            )
    rows.extend(
        audit_published_ood(
            val,
            train,
            v0,
            memory_log,
            train_embeddings,
            val_embeddings,
            train_tokens,
            train_signatures,
        )
    )
    audit = pd.DataFrame(rows)
    for channel in ("task_heldout_productivity", "task_heldout_systematicity"):
        overlaps = int(
            audit.loc[audit["channel"] == channel, "full_signature_overlap"].sum()
        )
        if overlaps:
            raise ValueError(f"C2 leakage gate failed: {channel} overlaps={overlaps}")
    summary = {
        "task": "C2",
        "gate": {
            "status": "PASS",
            "task_heldout_full_signature_overlap_rows": 0,
        },
        "channels": {
            channel: summarize(group) for channel, group in audit.groupby("channel")
        },
        "metrics": {
            "instruction": "maximum all-mpnet-base-v2 cosine similarity",
            "plan_edit": "minimum Levenshtein distance over grounded action tokens",
            "plan_ngram": "maximum trigram Jaccard over grounded action tokens",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "c2_leakage_audit.parquet"
    csv_path = args.output_dir / "c2_leakage_audit.csv"
    summary_path = args.output_dir / "c2_leakage_audit.summary.json"
    audit.to_parquet(parquet_path, index=False)
    audit.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "parquet": str(parquet_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
                **summary,
            },
            indent=2,
        )
    )
    return summary


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit VIKI composition leakage")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_audit(parse_args())
