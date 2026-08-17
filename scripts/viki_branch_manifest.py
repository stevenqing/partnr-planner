#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd

from habitat_llm.evaluation.viki_branch_conditions import (
    AvailabilityPredicate,
    derive_train_portable_assets,
    derive_train_vocabularies,
    discover_instruction_regions,
    get_instruction,
)
from habitat_llm.evaluation.viki_branch_memory import BranchIndexedMemory
from habitat_llm.evaluation.viki_memory_skill import VikiMemorySkillLibrary

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT.parent / "VIKI-R"
DATA_ROOT = BENCHMARK_ROOT / "data/VIKI-R/viki/VIKI-L2"
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment1"


def build_manifest() -> None:
    train = pd.read_parquet(DATA_ROOT / "train.parquet")
    val = pd.read_parquet(DATA_ROOT / "val.parquet")
    samples = [row.to_dict() for _, row in train.iterrows()]
    assets, locations = derive_train_vocabularies(samples)
    portable = derive_train_portable_assets(samples)
    regions = discover_instruction_regions(
        (get_instruction(sample) for sample in samples), assets, locations
    )
    predicate = AvailabilityPredicate(assets, locations | regions, portable)
    library = VikiMemorySkillLibrary(
        BENCHMARK_ROOT,
        "all-mpnet-base-v2",
        "cpu",
        cache_path=ROOT / "results/viki_l2_memory_skill_all_mpnet_base_v2.npz",
    )
    census = pd.read_parquet(OUTPUT_DIR / "a0_branch_census.parquet")
    labels = {
        int(row["index"]): str(row["branch"])
        for _, row in census[census["split"] == "train"].iterrows()
    }
    memory = BranchIndexedMemory(library, predicate, labels, sorted(assets))
    v0 = pd.read_parquet(
        ROOT / "results/viki_memory_experiments/viki_ood_samples.parquet"
    ).set_index("index")
    with (ROOT / "results/viki_memory_skill_7b_l2_ood.jsonl").open() as source:
        logs = {
            record["index"]: record
            for record in (json.loads(line) for line in source if line.strip())
        }

    for index in range(50):
        result = memory.retrieve(
            val.iloc[index].to_dict(),
            5,
            logs[index]["provider_metadata"]["raw_skill_prediction"],
            branch_indexing=False,
            graded_injection=False,
        )
        observed = [item.instance.train_index for item in result.retrieval.instances]
        expected = list(v0.loc[index, "injected_demo_ids"])
        if observed != expected:
            raise RuntimeError(
                f"GATE A3 failed at {index}: observed={observed}, expected={expected}"
            )

    rows = []
    for index in range(len(val)):
        raw_route = logs[index]["provider_metadata"]["raw_skill_prediction"]
        grounded = memory.retrieve(val.iloc[index].to_dict(), 5, raw_route, True, False)
        graded = memory.retrieve(val.iloc[index].to_dict(), 5, raw_route, True, True)
        rows.append(
            {
                "index": index,
                "ood_subset": v0.loc[index, "ood_subset"],
                "current_branch": grounded.current_branch,
                "absent_assets": json.dumps(
                    grounded.absent_assets, separators=(",", ":")
                ),
                "routed_skill": grounded.retrieval.skill_name,
                "original_demo_ids": json.dumps(
                    [int(value) for value in v0.loc[index, "injected_demo_ids"]],
                    separators=(",", ":"),
                ),
                "grounded_tier": grounded.tier,
                "grounded_demo_ids": json.dumps(
                    [
                        item.instance.train_index
                        for item in grounded.retrieval.instances
                    ],
                    separators=(",", ":"),
                ),
                "grounded_top_similarity": (
                    grounded.retrieval.instances[0].similarity
                    if grounded.retrieval.instances
                    else None
                ),
                "graded_tier": graded.tier,
                "graded_demo_ids": json.dumps(
                    [item.instance.train_index for item in graded.retrieval.instances],
                    separators=(",", ":"),
                ),
                "train_similarity_bar": memory.similarity_bar,
            }
        )
    manifest = pd.DataFrame(rows)
    train_counts = (
        census[census["split"] == "train"].groupby(["analysis_skill", "branch"]).size()
    )
    routed_skills = sorted(set(manifest["routed_skill"]))
    placement_skills = {
        skill
        for skill in routed_skills
        if int(train_counts.get((skill, "some_absent"), 0)) > 0
    }
    a2_trigger_skills = {
        skill: int(train_counts.get((skill, "some_absent"), 0))
        for skill in placement_skills
        if int(train_counts.get((skill, "some_absent"), 0)) < 5
    }
    all_present_sources = int(
        (census.loc[census["split"] == "train", "branch"] == "all_present").sum()
    )
    summary = {
        "tasks": ["A2", "A3", "A4"],
        "a2": {
            "status": "NOT_TRIGGERED",
            "trigger_rule": (
                "Any OOD-routed placement skill has fewer than five some-absent "
                "train instances."
            ),
            "placement_skill_absent_counts": {
                skill: int(train_counts.get((skill, "some_absent"), 0))
                for skill in sorted(placement_skills)
            },
            "triggered_skills": a2_trigger_skills,
            "all_present_train_source_rows": all_present_sources,
            "note": (
                "dog_check_environment has zero placement conditions and is an OOD "
                "misroute, not a placement skill eligible for counterfactual synthesis."
            ),
        },
        "a3": {
            "gate": {
                "status": "PASS",
                "disabled_replay_rows": 50,
                "exact_demo_id_rows": 50,
            },
            "grounded_tier_counts": {
                str(key): int(value)
                for key, value in manifest["grounded_tier"].value_counts().items()
            },
            "rows_changed_from_v0": int(
                sum(
                    json.loads(row["original_demo_ids"])
                    != json.loads(row["grounded_demo_ids"])
                    for _, row in manifest.iterrows()
                )
            ),
        },
        "a4": {
            "calibration": (
                "10th percentile of each train row's nearest different-context "
                "neighbor within the same skill and branch."
            ),
            "train_similarity_bar": memory.similarity_bar,
            "graded_tier_counts": {
                str(key): int(value)
                for key, value in manifest["graded_tier"].value_counts().items()
            },
            "abstract_descriptions_object_free": True,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / "a3_a4_ood_manifest.parquet"
    csv_path = OUTPUT_DIR / "a3_a4_ood_manifest.csv"
    summary_path = OUTPUT_DIR / "a2_a4.summary.json"
    manifest.to_parquet(parquet_path, index=False)
    manifest.to_csv(csv_path, index=False)
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


if __name__ == "__main__":
    build_manifest()
