# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import random
import re
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habitat_llm.evaluation.viki_bench import (
    PartnrTraceProvider,
    PredictionProvider,
    convert_partnr_trace_to_viki,
    evaluate,
    get_messages,
    load_official_scorer,
    score_response,
)
from habitat_llm.evaluation.viki_memory_skill import (
    VikiMemorySkillLibrary,
    add_memory_to_messages,
    format_memory_prompt,
    get_skill_prediction_messages,
)
from habitat_llm.evaluation.viki_partnr_planner import (
    PartnrOracleStatePlannerProvider,
    VikiEndpointLLM,
    _guided_action_regex,
)

VIKI_ROOT = Path(__file__).resolve().parents[3] / "VIKI-R"


@pytest.mark.skipif(
    not (VIKI_ROOT / "data/VIKI-R/viki").is_dir(),
    reason="VIKI-Bench data is not installed",
)
@pytest.mark.parametrize("level", [1, 2, 3])
def test_viki_oracle_scores_one(level, tmp_path):
    summary = evaluate(
        Namespace(
            level=level,
            split="test",
            provider="oracle",
            benchmark_root=VIKI_ROOT,
            output=tmp_path / f"viki_l{level}.jsonl",
            start=0,
            limit=1,
            workers=1,
            seed=0,
            resume=False,
            base_url="",
            api_key_env="",
            model="",
            max_tokens=1,
            max_retries=0,
            temperature=0.0,
            llm_name="",
            engine=None,
            predictions=None,
        )
    )

    assert summary["samples"] == 1
    assert summary["mean_score"] == 1.0
    assert summary["mean_format_score"] == 1.0
    assert summary["mean_task_score"] == 1.0
    assert summary["errors"] == 0
    if level == 3:
        assert summary["mean_rmse"] == 0.0
        assert summary["mean_hausdorff"] == 0.0
        assert summary["mean_discrete_frechet"] == 0.0


def test_get_messages_embeds_the_official_image():
    sample = {
        "prompt": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "<image>task prompt"},
        ],
        "images": [{"bytes": b"image-bytes"}],
    }

    messages = get_messages(sample)

    assert messages[0] == {"role": "system", "content": "system prompt"}
    assert messages[1]["content"][0]["type"] == "image_url"
    assert messages[1]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert messages[1]["content"][1] == {"type": "text", "text": "task prompt"}


class FakeEmbeddingModel:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, normalize_embeddings=False, **_):
        embeddings = np.asarray([self.vectors[text] for text in texts], dtype=float)
        if normalize_embeddings:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings


def _memory_sample(instruction, task_name=None, actions=None):
    sample = {
        "prompt": [
            {
                "role": "system",
                "content": "Available robot set: {'R1': 'fetch'}\n"
                "Their available operation APIs: {'fetch': ['Move', 'Reach', "
                "'Grasp', 'Place']}",
            },
            {"role": "user", "content": f"<image>{instruction}"},
        ],
        "images": [{"bytes": b"image-bytes"}],
    }
    if task_name is not None:
        time_steps = [
            {"step": index, "actions": {"R1": action}}
            for index, action in enumerate(actions, 1)
        ]
        sample["reward_model"] = {
            "ground_truth": {
                "task_name": task_name,
                "time_steps": time_steps,
            }
        }
    return sample


def test_memory_skill_routes_then_filters_for_executability(monkeypatch):
    relocate_actions = [
        ["Move", "apple"],
        ["Reach", "apple"],
        ["Grasp", "apple"],
        ["Move", "table"],
        ["Place", "table"],
    ]
    inspect_sample = _memory_sample(
        "Check the room.", "inspect", [["Move", "room"], ["Interact", "room"]]
    )
    inspect_sample["prompt"][0]["content"] = (
        "Available robot set: {'R1': 'fetch'}\n"
        "Their available operation APIs: {'fetch': ['Move', 'Interact']}"
    )
    frame = pd.DataFrame(
        [
            _memory_sample("Put the apple on the table.", "relocate", relocate_actions),
            inspect_sample,
        ]
    )
    monkeypatch.setattr(
        "habitat_llm.evaluation.viki_memory_skill.pd.read_parquet",
        lambda *_, **__: frame,
    )
    embedding_model = FakeEmbeddingModel(
        {
            "Put the apple on the table.": [1.0, 0.0],
            "Check the room.": [0.0, 1.0],
            "Put the bowl on the table.": [1.0, 0.0],
            "inspect": [0.0, 1.0],
            "relocate": [1.0, 0.0],
        }
    )
    library = VikiMemorySkillLibrary(
        VIKI_ROOT,
        "unused",
        embedding_model=embedding_model,
    )
    target = _memory_sample("Put the bowl on the table.")

    retrieval = library.retrieve(target, top_k=1, similarity_threshold=0.3)

    assert retrieval.skill_name == "relocate"
    assert retrieval.skill_similarity == pytest.approx(1.0)
    assert [item.instance.train_index for item in retrieval.instances] == [0]
    assert retrieval.to_metadata()["method"] == "memory-as-skill-individual"
    assert "Interact" not in retrieval.instances[0].instance.required_actions

    fallback = library.retrieve(target, top_k=1, similarity_threshold=1.01)
    assert fallback.skill_name is None
    assert fallback.instances == []
    assert fallback.skill_similarity == pytest.approx(1.0)

    predicted = library.retrieve(
        target,
        top_k=1,
        similarity_threshold=0.3,
        predicted_skill="inspect",
    )
    assert predicted.skill_name is None
    assert predicted.instances == []


def test_memory_prompt_preserves_image_and_places_current_task_last(monkeypatch):
    frame = pd.DataFrame(
        [
            _memory_sample(
                "Put the apple on the table.",
                "relocate",
                [["Move", "apple"], ["Place", "table"]],
            )
        ]
    )
    monkeypatch.setattr(
        "habitat_llm.evaluation.viki_memory_skill.pd.read_parquet",
        lambda *_, **__: frame,
    )
    embedding_model = FakeEmbeddingModel(
        {
            "Put the apple on the table.": [1.0],
            "Put the bowl on the table.": [1.0],
            "relocate": [1.0],
        }
    )
    library = VikiMemorySkillLibrary(
        VIKI_ROOT,
        "unused",
        embedding_model=embedding_model,
    )
    target = _memory_sample("Put the bowl on the table.")
    memory_prompt = format_memory_prompt(
        library.retrieve(target, top_k=1, similarity_threshold=0.3)
    )

    messages = add_memory_to_messages(get_messages(target), memory_prompt)

    assert messages[1]["content"][0]["type"] == "image_url"
    text = messages[1]["content"][1]["text"]
    assert "Predicted abstract skill: relocate" in text
    assert text.endswith("Current task:\nPut the bowl on the table.")


def test_skill_prediction_messages_use_only_task_context():
    sample = _memory_sample("Put the bowl on the table.")
    sample["reward_model"] = {"ground_truth": {"secret_answer": "do not expose"}}

    messages = get_skill_prediction_messages(
        sample,
        {
            "relocate": ["Move one object to a target."],
            "inspect": ["Inspect the room."],
        },
    )

    serialized = json.dumps(messages)
    assert isinstance(messages[1]["content"], str)
    assert "Put the bowl on the table." in serialized
    assert "relocate" in serialized
    assert "inspect" in serialized
    assert "secret_answer" not in serialized
    assert "do not expose" not in serialized
    assert "image_url" not in serialized


def test_prediction_provider_uses_explicit_indices(tmp_path):
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps({"index": 4, "response": "tagged response"}) + "\n"
    )

    provider = PredictionProvider(prediction_path)

    assert provider.generate({}, 4) == "tagged response"
    with pytest.raises(KeyError):
        provider.generate({}, 0)


@pytest.mark.skipif(
    not (VIKI_ROOT / "data/VIKI-R/viki").is_dir(),
    reason="VIKI-Bench data is not installed",
)
def test_viki_resume_skips_completed_sample(tmp_path):
    output = tmp_path / "viki_l1.jsonl"
    args = Namespace(
        level=1,
        split="test",
        provider="oracle",
        benchmark_root=VIKI_ROOT,
        output=output,
        start=0,
        limit=1,
        workers=1,
        seed=0,
        resume=False,
        base_url="",
        api_key_env="",
        model="",
        max_tokens=1,
        max_retries=0,
        temperature=0.0,
        llm_name="",
        engine=None,
        predictions=None,
    )
    evaluate(args)
    original = output.read_text()

    args.resume = True
    summary = evaluate(args)

    assert output.read_text() == original
    assert summary["samples"] == 1
    assert summary["mean_score"] == 1.0


@pytest.mark.skipif(
    not (VIKI_ROOT / "data/VIKI-R/viki").is_dir(),
    reason="VIKI-Bench data is not installed",
)
def test_viki_resume_preserves_other_shards_and_rejects_mismatch(tmp_path):
    output = tmp_path / "viki_l1.jsonl"
    args = Namespace(
        level=1,
        split="test",
        provider="oracle",
        benchmark_root=VIKI_ROOT,
        output=output,
        start=0,
        limit=1,
        workers=1,
        seed=0,
        resume=False,
        base_url="",
        api_key_env="",
        model="",
        max_tokens=1,
        max_retries=0,
        temperature=0.0,
        llm_name="",
        engine=None,
        predictions=None,
    )
    evaluate(args)

    args.start = 1
    args.resume = True
    evaluate(args)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["index"] for record in records] == [0, 1]
    assert all(record["run_fingerprint"] for record in records)

    args.seed = 1
    with pytest.raises(ValueError, match="configuration does not match"):
        evaluate(args)


def test_convert_partnr_trace_expands_parallel_actions():
    trace = [
        {
            0: ("Pick", "apple", ""),
            1: ("Navigate", "bowl", ""),
        },
        {
            0: ("Navigate", "table", ""),
            1: ("Wait", "", ""),
        },
        {
            0: ("Place", "apple, on, table, none, none", ""),
            1: ("Done", "", ""),
        },
    ]

    result = convert_partnr_trace_to_viki(
        trace,
        {0: "R1", 1: "R2"},
        {"apple": "apple", "bowl": "bowl", "table": "table"},
    )

    assert result == [
        {
            "step": 1,
            "actions": {"R1": ["Move", "apple"], "R2": ["Move", "bowl"]},
        },
        {"step": 2, "actions": {"R1": ["Reach", "apple"]}},
        {"step": 3, "actions": {"R1": ["Grasp", "apple"]}},
        {"step": 4, "actions": {"R1": ["Move", "table"]}},
        {"step": 5, "actions": {"R1": ["Place", "table"]}},
    ]


def test_convert_partnr_trace_validates_robot_capabilities():
    with pytest.raises(ValueError, match="cannot perform Reach"):
        convert_partnr_trace_to_viki(
            [{0: ("Pick", "apple", "")}],
            {0: "R1"},
            {"apple": "apple"},
            {"R1": ["Move", "Push", "Interact"]},
        )


def test_convert_partnr_trace_requires_explicit_entity_mapping():
    with pytest.raises(ValueError, match="No VIKI entity mapping"):
        convert_partnr_trace_to_viki(
            [{0: ("Navigate", "apple_0", "")}],
            {0: "R1"},
            {},
        )


@pytest.mark.skipif(
    not (VIKI_ROOT / "verl/verl/utils/reward_score").is_dir(),
    reason="VIKI-Bench scorer is not installed",
)
def test_converted_open_satisfies_official_viki_scorer():
    plan = convert_partnr_trace_to_viki(
        [{0: ("Open", "cabinet_39", "")}],
        {0: "R1"},
        {"cabinet_39": "cabinet"},
        {"R1": ["Move", "Reach", "Open"]},
    )
    ground_truth = {
        "robots": {"R1": "fetch"},
        "init_pos": {"cabinet_0": ["room_cabinet"]},
        "goal_constraints": [
            [
                {
                    "is_satisfied": True,
                    "name": "cabinet",
                    "status": {"container_position.isolated": False},
                    "type": "asset",
                }
            ]
        ],
        "temporal_constraints": [],
        "time_steps": [
            {"step": 1, "actions": {"R1": ["Move", "cabinet"]}},
            {"step": 2, "actions": {"R1": ["Reach", "cabinet"]}},
            {"step": 3, "actions": {"R1": ["Open", "cabinet"]}},
        ],
    }
    response = f"<think>open cabinet</think><answer>{plan!r}</answer>"
    scorer = load_official_scorer(2, VIKI_ROOT)
    random_state = random.getstate()

    metrics = score_response(scorer, 2, response, ground_truth, seed=0)

    assert plan == [
        {"step": 1, "actions": {"R1": ["Move", "cabinet"]}},
        {"step": 2, "actions": {"R1": ["Reach", "cabinet"]}},
        {"step": 3, "actions": {"R1": ["Open", "cabinet"]}},
    ]
    assert metrics == {"score": 1.0, "format_score": 1.0, "task_score": 1.0}
    assert random.getstate() == random_state


def test_oracle_state_provider_runs_partnr_planner(monkeypatch):
    outputs = iter(
        [
            "Thought: approach the wine\nAgent_0_Action: Navigate[wine_0]",
            "Thought: take the wine\nAgent_0_Action: Pick[wine_0]",
            "Thought: approach the target\n"
            "Agent_0_Action: Navigate[kitchen work area_0]",
            "Thought: place the wine\n"
            "Agent_0_Action: Place[wine_0, on, kitchen work area_0, None, None]",
            "Final Thought: the requested placement is complete\nDone[]",
        ]
    )
    monkeypatch.setattr(
        VikiEndpointLLM,
        "generate",
        lambda self, prompt, stop=None, max_length=None, generation_args=None: next(
            outputs
        ),
    )
    sample = {
        "prompt": [
            {
                "role": "system",
                "content": "Available robot set: {'R1': 'fetch'}\n"
                "Their available operation APIs: {'fetch': ['Move', 'Reach', "
                "'Grasp', 'Place', 'Open', 'Close', 'Interact']}",
            },
            {
                "role": "user",
                "content": "<image>Place the wine on the kitchen work area.",
            },
        ],
        "reward_model": {
            "ground_truth": {
                "init_pos": {
                    "R1": None,
                    "wine_0": ["kitchen island area"],
                }
            }
        },
    }
    provider = PartnrOracleStatePlannerProvider(
        VIKI_ROOT,
        "http://127.0.0.1:8000/v1",
        "EMPTY",
        "local-model",
        250,
        0.0,
        0,
        10,
        {"cabinet", "kitchen island area", "kitchen work area"},
    )

    response = provider.generate(sample, 0)

    assert "['Move', 'wine']" in response
    assert "['Reach', 'wine']" in response
    assert "['Grasp', 'wine']" in response
    assert "['Move', 'kitchen work area']" in response
    assert "['Place', 'kitchen work area']" in response
    metadata = provider.get_metadata(0)
    assert metadata["track"] == "oracle-state-planner-ablation"
    assert metadata["planner_stopped"] is True
    assert metadata["termination_reason"] == "model_done"
    assert "cabinet_0" not in metadata["entity_map"]
    assert metadata["entity_map"]["room_0"] == "room"
    assert "wine_0 is held by this agent" in metadata["partnr_trace"]
    assert "wine_0 is now on kitchen work area_0" in metadata["partnr_trace"]
    assert len(metadata["partnr_actions"]) == 4


def test_oracle_state_planner_guided_regex_uses_prompt_tools():
    prompt = """World model:
Furniture:
room_0: kitchen island area_0, kitchen work area_0

The following furnitures have a faucet:
Objects:
wine_0: kitchen island area_0 in room_0

Active agents and tools:
Agent ID: 0
- Navigate: go somewhere
- Pick: take an object
Agent ID: 1
- Wait: remain idle
"""

    pattern = _guided_action_regex(prompt)

    assert re.fullmatch(
        pattern,
        "Thought: collect the wine\n"
        "Agent_0_Action: Navigate[wine_0]\nAgent_1_Action: Wait[]",
    )
    assert re.fullmatch(pattern, "Final Thought: task complete\nDone[]")
    assert not re.fullmatch(pattern, "Agent_0_Action: Fly[wine_0]")
    assert not re.fullmatch(
        pattern,
        "Thought: bad arguments\n"
        "Agent_0_Action: Navigate[wine_0, None]\nAgent_1_Action: Wait[]",
    )


def test_partnr_trace_provider_converts_json_keys(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "index": 3,
                "agent_map": {"0": "R1"},
                "entity_map": {"apple_0": "apple"},
                "trace": [{"0": ["Navigate", "apple_0", ""]}],
                "available_actions": {"R1": ["Move"]},
            }
        )
        + "\n"
    )

    response = PartnrTraceProvider(trace_path).generate({}, 3)

    assert response.endswith(
        "<answer>[{'step': 1, 'actions': {'R1': ['Move', 'apple']}}]</answer>"
    )
