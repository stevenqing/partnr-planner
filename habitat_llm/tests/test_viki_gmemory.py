from typing import Any, Dict, List

import numpy as np

from habitat_llm.evaluation.viki_gmemory import (
    GMemoryState,
    parse_numbered_list,
    parse_relevance_score,
    parse_rule_operations,
    render_retrieval_prompt,
    reward_retrieved_insights,
    update_rules,
)


def test_rule_parser_and_updates_follow_author_contract() -> None:
    insights: List[Dict[str, Any]] = []
    operations = parse_rule_operations(
        "ADD: Verify capabilities before assigning actions, because unavailable APIs fail."
    )
    update_rules(insights, ["task-a"], operations)
    assert insights == [
        {
            "rule": "Verify capabilities before assigning actions, because unavailable APIs fail.",
            "score": 2,
            "positive_correlation_tasks": ["task-a"],
            "negative_correlation_tasks": [],
        }
    ]
    update_rules(
        insights,
        ["task-b"],
        [("AGREE 1", insights[0]["rule"])],
        local_insight_ids=[0],
    )
    assert insights[0]["score"] == 3
    assert insights[0]["positive_correlation_tasks"] == ["task-a", "task-b"]
    reward_retrieved_insights(insights, [insights[0]["rule"]], success=False)
    assert insights[0]["score"] == 1


def test_query_graph_retrieval_and_hash_are_deterministic() -> None:
    state = GMemoryState()
    state.add_record(
        {
            "memory_id": "train:0",
            "task_main": "place fruit",
            "task_description": "place fruit",
            "trajectory": "plan zero",
            "key_steps": "move and place",
            "label": True,
        },
        np.asarray([1.0, 0.0], dtype=np.float32),
    )
    initial_hash = state.hierarchy_sha256()
    state.add_record(
        {
            "memory_id": "train:1",
            "task_main": "put fruit away",
            "task_description": "put fruit away",
            "trajectory": "plan one",
            "key_steps": "grasp and place",
            "label": True,
        },
        np.asarray([0.9, 0.1], dtype=np.float32),
    )
    state.add_record(
        {
            "memory_id": "train:2",
            "task_main": "arrange fruit",
            "task_description": "arrange fruit",
            "trajectory": "plan two",
            "key_steps": "organize and finish",
            "label": True,
        },
        np.asarray([0.8, 0.2], dtype=np.float32),
    )
    assert state.nearest_record_indices(
        np.asarray([1.0, 0.0]), count=2, label=True
    ) == [0, 1]
    assert state.raw_success_candidates(np.asarray([1.0, 0.0]), 2) == [0, 1]
    assert state.hierarchy_sha256() != initial_hash
    assert state.hierarchy_sha256() == state.hierarchy_sha256()
    restored = GMemoryState.from_canonical_state(
        state.canonical_state(), state.embeddings
    )
    assert list(restored.graph) == [
        "place fruit",
        "put fruit away",
        "arrange fruit",
    ]
    assert restored.hierarchy_sha256() == state.hierarchy_sha256()


def test_prompt_and_numbered_list_keep_retrieval_sections() -> None:
    record = {
        "task_description": "Task instruction: clear the counter",
        "key_steps": "Move, grasp, and place.",
        "trajectory": "R1 executes the plan.",
    }
    prompt = render_retrieval_prompt(
        [record], ["Respect robot capabilities, because invalid APIs cannot execute."]
    )
    assert "Your Own Past Successes" in prompt
    assert "Key Insights" in prompt
    assert "activated robot APIs" in prompt
    assert parse_numbered_list("1. First rule\n2. Second rule") == [
        "First rule",
        "Second rule",
    ]
    assert parse_relevance_score("Score: 9") == 9
    assert parse_relevance_score("No score") == 0
