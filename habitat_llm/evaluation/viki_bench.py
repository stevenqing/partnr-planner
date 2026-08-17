# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import argparse
import base64
import hashlib
import importlib
import importlib.util
import json
import os
import random
import sys
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[3] / "VIKI-R"
VIKI_ACTIONS = {
    "Move",
    "Open",
    "Close",
    "Reach",
    "Grasp",
    "Place",
    "Push",
    "Interact",
}
SCORER_LOCK = Lock()


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if hasattr(value, "tolist"):
        return to_native(value.tolist())
    return value


def get_ground_truth(sample: Dict[str, Any]) -> Any:
    return to_native(sample["reward_model"]["ground_truth"])


def get_image_data_url(sample: Dict[str, Any]) -> str:
    image = to_native(sample["images"])[0]
    image_bytes = image["bytes"]
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = to_native(sample["prompt"])
    image_url = get_image_data_url(sample)
    result = []
    for message in messages:
        content = message["content"]
        if "<image>" in content:
            content = content.replace("<image>", "", 1).strip()
            result.append(
                {
                    "role": message["role"],
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": content},
                    ],
                }
            )
        else:
            result.append({"role": message["role"], "content": content})
    return result


def get_partnr_prompt(sample: Dict[str, Any]) -> List[Any]:
    messages = to_native(sample["prompt"])
    text = "\n\n".join(
        f"{message['role'].upper()}:\n{message['content'].replace('<image>', '').strip()}"
        for message in messages
    )
    return [("image", get_image_data_url(sample)), ("text", text)]


def _split_action_args(arguments: Optional[str]) -> List[str]:
    if arguments is None:
        return []
    return [item.strip() for item in str(arguments).split(",")]


def _map_entity(entity: str, entity_map: Dict[str, str]) -> str:
    if entity not in entity_map:
        raise ValueError(f"No VIKI entity mapping for PARTNR entity {entity!r}")
    return entity_map[entity]


def _partnr_action_to_viki(
    action: str,
    arguments: Optional[str],
    entity_map: Dict[str, str],
    current_position: Optional[str],
) -> List[List[str]]:
    args = _split_action_args(arguments)
    if action in {"Wait", "Done"}:
        return []
    target = _map_entity(args[0], entity_map) if args else ""
    if action == "Navigate":
        return [["Move", target]]
    if action == "Pick":
        move = [] if current_position == target else [["Move", target]]
        return move + [["Reach", target], ["Grasp", target]]
    if action == "Place":
        if len(args) < 3:
            raise ValueError(f"Invalid PARTNR Place arguments: {arguments!r}")
        destination = _map_entity(args[2], entity_map)
        move = [] if current_position == destination else [["Move", destination]]
        return move + [["Place", destination]]
    if action == "Rearrange":
        if len(args) < 3:
            raise ValueError(f"Invalid PARTNR Rearrange arguments: {arguments!r}")
        destination = _map_entity(args[2], entity_map)
        return [
            ["Move", target],
            ["Reach", target],
            ["Grasp", target],
            ["Move", destination],
            ["Place", destination],
        ]
    if action in {"Open", "Close"}:
        move = [] if current_position == target else [["Move", target]]
        return move + [["Reach", target], [action, target]]
    if action in {
        "Explore",
        "Clean",
        "Fill",
        "Pour",
        "PowerOn",
        "PowerOff",
    }:
        move = [] if current_position == target else [["Move", target]]
        return move + [["Interact", target]]
    if action in VIKI_ACTIONS:
        mapped_args = [_map_entity(item, entity_map) for item in args]
        return [[action, *mapped_args]]
    raise ValueError(f"Unsupported PARTNR action for VIKI-L2: {action}")


def convert_partnr_trace_to_viki(
    trace: List[Dict[int, Any]],
    agent_map: Dict[int, str],
    entity_map: Dict[str, str],
    available_actions: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Convert PARTNR planner high-level action timesteps to VIKI-L2 format."""
    result = []
    step_number = 1
    current_positions: Dict[str, str] = {}
    for partnr_step in trace:
        expanded = {}
        for agent_id, action_data in partnr_step.items():
            if agent_id not in agent_map:
                raise ValueError(f"No VIKI agent mapping for PARTNR agent {agent_id}")
            if len(action_data) < 2:
                raise ValueError(f"Invalid PARTNR action tuple: {action_data!r}")
            action, arguments = action_data[:2]
            error = action_data[2] if len(action_data) > 2 else ""
            if error:
                raise ValueError(f"PARTNR action contains an error: {error}")
            viki_agent = agent_map[agent_id]
            expanded[viki_agent] = _partnr_action_to_viki(
                action,
                arguments,
                entity_map,
                current_positions.get(viki_agent),
            )

        substep_count = max((len(actions) for actions in expanded.values()), default=0)
        for substep in range(substep_count):
            actions = {
                agent: agent_actions[substep]
                for agent, agent_actions in expanded.items()
                if substep < len(agent_actions)
            }
            if available_actions is not None:
                for agent, action_data in actions.items():
                    if action_data[0] not in available_actions.get(agent, []):
                        raise ValueError(
                            f"VIKI agent {agent} cannot perform {action_data[0]}"
                        )
            if actions:
                result.append({"step": step_number, "actions": actions})
                for agent, action_data in actions.items():
                    if action_data[0] == "Move":
                        current_positions[agent] = action_data[1]
                step_number += 1
    return result


def _register_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_official_scorer(level: int, benchmark_root: Path):
    reward_root = benchmark_root / "verl/verl/utils/reward_score"
    if not reward_root.is_dir():
        raise FileNotFoundError(f"VIKI reward directory not found: {reward_root}")

    _register_package("verl", reward_root.parents[1])
    _register_package("verl.utils", reward_root.parent)
    _register_package("verl.utils.reward_score", reward_root)
    _register_package("verl.utils.reward_score.utils", reward_root / "utils")
    _register_package("verl.utils.reward_score.utils.eval", reward_root / "utils/eval")
    if level == 2:
        importlib.import_module("verl.utils.reward_score.utils.eval.eval_viki_2")

    module_name = f"verl.utils.reward_score.viki_{level}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, reward_root / f"viki_{level}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load the VIKI-L{level} scorer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def score_response(
    scorer: Any,
    level: int,
    response: str,
    ground_truth: Any,
    seed: int,
) -> Dict[str, float]:
    if level == 2:
        evaluator_globals = scorer.eval_single.__globals__
        with SCORER_LOCK:
            original_random = evaluator_globals["random"]
            try:
                evaluator_globals["random"] = random.Random(seed)
                score = float(scorer.compute_score(response, ground_truth))
                evaluator_globals["random"] = random.Random(seed)
                task_score = float(scorer.acc_reward(response, ground_truth))
            finally:
                evaluator_globals["random"] = original_random
        format_score = float(scorer.format_reward(response))
    elif level == 3:
        score = float(scorer.compute_score(response, ground_truth))
        format_score = float(scorer.format_reward(response))
        task_score = float(scorer.acc_reward(response, ground_truth))
        raw_scorer = importlib.import_module("verl.utils.reward_score.viki_3_re")
        _, rmse, hausdorff, discrete_frechet = raw_scorer.compute_score(
            response, ground_truth
        )
    else:
        score = float(scorer.compute_score(response, ground_truth))
        format_score = float(scorer.format_reward(response))
        task_score = float(scorer.acc_reward(response, ground_truth))
    metrics = {
        "score": score,
        "format_score": format_score,
        "task_score": task_score,
    }
    if level == 3:
        metrics.update(
            {
                "rmse": float(rmse),
                "hausdorff": float(hausdorff),
                "discrete_frechet": float(discrete_frechet),
            }
        )
    return metrics


def oracle_response(level: int, sample: Dict[str, Any]) -> str:
    ground_truth = get_ground_truth(sample)
    if level == 2:
        plan = []
        for step in ground_truth["time_steps"]:
            actions = {
                robot: action
                for robot, action in step["actions"].items()
                if action is not None
            }
            plan.append({"step": step["step"], "actions": actions})
        answer = repr(plan)
    else:
        answer = str(ground_truth)
    return f"<think>oracle wiring check</think><answer>{answer}</answer>"


class EndpointProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _generate(
        self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content or ""

    def generate(self, sample: Dict[str, Any], _: int) -> str:
        return self._generate(get_messages(sample))


class MemorySkillEndpointProvider(EndpointProvider):
    def __init__(
        self,
        benchmark_root: Path,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int,
        memory_top_k: int,
        memory_similarity_threshold: float,
        memory_prediction_max_tokens: int,
        memory_embedding_model: str,
        memory_device: str,
        memory_cache: Optional[Path],
    ) -> None:
        super().__init__(
            base_url,
            api_key,
            model,
            max_tokens,
            temperature,
            max_retries,
        )
        from habitat_llm.evaluation.viki_memory_skill import VikiMemorySkillLibrary

        self.memory = VikiMemorySkillLibrary(
            benchmark_root,
            memory_embedding_model,
            memory_device,
            cache_path=memory_cache,
        )
        self.memory_top_k = memory_top_k
        self.memory_similarity_threshold = memory_similarity_threshold
        self.memory_prediction_max_tokens = memory_prediction_max_tokens
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.metadata_lock = Lock()
        self.skill_prediction_cache: Dict[str, str] = {}
        self.skill_prediction_lock = Lock()

    def generate(self, sample: Dict[str, Any], index: int) -> str:
        from habitat_llm.evaluation.viki_memory_skill import (
            add_memory_to_messages,
            format_memory_prompt,
            get_skill_prediction_messages,
        )

        skill_descriptions = self.memory.executable_skill_descriptions(sample)
        prediction_messages = get_skill_prediction_messages(sample, skill_descriptions)
        prediction_key = json.dumps(
            prediction_messages, sort_keys=True, separators=(",", ":")
        )
        with self.skill_prediction_lock:
            predicted_skill = self.skill_prediction_cache.get(prediction_key)
        prediction_cached = predicted_skill is not None
        if predicted_skill is None:
            predicted_skill = self._generate(
                prediction_messages,
                max_tokens=self.memory_prediction_max_tokens,
            )
            with self.skill_prediction_lock:
                self.skill_prediction_cache[prediction_key] = predicted_skill
        retrieval = self.memory.retrieve(
            sample,
            self.memory_top_k,
            self.memory_similarity_threshold,
            predicted_skill,
        )
        messages = add_memory_to_messages(
            get_messages(sample), format_memory_prompt(retrieval)
        )
        with self.metadata_lock:
            self.metadata[index] = {
                **retrieval.to_metadata(),
                "raw_skill_prediction": predicted_skill,
                "skill_prediction_cached": prediction_cached,
            }
        return self._generate(messages)

    def get_metadata(self, index: int) -> Dict[str, Any]:
        with self.metadata_lock:
            return self.metadata[index]


class PartnrProvider:
    def __init__(
        self,
        llm_name: str,
        engine: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> None:
        from habitat_llm.llm import instantiate_llm

        generation_params: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }
        if engine:
            generation_params["engine"] = engine
        self.llm = instantiate_llm(llm_name, generation_params=generation_params)

    def generate(self, sample: Dict[str, Any], _: int) -> str:
        return self.llm.generate(
            get_partnr_prompt(sample),
            stop="<|viki_generation_end|>",
        )


class PredictionProvider:
    def __init__(self, path: Path) -> None:
        self.responses: Dict[int, str] = {}
        with path.open() as source:
            for line_number, line in enumerate(source):
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    index, response = line_number, item
                else:
                    index = int(item.get("index", line_number))
                    response = item.get("response", item.get("prediction"))
                if not isinstance(response, str):
                    raise ValueError(f"Missing response at {path}:{line_number + 1}")
                self.responses[index] = response

    def generate(self, _: Dict[str, Any], index: int) -> str:
        if index not in self.responses:
            raise KeyError(f"No prediction found for sample index {index}")
        return self.responses[index]


class PartnrTraceProvider:
    def __init__(self, path: Path) -> None:
        self.responses = {}
        with path.open() as source:
            for line_number, line in enumerate(source):
                if not line.strip():
                    continue
                item = json.loads(line)
                index = int(item.get("index", line_number))
                agent_map = {
                    int(agent) if str(agent).isdigit() else agent: viki_agent
                    for agent, viki_agent in item["agent_map"].items()
                }
                trace = [
                    {
                        int(agent) if str(agent).isdigit() else agent: action
                        for agent, action in step.items()
                    }
                    for step in item["trace"]
                ]
                plan = convert_partnr_trace_to_viki(
                    trace,
                    agent_map,
                    item["entity_map"],
                    item.get("available_actions"),
                )
                self.responses[index] = (
                    "<think>PARTNR planner trace converted to VIKI actions.</think>"
                    f"<answer>{plan!r}</answer>"
                )

    def generate(self, _: Dict[str, Any], index: int) -> str:
        if index not in self.responses:
            raise KeyError(f"No PARTNR trace found for sample index {index}")
        return self.responses[index]


class OracleProvider:
    def __init__(self, level: int) -> None:
        self.level = level

    def generate(self, sample: Dict[str, Any], _: int) -> str:
        return oracle_response(self.level, sample)


def create_provider(args: argparse.Namespace):
    if args.provider == "endpoint":
        api_key = os.environ.get(args.api_key_env, "EMPTY")
        return EndpointProvider(
            args.base_url,
            api_key,
            args.model,
            args.max_tokens,
            args.temperature,
            args.max_retries,
        )
    if args.provider == "memory-endpoint":
        if args.level != 2:
            raise ValueError("The memory endpoint provider only supports VIKI-L2")
        api_key = os.environ.get(args.api_key_env, "EMPTY")
        return MemorySkillEndpointProvider(
            args.benchmark_root,
            args.base_url,
            api_key,
            args.model,
            args.max_tokens,
            args.temperature,
            args.max_retries,
            args.memory_top_k,
            args.memory_similarity_threshold,
            args.memory_prediction_max_tokens,
            args.memory_embedding_model,
            args.memory_device,
            args.memory_cache,
        )
    if args.provider == "partnr":
        return PartnrProvider(
            args.llm_name,
            args.engine,
            args.max_tokens,
            args.temperature,
        )
    if args.provider == "predictions":
        if args.predictions is None:
            raise ValueError("--predictions is required for the predictions provider")
        return PredictionProvider(args.predictions)
    if args.provider == "partnr-traces":
        if args.level != 2:
            raise ValueError("The PARTNR trace provider only supports VIKI-L2")
        if args.predictions is None:
            raise ValueError("--predictions is required for the PARTNR trace provider")
        return PartnrTraceProvider(args.predictions)
    if args.provider == "partnr-planner-oracle-state":
        if args.level != 2:
            raise ValueError("The PARTNR planner provider only supports VIKI-L2")
        from habitat_llm.evaluation.viki_partnr_planner import (
            PartnrOracleStatePlannerProvider,
        )

        return PartnrOracleStatePlannerProvider(
            args.benchmark_root,
            args.base_url,
            os.environ.get(args.api_key_env, "EMPTY"),
            args.model,
            args.max_tokens,
            args.temperature,
            args.max_retries,
            args.planner_max_steps,
        )
    return OracleProvider(args.level)


def generate_responses(
    provider: Any,
    samples: List[Dict[str, Any]],
    indices: List[int],
    workers: int,
) -> Iterable[Dict[str, Any]]:
    if workers == 1:
        for index, sample in zip(indices, samples):
            try:
                response = provider.generate(sample, index)
                item = {"index": index, "response": response}
                if hasattr(provider, "get_metadata"):
                    item["provider_metadata"] = provider.get_metadata(index)
                yield item
            except Exception as error:
                yield {"index": index, "error": repr(error), "response": ""}
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(provider.generate, sample, index): index
            for index, sample in zip(indices, samples)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                item = {"index": index, "response": future.result()}
                if hasattr(provider, "get_metadata"):
                    item["provider_metadata"] = provider.get_metadata(index)
                yield item
            except Exception as error:
                yield {"index": index, "error": repr(error), "response": ""}


def get_run_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    predictions = str(args.predictions.resolve()) if args.predictions else None
    endpoint_provider = args.provider in {"endpoint", "memory-endpoint"}
    return {
        "level": args.level,
        "split": args.split,
        "provider": args.provider,
        "benchmark_root": str(args.benchmark_root.resolve()),
        "seed": args.seed,
        "base_url": args.base_url if endpoint_provider else None,
        "model": args.model if endpoint_provider else None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "llm_name": args.llm_name if args.provider == "partnr" else None,
        "engine": args.engine if args.provider == "partnr" else None,
        "predictions": predictions,
        "planner_max_steps": (
            args.planner_max_steps
            if args.provider == "partnr-planner-oracle-state"
            else None
        ),
        "score_attempts": getattr(args, "score_attempts", 1),
        "memory_top_k": (
            args.memory_top_k if args.provider == "memory-endpoint" else None
        ),
        "memory_similarity_threshold": (
            args.memory_similarity_threshold
            if args.provider == "memory-endpoint"
            else None
        ),
        "memory_prediction_max_tokens": (
            args.memory_prediction_max_tokens
            if args.provider == "memory-endpoint"
            else None
        ),
        "memory_embedding_model": (
            args.memory_embedding_model if args.provider == "memory-endpoint" else None
        ),
        "memory_device": (
            args.memory_device if args.provider == "memory-endpoint" else None
        ),
        "memory_cache": (
            str(args.memory_cache.resolve())
            if args.provider == "memory-endpoint" and args.memory_cache
            else None
        ),
    }


def get_run_fingerprint(metadata: Dict[str, Any]) -> str:
    serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = (
        args.benchmark_root
        / "data/VIKI-R/viki"
        / f"VIKI-L{args.level}"
        / f"{args.split}.parquet"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"VIKI-L{args.level} split '{args.split}' not found: {dataset_path}"
        )
    frame = pd.read_parquet(dataset_path)
    stop = (
        len(frame) if args.limit is None else min(len(frame), args.start + args.limit)
    )
    indices = list(range(args.start, stop))
    samples_by_index = {
        index: to_native(frame.iloc[index].to_dict()) for index in indices
    }

    if args.provider in {"partnr", "partnr-planner-oracle-state"} and args.workers != 1:
        raise ValueError("PARTNR providers only support --workers 1")

    scorer = load_official_scorer(args.level, args.benchmark_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    run_metadata = get_run_metadata(args)
    run_fingerprint = get_run_fingerprint(run_metadata)
    metadata_path = args.output.with_suffix(args.output.suffix + ".run.json")
    if args.resume and args.output.exists():
        if not metadata_path.is_file():
            raise ValueError(f"Cannot resume {args.output}: run metadata is missing")
        existing_metadata = json.loads(metadata_path.read_text())
        if existing_metadata != run_metadata:
            raise ValueError(
                f"Cannot resume {args.output}: run configuration does not match"
            )
    else:
        _write_json_atomic(metadata_path, run_metadata)

    records_by_index = {}
    if args.resume and args.output.exists():
        with args.output.open() as existing_output:
            for line in existing_output:
                if line.strip():
                    record = json.loads(line)
                    if record.get("run_fingerprint") != run_fingerprint:
                        raise ValueError(
                            f"Cannot resume {args.output}: record fingerprint does not match"
                        )
                    records_by_index[int(record["index"])] = record

    pending_indices = [
        index
        for index in indices
        if index not in records_by_index or "error" in records_by_index[index]
    ]
    mode = "a" if args.resume else "w"
    with args.output.open(mode) as output:
        if pending_indices:
            provider = create_provider(args)
            generated = generate_responses(
                provider,
                [samples_by_index[index] for index in pending_indices],
                pending_indices,
                args.workers,
            )
        else:
            generated = []
        for generated_item in generated:
            index = generated_item["index"]
            sample = samples_by_index[index]
            response = generated_item["response"]
            metrics = score_response(
                scorer,
                args.level,
                response,
                get_ground_truth(sample),
                args.seed + index,
            )
            attempts = 1
            max_attempts = getattr(args, "score_attempts", 1)
            while (
                args.level == 3
                and metrics["score"] == 0
                and attempts < max_attempts
                and "error" not in generated_item
            ):
                attempts += 1
                response = provider.generate(sample, index)
                metrics = score_response(
                    scorer,
                    args.level,
                    response,
                    get_ground_truth(sample),
                    args.seed + index,
                )
            record = {
                "index": index,
                "level": args.level,
                "split": args.split,
                "response": response,
                "run_fingerprint": run_fingerprint,
                "attempts": attempts,
                **metrics,
            }
            if "error" in generated_item:
                record["error"] = generated_item["error"]
            if "provider_metadata" in generated_item:
                record["provider_metadata"] = generated_item["provider_metadata"]
            output.write(json.dumps(record, ensure_ascii=True) + "\n")
            output.flush()
            records_by_index[index] = record

    records = [
        records_by_index[index] for index in indices if index in records_by_index
    ]
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    with temporary_output.open("w") as output:
        for _, record in sorted(records_by_index.items()):
            output.write(json.dumps(record, ensure_ascii=True) + "\n")
    temporary_output.replace(args.output)

    count = len(records)
    summary = {
        "level": args.level,
        "split": args.split,
        "provider": args.provider,
        "samples": count,
        "mean_score": sum(item["score"] for item in records) / count if count else 0,
        "mean_format_score": (
            sum(item["format_score"] for item in records) / count if count else 0
        ),
        "mean_task_score": (
            sum(item["task_score"] for item in records) / count if count else 0
        ),
        "errors": sum("error" in item for item in records),
        "results": str(args.output),
    }
    if args.level == 3 and count:
        summary.update(
            {
                "mean_rmse": sum(item["rmse"] for item in records) / count,
                "mean_hausdorff": (sum(item["hausdorff"] for item in records) / count),
                "mean_discrete_frechet": (
                    sum(item["discrete_frechet"] for item in records) / count
                ),
            }
        )
    summary_path = args.output.with_suffix(".summary.json")
    _write_json_atomic(summary_path, summary)
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PARTNR on VIKI-Bench")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--provider",
        choices=(
            "partnr",
            "endpoint",
            "memory-endpoint",
            "predictions",
            "partnr-traces",
            "partnr-planner-oracle-state",
            "oracle",
        ),
        default="endpoint",
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--llm-name", default="multimodal_llama")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--planner-max-steps", type=int, default=10)
    parser.add_argument("--score-attempts", type=int, default=1)
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--memory-similarity-threshold", type=float, default=0.3)
    parser.add_argument("--memory-prediction-max-tokens", type=int, default=512)
    parser.add_argument("--memory-embedding-model", default="all-mpnet-base-v2")
    parser.add_argument("--memory-device", default="cpu")
    parser.add_argument(
        "--memory-cache",
        type=Path,
        default=Path("results/viki_l2_memory_skill_all_mpnet_base_v2.npz"),
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path(
            f"results/viki_l{args.level}_{args.split}_{args.provider}.jsonl"
        )
    if args.start < 0 or args.limit is not None and args.limit < 1:
        parser.error("--start must be non-negative and --limit must be positive")
    if args.planner_max_steps < 1:
        parser.error("--planner-max-steps must be positive")
    if args.score_attempts < 1:
        parser.error("--score-attempts must be positive")
    if args.memory_top_k < 1:
        parser.error("--memory-top-k must be positive")
    if not -1 <= args.memory_similarity_threshold <= 1:
        parser.error("--memory-similarity-threshold must be between -1 and 1")
    if args.memory_prediction_max_tokens < 1:
        parser.error("--memory-prediction-max-tokens must be positive")
    return args


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
