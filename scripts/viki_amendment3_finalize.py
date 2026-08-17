#!/usr/bin/env python3

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/viki_memory_experiments/amendment3"
AMENDMENT1_DIR = ROOT / "results/viki_memory_experiments/amendment1"
AMENDMENT2_DIR = ROOT / "results/viki_memory_experiments/amendment2"
REPORT_PATH = ROOT / "VIKI_RESULTS_4.md"
FINAL_JSON_PATH = OUTPUT_DIR / "f2_final_results.json"
FINAL_CSV_PATH = OUTPUT_DIR / "f2_final_tables.csv"
FIGURE_PATH = OUTPUT_DIR / "f2_three_regime_curve.png"
SEED = 20260814
BOOTSTRAP_SAMPLES = 100000
ALPHA = 0.05


class GateFailure(RuntimeError):
    pass


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if hasattr(value, "tolist"):
        return native(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(native(value), indent=2) + "\n")
    temporary.replace(path)


def clopper_pearson(successes: int, samples: int) -> Tuple[float, float]:
    if samples < 1 or not 0 <= successes <= samples:
        raise ValueError((successes, samples))
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(ALPHA / 2, successes, samples - successes + 1))
    )
    upper = (
        1.0
        if successes == samples
        else float(beta.ppf(1 - ALPHA / 2, successes + 1, samples - successes))
    )
    return lower, upper


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    return min(
        1.0,
        2
        * sum(math.comb(discordant, value) for value in range(tail + 1))
        / (2**discordant),
    )


def paired_bootstrap_interval(
    samples: int,
    both_fail: int,
    left_only: int,
    right_only: int,
    both_success: int,
    seed_offset: int,
) -> Tuple[float, float]:
    counts = np.asarray([both_fail, left_only, right_only, both_success])
    if int(counts.sum()) != samples:
        raise ValueError((samples, counts.tolist()))
    probabilities = counts / samples
    rng = np.random.default_rng(SEED + seed_offset)
    draws = rng.multinomial(samples, probabilities, size=BOOTSTRAP_SAMPLES)
    deltas = (draws[:, 2] - draws[:, 1]) / samples
    lower, upper = np.quantile(deltas, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lower), float(upper)


def paired_result_from_counts(
    label: str,
    samples: int,
    left_successes: int,
    right_successes: int,
    left_only: int,
    right_only: int,
    seed_offset: int,
    left_format: float = 1.0,
    right_format: float = 1.0,
) -> Dict[str, Any]:
    both_success = left_successes - left_only
    both_fail = samples - both_success - left_only - right_only
    if right_successes != both_success + right_only:
        raise ValueError(label)
    left_interval = clopper_pearson(left_successes, samples)
    right_interval = clopper_pearson(right_successes, samples)
    delta_interval = paired_bootstrap_interval(
        samples,
        both_fail,
        left_only,
        right_only,
        both_success,
        seed_offset,
    )
    return {
        "label": label,
        "samples": samples,
        "zero_shot_successes": left_successes,
        "memory_successes": right_successes,
        "zero_shot_accuracy": left_successes / samples,
        "memory_accuracy": right_successes / samples,
        "zero_shot_interval": left_interval,
        "memory_interval": right_interval,
        "absolute_delta": (right_successes - left_successes) / samples,
        "delta_interval": delta_interval,
        "success_to_fail": left_only,
        "fail_to_success": right_only,
        "discordant_pairs": left_only + right_only,
        "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
        "zero_shot_format_compliance": left_format,
        "memory_format_compliance": right_format,
    }


def paired_result_from_frame(
    label: str,
    frame: pd.DataFrame,
    left: str,
    right: str,
    seed_offset: int,
) -> Dict[str, Any]:
    left_values = frame[f"{left}_task_score"].to_numpy() == 1
    right_values = frame[f"{right}_task_score"].to_numpy() == 1
    left_only = int((left_values & ~right_values).sum())
    right_only = int((~left_values & right_values).sum())
    return paired_result_from_counts(
        label,
        len(frame),
        int(left_values.sum()),
        int(right_values.sum()),
        left_only,
        right_only,
        seed_offset,
        float((frame[f"{left}_format_score"] == 1).mean()),
        float((frame[f"{right}_format_score"] == 1).mean()),
    )


def paired_result_from_boolean_frame(
    label: str,
    frame: pd.DataFrame,
    left: str,
    right: str,
    seed_offset: int,
    left_format: float,
    right_format: float,
) -> Dict[str, Any]:
    left_values = frame[left].astype(bool).to_numpy()
    right_values = frame[right].astype(bool).to_numpy()
    left_only = int((left_values & ~right_values).sum())
    right_only = int((~left_values & right_values).sum())
    return paired_result_from_counts(
        label,
        len(frame),
        int(left_values.sum()),
        int(right_values.sum()),
        left_only,
        right_only,
        seed_offset,
        left_format,
        right_format,
    )


def require_f2() -> Dict[str, Dict[str, Any]]:
    names = (
        "f2_ood",
        "f2_id",
        "f2_cprime_instance",
        "f2_cprime_productivity",
    )
    summaries = {}
    missing = []
    for name in names:
        path = OUTPUT_DIR / f"{name}.summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        summary = json.loads(path.read_text())
        if summary.get("status") != "PASS":
            raise GateFailure(f"F2 artifact did not pass: {path}")
        summaries[name] = summary
    if missing:
        raise GateFailure("F2 is incomplete: " + ", ".join(missing))
    return summaries


def fingerprint(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_raw_artifact(
    stem: str,
    expected_indices: Sequence[int],
    required_arms: Sequence[str],
) -> None:
    run_path = OUTPUT_DIR / f"{stem}.jsonl.run.json"
    jsonl_path = OUTPUT_DIR / f"{stem}.jsonl"
    if not run_path.is_file() or not jsonl_path.is_file():
        raise GateFailure(f"Missing raw F2 artifact: {stem}")
    run_hash = fingerprint(json.loads(run_path.read_text()))
    records = []
    with jsonl_path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise GateFailure(
                    f"Malformed raw F2 JSONL: {stem}:{line_number}"
                ) from error
    indices = [int(record["index"]) for record in records]
    if indices != list(expected_indices) or len(indices) != len(set(indices)):
        raise GateFailure(f"Raw F2 ordering/index mismatch: {stem}")
    for record in records:
        index = int(record["index"])
        if record.get("run_fingerprint") != run_hash:
            raise GateFailure(f"Raw F2 fingerprint mismatch: {stem}:{index}")
        if record.get("endpoint_error"):
            raise GateFailure(f"Raw F2 endpoint error: {stem}:{index}")
        arms = record.get("arms", {})
        if set(required_arms) - set(arms):
            raise GateFailure(f"Raw F2 arm mismatch: {stem}:{index}")
        for arm in required_arms:
            if arms[arm].get("task_score") not in (0, 1, 0.0, 1.0):
                raise GateFailure(f"Raw F2 task score mismatch: {stem}:{index}")
            if arms[arm].get("format_score") not in (0, 1, 0.0, 1.0):
                raise GateFailure(f"Raw F2 format score mismatch: {stem}:{index}")


def validate_binary_columns(
    frame: pd.DataFrame, columns: Sequence[str], label: str
) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
            raise GateFailure(f"Non-binary or non-finite {label} column: {column}")


def validate_summary_agreement(
    name: str, frame: pd.DataFrame, summary: Mapping[str, Any]
) -> None:
    if name in ("ood", "id"):
        overall = summary["overall"]
        checks = {
            "zero_shot_successes": int((frame["zero_shot_task_score"] == 1).sum()),
            "skill_memory_successes": int(
                (frame["skill_memory_task_score"] == 1).sum()
            ),
        }
        if overall["zero_shot_successes"] != checks["zero_shot_successes"]:
            raise GateFailure(f"F2 summary disagreement: {name}:zero_shot")
        if overall["skill_memory_successes"] != checks["skill_memory_successes"]:
            raise GateFailure(f"F2 summary disagreement: {name}:skill_memory")
        if summary["samples"] != len(frame) or summary["endpoint_errors"] != 0:
            raise GateFailure(f"F2 summary count disagreement: {name}")
        if summary["token_budget"]["truncated_rows"] != 0:
            raise GateFailure(f"F2 summary reports truncation: {name}")
        return
    for arm in ("zero_shot", "skill_memory", "flat_memory"):
        successes = int((frame[f"{arm}_task_score"] == 1).sum())
        formats = int((frame[f"{arm}_format_score"] == 1).sum())
        if summary["arms"][arm]["successes"] != successes:
            raise GateFailure(f"F2 C-prime summary disagreement: {name}:{arm}")
        if summary["arms"][arm]["format_successes"] != formats:
            raise GateFailure(f"F2 C-prime format disagreement: {name}:{arm}")
    if summary["flat_token_match"]["violations"] != 0:
        raise GateFailure(f"F2 C-prime token-match violation: {name}")


def validate_f2_frames(
    summaries: Mapping[str, Mapping[str, Any]]
) -> Dict[str, pd.DataFrame]:
    paths = {
        "ood": OUTPUT_DIR / "f2_ood.parquet",
        "id": OUTPUT_DIR / "f2_id.parquet",
        "instance": OUTPUT_DIR / "f2_cprime_instance.parquet",
        "productivity": OUTPUT_DIR / "f2_cprime_productivity.parquet",
    }
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    expected = {"ood": 1218, "id": 300, "instance": 400, "productivity": 400}
    for name, frame in frames.items():
        if len(frame) != expected[name] or frame["index"].duplicated().any():
            raise GateFailure(f"Invalid final F2 frame: {name}")
    if list(frames["ood"]["index"]) != list(range(1218)):
        raise GateFailure("F2 OOD indices are not exactly 0..1217")
    id_manifest = pd.read_parquet(AMENDMENT1_DIR / "a5_id_safety_manifest.parquet")
    if set(frames["id"]["index"]) != set(id_manifest["index"]):
        raise GateFailure("F2 ID indices do not match the frozen 300-row manifest")
    cprime_manifest = pd.read_parquet(OUTPUT_DIR / "f2_cprime_manifest.parquet")
    for channel in ("instance", "productivity"):
        expected_indices = set(
            cprime_manifest[cprime_manifest["channel"] == channel]["index"]
        )
        if set(frames[channel]["index"]) != expected_indices:
            raise GateFailure(f"F2 C-prime indices do not match: {channel}")
        if (
            not np.isfinite(
                frames[channel]["flat_token_relative_difference"].to_numpy(dtype=float)
            ).all()
            or frames[channel]["flat_token_relative_difference"].max() > 0.05
        ):
            raise GateFailure(f"F2 flat token-match violation: {channel}")
    paired_columns = (
        "zero_shot_task_score",
        "zero_shot_format_score",
        "skill_memory_task_score",
        "skill_memory_format_score",
    )
    cprime_columns = paired_columns + (
        "flat_memory_task_score",
        "flat_memory_format_score",
    )
    validate_binary_columns(frames["ood"], paired_columns, "ood")
    validate_binary_columns(frames["id"], paired_columns, "id")
    validate_binary_columns(frames["instance"], cprime_columns, "instance")
    validate_binary_columns(frames["productivity"], cprime_columns, "productivity")
    validate_raw_artifact(
        "f2_ood", list(frames["ood"]["index"]), ("zero_shot", "skill_memory")
    )
    validate_raw_artifact(
        "f2_id", list(frames["id"]["index"]), ("zero_shot", "skill_memory")
    )
    for channel in ("instance", "productivity"):
        validate_raw_artifact(
            f"f2_cprime_{channel}",
            list(frames[channel]["index"]),
            ("zero_shot", "skill_memory", "flat_memory"),
        )
    validate_summary_agreement("ood", frames["ood"], summaries["f2_ood"])
    validate_summary_agreement("id", frames["id"], summaries["f2_id"])
    validate_summary_agreement(
        "instance", frames["instance"], summaries["f2_cprime_instance"]
    )
    validate_summary_agreement(
        "productivity",
        frames["productivity"],
        summaries["f2_cprime_productivity"],
    )
    return frames


def build_regimes(
    frames: Mapping[str, pd.DataFrame]
) -> Dict[str, List[Dict[str, Any]]]:
    stock_id = pd.read_parquet(AMENDMENT2_DIR / "b1_id.parquet")
    stock_ood = pd.read_parquet(AMENDMENT2_DIR / "b1_ood.parquet")
    rl_id_frame = pd.read_parquet(AMENDMENT1_DIR / "a5_id_safety.parquet")
    rl_ood_summary = json.loads(
        (ROOT / "results/viki_memory_skill_7b_l2_ood.summary.json").read_text()
    )
    if set(stock_id["index"]) != set(frames["id"]["index"]):
        raise GateFailure("Stock and 72B ID windows differ")
    if set(stock_ood["index"]) != set(frames["ood"]["index"]):
        raise GateFailure("Stock and 72B OOD windows differ")
    id_regimes = [
        paired_result_from_frame(
            "Stock 7B segment memory", stock_id, "base", "memory", 1
        ),
        paired_result_from_frame(
            "Local 72B segment memory",
            frames["id"],
            "zero_shot",
            "skill_memory",
            2,
        ),
        paired_result_from_boolean_frame(
            "RL 7B graded memory",
            rl_id_frame,
            "baseline_success",
            "graded_success",
            3,
            1.0,
            float((rl_id_frame["graded_format_score"] == 1).mean()),
        ),
    ]
    ood_regimes = [
        paired_result_from_frame(
            "Stock 7B segment memory", stock_ood, "base", "memory", 4
        ),
        paired_result_from_frame(
            "Local 72B segment memory",
            frames["ood"],
            "zero_shot",
            "skill_memory",
            5,
        ),
        paired_result_from_counts(
            "RL 7B legacy skill memory",
            int(rl_ood_summary["samples"]),
            int(rl_ood_summary["baseline_successes"]),
            int(rl_ood_summary["memory_successes"]),
            int(rl_ood_summary["success_to_fail"]),
            int(rl_ood_summary["fail_to_success"]),
            6,
            1.0,
            float(rl_ood_summary["mean_format_score"]),
        ),
    ]
    return {"id": id_regimes, "ood": ood_regimes}


def build_family_results(frame: pd.DataFrame, seed_offset: int) -> List[Dict[str, Any]]:
    rows = []
    for position, (task, group) in enumerate(frame.groupby("task_name", sort=True)):
        result = paired_result_from_frame(
            str(task),
            group,
            "zero_shot",
            "skill_memory",
            seed_offset + position,
        )
        result["task_name"] = str(task)
        rows.append(result)
    return rows


def cprime_result(
    channel: str, frame: pd.DataFrame, seed_offset: int
) -> Dict[str, Any]:
    arms = {}
    for arm in ("zero_shot", "skill_memory", "flat_memory"):
        successes = int((frame[f"{arm}_task_score"] == 1).sum())
        arms[arm] = {
            "samples": len(frame),
            "successes": successes,
            "accuracy": successes / len(frame),
            "interval": clopper_pearson(successes, len(frame)),
            "format_compliance": float((frame[f"{arm}_format_score"] == 1).mean()),
        }
    comparisons = {
        "zero_shot_to_skill": paired_result_from_frame(
            f"{channel}: zero to skill",
            frame,
            "zero_shot",
            "skill_memory",
            seed_offset,
        ),
        "zero_shot_to_flat": paired_result_from_frame(
            f"{channel}: zero to flat",
            frame,
            "zero_shot",
            "flat_memory",
            seed_offset + 1,
        ),
        "flat_to_skill": paired_result_from_frame(
            f"{channel}: flat to skill",
            frame,
            "flat_memory",
            "skill_memory",
            seed_offset + 2,
        ),
    }
    return {
        "channel": channel,
        "arms": arms,
        "comparisons": comparisons,
        "flat_token_match": {
            "maximum_relative_difference": float(
                frame["flat_token_relative_difference"].max()
            ),
            "mean_relative_difference": float(
                frame["flat_token_relative_difference"].mean()
            ),
            "truncated_prompts": int(frame["flat_truncated"].sum()),
        },
    }


def flatten_tables(results: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for split, regimes in results["regimes"].items():
        for item in regimes:
            rows.append({"table": "regime", "split": split, **item})
    for item in results["ood_families"]:
        rows.append({"table": "ood_family", "split": "ood", **item})
    for channel, value in results["cprime"].items():
        for comparison, item in value["comparisons"].items():
            rows.append(
                {
                    "table": "cprime_comparison",
                    "split": channel,
                    "comparison": comparison,
                    **item,
                }
            )
    return pd.DataFrame(rows)


def plot_regimes(regimes: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"zero": "#335c67", "memory": "#9e2a2b"}
    for axis, split in zip(axes, ("id", "ood")):
        items = regimes[split]
        x_values = np.arange(len(items))
        zero = np.asarray([item["zero_shot_accuracy"] for item in items])
        memory = np.asarray([item["memory_accuracy"] for item in items])
        zero_intervals = np.asarray([item["zero_shot_interval"] for item in items])
        memory_intervals = np.asarray([item["memory_interval"] for item in items])
        axis.errorbar(
            x_values - 0.08,
            zero,
            yerr=np.vstack([zero - zero_intervals[:, 0], zero_intervals[:, 1] - zero]),
            fmt="o-",
            color=colors["zero"],
            capsize=4,
            label="Zero-shot",
        )
        axis.errorbar(
            x_values + 0.08,
            memory,
            yerr=np.vstack(
                [memory - memory_intervals[:, 0], memory_intervals[:, 1] - memory]
            ),
            fmt="s-",
            color=colors["memory"],
            capsize=4,
            label="Memory",
        )
        for position, item in enumerate(items):
            lower, upper = item["delta_interval"]
            axis.annotate(
                f"d={item['absolute_delta'] * 100:+.1f} pp\n"
                f"[{lower * 100:+.1f}, {upper * 100:+.1f}]",
                (position, max(zero[position], memory[position])),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_title(f"VIKI L2 {split.upper()}")
        axis.set_xticks(x_values, ["Stock 7B", "Local 72B", "RL 7B"])
        axis.set_ylabel("Official task success")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left")
    figure.suptitle("Three measured regimes (95% Clopper-Pearson arm intervals)")
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def comparison_row(item: Mapping[str, Any]) -> str:
    lower, upper = item["delta_interval"]
    return (
        f"| {item['label']} | {item['samples']} | "
        f"{item['zero_shot_successes']}/{item['samples']} "
        f"({percent(item['zero_shot_accuracy'])}) | "
        f"{item['memory_successes']}/{item['samples']} "
        f"({percent(item['memory_accuracy'])}) | "
        f"{item['absolute_delta'] * 100:+.2f} pp "
        f"[{lower * 100:+.2f}, {upper * 100:+.2f}] | "
        f"{item['fail_to_success']} | {item['success_to_fail']} | "
        f"{item['mcnemar_exact_p']:.6g} |"
    )


def build_report(results: Mapping[str, Any]) -> str:
    ood_window = results["regimes"]["ood"][1]
    id_window = results["regimes"]["id"][1]
    prediction_one = ood_window["absolute_delta"] > 0
    lines = [
        "# VIKI Amendment 3: Local 72B Window Round",
        "",
        "## Status",
        "",
        "This round is a post-F1 local-model amendment. The preregistered F1 pick "
        "selected Gemini-2.5-Flash after no open model cleared every threshold; "
        "the user then explicitly authorized Qwen2.5-VL-72B for F2. The original "
        "pick artifact remains unchanged.",
        "",
        (
            "The primary OOD prediction "
            + ("held." if prediction_one else "failed.")
            + f" Zero-shot was {ood_window['zero_shot_successes']}/"
            f"{ood_window['samples']} and segment memory was "
            f"{ood_window['memory_successes']}/{ood_window['samples']} "
            f"(delta {ood_window['absolute_delta'] * 100:+.2f} pp, "
            f"exact McNemar p={ood_window['mcnemar_exact_p']:.6g})."
        ),
        "",
        "## Paired Results",
        "",
        "| Regime | N | Zero-shot | Memory | Delta [paired bootstrap 95%] | F->S | S->F | Exact p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("id", "ood"):
        lines.append(f"| **{split.upper()}** | | | | | | | |")
        lines.extend(comparison_row(item) for item in results["regimes"][split])
    lines.extend(
        [
            "",
            "The three regimes are descriptive rather than treatment-controlled: "
            "the stock and local-72B arms use the Amendment 2 segment bank, while "
            "the RL checkpoint uses earlier branch/legacy skill memories and a "
            "different context-budget policy.",
            "",
            "## Local 72B OOD Families",
            "",
            "| Family | N | Zero-shot | Memory | Delta [95%] | F->S | S->F | Exact p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(comparison_row(item) for item in results["ood_families"])
    lines.extend(
        [
            "",
            "## C-prime",
            "",
            "Systematicity remains excluded as grammar-impossible. Instance and "
            "productivity each use 400 frozen rows; flat prompts are token-matched "
            "to skill-memory prompts within 5%.",
            "",
            "| Channel | Arm | Success | Format |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for channel, value in results["cprime"].items():
        for arm, arm_value in value["arms"].items():
            lines.append(
                f"| {channel} | {arm} | {arm_value['successes']}/"
                f"{arm_value['samples']} ({percent(arm_value['accuracy'])}) | "
                f"{percent(arm_value['format_compliance'])} |"
            )
    lines.extend(
        [
            "",
            "| Channel | Comparison | Delta [paired bootstrap 95%] | F->S | S->F | Exact p |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for channel, value in results["cprime"].items():
        for comparison, item in value["comparisons"].items():
            lower, upper = item["delta_interval"]
            lines.append(
                f"| {channel} | {comparison} | "
                f"{item['absolute_delta'] * 100:+.2f} pp "
                f"[{lower * 100:+.2f}, {upper * 100:+.2f}] | "
                f"{item['fail_to_success']} | {item['success_to_fail']} | "
                f"{item['mcnemar_exact_p']:.6g} |"
            )
        token_match = value["flat_token_match"]
        lines.append(
            f"| {channel} | flat token match | max "
            f"{token_match['maximum_relative_difference'] * 100:.2f}%; "
            f"prefix-trimmed {token_match['truncated_prompts']} | | | |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"On local-72B ID, zero-shot was {id_window['zero_shot_successes']}/"
            f"{id_window['samples']} and memory was {id_window['memory_successes']}/"
            f"{id_window['samples']}. No skill-memory instance was removed for the "
            "16K budget. Detailed retained-instance and token-headroom distributions "
            "are stored in `f2_ood.summary.json`, `f2_id.summary.json`, and both "
            "C-prime summaries.",
            "",
            f"![Three measured regimes](results/viki_memory_experiments/amendment3/{FIGURE_PATH.name})",
            "",
            "## Artifacts",
            "",
            "- `results/viki_memory_experiments/amendment3/f2_local_override.json`",
            "- `results/viki_memory_experiments/amendment3/f2_final_results.json`",
            "- `results/viki_memory_experiments/amendment3/f2_final_tables.csv`",
            "- `results/viki_memory_experiments/amendment3/f2_three_regime_curve.png`",
            "",
        ]
    )
    return "\n".join(lines)


def finalize() -> Dict[str, Any]:
    summaries = require_f2()
    frames = validate_f2_frames(summaries)
    regimes = build_regimes(frames)
    ood_families = build_family_results(frames["ood"], 100)
    cprime = {
        "instance": cprime_result("instance", frames["instance"], 200),
        "productivity": cprime_result("productivity", frames["productivity"], 300),
    }
    results = {
        "task": "Amendment3_F2_final",
        "status": "PASS",
        "methodological_status": (
            "Post-F1 user-authorized local-model amendment; original F1 pick retained"
        ),
        "intervals": {
            "arm": "two-sided 95% Clopper-Pearson",
            "paired_delta": (
                "paired nonparametric multinomial bootstrap, 100000 draws, "
                "seed 20260814"
            ),
        },
        "regimes": regimes,
        "ood_families": ood_families,
        "cprime": cprime,
        "retained_instances": {
            "ood": summaries["f2_ood"]["retained_instances"],
            "id": summaries["f2_id"]["retained_instances"],
            "instance": summaries["f2_cprime_instance"]["retained_instances"],
            "productivity": summaries["f2_cprime_productivity"]["retained_instances"],
        },
        "predictions": {
            "memory_ood_exceeds_zero_shot": (regimes["ood"][1]["absolute_delta"] > 0),
            "bowl_missing_moves": next(
                item
                for item in ood_families
                if item["task_name"] == "bring_bowl_to_table_plate_already_there"
            )["absolute_delta"]
            != 0,
            "plate_missing_moves": next(
                item
                for item in ood_families
                if item["task_name"] == "bring_plate_to_table_bowl_already_there"
            )["absolute_delta"]
            != 0,
            "productivity_skill_over_flat_at_least_instance": (
                cprime["productivity"]["comparisons"]["flat_to_skill"]["absolute_delta"]
                >= cprime["instance"]["comparisons"]["flat_to_skill"]["absolute_delta"]
            ),
        },
    }
    table = flatten_tables(results)
    atomic_json(FINAL_JSON_PATH, results)
    table.to_csv(FINAL_CSV_PATH, index=False)
    plot_regimes(regimes)
    temporary_report = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
    temporary_report.write_text(build_report(results))
    temporary_report.replace(REPORT_PATH)
    return results


def main() -> None:
    try:
        results = finalize()
    except GateFailure as error:
        print(f"GATE FAILURE: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(native(results["predictions"]), indent=2))


if __name__ == "__main__":
    main()
