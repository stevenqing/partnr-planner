import hashlib
import json


def run_check():
    # 1. Read files
    jsonl_path = "results/viki_memory_experiments/amendment5/qwen2_5_vl_7b_stock.jsonl"
    run_path = (
        "results/viki_memory_experiments/amendment5/qwen2_5_vl_7b_stock.jsonl.run.json"
    )
    summary_path = (
        "results/viki_memory_experiments/amendment5/qwen2_5_vl_7b_stock.summary.json"
    )

    with open(run_path) as f:
        run_data = json.load(f)
    with open(summary_path) as f:
        json.load(f)

    canonical_run = json.dumps(run_data, sort_keys=True, separators=(",", ":"))
    run_hash = hashlib.sha256(canonical_run.encode("utf-8")).hexdigest()

    print(f"Computed run file canonical SHA256 hash: {run_hash}")

    # 2. Check jsonl records
    indices = []
    has_endpoint_error = False

    zero_shot_success_cnt = 0
    segment_success_cnt = 0
    zero_shot_format_score_sum = 0
    segment_format_score_sum = 0

    # Track arms present across records
    row_arm_counts = {}

    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f]

    print(f"Total lines read from JSONL: {len(records)}")

    fingerprints = set()

    # Let's inspect the target requirements:
    # "require exactly indices 0..1217, no duplicate/missing records,"
    # "no endpoint_error,"
    # "exactly arms zero_shot/composed/segment each row,"
    # "one run_fingerprint matching SHA256 canonical JSON of the .run.json metadata,"
    # "source_results_sha256 present,"
    # "summary status PASS/samples 1218/model_generation_calls 1218/new_run_fingerprint matching,"
    # "runtime served model id/root/max_model_len qwen2.5-vl-7b-amendment5 / Qwen/Qwen2.5-VL-7B-Instruct / 16384,"
    # "and independently recompute zero/segment success and format compliance from row arm fields to match summary."
    # Wait, "exactly arms zero_shot/composed/segment each row". Let's check which arms are actually present in the rows.
    # Are we sure there are zero_shot, segment, AND composed? Wait, could "composed" be present? Let's check keys.

    for _idx, r in enumerate(records):
        indices.append(r.get("index"))
        if "endpoint_error" in r or any(
            "endpoint_error" in (r.get("arms", {}).get(arm, {}) or {})
            for arm in r.get("arms", {})
        ):
            has_endpoint_error = True

        fingerprints.add(r.get("run_fingerprint"))

        # Check arms present
        arms = r.get("arms", {})
        arm_keys = set(arms.keys())
        # Let's see if this matches zero_shot, composed, segment
        row_arm_counts[tuple(sorted(arm_keys))] = (
            row_arm_counts.get(tuple(sorted(arm_keys)), 0) + 1
        )

        # Zero shot details
        zs = arms.get("zero_shot", {})
        # Note: success can be score or tasks or success. Let's see how success is derived.
        # Let's check what keys are in zs.
        zs_success = (
            zs.get("success", False) or zs.get("score") == 1.0
        )  # typically score == 1.0 means success
        zs_format = zs.get("format_score", 0.0)

        # Segment details
        seg = arms.get("segment", {})
        seg_success = seg.get("success", False) or seg.get("score") == 1.0
        seg_format = seg.get("format_score", 0.0)

        if zs_success:
            zero_shot_success_cnt += 1
        if seg_success:
            segment_success_cnt += 1

        zero_shot_format_score_sum += zs_format
        segment_format_score_sum += seg_format

    print("Index checks:")
    sorted_indices = sorted(indices)
    expected_indices = list(range(1218))
    print(f"Indices match expected exactly: {sorted_indices == expected_indices}")
    print(f"Has duplicates: {len(indices) != len(set(indices))}")
    print(f"Has endpoint_error: {has_endpoint_error}")
    print(f"Unique arm patterns in JSONL across rows: {row_arm_counts}")
    print(f"Unique fingerprints in JSONL: {fingerprints}")

    zs_format_compliance = zero_shot_format_score_sum / len(records) if records else 0
    seg_format_compliance = segment_format_score_sum / len(records) if records else 0

    print(
        f"Recomputed Zero Shot Successes: {zero_shot_success_cnt}, Format Compliance: {zs_format_compliance}"
    )
    print(
        f"Recomputed Segment Successes: {segment_success_cnt}, Format Compliance: {seg_format_compliance}"
    )


run_check()
