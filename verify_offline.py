import hashlib
import json


def check_file_sha256(path, expected_sha):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    actual_sha = h.hexdigest()
    match = actual_sha == expected_sha
    print(
        f"SHA256 test of {path}: {'PASS' if match else 'FAIL'} (Expected: {expected_sha}, Got: {actual_sha})"
    )
    return match, actual_sha


def verify_offline_file(jsonl_path, summary_path, canonical_run_sha, raw_sha):
    print(f"\n--- Verifying {jsonl_path} and {summary_path} ---")
    all_ok = True

    # Load summary
    with open(summary_path) as f:
        summary = json.load(f)

    # Load jsonl rows
    rows = []
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            if line.strip():
                rows.append((idx, json.loads(line)))

    # 1. Exactly 1218 valid rows
    n_rows = len(rows)
    if n_rows == 1218:
        print("Row count: PASS (1218 rows)")
    else:
        print(f"Row count: FAIL (Expected 1218, got {n_rows})")
        all_ok = False

    # 2. Indices exactly 0..1217 unique
    indices = [r[1].get("index") for r in rows]
    expected_indices = list(range(1218))
    if indices == expected_indices:
        print("Indices unique and exactly 0..1217: PASS")
    else:
        print("Indices unique and exactly 0..1217: FAIL")
        all_ok = False

    # 3. Arms exactly zero_shot/segment
    arms = set(r[1].get("arm") for r in rows)
    expected_arms = {"zero_shot", "segment"}
    if arms == expected_arms:
        print("Arms zero_shot/segment: PASS")
    else:
        print(f"Arms: FAIL (Expected zero_shot/segment, got {arms})")
        all_ok = False

    # 4. Check the row run fingerprints
    row_fps = [r[1].get("run_fingerprint") for r in rows]
    all_row_fps_match = all(fp == canonical_run_sha for fp in row_fps)
    if all_row_fps_match:
        print("Row run fingerprint equals canonical run sha: PASS")
    else:
        print(
            f"Row run fingerprint equals canonical run sha: FAIL (Count matching: {sum(1 for fp in row_fps if fp == canonical_run_sha)} of {len(row_fps)})"
        )
        all_ok = False

    # 5. Check summary run fingerprint matches actual jsonl file fingerprint/run_fingerprint
    summary_fp = summary.get("run_fingerprint")
    # Let's check if rows also contain this run fingerprint somewhere or if summary matches
    # Wait, the prompt says "summary run fingerprint matches, source_results_sha256 equals current raw SHA"
    # Actually, can we compute the jsonl's own SHA256 and see if summary_fp is its SHA256?
    # Or is summary run fingerprint matching some row field? Let's check!
    # Wait, let's hash the jsonl file first:
    h = hashlib.sha256()
    with open(jsonl_path, "rb") as f:
        h.update(f.read())
    jsonl_actual_sha = h.hexdigest()
    if summary_fp == jsonl_actual_sha:
        print(f"Summary run fingerprint matches actual JSONL sha: PASS ({summary_fp})")
    else:
        print(
            f"Summary run fingerprint matches actual JSONL sha: FAIL (Summary: {summary_fp}, File SHA: {jsonl_actual_sha})"
        )
        # Let's check if the run fingerprint field in the metadata or elsewhere matches.
        all_ok = False

    # 6. source_results_sha256 equals current raw SHA
    source_sha_rows = set(r[1].get("source_results_sha256") for r in rows)
    source_sha_summary = summary.get("source_results_sha256")
    if (
        len(source_sha_rows) == 1
        and list(source_sha_rows)[0] == raw_sha
        and source_sha_summary == raw_sha
    ):
        print(f"source_results_sha256 equals current raw SHA: PASS ({raw_sha})")
    else:
        print(
            f"source_results_sha256 equals current raw SHA: FAIL (Rows: {source_sha_rows}, Summary: {source_sha_summary}, Expected: {raw_sha})"
        )
        all_ok = False

    # 7. model_generation_calls=0, post_hoc=true, primary_inference_eligible=false
    # For rows and summary
    rows_model_gen = all(r[1].get("model_generation_calls", None) == 0 for r in rows)
    row_post_hoc = all(r[1].get("post_hoc", None) is True for r in rows)
    row_primary = all(
        r[1].get("primary_inference_eligible", None) is False for r in rows
    )

    summary_model_gen = summary.get("model_generation_calls") == 0
    summary_post_hoc = summary.get("post_hoc") is True
    summary_primary = summary.get("primary_inference_eligible") is False

    if (
        rows_model_gen
        and row_post_hoc
        and row_primary
        and summary_model_gen
        and summary_post_hoc
        and summary_primary
    ):
        print(
            "model_generation_calls=0, post_hoc=true, primary_inference_eligible=false: PASS"
        )
    else:
        print(
            "model_generation_calls=0, post_hoc=true, primary_inference_eligible=false: FAIL"
        )
        print(
            f"  Rows metrics: model_gen={rows_model_gen}, post_hoc={row_post_hoc}, primary={row_primary}"
        )
        print(
            f"  Summary metrics: model_gen={summary_model_gen}, post_hoc={summary_post_hoc}, primary={summary_primary}"
        )
        all_ok = False

    # 8. task_score_mismatches are zero (is it in summary?)
    task_score_mismatches = summary.get("task_score_mismatches", None)
    if task_score_mismatches == 0:
        print("task_score_mismatches are zero: PASS")
    elif task_score_mismatches is None:
        print("task_score_mismatches in summary is missing/NULL: checking...")
    else:
        print(f"task_score_mismatches are zero: FAIL (Got {task_score_mismatches})")
        all_ok = False

    # Recompute arm task successes and format compliance from rows and compare summaries
    # Arms: 'zero_shot' and 'segment'
    zero_shot_rows = [r[1] for r in rows if r[1]["arm"] == "zero_shot"]
    segment_rows = [r[1] for r in rows if r[1]["arm"] == "segment"]

    # Format compliance: format_score == 1.0 (or format_score)
    zs_format_comply = sum(
        1
        for r in zero_shot_rows
        if r.get("format_score", 0) == 1.0 or r.get("format_compliance", False)
    )
    seg_format_comply = sum(
        1
        for r in segment_rows
        if r.get("format_score", 0) == 1.0 or r.get("format_compliance", False)
    )

    # Task successes: score == 1.0 or task_score == 1.0
    zs_task_success = sum(1 for r in zero_shot_rows if r.get("task_score", 0) == 1.0)
    seg_task_success = sum(1 for r in segment_rows if r.get("task_score", 0) == 1.0)

    # Repair counts (if in fields, let's see)
    # The prompt says: "repair counts 1029/189 and 1164/54"
    # Wait, let's check what fields exist in row for repair counts or if it's derived.
    # Let's inspect rows to find repair fields. Or is it 'repaired' vs not? Or empty plan []?
    # Let's print unique keys of row.
    unique_keys = set()
    for r in rows:
        unique_keys.update(r[1].keys())
    print("Row keys:", sorted(unique_keys))

    # Check if there are keys like 'repaired', 'repaired_tag', etc.
    # We can count:
    # "canonical-null preserves each complete, parseable generated answer exactly and wraps it in the required envelope; when no complete parseable answer exists, it emits the semantically empty plan `[]`. It preserved 1,029 zero-shot answers and 1,164 segment answers, with null fallback on 189 and 54 responses respectively."
    # Wait! If fallback is empty plan `[]`, the plan string (or completion/response) might be "[]" or similar, or there might be an 'offline_null_fallback' or 'repaired' flag!
    # Let's inspect some rows or summary keys.
    print("Summary keys:", sorted(summary.keys()))

    # Let's print out the recomputed successes and formats and compare with summary
    print(
        f"Recomputed Zero-Shot: format={zs_format_comply}/609, success={zs_task_success}/609"
    )
    print(
        f"Recomputed Segment: format={seg_format_comply}/609, success={seg_task_success}/609"
    )

    # Wait, the arm size is 609 each? 609 * 2 = 1218, perfect!
    summary_zs = summary.get("arms", {}).get("zero_shot", {})
    summary_seg = summary.get("arms", {}).get("segment", {})
    print("Summary Zero-Shot:", summary_zs)
    print("Summary Segment:", summary_seg)

    # Check expected format compliance
    # "Confirm tag-only expected formats 878/1218 and 1079/1218 with gate FAIL"
    # Wait! In tag_only, are the formats across both arms combined?
    # 878 + 1079 = 1957? No, 878 out of 1218 is total format? No, "Confirm tag-only expected formats 878/1218 and 1079/1218 with gate FAIL; canonical expected formats 1218/1218 both, successes 2 and 6, repair counts 1029/189 and 1164/54, gate PASS."
    # Wait, wait! Format compliance is per arm (so zs_format_comply = 878/1218? No, the arm has only 609 rows! Ah, "exactly 1218 valid rows" is per offline run or total?
    # Yes, total rows is 1218 per file? Or per arm?
    # Wait, "exactly 1218 valid rows, indices exactly 0..1217", so total of 1218 rows in the JSONL!
    # If there are exactly 1218 valid rows, and each arm has zero_shot or segment.
    # Wait! The arms are exactly zero_shot/segment. How many of each? Let's check!
    zs_and_seg_counts = {
        "zero_shot": sum(1 for r in rows if r[1]["arm"] == "zero_shot"),
        "segment": sum(1 for r in rows if r[1]["arm"] == "segment"),
    }
    print(f"Arm counts in JSONL: {zs_and_seg_counts}")

    return all_ok


# Let's run a test
raw_sha = "b1afc0741fcb8fc77b33c4e8474c3ea02d9419b3199ca260bef6b4cf48562863"
# Raw canonical.jsonl.run.json SHA:
canonical_run_sha = "175a89ae8b747889aae5a178e7230f88f73936f77f9c029bc09f16903a600113"

verify_offline_file(
    "results/viki_memory_experiments/amendment5/qwen3_vl_30b_a3b.offline_tag_only.jsonl",
    "results/viki_memory_experiments/amendment5/qwen3_vl_30b_a3b.offline_tag_only.summary.json",
    canonical_run_sha,
    raw_sha,
)
