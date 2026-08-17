# Amendment 3.1: C-prime Token-Matching Clarification

## Scope

This amendment applies only to the flat token-matched control builder in F2 task 4. It changes no tolerance, threshold, retrieval setting, or completed result.

## Binding criterion

C-prime token matching is evaluated per row. Every retained flat control must be within 5 percent of its corresponding skill-memory prompt. This preserves the stronger guarantee already reported by the published C3 tables.

The halted instance run was correct: one row produced 4,009 flat tokens against 4,323 skill tokens, a 7.2635 percent difference. The checkpoint contained 18 of 400 rows with no duplicates, endpoint errors, route parse errors, summary, parquet, or residual process. Productivity and finalization had not started.

## Builder ruling

The failure was a builder capability defect. The builder now:

1. Ranks allowed train rows by context similarity.
2. Appends the next-nearest whole train row until the flat prompt first reaches or exceeds the row's skill-memory token target.
3. Truncates the resulting flat block at token level down into the unchanged 5 percent band.
4. Symmetrically drops the row from all channel arms only if the entire allowed pool is exhausted below the lower bound.

Any symmetric drops are recorded by channel. The expected drop count is zero.

## Mandatory preflight

Before C-prime arm generation resumes, the runner must execute exact token accounting for every row in both channels:

```bash
TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/viki_amendment3_f2.py preflight-cprime --base-url http://127.0.0.1:8050/v1 --workers 8
```

The command writes channel-level preflight records and a combined PASS certificate. Formal C-prime generation refuses to run without that certificate and matching artifact hashes.

## Checkpoint reuse

The completed zero-shot and skill-memory outputs are reused only when the old run fingerprint is valid and their reconstructed prompts match the preflight prompt SHA-256 values. Flat prompts are rebuilt for every row, prior flat outputs are discarded, and migrated records receive the new Amendment 3.1 run fingerprint.

The original under-budget deviation favored the skill arm. Halting and repairing builder capability preserves attribution of the skill-versus-flat comparison to structure.
