"""Edits to main.tex (pass8) for the in-context-library spec, 2026-09-22.

Applied to src/pass8_excerpts.tex (verbatim excerpts of the pass8 file the user pasted); the
resulting unified diff applies to the full main.tex with offsets. Every anchor must occur once.

- A3 all numbers match -> drop TODO[gmemory-72b].
- A2 PASS by the user's decision (task_id collisions, different episodes) -> rewrite the CG audit
  paragraph with the numbers of the Figure 2 library, drop TODO[cg-audit].
- A1 is S2 -> the RQ1 sentence "hold the interface and the planner fixed ..." is left alone.
- B: the main cell (CG w/o Image, JSON-tolerant) is P 9/297 vs G-Memory 10/297 (p=1.0) and vs
  zero-shot 1/297 (p=.0078); CG w/ Image P 0/297 vs G-Memory 14/297 (p=1.2e-4). No pre-written
  outcome (E1/E2/E3) fits; the E2 structure is used with its false clause replaced (see report).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "iclr_todo"))
from patch_util import replace_once  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "results/incontext_library_2026-09-21/src"
t = (SRC / "pass8_excerpts.tex").read_text()

# A3
t = replace_once(t, "% TODO[gmemory-72b] The G-Memory 72B values 51.0/20.8/4.7/3.4 come from the Sep 2 and Sep 8 result\n"
                    "% documents (JSON-tolerant scoring). Check them against the data behind Figure 2.\n", "")

# B, RQ1 last sentence + TODO
t = replace_once(
    t,
    "Whether a baseline memory would gain from the same interface is not tested.\n"
    "% TODO[in-context-library] (formerly skill-call-baseline) Put the admitted library in the prompt,\n"
    "% let the model write primitive plans, score like the baselines. 72B, two CG splits, 594 calls.\n"
    "% See specs/incontext_library_spec.md. The result replaces the last sentence of this paragraph.\n",
    "With the admitted library placed in the prompt and primitive plans as output, success on the two CG "
    "splits is 0.0\\% with images and 3.0\\% without, not above G-Memory under the same output format "
    "(Appendix~\\ref{sec:viki_audit}). On VIKI-L2 the gain therefore requires the planner that consumes "
    "the library, and we treat the planner as part of the method.\n",
)

# B, Limitations
t = replace_once(
    t,
    "no single setting combines decentralized cooperation with a well-powered compositional test. ",
    "no single setting combines decentralized cooperation with a well-powered compositional test. "
    "On VIKI-L2 the library gains over G-Memory only when the deterministic planner consumes it. ",
)

# A2 PASS, rewritten audit paragraph
t = replace_once(
    t,
    "A read-only audit compares the CG task identifiers with the nine episode pools used to build the library and finds no overlap. "
    "The 180 episodes that the proposing agent opens all lie in the induction half, and none of its 884 transcripts contains a CG task identifier. ",
    "A read-only audit compares the CG tasks with every episode pool used to build the library evaluated in Figure~\\ref{fig:viki_main}, "
    "namely the seeds, holdouts, and tool calls of its 896 induction runs, the support probe, and the 3{,}598-episode pool from which the "
    "ordering rules are mined. All 1{,}242 training episodes that the proposing agent addresses through tools lie in the induction half, "
    "and none of the 896 transcripts contains a CG task identifier. Four of the 295 CG task identifiers also label training episodes, two "
    "of them in the induction half, because VIKI-L2 reuses identifiers across its training and test files. Each paired episode differs "
    "from its CG task in instruction, initial positions, goals, and reference plan, and none contains the held-out combination. ",
)
t = replace_once(t, "% TODO[cg-audit] Numbers are from the 2026-09-05 audit. Confirm that the nine pools are those of the\n"
                    "% library evaluated in Figure 2 (the 14-family build) and rerun the id intersection if they differ.\n", "")

# B, appendix control paragraph (before the sibling-group paragraph)
t = replace_once(
    t,
    "\\paragraph{Sibling-group-held-out OOD.}\n",
    "\\paragraph{In-context library control.}\n"
    "This arm places the admitted library in the prompt of Qwen2.5-VL-72B-Instruct and asks for a primitive plan. It keeps the "
    "G-Memory prompt, model, decoding, and scoring of Figure~\\ref{fig:viki_main} and replaces only the memory block with one fixed "
    "rendering of the eight operators and the three ordering rules, identical for every row. The G-Memory and zero-shot responses are "
    "those of Figure~\\ref{fig:viki_main}. On CG w/o Image the arm succeeds on 9 of 297 rows under both the JSON-tolerant parser and "
    "the official scorer, against 10 for G-Memory (9 vs.\\ 10 rows solved by only one arm, $p=1.0$) and 1 for zero-shot (8 vs.\\ 0, "
    "$p=.0078$). On CG w/ Image it succeeds on no row under either criterion, against 14 for G-Memory (0 vs.\\ 14, "
    "$p=1.2\\times10^{-4}$) and none for zero-shot. Our method solves 247 and 237 rows that this arm misses and loses none "
    "($p<10^{-70}$ on each split). Dropping the second occurrence of the two repeated tasks changes neither direction nor "
    "significance. Rerunning the first 20 rows of CG w/o Image changes 19 of the 20 responses and their success count from 0 to 3, "
    "so single-run differences of a few rows are within run-to-run variation.\n\n"
    "\\paragraph{Sibling-group-held-out OOD.}\n",
)

(SRC / "pass8_excerpts.patched.tex").write_text(t)
print("ok")
