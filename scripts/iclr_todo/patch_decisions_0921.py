"""Apply the four decisions the user made on 2026-09-21 to the working copy of main.tex.

1. PARTNR leak: disclose in the paper (no rebuild). Checked 09-21: the Table 1/2 targets
   (R_S, H_R_T, H_R_S_T) share no episode with their source memories, for ours, RAG and
   MEMENTO. Only H_R evaluated from the H_R memory overlaps (196/197 ours, 196/197 RAG,
   197/197 MEMENTO source set).
3. cells: 72 + 36 + 36 = 144, sibling 18 separately.
4. handwritten-7b / cells / retrieval-impl: all three edited.

Run once on src/main.tex (it asserts every anchor occurs exactly once).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patch_util import drop_todo, replace_once

P = Path(__file__).resolve().parents[2] / "results/iclr_todo_2026-09-21/src/main.tex"
t = P.read_text()

# --- retrieval-impl: the two sentences become code facts ---
t = replace_once(
    t,
    "The PARTNR implementation adds two steps. A proposition-based task-type classifier first routes the query to the skill bank of the predicted type (Appendix~\\ref{sec:task_classification_appendix}). The skill-level score then adds a fixed boost of 0.15 when an object type named in the instruction also appears in the skill, so reported scores can exceed 1 (Appendix~\\ref{subsec:retrieval_examples}).",
    "In the PARTNR runs, the skill banks searched are fixed by each run's configuration and stated with each result, and no classifier selects a bank at test time. Both retrieval levels score by raw cosine similarity with no additional term.",
)
t = drop_todo(t, "retrieval-impl")

# routing ablation: pointer in RQ3, the appendix intro sentence, the figure and its paragraph
t = replace_once(
    t,
    " Additional routing diagnostics appear in Appendix~\\ref{sec:detailed_partnr_diagnostics}.",
    "",
)
t = replace_once(
    t,
    "Two supplementary diagnostics help interpret RQ3. Figure~\\ref{fig:tasktype} evaluates task-aware routing, and Figure~\\ref{fig:one_agent} isolates the effect of assigning the learned library to only one role.",
    "A supplementary diagnostic helps interpret RQ3. Figure~\\ref{fig:one_agent} isolates the effect of assigning the learned library to only one role.",
)
start = t.index("\\begin{figure}[h]\n    \\centering\n    \\includegraphics[width=0.72\\linewidth]{figs/tasktype_ablation.pdf}")
end = t.index("not from classifier headroom alone.\n", start) + len("not from classifier headroom alone.\n")
assert t.count("figs/tasktype_ablation.pdf") == 1
t = t[:start] + t[end:].lstrip("\n")

# appendix example (1): delete; (2) is the only one left, so drop the numbering
t = replace_once(
    t,
    "To complement the aggregate statistics, we extract concrete retrieval cases from evaluation logs that illustrate how the hierarchical retrieval mechanism operates in practice. \\textbf{(1) Task-aware boosting.} For the instruction ``move kettle from living room to kitchen, fill and turn on,'' the retrieval system returns \\texttt{kettle\\_operation} with combined score 1.117 (cosine similarity 0.967 plus object-type boost 0.150 from matching the ``kettle'' token). Lexical and contextual signals supplement raw embedding distance, providing discriminative power when multiple skills have similar descriptions but apply to different object classes. \\textbf{(2) Decomposition into complementary skills.}",
    "To complement the aggregate statistics, we describe a retrieval case that illustrates how the hierarchical retrieval mechanism operates in practice. \\textbf{Decomposition into complementary skills.}",
)

# still inconsistent with "no additional term", outside the authorised edit: flag, do not rewrite
anchor = "with median post-boost scores of $0.836$ on R\\_S"
assert t.count(anchor) == 1
i = t.index("\n", t.index(anchor))
t = t[:i] + (
    "\n% TODO[retrieval-impl-fallback] \"post-boost\" medians of 1.162 and 1.226 are impossible"
    "\n% without a boost; the code has none and the highest logged score is 0.965."
    "\n% Source of these medians not found (Isambard outputs). Fix or remove."
) + t[i:]
anchor = "supporting the use of predicted types for routing"
assert t.count(anchor) == 1
i = t.index("\n", t.index(anchor))
t = t[:i] + (
    "\n% TODO[retrieval-impl-gt] No test-time routing exists in the code: the bank is fixed"
    "\n% per run. Say what the classifier-vs-GT runs below actually varied, or remove them."
) + t[i:]

# --- PARTNR leak disclosure (decision 1: option one) ---
t = replace_once(
    t,
    "The gate is evaluated on VIKI-L2 and on one targeted PARTNR skill (Section~\\ref{subsec:induction_validation}).",
    "The gate is evaluated on VIKI-L2 and on one targeted PARTNR skill (Section~\\ref{subsec:induction_validation}). Every PARTNR memory, including those of RAG and MEMENTO, is built offline from heuristic-agent trajectories on the episodes of its source task group, and retrieval does not exclude the current episode. No target episode in Tables~\\ref{tab:main_results_task} and~\\ref{tab:partnr_composition} occurs in its source memory. The H\\_R memory, however, is built from 196 of the 197 H\\_R episodes, so any H\\_R result that retrieves from it (Appendices~\\ref{sec:hr_failure_appendix} and~\\ref{sec:n_agent_extension}) measures reuse on seen episodes rather than transfer.",
)
anchor = "\\subsection{Cooperation Failure Metrics under Asymmetric Memory}\n\\label{sec:hr_failure_appendix}\n"
assert t.count(anchor) == 1
t = t.replace(anchor, anchor + (
    "% TODO[hr-leak] Which memory the one-agent and N=3 H_R runs retrieve from is not on disk\n"
    "% (Isambard outputs). If it is the H_R memory, say so here; the protocol paragraph\n"
    "% already states that such runs reuse seen episodes.\n"
))

# --- cells ---
t = replace_once(
    t,
    " The 30B ablation covers ID and OOD only, so its compositional cells are marked as not run.}",
    "}\n% TODO[cells-fig] figs/ for this figure still marks the four 30B CG cells as not run.\n% They exist (2026-09-14): w/ Image no_grounding 0.024, no_order 0.552 (full 0.650);\n% w/o Image no_grounding 0.724, no_order 0.643 (full 0.724). Regenerate the figure.",
)
t = replace_once(
    t,
    "They comprise 72 main-comparison cells, 36 RQ2 cells, and 32 RQ3 cells. Four 30B compositional ablation cells are not run.",
    "They comprise 72 main-comparison cells, 36 RQ2 cells, and 36 RQ3 cells, 144 in total, plus 18 sibling-group comparison cells.",
)
t = drop_todo(t, "cells")
t = replace_once(
    t,
    "The main VIKI comparison contains all 90 expected combinations of model, split, and method.",
    "The main VIKI comparison contains all 72 expected combinations of three models, four splits, and six methods. The sibling-group split adds 18 cells for the same three models. ToM is not run on that split, and its sixth arm is an earlier skill-memory variant of our method.",
)

# --- handwritten-7b ---
t = replace_once(
    t,
    "A hand-written 19-operator library with the same interface exceeds the learned library on 72B OOD (67.4\\% vs.\\ 54.2\\%), while the learned library leads on 72B ID (Appendix~\\ref{sec:viki_audit}).",
    "A 19-operator reference library with the same interface, mined by simulator replay of the same even-indexed training episodes without LLM proposal, exceeds the learned library on 72B OOD even when the held-out family is withheld from it too (65.6\\% vs.\\ 54.2\\%), while the learned library leads on 72B ID (Appendix~\\ref{sec:viki_audit}).",
)
t = replace_once(
    t,
    "A 19-operator hand-written reference library further cautions against attributing every score to induction. It outperforms the learned eight-operator library on 72B OOD (0.674 vs.\\ 0.542) and 7B ID/OOD (0.160 vs.\\ 0.139/0.117), whereas the learned library leads on 72B ID and 30B ID.",
    "A 19-operator reference library, mined by simulator replay of the same even-indexed training episodes without LLM proposal, further cautions against attributing every score to induction. On single-family OOD we use its per-fold versions, which withhold the held-out family. It outperforms the learned eight-operator library on OOD at all three scales (72B 0.656 vs.\\ 0.542, 30B 0.460 vs.\\ 0.354, 7B 0.145 vs.\\ 0.117) and on 7B ID (0.160 vs.\\ 0.139), whereas the learned library leads on 72B ID (0.780 vs.\\ 0.674) and 30B ID (0.574 vs.\\ 0.510).",
)
t = drop_todo(t, "handwritten-7b")

P.write_text(t)
print("ok", P, len(t.splitlines()), "lines")
