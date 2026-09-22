"""B2 (memento-hrt): replace the unreproducible MEMENTO H_R_T 14.5% with the 2026-09-21 rerun.

Rerun: all 11 H_R_T episodes, memory H_R+H_T as Table 2 states, Llama-3.1-8B-Instruct, old
pipeline unchanged (scripts/iclr_todo/run_b2_memento_hrt.sh). 1/11 success, completion 0.393.
Per-episode file: results/iclr_todo_2026-09-21/b2/episode_result_log.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patch_util import drop_todo, replace_once

P = Path(__file__).resolve().parents[2] / "results/iclr_todo_2026-09-21/src/main.tex"
t = P.read_text()

t = replace_once(
    t,
    "Memory-as-Skill reaches 27.7\\% success on H\\_R\\_T, compared with 19.8\\% for G-Memory, 14.5\\% for MEMENTO, and 11.1\\% for RAG.",
    "Memory-as-Skill reaches 27.7\\% success on H\\_R\\_T, compared with 19.8\\% for G-Memory, 11.1\\% for RAG, and 9.1\\% for MEMENTO.",
)
t = replace_once(
    t,
    "H\\_R\\_T & H\\_R+H\\_T & \\textbf{27.7} & 11.1 & 14.5 & 0.0 & 9.2 & 19.8 \\\\",
    "H\\_R\\_T & H\\_R+H\\_T & \\textbf{27.7} & 11.1 & 9.1 & 0.0 & 9.2 & 19.8 \\\\",
)
t = replace_once(
    t,
    "The MEMENTO H\\_R\\_T value of 14.5\\% comes from an earlier run and is not repeated in the final evaluation pass.",
    "The MEMENTO H\\_R\\_T cell is a rerun of the same code and memory on all 11 H\\_R\\_T episodes (1/11 successes, mean completion 39.3\\%); the earlier value of 14.5\\% could not be matched to any run.",
)
t = drop_todo(t, "memento-hrt")

P.write_text(t)
print("ok")
