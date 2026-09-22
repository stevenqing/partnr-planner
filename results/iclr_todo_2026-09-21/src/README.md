# ICLR 2027 transfer

`main.tex` is the NeurIPS 2026 paper transferred to the supplied ICLR 2027 style files.

- The paper body, equations, tables, figures, appendix, and bibliography entries are preserved.
- The document now loads `iclr2027_conference.sty` and uses `iclr2027_conference.bst`.
- The author block is anonymous for double-blind submission because `\\iclrfinalcopy` remains commented.
- The NeurIPS-only checklist page has been removed; ICLR has different submission requirements.
- Figure paths are kept relative to the project root (`figs/` and `figs_append/`).

For a camera-ready version, add `\\iclrfinalcopy` before `\\begin{document}` and replace the empty `\\author{}` block with the final author information.

The workspace now has MiKTeX 25.12 installed. The paper was compiled successfully with `pdflatex`, `bibtex`, and two final `pdflatex` passes. The generated PDF is 29 pages. The former introductory wrapfigure was removed, as requested, and the corresponding text reference was removed as well.

The compiler reports only layout warnings from long appendix/listing lines (overfull boxes); there are no fatal errors, undefined citations, or undefined references.

The reference audit also repaired the corrupted `Memory^3` BibTeX entry, corrected author/title/year metadata for Reflexion, the agent-memory survey, MemoryBank, PARTNR, MEMENTO, and the Theory-of-Mind evaluation paper, and softened the MEMENTO/Theory-of-Mind claims in the main text so they match the cited sources.
