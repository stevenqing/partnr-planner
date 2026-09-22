"""Helpers to edit the working copy of main.tex for the ICLR TODO spec (2026-09-21)."""
from pathlib import Path


def drop_todo(text: str, tag: str) -> str:
    """Remove the `% TODO[tag]` comment block: that line and the `%` lines that follow it."""
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(f"% TODO[{tag}]"))
    end = start + 1
    while end < len(lines) and lines[end].startswith("%") and not lines[end].startswith("% TODO["):
        end += 1
    return "\n".join(lines[:start] + lines[end:])


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)
