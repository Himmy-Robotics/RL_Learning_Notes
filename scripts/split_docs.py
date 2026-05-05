#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE = ROOT_DIR / "RL_Learning_Notes.md"
OUT_DIR = ROOT_DIR / "docs"

CHAPTER_FILES = [
    "01-basic-concepts.md",
    "02-value-bellman.md",
    "03-optimal-bellman.md",
    "04-value-policy-iteration.md",
    "05-monte-carlo.md",
    "06-stochastic-approximation.md",
    "07-td-methods.md",
    "08-function-approximation.md",
    "09-policy-gradient.md",
    "10-actor-critic.md",
    "11-ppo.md",
]


def main() -> int:
    if not SOURCE.exists():
        print(f"Missing source: {SOURCE}", file=sys.stderr)
        return 1

    lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    chapter_starts = [
        idx for idx, line in enumerate(lines) if re.match(r"^#\s+\d+\.", line)
    ]

    if len(chapter_starts) != len(CHAPTER_FILES):
        print(
            f"Expected {len(CHAPTER_FILES)} chapters, found {len(chapter_starts)}",
            file=sys.stderr,
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for chapter_index, start in enumerate(chapter_starts):
        end = chapter_starts[chapter_index + 1] if chapter_index + 1 < len(chapter_starts) else len(lines)
        content = "".join(lines[start:end]).rstrip() + "\n"
        (OUT_DIR / CHAPTER_FILES[chapter_index]).write_text(content, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
