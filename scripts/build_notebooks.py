"""Convert percent-format notebook sources (notebooks/src/*.py) into .ipynb files.

Cells start with ``# %%`` (code) or ``# %% [markdown]`` (markdown; lines are un-commented).
The .py files are the reviewed source of truth; the .ipynb files are what Colab opens.
A line ``# colab-runtime: cpu`` anywhere in the source makes the notebook open on a CPU runtime
(no GPU compute units); otherwise it asks for an L4.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "src"
OUT = ROOT / "notebooks"
CELL_RE = re.compile(r"^# %%(?P<md> \[markdown\])?.*$")


def parse(text: str) -> list[dict]:
    cells: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            current = {"type": "markdown" if m.group("md") else "code", "lines": []}
            cells.append(current)
            continue
        if current is None:
            continue
        if current["type"] == "markdown":
            line = line[2:] if line.startswith("# ") else line.lstrip("#")
        current["lines"].append(line)
    for cell in cells:
        while cell["lines"] and not cell["lines"][-1].strip():
            cell["lines"].pop()
    return [c for c in cells if c["lines"]]


def to_ipynb(cells: list[dict], gpu: bool = True) -> dict:
    nb_cells = []
    for cell in cells:
        source = [line + "\n" for line in cell["lines"]]
        source[-1] = source[-1].rstrip("\n")
        entry = {"cell_type": cell["type"], "metadata": {}, "source": source}
        if cell["type"] == "code":
            entry.update(execution_count=None, outputs=[])
        nb_cells.append(entry)
    metadata = {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if gpu:
        metadata["accelerator"] = "GPU"
        metadata["colab"]["gpuType"] = "L4"
    return {
        "cells": nb_cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main(names: list[str]) -> None:
    sources = sorted(SRC.glob("*.py")) if not names else [SRC / f"{n}.py" for n in names]
    for src in sources:
        text = src.read_text(encoding="utf-8")
        nb = to_ipynb(parse(text), gpu="# colab-runtime: cpu" not in text)
        dest = OUT / f"{src.stem}.ipynb"
        dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{src.relative_to(ROOT)} -> {dest.relative_to(ROOT)} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main(sys.argv[1:])
