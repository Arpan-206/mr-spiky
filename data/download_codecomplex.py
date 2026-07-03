"""Download the CodeComplex Python subset — a genuine complexity-labeled Python
dataset — for a second validation pass in `src/calibrate.py`.

Unlike PyResBugs (which labels semantic bugs — invisible to structural features),
CodeComplex labels algorithmic time-complexity classes, which correlate strongly
with what Mr. Spiky measures (nesting depth, cyclomatic complexity, length).
So this is the dataset that *should* give us real discriminative accuracy.

Binarization:
    simple (label 0): {constant, logn, linear}
    complex (label 1): {nlogn, quadratic, cubic, np}

Writes: data/codecomplex_labeled.json
    [{"label": 0|1, "code": "...", "complexity": "<original>", "problem": ...}, ...]
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/sybaik1/CodeComplex-Data/main/python_data.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "codecomplex_labeled.json"

SIMPLE_CLASSES = {"constant", "logn", "linear"}
COMPLEX_CLASSES = {"nlogn", "quadratic", "cubic", "np"}

log = logging.getLogger("mrspiky.data.codecomplex")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("streaming %s ...", URL)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()

    out: list[dict] = []
    seen = 0
    parsed_fail = 0
    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        seen += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        complexity = row.get("complexity", "").strip().lower()
        src = row.get("src", "")
        if not isinstance(src, str) or not src.strip():
            continue

        if complexity in SIMPLE_CLASSES:
            label = 0
        elif complexity in COMPLEX_CLASSES:
            label = 1
        else:
            continue  # unknown class, skip

        try:
            ast.parse(src)
        except SyntaxError:
            parsed_fail += 1
            continue

        out.append({
            "label": label,
            "code": src,
            "complexity": complexity,
            "problem": row.get("problem"),
        })

    from collections import Counter
    counts = Counter(x["complexity"] for x in out)
    label_counts = Counter(x["label"] for x in out)
    log.info(
        "kept %d/%d rows (%d parse failures); labels %s; classes %s",
        len(out), seen, parsed_fail, dict(label_counts), dict(counts),
    )

    OUT_PATH.write_text(json.dumps(out))
    log.info("wrote -> %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
