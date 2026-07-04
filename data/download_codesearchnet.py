"""Download a small sample of Python functions from CodeSearchNet.

Tries HuggingFace `datasets` first. If that fails (network, gated dataset, etc.),
falls back to a curated micro-sample so the pipeline still runs end-to-end.

Writes `data/codesearchnet_python.json` — a list[str] of function source blocks.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

SAMPLE_SIZE = 250  # ~150-300 as spec'd
OUT_PATH = Path(__file__).resolve().parent / "codesearchnet_python.json"

log = logging.getLogger("mrspiky.data.csn")

_HF_CANDIDATES = [
    ("Nan-Do/code-search-net-python", None, "func_code_string"),
    ("Nan-Do/code-search-net-python", None, "whole_func_string"),
    ("espejelomar/code_search_net_python_10000_examples", None, "whole_func_string"),
    ("code-search-net/code_search_net", "python", "whole_func_string"),
]

_MICRO_FALLBACK = [
    "def add(a, b):\n    return a + b\n",
    "def is_even(n):\n    return n % 2 == 0\n",
    "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
    "def mean(xs):\n    return sum(xs) / len(xs)\n",
    "def reverse(s):\n    return s[::-1]\n",
    "def flatten(xss):\n    return [x for xs in xss for x in xs]\n",
    "def word_count(s):\n    return {w: s.split().count(w) for w in set(s.split())}\n",
]


def _try_hf() -> list[str] | None:
    try:
        from datasets import load_dataset  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        log.warning("datasets not available (%s)", e)
        return None

    for name, config, field in _HF_CANDIDATES:
        log.info("trying HF dataset %s (config=%s, field=%s)", name, config, field)
        out: list[str] = []
        try:
            ds = load_dataset(name, config, split="train", streaming=True) if config else \
                 load_dataset(name, split="train", streaming=True)
        except Exception as e:  # noqa: BLE001
            log.warning("  load_dataset failed: %s", e)
            continue
        try:
            for row in ds:
                if field in row and isinstance(row[field], str) and row[field].strip():
                    out.append(row[field])
                if len(out) >= SAMPLE_SIZE:
                    break
        except Exception as e:  # noqa: BLE001
            log.warning("  row iteration failed after %d rows: %s", len(out), e)
        if out:
            log.info("collected %d functions from %s", len(out), name)
            return out
        log.warning("  source yielded 0 usable rows (field %r missing?)", field)
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    samples = _try_hf()
    if not samples:
        log.warning("using micro fallback — pipeline will still run but weights won't reflect real code")
        samples = _MICRO_FALLBACK

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(samples))
    log.info("wrote %d samples -> %s", len(samples), OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())