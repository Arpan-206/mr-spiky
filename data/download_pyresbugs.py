"""Download PyResBugs (OSS-forge/PyResBugs on HuggingFace) and materialize a
Python-labeled calibration set for `src/calibrate.py`.

Each row of PyResBugs contains a `Faulty Code` (buggy) and `Fault Free Code`
(clean) version of the same Python function. We pair-encode them: label=1 for
faulty, label=0 for fault-free. This gives us `N` naturally-paired samples
(2N labeled functions total) which is what the calibration step wants.

Writes: data/pyresbugs_labeled.json
    [{"label": 0|1, "code": "...", "meta": {...}}, ...]

`calibrate.py` reads `data/mlcq_labeled.json` by default; we also write that
filename for zero-config compatibility, but the file is PyResBugs-sourced.
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path

SAMPLE_PAIRS = 2000  # → 4000 labeled functions (2000 buggy + 2000 clean)
HF_DATASET = "OSS-forge/PyResBugs"

OUT_PATH = Path(__file__).resolve().parent / "pyresbugs_labeled.json"
COMPAT_PATH = Path(__file__).resolve().parent / "mlcq_labeled.json"

log = logging.getLogger("mrspiky.data.pyresbugs")


def _clean(code: str) -> str | None:
    """Normalize whitespace and reject anything that doesn't parse."""
    if not isinstance(code, str) or not code.strip():
        return None
    # PyResBugs stores literal tabs; ast handles them fine, but some rows have
    # escaped `\\n` / `\\t` where the CSV round-trip mangled them. Try both.
    candidates = [code, code.replace("\\n", "\n").replace("\\t", "\t")]
    for c in candidates:
        try:
            ast.parse(c)
            return c
        except SyntaxError:
            continue
    return None


def _try_hf() -> list[dict] | None:
    try:
        from datasets import load_dataset  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        log.warning("datasets not available (%s)", e)
        return None

    log.info("streaming %s ...", HF_DATASET)
    try:
        ds = load_dataset(HF_DATASET, split="train", streaming=True)
    except Exception as e:  # noqa: BLE001
        log.warning("load_dataset failed: %s", e)
        return None

    out: list[dict] = []
    seen_pairs = 0
    kept_pairs = 0
    try:
        for row in ds:
            seen_pairs += 1
            faulty = _clean(row.get("Faulty Code", ""))
            clean = _clean(row.get("Fault Free Code", ""))
            if faulty is None or clean is None:
                continue
            if faulty.strip() == clean.strip():
                # Same code on both sides is useless as a labeled pair.
                continue

            meta_common = {
                "project": row.get("Project"),
                "commit": row.get("Commit_sha"),
                "bug_type": row.get("Bug_Type"),
                "fault_acronym": row.get("Fault_Acronym"),
            }
            out.append({"label": 1, "code": faulty, "meta": meta_common})
            out.append({"label": 0, "code": clean, "meta": meta_common})
            kept_pairs += 1
            if kept_pairs >= SAMPLE_PAIRS:
                break
    except Exception as e:  # noqa: BLE001
        log.warning("row iteration ended after %d pairs kept: %s", kept_pairs, e)

    log.info("kept %d pairs (%d functions) out of %d seen", kept_pairs, len(out), seen_pairs)
    return out or None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    samples = _try_hf()
    if not samples:
        log.error("failed to fetch PyResBugs — calibrate.py will fall back to its built-in set")
        return 1

    OUT_PATH.write_text(json.dumps(samples, indent=2))
    COMPAT_PATH.write_text(json.dumps(samples, indent=2))
    log.info("wrote %d labeled samples -> %s (and %s for calibrate.py compat)", len(samples), OUT_PATH, COMPAT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
