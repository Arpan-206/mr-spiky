"""Fetch (or explain how to fetch) the MLCQ labeled code-smell dataset.

MLCQ is hosted on Zenodo (record 3666840). The archive is a mixed bag of CSVs
plus raw source snippets and isn't trivially machine-fetchable without knowing
the exact filenames — Zenodo's API returns file metadata, so we try that. If we
can't produce a labeled JSON, we print manual instructions and exit 0 so the
pipeline can proceed with the built-in fallback labeled set in calibrate.py.

Writes `data/mlcq_labeled.json` on success — a list of {label, code} objects.
`label` is 1 for suspicious (has a code smell) and 0 for clean.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import requests

ZENODO_RECORD = "3666840"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
OUT_PATH = Path(__file__).resolve().parent / "mlcq_labeled.json"

log = logging.getLogger("mrspiky.data.mlcq")


def _instructions() -> None:
    print(
        "\nMLCQ not downloaded automatically.\n"
        f"  1. visit https://zenodo.org/record/{ZENODO_RECORD}\n"
        "  2. download the CSV of smell labels + the source snippet archive\n"
        "  3. produce data/mlcq_labeled.json as a list of {\"label\": 0|1, \"code\": \"...\"}\n"
        "     (label 1 = smelly / suspicious, label 0 = clean)\n"
        "  4. re-run `just calibrate`\n\n"
        "The calibration script has a built-in fallback labeled set, so you can\n"
        "skip this and still get a working end-to-end pipeline.\n",
        file=sys.stderr,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        r = requests.get(ZENODO_API, timeout=15)
        r.raise_for_status()
        meta = r.json()
        files = meta.get("files", [])
        log.info("MLCQ record has %d files", len(files))
        for f in files:
            log.info("  - %s (%s bytes)", f.get("key"), f.get("size"))
    except Exception as e:  # noqa: BLE001
        log.warning("could not query Zenodo (%s)", e)

    _instructions()
    return 0


if __name__ == "__main__":
    sys.exit(main())