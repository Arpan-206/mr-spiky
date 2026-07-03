"""Inference entry point.

`analyze(code)` returns the fixed response schema regardless of whether the SNN
weights exist yet. If `models/snn_weights.pt` is missing, we fall back to
scoring each line from its normalized AST features (mock mode) so the frontend
can integrate against a live API before training finishes.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from .encode import DEFAULT_STEPS, encode_batch
from .features import extract_line_features, normalize
from .model import SpikyNet, temporal_spike_attribution

log = logging.getLogger("mrspiky.infer")

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_WEIGHTS_PATH = _MODELS_DIR / "snn_weights.pt"
_THRESHOLD_PATH = _MODELS_DIR / "threshold.json"

# Weights used to collapse normalized features into a single suspicion score in
# mock mode. Hand-picked: nesting depth + cyclomatic complexity dominate,
# length contributes a bit, entropy features act as tie-breakers.
_MOCK_WEIGHTS: tuple[float, ...] = (0.40, 0.15, 0.10, 0.10, 0.25)
_MOCK_THRESHOLD = 0.55

# Populated lazily on first call if weights exist.
_snn_state: dict[str, Any] | None = None


def _mock_score(vector: list[float]) -> float:
    norm = normalize(vector)
    s = sum(w * v for w, v in zip(_MOCK_WEIGHTS, norm))
    return max(0.0, min(1.0, s))


def _load_snn() -> dict[str, Any] | None:
    """Lazy-load SNN weights + threshold. Returns None if not yet trained."""
    global _snn_state
    if _snn_state is not None:
        return _snn_state
    if not _WEIGHTS_PATH.exists():
        return None
    try:
        net = SpikyNet()
        net.load_state_dict(torch.load(_WEIGHTS_PATH, weights_only=True))
        net.eval()
        threshold = _MOCK_THRESHOLD
        if _THRESHOLD_PATH.exists():
            threshold = float(json.loads(_THRESHOLD_PATH.read_text())["threshold"])
        _snn_state = {"net": net, "threshold": threshold}
        log.info("loaded SNN weights + threshold=%.4f", threshold)
        return _snn_state
    except Exception as e:  # noqa: BLE001 — hackathon: don't crash the API
        log.warning("failed to load SNN artifacts (%s), staying in mock mode", e)
        return None


def _snn_scores(net: SpikyNet, vectors: list[list[float]]) -> list[float]:
    """Run all line vectors through the SNN in one batch, return TSA per line."""
    if not vectors:
        return []
    spikes = encode_batch(vectors, num_steps=DEFAULT_STEPS)
    with torch.no_grad():
        out = net(spikes)
    intensities = temporal_spike_attribution(out.hidden_spikes)  # (B,) already in ~[0,1]
    return [max(0.0, min(1.0, float(v))) for v in intensities.tolist()]


def _verdict(flagged_count: int) -> str:
    if flagged_count == 0:
        return "no suspicious spikes detected"
    if flagged_count == 1:
        return "1 high-intensity spike detected"
    return f"{flagged_count} high-intensity spikes detected"


def analyze(code: str) -> dict[str, Any]:
    """Return the fixed JSON schema for a given code string."""
    line_feats = extract_line_features(code)

    snn = _load_snn()
    if snn is None:
        log.warning("mock mode: models/snn_weights.pt missing — scoring from features only")
        threshold = _MOCK_THRESHOLD
        scores = [(lf.line, _mock_score(lf.vector)) for lf in line_feats]
    else:
        threshold = snn["threshold"]
        raw = _snn_scores(snn["net"], [lf.vector for lf in line_feats])
        scores = list(zip((lf.line for lf in line_feats), raw))

    lines_out = [
        {"line": ln, "score": round(sc, 4), "flag": sc >= threshold}
        for ln, sc in scores
    ]
    flagged = [ln for ln, sc in scores if sc >= threshold]
    top_flagged = [
        ln for ln, _ in sorted(
            ((ln, sc) for ln, sc in scores if sc >= threshold),
            key=lambda pair: pair[1],
            reverse=True,
        )[:10]
    ]

    return {
        "verdict": _verdict(len(flagged)),
        "lines": lines_out,
        "top_flagged": top_flagged,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sample = sys.stdin.read() if not sys.stdin.isatty() else "def f(x):\n    if x:\n        for i in range(x):\n            print(i)\n"
    print(json.dumps(analyze(sample), indent=2))