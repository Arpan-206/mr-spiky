"""Stage 1 — unsupervised STDP pretraining.

Rate-coded training (the sequence-mode STDP experiment saturated fc2 to zero
because dense graded-current inputs overpower the multiplicative depression
rule that was tuned for sparse binary spikes — see the notes in encode.py
about `encode_sequence`. Inference still uses sequence mode via
`per_timestep_attribution` in model.py, so lines influence each other at
inference time even though STDP itself was trained line-independently).

STDP rule:
    Δw⁺ = +A⁺ · pre_trace · post_spike                 (potentiation)
    Δw⁻ = -A⁻ · post_trace · pre_spike · current_w     (multiplicative depression)

Output: models/snn_weights.pt (state_dict of SpikyNet).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .encode import DEFAULT_STEPS, encode_batch
from .features import extract_function_features
from .model import SpikyNet

log = logging.getLogger("mrspiky.train")

ROOT = Path(__file__).resolve().parent.parent
# The pretraining corpus is what the SNN's STDP defines as "normal." We prefer
# the curated senior-approved corpus (CPython/Django/FastAPI/etc — code that
# shipped past reviewers at organizations with strong review culture). If it's
# not present, fall back to CodeSearchNet (generic scraped Python). This
# choice matters for the pitch: it's the difference between the SNN encoding
# "senior engineer intuition" vs "typical GitHub Python."
_SENIOR_CORPUS = ROOT / "data" / "senior_corpus.json"
_CSN_CORPUS = ROOT / "data" / "codesearchnet_python.json"
DATA_PATH = _SENIOR_CORPUS if _SENIOR_CORPUS.exists() else _CSN_CORPUS
WEIGHTS_PATH = ROOT / "models" / "snn_weights.pt"

# STDP hyperparameters. Depression dominates potentiation and scales with the
# current weight (multiplicative depression) so weights don't stampede to W_MAX.
A_PLUS = 0.005
A_MINUS = 0.020
TAU_TRACE = 20.0
W_MIN = 0.0
W_MAX = 1.0
WEIGHT_DECAY = 0.001
EPOCHS = 6
BATCH_SIZE = 16


def _stdp_update(
    layer_weight: torch.Tensor,
    pre_spikes: torch.Tensor,   # (T, B, in) — B is always 1 in sequence mode
    post_spikes: torch.Tensor,  # (T, B, out)
) -> None:
    """Pair-based STDP, in-place. layer_weight shape: (out, in)."""
    T, B, in_dim = pre_spikes.shape
    _, _, out_dim = post_spikes.shape
    decay = float(torch.exp(torch.tensor(-1.0 / TAU_TRACE)))

    pre_trace = torch.zeros(B, in_dim)
    post_trace = torch.zeros(B, out_dim)
    dw = torch.zeros_like(layer_weight)

    for t in range(T):
        pre_trace = pre_trace * decay + pre_spikes[t]
        post_trace = post_trace * decay + post_spikes[t]

        pot = torch.bmm(post_spikes[t].unsqueeze(2), pre_trace.unsqueeze(1)).sum(0)
        dep = torch.bmm(post_trace.unsqueeze(2), pre_spikes[t].unsqueeze(1)).sum(0)
        dw += A_PLUS * pot - A_MINUS * dep * layer_weight

    with torch.no_grad():
        layer_weight.mul_(1.0 - WEIGHT_DECAY)
        layer_weight.add_(dw / max(B, 1))
        layer_weight.clamp_(W_MIN, W_MAX)


def _load_functions() -> list[list[float]]:
    if not DATA_PATH.exists():
        log.warning(
            "no dataset at %s — falling back to a tiny built-in sample.",
            DATA_PATH,
        )
        sources = [
            "def add(a, b):\n    return a + b\n",
            "def is_even(n):\n    return n % 2 == 0\n",
            "def clamp(x, lo, hi):\n    if x < lo: return lo\n    if x > hi: return hi\n    return x\n",
            "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)\n",
            "def fizzbuzz(n):\n    for i in range(n):\n        if i % 15 == 0: print('fb')\n        elif i % 3 == 0: print('f')\n        elif i % 5 == 0: print('b')\n",
        ]
    else:
        sources = json.loads(DATA_PATH.read_text())

    vectors: list[list[float]] = []
    for src in sources:
        for fn in extract_function_features(src):
            vectors.append(fn.vector)
    log.info("loaded %d function feature vectors from %d source samples", len(vectors), len(sources))
    return vectors


def train() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    vectors = _load_functions()
    if not vectors:
        raise SystemExit("no training vectors — check dataset")

    net = SpikyNet()
    net.eval()

    for epoch in range(EPOCHS):
        perm = torch.randperm(len(vectors)).tolist()
        for start in range(0, len(vectors), BATCH_SIZE):
            batch = [vectors[i] for i in perm[start : start + BATCH_SIZE]]
            spikes_in = encode_batch(batch, num_steps=DEFAULT_STEPS)
            with torch.no_grad():
                out = net(spikes_in)
            _stdp_update(net.fc1.weight.data, spikes_in, out.hidden_spikes)
            _stdp_update(net.fc2.weight.data, out.hidden_spikes, out.output_spikes)

        log.info(
            "epoch %d/%d  fc1 mean=%.3f std=%.3f  fc2 mean=%.3f std=%.3f",
            epoch + 1, EPOCHS,
            net.fc1.weight.data.mean().item(), net.fc1.weight.data.std().item(),
            net.fc2.weight.data.mean().item(), net.fc2.weight.data.std().item(),
        )

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), WEIGHTS_PATH)
    log.info("saved weights -> %s", WEIGHTS_PATH)


if __name__ == "__main__":
    train()
