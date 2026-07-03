"""Stage 1 — unsupervised STDP pretraining.

Loads the CodeSearchNet Python sample cached by `data/download_codesearchnet.py`,
extracts function-level features, encodes them as spike trains, and trains
`SpikyNet` weights with a classical pair-based STDP rule:

    Δw = +A+ * pre_trace * post_spike      (potentiation on post-fire)
    Δw = -A- * post_trace * pre_spike      (depression on pre-fire)

Pre/post traces are exponentially decaying counters of recent spikes. There is
no target signal — the network self-organizes toward the statistics of "normal"
code, so anomalous code later produces atypical spike patterns.

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
DATA_PATH = ROOT / "data" / "codesearchnet_python.json"
WEIGHTS_PATH = ROOT / "models" / "snn_weights.pt"

# STDP hyperparameters. Depression dominates potentiation and scales with the
# current weight (multiplicative depression) so weights don't stampede to W_MAX.
# Empirically this is the standard fix for saturation in additive STDP.
A_PLUS = 0.005
A_MINUS = 0.020  # > A_PLUS so competition drives selectivity
TAU_TRACE = 20.0  # trace decay time-constant (in timesteps)
W_MIN = 0.0
W_MAX = 1.0
WEIGHT_DECAY = 0.001  # per-batch multiplicative decay toward 0
EPOCHS = 3
BATCH_SIZE = 16


def _stdp_update(
    layer_weight: torch.Tensor,
    pre_spikes: torch.Tensor,  # (T, B, in)
    post_spikes: torch.Tensor,  # (T, B, out)
) -> None:
    """Apply pair-based STDP to `layer_weight` in-place.

    layer_weight shape: (out, in) — matches nn.Linear convention.
    """
    T, B, in_dim = pre_spikes.shape
    _, _, out_dim = post_spikes.shape
    decay = float(torch.exp(torch.tensor(-1.0 / TAU_TRACE)))

    pre_trace = torch.zeros(B, in_dim)
    post_trace = torch.zeros(B, out_dim)
    dw = torch.zeros_like(layer_weight)

    for t in range(T):
        pre_trace = pre_trace * decay + pre_spikes[t]
        post_trace = post_trace * decay + post_spikes[t]

        # Potentiation (additive): post fires now, correlate with recent pre.
        # (B, out, 1) * (B, 1, in) -> (B, out, in) -> sum over batch
        pot = torch.bmm(post_spikes[t].unsqueeze(2), pre_trace.unsqueeze(1)).sum(0)
        # Depression (multiplicative): pre fires now, correlate with recent post.
        # Scaling by current weight prevents saturation at W_MAX.
        dep = torch.bmm(post_trace.unsqueeze(2), pre_spikes[t].unsqueeze(1)).sum(0)
        dw += A_PLUS * pot - A_MINUS * dep * layer_weight

    with torch.no_grad():
        layer_weight.mul_(1.0 - WEIGHT_DECAY)
        layer_weight.add_(dw / max(B, 1))
        layer_weight.clamp_(W_MIN, W_MAX)


def _load_functions() -> list[list[float]]:
    if not DATA_PATH.exists():
        log.warning(
            "no dataset at %s — falling back to a tiny built-in sample. "
            "run `just data-pretrain` for real training.",
            DATA_PATH,
        )
        fallback = [
            "def add(a, b):\n    return a + b\n",
            "def is_even(n):\n    return n % 2 == 0\n",
            "def clamp(x, lo, hi):\n    if x < lo: return lo\n    if x > hi: return hi\n    return x\n",
            "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)\n",
            "def fizzbuzz(n):\n    for i in range(n):\n        if i % 15 == 0: print('fizzbuzz')\n        elif i % 3 == 0: print('fizz')\n        elif i % 5 == 0: print('buzz')\n",
        ]
        sources = fallback
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
    net.eval()  # STDP doesn't use autograd; we mutate weights directly.

    for epoch in range(EPOCHS):
        # Shuffle each epoch.
        perm = torch.randperm(len(vectors)).tolist()
        for start in range(0, len(vectors), BATCH_SIZE):
            batch_idx = perm[start : start + BATCH_SIZE]
            batch = [vectors[i] for i in batch_idx]
            spikes_in = encode_batch(batch, num_steps=DEFAULT_STEPS)  # (T, B, F)

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