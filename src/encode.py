"""Encode feature vectors as inputs to the SNN.

Two modes:

`encode_batch(vectors, num_steps)` — original mode, still used by the
line-independent inference path in the mock/fallback code. Each vector is
rate-coded into `num_steps` Bernoulli spike samples.

`encode_sequence(vectors)` — Tier 3 sequence mode. Each vector's normalized
values become a **graded current input** at one timestep, so the SNN sees the
lines *in order* and its LIF membrane state carries context across lines.
This is the mode that actually uses the SNN's temporal dynamics — it lets STDP
learn line-to-line correlations rather than treating each line in isolation.

Shapes:
    encode_batch:    (T=num_steps,  B=len(vectors), F)   — rate-coded spikes {0,1}
    encode_sequence: (T=len(vectors), B=1,           F)   — graded currents in [0,1]
"""

from __future__ import annotations

import torch
from snntorch import spikegen

from .features import NUM_FEATURES, normalize

DEFAULT_STEPS = 25


def encode_vector(vector: list[float], num_steps: int = DEFAULT_STEPS) -> torch.Tensor:
    """Rate-encode a single feature vector. Returns (num_steps, num_features)."""
    norm = normalize(vector)
    data = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)  # (1, F)
    spikes = spikegen.rate(data, num_steps=num_steps)  # (T, 1, F)
    return spikes.squeeze(1)  # (T, F)


def encode_batch(vectors: list[list[float]], num_steps: int = DEFAULT_STEPS) -> torch.Tensor:
    """Batch rate-coded encoding. Returns (num_steps, batch, num_features).

    Used by the *line-independent* path (each line scored separately). Kept for
    backward compat and for the mock-mode / fallback branches.
    """
    if not vectors:
        return torch.zeros(num_steps, 0, NUM_FEATURES, dtype=torch.float32)
    norm = torch.tensor([normalize(v) for v in vectors], dtype=torch.float32)  # (B, F)
    return spikegen.rate(norm, num_steps=num_steps)  # (T, B, F)


def encode_sequence(vectors: list[list[float]]) -> torch.Tensor:
    """Sequence encoding: each vector's normalized values are the graded input
    current at one timestep. Returns (T=len(vectors), B=1, F).

    The SNN then processes the lines *in order*, with LIF membrane state
    carrying context from line i-1 into line i. This is what unlocks
    line-to-line temporal correlations that the rate-coded path can't see.
    """
    if not vectors:
        return torch.zeros(0, 1, NUM_FEATURES, dtype=torch.float32)
    norm = torch.tensor([normalize(v) for v in vectors], dtype=torch.float32)  # (T, F)
    return norm.unsqueeze(1)  # (T, 1, F)