"""Rate-code normalized feature vectors into spike trains for the SNN.

Each normalized feature in [0, 1] is treated as a Bernoulli firing probability
per timestep. Output shape: (num_steps, num_features), dtype float32.
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
    """Batch version. Returns (num_steps, batch, num_features)."""
    if not vectors:
        return torch.zeros(num_steps, 0, NUM_FEATURES, dtype=torch.float32)
    norm = torch.tensor([normalize(v) for v in vectors], dtype=torch.float32)  # (B, F)
    return spikegen.rate(norm, num_steps=num_steps)  # (T, B, F)