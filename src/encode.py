"""Encode feature vectors as inputs to the SNN, with optional whitening.

The 9 raw features are heavily correlated on real Python (e.g. length ↔
token_entropy at r=0.84, cyclomatic_proxy ↔ call_graph_shape at r=0.92), so
STDP effectively saw only ~4 independent signal dimensions and homogenized
its hidden layer into 2 clusters. Applying a ZCA whitening transform (fitted
once on the senior corpus, stored in models/whitening.pt) before encoding
gives STDP 9 orthogonal dimensions to specialize on — the fix at the
mathematical level rather than throwing more neurons at the problem.

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

from pathlib import Path

import torch
from snntorch import spikegen

from .features import NUM_FEATURES, normalize

DEFAULT_STEPS = 25

_WHITENING_PATH = Path(__file__).resolve().parent.parent / "models" / "whitening.pt"
_whitening_cache: dict[str, torch.Tensor] | None = None


def _load_whitening() -> dict[str, torch.Tensor] | None:
    """Lazy-load the whitening transform. Returns dict{"mean": (F,), "W": (F, F)}
    or None if not fitted yet (then encoding falls back to unwhitened normalize)."""
    global _whitening_cache
    if _whitening_cache is not None:
        return _whitening_cache
    if not _WHITENING_PATH.exists():
        return None
    _whitening_cache = torch.load(_WHITENING_PATH, weights_only=True)
    return _whitening_cache


def fit_whitening(vectors: list[list[float]], epsilon: float = 1e-3) -> dict[str, torch.Tensor]:
    """Compute a ZCA whitening transform from normalized feature vectors.

    Steps: center, compute covariance, take symmetric inverse square root.
    Applying W · (x - mean) yields x_white with identity covariance — every
    output dimension is uncorrelated with every other and has unit variance.

    The result is *rescaled* back into a roughly [0, 1] range (divided by
    2·max_abs) so it's a valid rate-coding input for the SNN.
    """
    X = torch.tensor([normalize(v) for v in vectors], dtype=torch.float32)  # (N, F)
    mean = X.mean(dim=0)                                                     # (F,)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)                              # (F, F)
    # ZCA: W = Cov^(-1/2). Compute via eigen-decomposition (symmetric matrix).
    eigvals, eigvecs = torch.linalg.eigh(cov)                                # ascending
    inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals.clamp(min=epsilon)))
    W = eigvecs @ inv_sqrt @ eigvecs.T                                       # (F, F)
    # Rescale so whitened outputs land back in a similar range to normalized
    # ones (Bernoulli rate-coding wants inputs in [0, 1]).
    scale = float((Xc @ W.T).abs().quantile(0.99)) * 2.0
    W = W / max(scale, 1e-6)
    return {"mean": mean, "W": W}


def whiten(vector: list[float], transform: dict[str, torch.Tensor] | None = None) -> list[float]:
    """Apply whitening transform. Falls back to normalize() if no transform."""
    norm = normalize(vector)
    t = transform if transform is not None else _load_whitening()
    if t is None:
        return norm
    x = torch.tensor(norm, dtype=torch.float32) - t["mean"]
    y = t["W"] @ x
    # Shift + clamp back into [0, 1] (whitened values are ~ mean-zero).
    y = (y + 0.5).clamp(0.0, 1.0)
    return [float(v) for v in y.tolist()]


def encode_vector(vector: list[float], num_steps: int = DEFAULT_STEPS) -> torch.Tensor:
    """Rate-encode a single feature vector. Returns (num_steps, num_features)."""
    norm = normalize(vector)
    data = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)  # (1, F)
    spikes = spikegen.rate(data, num_steps=num_steps)  # (T, 1, F)
    return spikes.squeeze(1)  # (T, F)


def encode_batch(vectors: list[list[float]], num_steps: int = DEFAULT_STEPS) -> torch.Tensor:
    """Batch rate-coded encoding. Returns (num_steps, batch, num_features).

    Used by the STDP training loop and the line-independent inference path.
    Whitens vectors (if the transform is fitted) so STDP sees decorrelated
    features and hidden neurons can specialize on independent signal
    dimensions instead of collapsing into a few clusters.
    """
    if not vectors:
        return torch.zeros(num_steps, 0, NUM_FEATURES, dtype=torch.float32)
    transform = _load_whitening()
    encoded = torch.tensor(
        [whiten(v, transform) for v in vectors],
        dtype=torch.float32,
    )  # (B, F)
    return spikegen.rate(encoded, num_steps=num_steps)  # (T, B, F)


def encode_sequence(vectors: list[list[float]]) -> torch.Tensor:
    """Sequence encoding: each vector's whitened values are the graded input
    current at one timestep. Returns (T=len(vectors), B=1, F).

    The SNN then processes the lines *in order*, with LIF membrane state
    carrying context from line i-1 into line i. This is what unlocks
    line-to-line temporal correlations that the rate-coded path can't see.

    Whitening (applied here if models/whitening.pt exists) decorrelates the
    input so STDP sees 9 orthogonal signal dimensions instead of ~4 tangled
    ones. Falls back to plain `normalize` for the first-time bootstrap.
    """
    if not vectors:
        return torch.zeros(0, 1, NUM_FEATURES, dtype=torch.float32)
    transform = _load_whitening()
    encoded = torch.tensor(
        [whiten(v, transform) for v in vectors],
        dtype=torch.float32,
    )  # (T, F)
    return encoded.unsqueeze(1)  # (T, 1, F)