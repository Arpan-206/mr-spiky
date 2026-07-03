"""Small 2-layer LIF SNN. Kept tiny on purpose — this is a hackathon and the
signal we care about is the spike pattern, not raw predictive accuracy.

Input: (T, B, NUM_FEATURES) spike tensor from encode.encode_batch
Output: (T, B, HIDDEN) hidden-layer spikes, plus the final readout spikes.

We expose the hidden-layer spike trace because Temporal Spike Attribution (TSA)
needs it — TSA sums spikes weighted by their timestep to produce a per-neuron
"intensity" score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from snntorch import surrogate
from snntorch import Leaky

from .features import NUM_FEATURES

HIDDEN = 64   # bumped from 16 — with 64 neurons, mean-of-firing has enough
              # granularity to produce continuous-looking scores. With 16
              # neurons the mean collapses to ~5 discrete values which
              # saturated the ECDF calibration on real code.
OUTPUT = 16   # bumped 4 -> 16 for the same reason
BETA = 0.9    # membrane decay
LIF1_THRESHOLD = 1.0
LIF2_THRESHOLD = 1.0


@dataclass
class SNNOutput:
    hidden_spikes: torch.Tensor  # (T, B, HIDDEN)  — binary
    output_spikes: torch.Tensor  # (T, B, OUTPUT)  — binary
    hidden_mem: torch.Tensor     # (T, B, HIDDEN)  — continuous membrane
    output_mem: torch.Tensor     # (T, B, OUTPUT)  — continuous membrane


class SpikyNet(nn.Module):
    def __init__(self, input_dim: int = NUM_FEATURES, hidden: int = HIDDEN, output: int = OUTPUT):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(input_dim, hidden, bias=False)
        self.lif1 = Leaky(beta=BETA, spike_grad=spike_grad, threshold=LIF1_THRESHOLD)
        self.fc2 = nn.Linear(hidden, output, bias=False)
        self.lif2 = Leaky(beta=BETA, spike_grad=spike_grad, threshold=LIF2_THRESHOLD)

        # STDP-style init: small non-negative weights so early training is stable.
        nn.init.uniform_(self.fc1.weight, 0.0, 0.5)
        nn.init.uniform_(self.fc2.weight, 0.0, 0.5)

    def forward(self, spikes_in: torch.Tensor) -> SNNOutput:
        """spikes_in: (T, B, F). Returns SNNOutput with spike traces + membrane
        traces. Membranes are the continuous 'about-to-fire' potentials — used
        for smooth per-line attribution that doesn't collapse to discrete
        firing-rate values."""
        T, B, _ = spikes_in.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        h_spk_trace = []
        o_spk_trace = []
        h_mem_trace = []
        o_mem_trace = []
        for t in range(T):
            cur1 = self.fc1(spikes_in[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            h_spk_trace.append(spk1)
            o_spk_trace.append(spk2)
            h_mem_trace.append(mem1)
            o_mem_trace.append(mem2)

        return SNNOutput(
            hidden_spikes=torch.stack(h_spk_trace, dim=0),
            output_spikes=torch.stack(o_spk_trace, dim=0),
            hidden_mem=torch.stack(h_mem_trace, dim=0),
            output_mem=torch.stack(o_mem_trace, dim=0),
        )


def temporal_spike_attribution(hidden_spikes: torch.Tensor) -> torch.Tensor:
    """TSA collapsed to a scalar per batch item.

    Used by the *line-independent* rate-coded path: each batch item is one
    line, we sum over T=25 rate-coded steps to get a single intensity.

    hidden_spikes: (T, B, H) -> intensity: (B,) in [0, 1].
    """
    T = hidden_spikes.shape[0]
    if T == 0:
        return torch.zeros(hidden_spikes.shape[1])
    time_weights = torch.linspace(0.5, 1.0, T, device=hidden_spikes.device).view(T, 1, 1)
    weighted = (hidden_spikes * time_weights).sum(dim=(0, 2))  # (B,)
    denom = float(T * hidden_spikes.shape[2])
    return weighted / max(denom, 1.0)


def ecdf_rescale(raw_scores: list[float], sorted_ref: torch.Tensor) -> list[float]:
    """Map raw SNN scores to their percentile rank in a reference distribution.

    `sorted_ref` is the sorted array of raw scores over the senior corpus.
    A raw score at the 50th-percentile of the ref becomes 0.5; at the 99th,
    0.99; below the lowest, 0.0. This turns the discrete/saturated raw output
    into a smooth [0,1] scale where the threshold means 'top-X% for senior code.'
    """
    if not raw_scores or sorted_ref.numel() == 0:
        return raw_scores
    n = float(sorted_ref.numel())
    out: list[float] = []
    for s in raw_scores:
        # right-search position gives count of ref-scores <= s
        pos = float(torch.searchsorted(sorted_ref, torch.tensor(s), right=True))
        out.append(max(0.0, min(1.0, pos / n)))
    return out


def per_timestep_attribution(
    hidden_spikes: torch.Tensor,
    output_spikes: torch.Tensor | None = None,
    hidden_baseline: torch.Tensor | None = None,
    output_baseline: torch.Tensor | None = None,
    hidden_mem: torch.Tensor | None = None,
    output_mem: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sequence-mode attribution: one score per timestep (line).

    If membrane traces are given (hidden_mem, output_mem), we use those
    directly — membrane potentials are continuous, so scores don't collapse
    to the ~5 discrete values that binary spike averages produce. This is
    what makes per-line scores actually distribute smoothly across [0, 1].

    Membrane is *how close the neuron got to firing* — a strictly richer
    signal than the binary spike output. A neuron that hovers at 0.9 threshold
    but doesn't quite spike is still "excited by this line," and the
    membrane captures that.

    Fallback: if only spikes are given, we compute the mean-firing attribution
    (used to be the primary path). Still supported for legacy calls.

    hidden_baseline / output_baseline: per-neuron mean firing rate on the
    training corpus. Score is how much this line's activation exceeds those
    baselines — the SNN's "unusual for senior code" signal.

    Returns (T, B) in [0, 1].
    """
    T = hidden_spikes.shape[0]
    if T == 0:
        return torch.zeros(T, hidden_spikes.shape[1])

    if hidden_mem is not None:
        # Membrane potentials can range above 1.0 (they accumulate). Squash
        # via sigmoid centered at the LIF threshold so 0.5 = "at threshold."
        h_activation = torch.sigmoid(4.0 * (hidden_mem - LIF1_THRESHOLD))  # (T,B,H)
        if hidden_baseline is not None:
            # Baseline is a per-neuron firing rate; convert to same
            # sigmoid-activation space by using it as a scalar shift.
            base = hidden_baseline.view(1, 1, -1).clamp(0.0, 1.0)
            h_activation = (h_activation - base).clamp(min=0.0) / (1.0 - base + 1e-6)
        h_score = h_activation.mean(dim=2)  # (T, B)
        if output_mem is not None:
            o_activation = torch.sigmoid(4.0 * (output_mem - LIF2_THRESHOLD))
            if output_baseline is not None:
                base = output_baseline.view(1, 1, -1).clamp(0.0, 1.0)
                o_activation = (o_activation - base).clamp(min=0.0) / (1.0 - base + 1e-6)
            o_score = o_activation.mean(dim=2)
            return (0.5 * h_score + 0.5 * o_score).clamp(0.0, 1.0)
        return h_score.clamp(0.0, 1.0)

    # Legacy spike-only path.
    if hidden_baseline is not None:
        excess = (hidden_spikes - hidden_baseline.view(1, 1, -1)).clamp(min=0.0)
        max_excess = (1.0 - hidden_baseline).clamp(min=1e-6)
        hidden_score = (excess / max_excess.view(1, 1, -1)).mean(dim=2)
        if output_spikes is not None and output_baseline is not None:
            o_excess = (output_spikes - output_baseline.view(1, 1, -1)).clamp(min=0.0)
            o_max = (1.0 - output_baseline).clamp(min=1e-6)
            output_score = (o_excess / o_max.view(1, 1, -1)).mean(dim=2)
            return (0.6 * hidden_score + 0.4 * output_score).clamp(0.0, 1.0)
        return hidden_score.clamp(0.0, 1.0)

    hidden = hidden_spikes.mean(dim=2)
    if output_spikes is None or output_spikes.numel() == 0:
        return hidden
    out = output_spikes.mean(dim=2)
    return 0.7 * hidden + 0.3 * out