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

HIDDEN = 16
OUTPUT = 4
BETA = 0.9  # membrane decay


@dataclass
class SNNOutput:
    hidden_spikes: torch.Tensor  # (T, B, HIDDEN)
    output_spikes: torch.Tensor  # (T, B, OUTPUT)


class SpikyNet(nn.Module):
    def __init__(self, input_dim: int = NUM_FEATURES, hidden: int = HIDDEN, output: int = OUTPUT):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(input_dim, hidden, bias=False)
        self.lif1 = Leaky(beta=BETA, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden, output, bias=False)
        self.lif2 = Leaky(beta=BETA, spike_grad=spike_grad)

        # STDP-style init: small non-negative weights so early training is stable.
        nn.init.uniform_(self.fc1.weight, 0.0, 0.5)
        nn.init.uniform_(self.fc2.weight, 0.0, 0.5)

    def forward(self, spikes_in: torch.Tensor) -> SNNOutput:
        """spikes_in: (T, B, F). Returns SNNOutput with spike traces."""
        T, B, _ = spikes_in.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        h_trace = []
        o_trace = []
        for t in range(T):
            cur1 = self.fc1(spikes_in[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            h_trace.append(spk1)
            o_trace.append(spk2)

        return SNNOutput(
            hidden_spikes=torch.stack(h_trace, dim=0),
            output_spikes=torch.stack(o_trace, dim=0),
        )


def temporal_spike_attribution(hidden_spikes: torch.Tensor) -> torch.Tensor:
    """TSA: weight each spike by its (normalized) timestep and sum across time
    and neurons to get a scalar 'intensity' per batch item.

    Later timesteps get higher weight — the idea is that persistent late-stage
    firing is more indicative of anomaly than transient early firing.

    hidden_spikes: (T, B, H) -> intensity: (B,) in [0, 1]-ish after normalize.
    """
    T = hidden_spikes.shape[0]
    if T == 0:
        return torch.zeros(hidden_spikes.shape[1])
    time_weights = torch.linspace(0.5, 1.0, T, device=hidden_spikes.device).view(T, 1, 1)
    weighted = (hidden_spikes * time_weights).sum(dim=(0, 2))  # (B,)
    # Normalize by max possible: T * HIDDEN * 1.0
    denom = float(T * hidden_spikes.shape[2])
    return weighted / max(denom, 1.0)