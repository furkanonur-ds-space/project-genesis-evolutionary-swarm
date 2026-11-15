"""Baseline: a standard GRU network (Cho et al. 2014).

One monolithic recurrent network with a single hidden state vector.
Trained with the same ES as the colony so that only the architecture
differs in the comparison.
"""

from __future__ import annotations

import torch

from egg.genome import GenomeSpec, ParamDef


def gru_step(x_proj: torch.Tensor, h: torch.Tensor, w_h: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    """One batched GRU update.

    x_proj: (P, R, 3H) input already projected (includes input bias)
    h:      (P, R, H)
    w_h:    (P, H, 3H) recurrent weights, b_h: (P, 3H)
    """
    hidden = h.shape[-1]
    h_proj = torch.bmm(h, w_h) + b_h[:, None, :]
    xr, xz, xn = x_proj.split(hidden, dim=-1)
    hr, hz, hn = h_proj.split(hidden, dim=-1)
    r = torch.sigmoid(xr + hr)          # reset gate: how much old state to use
    z = torch.sigmoid(xz + hz)          # update gate: keep old vs take new
    n = torch.tanh(xn + r * hn)         # candidate new state
    return (1 - z) * n + z * h


class GRUBaseline:
    def __init__(self, num_inputs: int, num_outputs: int, hidden: int, learn_h0: bool = False) -> None:
        self.learn_h0 = learn_h0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden = hidden
        h3 = 3 * hidden
        extra = [ParamDef("h0", (hidden,), "zeros")] if learn_h0 else []
        self.spec = GenomeSpec(extra + [
            ParamDef("w_x", (num_inputs, h3)),
            ParamDef("b_x", (h3,), "zeros"),
            ParamDef("w_h", (hidden, h3)),
            ParamDef("b_h", (h3,), "zeros"),
            ParamDef("w_out", (hidden, num_outputs)),
            ParamDef("b_out", (num_outputs,), "zeros"),
        ])

    def forward(self, genomes: torch.Tensor, tokens: torch.Tensor, ticks: int | None = None) -> torch.Tensor:
        """ticks is accepted for interface compatibility with Colony and ignored."""
        w = self.spec.unflatten(genomes)
        pop = genomes.shape[0] if genomes.dim() == 2 else 1
        batch, steps = tokens.shape
        h = torch.zeros(pop, batch, self.hidden, device=genomes.device)
        if self.learn_h0:
            h = h + w["h0"][:, None, :]
        # One-hot input times w_x is just a row lookup.
        x_proj_all = w["w_x"][:, tokens] + w["b_x"][:, None, None, :]  # (P, B, T, 3H)
        outputs = []
        for t in range(steps):
            h = gru_step(x_proj_all[:, :, t], h, w["w_h"], w["b_h"])
            outputs.append(torch.bmm(h, w["w_out"]) + w["b_out"][:, None, :])
        return torch.stack(outputs, dim=2)
