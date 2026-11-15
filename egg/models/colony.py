"""The neuron-agent colony.

Layout: neurons sit on a rows x cols grid that wraps around at the edges
(a torus). Every neuron:

  - has a private state vector h (size `hidden`),
  - emits a message m = tanh(h W_msg + b_msg) (size `message`),
  - receives the messages of its 4 neighbours, in fixed slots
    (from above, below, left, right), so direction is known,
  - knows where it is (a fixed position code: sin/cos of row and column,
    plus "am I an input neuron" and "am I an output neuron" flags),
  - updates its state with a gated (GRU-style) rule.

ALL neurons use the SAME weights: the shared genome ("the soul"). They
differ only in their state, position and what reaches them.

Input neurons (left column) also receive the embedding of the current
input symbol. Output neurons (right column) are read out with a shared
linear map; their logits are averaged.

Per external time step (one input symbol in, one output symbol out) the
colony runs `ticks` internal update rounds, so signals can travel
`ticks` neurons across the grid per symbol.
"""

from __future__ import annotations

import math

import torch

from egg.genome import GenomeSpec, ParamDef
from egg.models.gru import gru_step

POS_FEATURES = 6


def position_code(rows: int, cols: int) -> torch.Tensor:
    """(rows*cols, 6): sin/cos of row angle, sin/cos of column angle, is_input, is_output."""
    r = torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, cols)
    c = torch.arange(cols, dtype=torch.float32)[None, :].expand(rows, cols)
    ra, ca = 2 * math.pi * r / rows, 2 * math.pi * c / cols
    is_in = (c == 0).float()
    is_out = (c == cols - 1).float()
    code = torch.stack([ra.sin(), ra.cos(), ca.sin(), ca.cos(), is_in, is_out], dim=-1)
    return code.reshape(rows * cols, POS_FEATURES)


class Colony:
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        rows: int = 6,
        cols: int = 6,
        hidden: int = 8,
        message: int = 4,
        embed: int = 8,
        ticks: int = 3,
    ) -> None:
        if cols < 2:
            raise ValueError("need at least 2 columns (input and output column)")
        self.num_inputs, self.num_outputs = num_inputs, num_outputs
        self.rows, self.cols = rows, cols
        self.hidden, self.message, self.embed, self.ticks = hidden, message, embed, ticks
        self.num_neurons = rows * cols
        self.in_features = 4 * message + embed + POS_FEATURES
        h3 = 3 * hidden
        self.spec = GenomeSpec([
            ParamDef("w_msg", (hidden, message)),
            ParamDef("b_msg", (message,), "zeros"),
            ParamDef("w_in", (num_inputs, embed)),
            ParamDef("w_x", (self.in_features, h3)),
            ParamDef("b_x", (h3,), "zeros"),
            ParamDef("w_h", (hidden, h3)),
            ParamDef("b_h", (h3,), "zeros"),
            ParamDef("w_out", (hidden, num_outputs)),
            ParamDef("b_out", (num_outputs,), "zeros"),
        ])
        self._pos = position_code(rows, cols)

    def _neighbour_messages(self, m: torch.Tensor) -> torch.Tensor:
        """m: (P, B, rows, cols, M) -> (P, B, rows, cols, 4M) in fixed direction slots."""
        from_above = torch.roll(m, shifts=1, dims=2)
        from_below = torch.roll(m, shifts=-1, dims=2)
        from_left = torch.roll(m, shifts=1, dims=3)
        from_right = torch.roll(m, shifts=-1, dims=3)
        return torch.cat([from_above, from_below, from_left, from_right], dim=-1)

    def forward(
        self, genomes: torch.Tensor, tokens: torch.Tensor, record: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the colony over a token sequence.

        With record=True also returns the state of every neuron after every
        tick, shape (P, B, time*ticks, rows, cols, hidden), for visualization.
        """
        w = self.spec.unflatten(genomes)
        device = genomes.device
        pop = w["w_x"].shape[0]
        batch, steps = tokens.shape
        R, C, H, N = self.rows, self.cols, self.hidden, self.num_neurons

        pos = self._pos.to(device)[None, None].expand(pop, batch, N, POS_FEATURES)
        input_mask = (pos[..., 4:5] > 0.5).float()          # (P, B, N, 1)
        out_cols = slice(C - 1, C)

        h = torch.zeros(pop, batch, R, C, H, device=device)
        token_emb = w["w_in"][:, tokens]                     # (P, B, T, E)
        logits, states = [], []
        for t in range(steps):
            inject = token_emb[:, :, t, None, :] * input_mask  # (P, B, N, E)
            for _ in range(self.ticks):
                m = torch.tanh(
                    torch.einsum("pbrch,phm->pbrcm", h, w["w_msg"]) + w["b_msg"][:, None, None, None, :]
                )
                msgs = self._neighbour_messages(m).reshape(pop, batch, N, 4 * self.message)
                x = torch.cat([msgs, inject, pos], dim=-1).reshape(pop, batch * N, self.in_features)
                x_proj = torch.bmm(x, w["w_x"]) + w["b_x"][:, None, :]
                h_flat = gru_step(x_proj, h.reshape(pop, batch * N, H), w["w_h"], w["b_h"])
                h = h_flat.reshape(pop, batch, R, C, H)
                if record:
                    states.append(h)
            out_h = h[:, :, :, out_cols, :].reshape(pop, batch, -1, H)   # (P, B, rows, H)
            out = torch.einsum("pbkh,pho->pbko", out_h, w["w_out"]).mean(dim=2)
            logits.append(out + w["b_out"][:, None, :])
        result = torch.stack(logits, dim=2)
        if record:
            return result, torch.stack(states, dim=2)
        return result
