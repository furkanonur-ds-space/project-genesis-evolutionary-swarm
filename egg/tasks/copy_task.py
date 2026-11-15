"""Serial copy task (as in Graves et al. 2014, Neural Turing Machines).

The model sees one symbol per step: first `length` random digits, then a
GO symbol, then BLANK symbols. From the GO step on, it must write the
digits back, one per step, in the same order.

    step:    0  1  2  3   4   5   6   7
    input:   3  1  4  2  GO   _   _   _
    target:  .  .  .  .   3   1   4   2      ("." = not scored)

The read phase can only be solved by remembering: when writing digit k,
the model no longer sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CopyBatch:
    tokens: torch.Tensor   # (B, T) input symbols
    targets: torch.Tensor  # (B, T) digit to write (0 where not scored)
    mask: torch.Tensor     # (B, T) 1.0 where the output is scored


class CopyTask:
    def __init__(self, digits: int, length: int) -> None:
        if digits < 2 or length < 1:
            raise ValueError("need digits >= 2 and length >= 1")
        self.digits = digits
        self.length = length
        self.go = digits
        self.blank = digits + 1
        self.num_inputs = digits + 2
        self.num_outputs = digits
        self.steps = 2 * length

    def sample(self, batch: int, generator: torch.Generator, device: torch.device | str = "cpu") -> CopyBatch:
        seq = torch.randint(0, self.digits, (batch, self.length), generator=generator)
        go = torch.full((batch, 1), self.go)
        blanks = torch.full((batch, self.length - 1), self.blank)
        tokens = torch.cat([seq, go, blanks], dim=1)
        targets = torch.cat([torch.zeros_like(seq), seq], dim=1)
        mask = torch.cat([torch.zeros(batch, self.length), torch.ones(batch, self.length)], dim=1)
        return CopyBatch(tokens.to(device), targets.to(device), mask.to(device))


def score(logits: torch.Tensor, batch: CopyBatch) -> dict[str, torch.Tensor]:
    """Per population member scores. logits: (P, B, T, digits).

    Returns tensors of shape (P,):
      log_prob:    mean log probability of the correct digit on scored steps
                   (smooth; used as fitness)
      symbol_acc:  fraction of scored digits written correctly
      seq_acc:     fraction of sequences copied with zero mistakes
    """
    logp = torch.log_softmax(logits, dim=-1)
    target = batch.targets[None, :, :, None].expand(logp.shape[0], -1, -1, 1)
    correct_logp = logp.gather(-1, target).squeeze(-1)              # (P, B, T)
    mask = batch.mask[None]
    denom = mask.sum(dim=(1, 2))
    log_prob = (correct_logp * mask).sum(dim=(1, 2)) / denom
    hit = (logits.argmax(dim=-1) == batch.targets[None]).float() * mask
    symbol_acc = hit.sum(dim=(1, 2)) / denom
    per_seq_hits = hit.sum(dim=2)
    seq_acc = (per_seq_hits == batch.mask.sum(dim=1)[None]).float().mean(dim=1)
    return {"log_prob": log_prob, "symbol_acc": symbol_acc, "seq_acc": seq_acc}
