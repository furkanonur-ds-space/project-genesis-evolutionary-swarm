"""Train a sequence model (colony or GRU baseline) on the serial copy task with ES.

Usage:
    python -m egg.train_copy --config configs/e001_colony.toml
    python -m egg.train_copy --config configs/e001_gru.toml --set run.seed=1

Fitness of a lifetime = mean log probability of the correct digit on the
write steps. All lifetimes in a generation see the same sequences
(common random numbers), so differences in fitness come from the genome,
not from luckier sequences. The center genome is scored on a fixed
validation set that never changes during training.
"""

from __future__ import annotations

import argparse
import time

import torch

from egg.checkpoint import load_checkpoint, save_checkpoint
from egg.config import Config, load_config
from egg.device import resolve_device
from egg.es import OpenAIES
from egg.models import Colony, GRUBaseline
from egg.run import RunDir
from egg.seeding import seed_everything
from egg.tasks.copy_task import CopyTask, score

VAL_SEED_OFFSET = 999_983


def build_model(config: Config, task: CopyTask):
    if config.model.kind == "colony":
        c = config.colony
        return Colony(task.num_inputs, task.num_outputs, rows=c.rows, cols=c.cols, hidden=c.hidden,
                      message=c.message, embed=c.embed, ticks=c.ticks, wrap_cols=c.wrap_cols,
                      readout_ticks=c.readout_ticks, learn_h0=c.learn_h0)
    if config.model.kind == "gru":
        return GRUBaseline(task.num_inputs, task.num_outputs, hidden=config.gru.hidden,
                           learn_h0=config.gru.learn_h0)
    raise ValueError(f"unknown model kind {config.model.kind!r}")


def evaluate(model, genomes: torch.Tensor, batch, chunk: int, ticks: int | None = None) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        if not chunk or chunk >= genomes.shape[0]:
            return score(model.forward(genomes, batch.tokens, ticks=ticks), batch)
        parts = [score(model.forward(g, batch.tokens, ticks=ticks), batch) for g in genomes.split(chunk)]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def train(config: Config, resume: str | None = None, quiet: bool = False):
    seed_everything(config.run.seed)
    device = resolve_device(config.run.device)
    task = CopyTask(config.copy.digits, config.copy.length)
    model = build_model(config, task)

    init_gen = torch.Generator().manual_seed(config.run.seed)
    es = OpenAIES(
        model.spec.init(init_gen).to(device),
        population=config.es.population, sigma=config.es.sigma, lr=config.es.lr,
        weight_decay=config.es.weight_decay, seed=config.run.seed,
        lr_decay=config.es.lr_decay, lr_min=config.es.lr_min,
        sigma_decay=config.es.sigma_decay, sigma_min=config.es.sigma_min,
    )
    if resume:
        es.load_state_dict(load_checkpoint(resume, map_location=device))

    val_gen = torch.Generator().manual_seed(config.run.seed + VAL_SEED_OFFSET)
    val_batch = task.sample(config.copy.val_batch, val_gen, device)
    # Generalization test: lengths never (or not all) seen in training (Graves et al. 2014).
    test_batches = {
        n: task.sample(config.copy.val_batch, torch.Generator().manual_seed(config.run.seed + VAL_SEED_OFFSET + n),
                       device, length=n)
        for n in config.copy.test_lengths
    }
    min_len = config.copy.min_length or config.copy.length
    if not 1 <= min_len <= config.copy.length:
        raise ValueError("copy.min_length must be between 1 and copy.length")
    c = config.colony
    ticks_hi = max(c.ticks, c.ticks_max)

    run = RunDir(config)
    if not quiet:
        print(f"model {config.model.kind}: {model.spec.size} parameters, device {device}")
    start = time.perf_counter()
    chunk = config.model.eval_chunk
    try:
        while es.generation < config.es.generations:
            gen = torch.Generator().manual_seed(config.run.seed * 7919 + es.generation)
            # Length and tick count are drawn once per generation, so every
            # lifetime in the generation faces the same conditions.
            # Draw only when a range is set, so fixed settings consume exactly
            # the same random numbers as before these options existed.
            length = config.copy.length
            if min_len < config.copy.length:
                length = int(torch.randint(min_len, config.copy.length + 1, (1,), generator=gen))
            ticks = c.ticks
            if ticks_hi > c.ticks:
                ticks = int(torch.randint(c.ticks, ticks_hi + 1, (1,), generator=gen))
            batch = task.sample(config.copy.batch, gen, device, length=length)
            candidates = es.ask()
            fitness = evaluate(model, candidates, batch, chunk, ticks=ticks)["log_prob"]
            stats = es.tell(fitness)
            g = es.generation
            if g % config.es.log_every == 0 or g == config.es.generations:
                val = evaluate(model, es.theta.detach()[None], val_batch, chunk)
                row = {
                    "generation": g,
                    "val_log_prob": round(float(val["log_prob"][0]), 5),
                    "val_symbol_acc": round(float(val["symbol_acc"][0]), 4),
                    "val_seq_acc": round(float(val["seq_acc"][0]), 4),
                    "val_pred_entropy": round(float(val["pred_entropy"][0]), 4),
                    "pop_fitness_mean": round(stats["fitness_mean"], 5),
                    "pop_fitness_max": round(stats["fitness_max"], 5),
                    "grad_norm": round(stats["grad_norm"], 5),
                    "theta_norm": round(stats["theta_norm"], 3),
                    "sigma": round(stats["sigma"], 6),
                    "lr": round(stats["lr"], 6),
                }
                for n, tb in test_batches.items():
                    res = evaluate(model, es.theta.detach()[None], tb, chunk)
                    row[f"len{n}_symbol_acc"] = round(float(res["symbol_acc"][0]), 4)
                    row[f"len{n}_seq_acc"] = round(float(res["seq_acc"][0]), 4)
                row["seconds"] = round(time.perf_counter() - start, 2)
                run.metrics.log(**row)
                if not quiet:
                    print(f"gen {g:5d}  val_logp {row['val_log_prob']:8.4f}  "
                          f"sym_acc {row['val_symbol_acc']:.3f}  seq_acc {row['val_seq_acc']:.3f}  "
                          f"{row['seconds']:7.1f}s")
            if config.es.checkpoint_every and g % config.es.checkpoint_every == 0:
                save_checkpoint(run.path / "checkpoint.pt", es.state_dict())
        save_checkpoint(run.path / "checkpoint.pt", es.state_dict())
    finally:
        run.close()
    return es, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    _, run = train(config, resume=args.resume)
    print(f"run folder: {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
