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
                      message=c.message, embed=c.embed, ticks=c.ticks)
    if config.model.kind == "gru":
        return GRUBaseline(task.num_inputs, task.num_outputs, hidden=config.gru.hidden)
    raise ValueError(f"unknown model kind {config.model.kind!r}")


def evaluate(model, genomes: torch.Tensor, batch, chunk: int) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        if not chunk or chunk >= genomes.shape[0]:
            return score(model.forward(genomes, batch.tokens), batch)
        parts = [score(model.forward(g, batch.tokens), batch) for g in genomes.split(chunk)]
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
    )
    if resume:
        es.load_state_dict(load_checkpoint(resume, map_location=device))

    val_gen = torch.Generator().manual_seed(config.run.seed + VAL_SEED_OFFSET)
    val_batch = task.sample(config.copy.val_batch, val_gen, device)

    run = RunDir(config)
    if not quiet:
        print(f"model {config.model.kind}: {model.spec.size} parameters, device {device}")
    start = time.perf_counter()
    chunk = config.model.eval_chunk
    try:
        while es.generation < config.es.generations:
            gen = torch.Generator().manual_seed(config.run.seed * 7919 + es.generation)
            batch = task.sample(config.copy.batch, gen, device)
            candidates = es.ask()
            fitness = evaluate(model, candidates, batch, chunk)["log_prob"]
            stats = es.tell(fitness)
            g = es.generation
            if g % config.es.log_every == 0 or g == config.es.generations:
                val = evaluate(model, es.theta.detach()[None], val_batch, chunk)
                row = {
                    "generation": g,
                    "val_log_prob": round(float(val["log_prob"][0]), 5),
                    "val_symbol_acc": round(float(val["symbol_acc"][0]), 4),
                    "val_seq_acc": round(float(val["seq_acc"][0]), 4),
                    "pop_fitness_mean": round(stats["fitness_mean"], 5),
                    "pop_fitness_max": round(stats["fitness_max"], 5),
                    "grad_norm": round(stats["grad_norm"], 5),
                    "theta_norm": round(stats["theta_norm"], 3),
                    "seconds": round(time.perf_counter() - start, 2),
                }
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
