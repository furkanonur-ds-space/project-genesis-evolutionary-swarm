"""Train the ES on a toy objective and log the run.

Usage:
    python -m egg.train_toy --config configs/es_sphere.toml
    python -m egg.train_toy --config configs/es_sphere.toml --resume runs/<folder>/checkpoint.pt
"""

from __future__ import annotations

import argparse
import time

import torch

from egg.checkpoint import load_checkpoint, save_checkpoint
from egg.config import Config, load_config
from egg.device import resolve_device
from egg.es import OpenAIES
from egg.run import RunDir
from egg.seeding import seed_everything
from egg.toy import TOY_FUNCTIONS


def build(config: Config, device: torch.device) -> tuple[OpenAIES, object]:
    if config.toy.function not in TOY_FUNCTIONS:
        raise ValueError(f"unknown toy function {config.toy.function!r}; options {sorted(TOY_FUNCTIONS)}")
    gen = torch.Generator().manual_seed(config.run.seed)
    theta0 = torch.randn(config.toy.dim, generator=gen) * config.toy.init_scale
    es = OpenAIES(
        theta0.to(device),
        population=config.es.population,
        sigma=config.es.sigma,
        lr=config.es.lr,
        weight_decay=config.es.weight_decay,
        seed=config.run.seed,
    )
    return es, TOY_FUNCTIONS[config.toy.function]


def train(config: Config, resume: str | None = None, quiet: bool = False) -> tuple[OpenAIES, RunDir]:
    seed_everything(config.run.seed)
    device = resolve_device(config.run.device)
    es, objective = build(config, device)
    if resume:
        es.load_state_dict(load_checkpoint(resume, map_location=device))

    run = RunDir(config)
    start = time.perf_counter()
    try:
        while es.generation < config.es.generations:
            candidates = es.ask()
            fitness = objective(candidates)
            stats = es.tell(fitness)
            g = es.generation
            if g % config.es.log_every == 0 or g == config.es.generations:
                center = float(objective(es.theta.detach()[None])[0])
                run.metrics.log(generation=g, center_fitness=center, **stats,
                                seconds=round(time.perf_counter() - start, 3))
                if not quiet:
                    print(f"gen {g:5d}  center {center:12.4f}  best {stats['fitness_max']:12.4f}")
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
    parser.add_argument("--resume", default=None, help="checkpoint.pt to continue from")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    _, run = train(config, resume=args.resume)
    print(f"run folder: {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
