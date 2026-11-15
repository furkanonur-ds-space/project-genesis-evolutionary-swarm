"""Smoke test of the full pipeline: config -> seed -> run folder -> metrics.

Usage:
    python -m egg.smoke --config configs/smoke.toml
    python -m egg.smoke --config configs/smoke.toml --set run.seed=7

It runs a toy random walk and logs it. Running it twice with the same
seed must give identical metrics.csv values.
"""

from __future__ import annotations

import argparse

from egg.config import load_config
from egg.device import describe_device
from egg.run import RunDir
from egg.seeding import seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to a TOML config file")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set run.seed=3 (repeatable)",
    )
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    rng = seed_everything(config.run.seed)
    print(f"device: {describe_device(config.run.device)}")

    with RunDir(config) as run:
        position = 0.0
        for step in range(args.steps):
            position += float(rng.normal())
            run.metrics.log(step=step, position=round(position, 6))
        print(f"run folder: {run.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
