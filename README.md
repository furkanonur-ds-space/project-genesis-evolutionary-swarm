# The Egg

Research project: can a colony of simple neuron-agents that all share one
genome learn to read and write serially, the way a human eye scans a line
and a hand writes one letter at a time?

Inspired by Andy Weir's short story "The Egg": every neuron is the same
soul living a different life. Evolution improves the shared genome across
lifetimes; each lifetime learns locally from its own experience.

## Status

Early stage. Infrastructure is in place (configs, seeding, run logging,
tests). The evolution trainer and the neuron-agent colony are next.

## Layout

```
egg/            main package
configs/        TOML experiment configs
tests/          unit tests (stdlib unittest)
tools/          helper scripts
legacy/phase0/  archived phase 0 prototype (git tag: phase0-archive)
runs/           experiment outputs (git ignored)
```

## Setup

Python 3.14. Create a virtual environment, then install NumPy and a
PyTorch build that matches your GPU:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install "numpy>=2.0"
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

CUDA 12.8 builds (cu128) or newer are required for RTX 50 series GPUs.
The code also runs on CPU.

## Quick start

```
.venv\Scripts\python.exe -m unittest discover -s tests -t .
.venv\Scripts\python.exe -m egg.smoke --config configs/smoke.toml
```

Override any config value from the command line:

```
.venv\Scripts\python.exe -m egg.smoke --config configs/smoke.toml --set run.seed=7
```

Each run writes `config.json`, `env.json` and `metrics.csv` to its own
folder under `runs/`.
