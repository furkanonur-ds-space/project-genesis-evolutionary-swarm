"""Summarize copy task runs: final validation metrics per run and per group.

Usage: python tools/summarize_runs.py runs/e001

Groups runs by name with the trailing "_s<seed>" removed, e.g.
e001a_colony_s0, e001a_colony_s1 -> group e001a_colony.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

THRESHOLD = 0.9


def load_run(path: Path) -> dict | None:
    metrics = path / "metrics.csv"
    if not metrics.exists():
        return None
    with open(metrics, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    last = rows[-1]
    reached = next((int(r["generation"]) for r in rows if float(r["val_seq_acc"]) >= THRESHOLD), None)
    return {
        "name": config["run"]["name"],
        "group": re.sub(r"_s\d+$", "", config["run"]["name"]),
        "generation": int(last["generation"]),
        "val_log_prob": float(last["val_log_prob"]),
        "val_symbol_acc": float(last["val_symbol_acc"]),
        "val_seq_acc": float(last["val_seq_acc"]),
        "best_seq_acc": max(float(r["val_seq_acc"]) for r in rows),
        "gen_to_90": reached,
        "seconds": float(last["seconds"]),
    }


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} +- {statistics.stdev(values):.3f}"


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "runs")
    runs = [r for p in sorted(root.iterdir()) if p.is_dir() and (r := load_run(p))]
    if not runs:
        print(f"no runs found in {root}")
        return 1

    print(f"{'run':24s} {'gen':>5s} {'logp':>8s} {'sym_acc':>8s} {'seq_acc':>8s} {'best_seq':>8s} {'gen>=90%':>9s} {'sec':>7s}")
    for r in runs:
        g90 = str(r["gen_to_90"]) if r["gen_to_90"] is not None else "-"
        print(f"{r['name']:24s} {r['generation']:5d} {r['val_log_prob']:8.4f} {r['val_symbol_acc']:8.3f} "
              f"{r['val_seq_acc']:8.3f} {r['best_seq_acc']:8.3f} {g90:>9s} {r['seconds']:7.0f}")

    print()
    print(f"{'group':18s} {'n':>2s}  {'final sym_acc':>16s}  {'final seq_acc':>16s}  {'reached 90% seq':>15s}")
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    for name, items in groups.items():
        reached = [r["gen_to_90"] for r in items if r["gen_to_90"] is not None]
        reach_txt = f"{len(reached)}/{len(items)}" + (f" (mean gen {statistics.mean(reached):.0f})" if reached else "")
        print(f"{name:18s} {len(items):2d}  {fmt([r['val_symbol_acc'] for r in items]):>16s}  "
              f"{fmt([r['val_seq_acc'] for r in items]):>16s}  {reach_txt:>15s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
