#!/usr/bin/env bash
# Run experiment E001: copy task, colony vs GRU, two settings, three seeds.
# Usage: bash tools/run_e001.sh   (from the repo root)
set -e
PY=.venv/Scripts/python.exe
OUT="'runs/e001'"
for seed in 0 1 2; do
  for kind in gru colony; do
    $PY -m egg.train_copy --config configs/e001_$kind.toml \
      --set run.output_dir=$OUT --set run.seed=$seed \
      --set run.name="'e001a_${kind}_s${seed}'"
    $PY -m egg.train_copy --config configs/e001_$kind.toml \
      --set run.output_dir=$OUT --set run.seed=$seed \
      --set run.name="'e001b_${kind}_s${seed}'" \
      --set copy.digits=8 --set copy.length=5 --set es.generations=3000
  done
done
echo "E001 done"
