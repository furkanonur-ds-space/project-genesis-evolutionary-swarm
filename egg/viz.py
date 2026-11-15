"""Export a self-contained HTML view of a trained colony at work.

Usage:
    python -m egg.viz runs/e001/<colony run folder>
    python -m egg.viz runs/e001/<folder> --examples 4 --out colony.html

Loads the run's config and checkpoint, runs the center genome on a few
copy-task sequences and records every neuron's state after every tick.
The HTML page (no external dependencies) shows the grid tick by tick:
cell colour = first state component (signed), cell size = state norm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from egg.checkpoint import load_checkpoint
from egg.config import config_from_dict
from egg.tasks.copy_task import CopyTask
from egg.train_copy import build_model

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Colony viewer</title>
<style>
body { font-family: monospace; background: #111; color: #ddd; margin: 16px; }
canvas { background: #000; display: block; margin: 8px 0; }
.row { margin: 4px 0; } .tok { display: inline-block; width: 2.2em; text-align: center; padding: 2px; }
.cur { outline: 2px solid #fc6; } .ok { color: #6f6; } .bad { color: #f66; }
</style></head><body>
<h3>Colony viewer: __TITLE__</h3>
<div class="row">example <select id="ex"></select>
 tick <input id="t" type="range" min="0" value="0" style="width:400px"> <span id="tl"></span>
 <button id="play">play</button></div>
<div class="row">input&nbsp; <span id="inp"></span></div>
<div class="row">output <span id="outp"></span></div>
<canvas id="c"></canvas>
<div class="row">left column = input neurons, right column = output neurons.
colour = first state component (blue negative, orange positive), size = state norm.</div>
<script>
const D = __DATA__;
const ex = document.getElementById('ex'), t = document.getElementById('t');
const cv = document.getElementById('c'), cx = cv.getContext('2d');
const CELL = 56; cv.width = D.cols * CELL; cv.height = D.rows * CELL;
D.examples.forEach((e, i) => { const o = document.createElement('option'); o.value = i; o.text = i; ex.appendChild(o); });
t.max = D.examples[0].states.length - 1;
function sym(v) { return v < D.digits ? String(v) : (v === D.digits ? 'GO' : '_'); }
function draw() {
  const e = D.examples[+ex.value], k = +t.value, step = Math.floor(k / D.ticks);
  document.getElementById('tl').textContent = `${k} (symbol ${step}, tick ${k % D.ticks})`;
  document.getElementById('inp').innerHTML = e.tokens.map((v, i) =>
    `<span class="tok ${i === step ? 'cur' : ''}">${sym(v)}</span>`).join('');
  document.getElementById('outp').innerHTML = e.preds.map((v, i) => {
    if (!e.mask[i]) return `<span class="tok ${i === step ? 'cur' : ''}">.</span>`;
    const cls = v === e.targets[i] ? 'ok' : 'bad';
    return `<span class="tok ${cls} ${i === step ? 'cur' : ''}">${v}</span>`;
  }).join('');
  cx.clearRect(0, 0, cv.width, cv.height);
  const s = e.states[k];
  for (let r = 0; r < D.rows; r++) for (let c = 0; c < D.cols; c++) {
    const [a, n] = s[r][c];
    const col = a >= 0 ? `rgba(255,170,60,${Math.min(1, Math.abs(a))})` : `rgba(80,150,255,${Math.min(1, Math.abs(a))})`;
    const sz = 8 + (CELL - 12) * Math.min(1, n / D.max_norm);
    cx.fillStyle = col;
    cx.fillRect(c * CELL + (CELL - sz) / 2, r * CELL + (CELL - sz) / 2, sz, sz);
    cx.strokeStyle = c === 0 ? '#6f6' : (c === D.cols - 1 ? '#f6f' : '#333');
    cx.strokeRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
  }
}
let timer = null;
document.getElementById('play').onclick = () => {
  if (timer) { clearInterval(timer); timer = null; return; }
  timer = setInterval(() => { t.value = (+t.value + 1) % (+t.max + 1); draw(); }, 250);
};
ex.onchange = draw; t.oninput = draw; draw();
</script></body></html>
"""


def export(run_dir: Path, examples: int, out: Path | None) -> Path:
    config = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    if config.model.kind != "colony":
        raise ValueError("viz only supports colony runs")
    task = CopyTask(config.copy.digits, config.copy.length)
    model = build_model(config, task)
    theta = load_checkpoint(run_dir / "checkpoint.pt")["theta"]
    batch = task.sample(examples, torch.Generator().manual_seed(12345))
    with torch.no_grad():
        logits, states = model.forward(theta[None], batch.tokens, record=True)
    preds = logits[0].argmax(dim=-1)
    st = states[0]                                   # (B, ticks_total, rows, cols, H)
    first = st[..., 0]
    norm = st.norm(dim=-1)
    data = {
        "rows": model.rows, "cols": model.cols, "ticks": model.ticks, "digits": task.digits,
        "max_norm": float(norm.max()),
        "examples": [
            {
                "tokens": batch.tokens[b].tolist(),
                "targets": batch.targets[b].tolist(),
                "mask": [int(v) for v in batch.mask[b].tolist()],
                "preds": preds[b].tolist(),
                "states": torch.stack([first[b], norm[b]], dim=-1).round(decimals=3).tolist(),
            }
            for b in range(examples)
        ],
    }
    html = PAGE.replace("__TITLE__", config.run.name).replace("__DATA__", json.dumps(data))
    out = out or run_dir / "colony_viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    print(f"wrote {export(args.run_dir, args.examples, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
