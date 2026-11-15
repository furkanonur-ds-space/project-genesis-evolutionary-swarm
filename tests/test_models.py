import tempfile
import unittest
from pathlib import Path

import torch

from egg.config import load_config
from egg.genome import GenomeSpec, ParamDef
from egg.models import Colony, GRUBaseline
from egg.models.colony import position_code
from egg.tasks.copy_task import CopyTask, score
from egg.train_copy import train
from egg.viz import export


class GenomeSpecTest(unittest.TestCase):
    def setUp(self):
        self.spec = GenomeSpec([ParamDef("a", (2, 3)), ParamDef("b", (3,), "zeros")])

    def test_size_and_roundtrip(self):
        self.assertEqual(self.spec.size, 9)
        g = torch.arange(18.0).reshape(2, 9)
        w = self.spec.unflatten(g)
        self.assertEqual(w["a"].shape, (2, 2, 3))
        self.assertEqual(w["b"][1].tolist(), [15.0, 16.0, 17.0])

    def test_init_zeros_part(self):
        g = self.spec.init(torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(g[6:], torch.zeros(3)))

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            GenomeSpec([ParamDef("a", (1,)), ParamDef("a", (2,))])

    def test_wrong_size_rejected(self):
        with self.assertRaises(ValueError):
            self.spec.unflatten(torch.zeros(2, 8))


class CopyTaskTest(unittest.TestCase):
    def test_layout(self):
        task = CopyTask(digits=5, length=3)
        b = task.sample(4, torch.Generator().manual_seed(0))
        self.assertEqual(b.tokens.shape, (4, 6))
        self.assertTrue(torch.equal(b.tokens[:, 3], torch.full((4,), task.go)))
        self.assertTrue(torch.equal(b.tokens[:, 4:], torch.full((4, 2), task.blank)))
        self.assertTrue(torch.equal(b.targets[:, 3:], b.tokens[:, :3]))
        self.assertEqual(b.mask.sum().item(), 12)

    def test_perfect_and_wrong_scores(self):
        task = CopyTask(digits=3, length=2)
        b = task.sample(5, torch.Generator().manual_seed(1))
        perfect = torch.nn.functional.one_hot(b.targets, 3).float()[None] * 50
        s = score(perfect, b)
        self.assertAlmostEqual(float(s["symbol_acc"][0]), 1.0)
        self.assertAlmostEqual(float(s["seq_acc"][0]), 1.0)
        wrong = torch.nn.functional.one_hot((b.targets + 1) % 3, 3).float()[None] * 50
        self.assertAlmostEqual(float(score(wrong, b)["symbol_acc"][0]), 0.0)

    def test_uniform_logits_give_chance_log_prob(self):
        task = CopyTask(digits=4, length=3)
        b = task.sample(3, torch.Generator().manual_seed(2))
        s = score(torch.zeros(2, 3, 6, 4), b)
        self.assertTrue(torch.allclose(s["log_prob"], torch.full((2,), -torch.log(torch.tensor(4.0)))))


class ModelShapeTest(unittest.TestCase):
    def _check(self, model, pop=3, batch=2, steps=6):
        g = torch.stack([model.spec.init(torch.Generator().manual_seed(i)) for i in range(pop)])
        tokens = torch.randint(0, model.num_inputs, (batch, steps))
        out = model.forward(g, tokens)
        self.assertEqual(out.shape, (pop, batch, steps, model.num_outputs))
        self.assertTrue(torch.isfinite(out).all())
        # Each population member must be computed independently of the others.
        single = model.forward(g[1:2], tokens)
        self.assertTrue(torch.allclose(out[1:2], single, atol=1e-5))

    def test_gru(self):
        self._check(GRUBaseline(num_inputs=6, num_outputs=4, hidden=5))

    def test_colony(self):
        self._check(Colony(num_inputs=6, num_outputs=4, rows=3, cols=4, hidden=4, message=2, embed=3, ticks=2))

    def test_colony_record_shape(self):
        col = Colony(num_inputs=6, num_outputs=4, rows=3, cols=4, hidden=4, message=2, embed=3, ticks=2)
        g = col.spec.init()[None]
        _, states = col.forward(g, torch.zeros(1, 5, dtype=torch.long), record=True)
        self.assertEqual(states.shape, (1, 1, 10, 3, 4, 4))

    def test_colony_is_causal(self):
        """Output at step t must not depend on inputs after t (serial reading)."""
        col = Colony(num_inputs=6, num_outputs=4, rows=3, cols=3, hidden=4, message=2, embed=3, ticks=2)
        g = col.spec.init(torch.Generator().manual_seed(0))[None]
        a = torch.tensor([[1, 2, 3, 0, 0]])
        b = torch.tensor([[1, 2, 3, 5, 4]])
        out_a, out_b = col.forward(g, a), col.forward(g, b)
        self.assertTrue(torch.allclose(out_a[:, :, :3], out_b[:, :, :3]))
        self.assertFalse(torch.allclose(out_a[:, :, 3:], out_b[:, :, 3:]))

    def test_position_code_flags(self):
        code = position_code(2, 3)
        self.assertEqual(code[:, 4].tolist(), [1, 0, 0, 1, 0, 0])
        self.assertEqual(code[:, 5].tolist(), [0, 0, 1, 0, 0, 1])


class TrainCopyTest(unittest.TestCase):
    def test_short_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("gru", "colony"):
                config = load_config(None, [
                    f"model.kind='{kind}'", "es.generations=3", "es.population=4",
                    "copy.batch=4", "copy.val_batch=8", "colony.rows=3", "colony.cols=3",
                    "run.device='cpu'", f"run.output_dir='{Path(tmp).as_posix()}'",
                ])
                es, run = train(config, quiet=True)
                self.assertEqual(es.generation, 3)
                self.assertTrue((run.path / "checkpoint.pt").exists())
                if kind == "colony":
                    html = export(run.path, examples=2, out=None)
                    text = html.read_text(encoding="utf-8")
                    self.assertIn('"rows": 3', text)
                    self.assertNotIn("__DATA__", text)


if __name__ == "__main__":
    unittest.main()
