import tempfile
import unittest
from pathlib import Path

import torch

from egg.checkpoint import load_checkpoint, save_checkpoint
from egg.config import load_config
from egg.es import OpenAIES, centered_ranks
from egg.toy import rastrigin, sphere
from egg.train_toy import train


class CenteredRanksTest(unittest.TestCase):
    def test_range_and_order(self):
        r = centered_ranks(torch.tensor([10.0, -5.0, 3.0, 100.0]))
        self.assertTrue(torch.allclose(r, torch.tensor([1 / 6, -0.5, -1 / 6, 0.5])))

    def test_scale_invariant(self):
        f = torch.tensor([1.0, 4.0, 2.0, 3.0])
        self.assertTrue(torch.equal(centered_ranks(f), centered_ranks(f * 1000 - 7)))


class OpenAIESTest(unittest.TestCase):
    def _es(self, seed=0, dim=10, pop=16):
        return OpenAIES(torch.full((dim,), 2.0), population=pop, sigma=0.1, lr=0.05, seed=seed)

    def test_rejects_odd_population(self):
        with self.assertRaises(ValueError):
            OpenAIES(torch.zeros(3), population=5, sigma=0.1, lr=0.1)

    def test_antithetic_pairs(self):
        es = self._es()
        c = es.ask()
        self.assertEqual(c.shape, (16, 10))
        mid = (c[0::2] + c[1::2]) / 2
        self.assertTrue(torch.allclose(mid, es.theta.detach().expand_as(mid), atol=1e-6))

    def test_tell_requires_ask(self):
        with self.assertRaises(RuntimeError):
            self._es().tell(torch.zeros(16))

    def test_same_seed_same_trajectory(self):
        thetas = []
        for _ in range(2):
            es = self._es(seed=3)
            for _ in range(5):
                es.tell(sphere(es.ask()))
            thetas.append(es.theta.detach().clone())
        self.assertTrue(torch.equal(thetas[0], thetas[1]))

    def test_different_seed_different_trajectory(self):
        a, b = self._es(seed=1), self._es(seed=2)
        for es in (a, b):
            es.tell(sphere(es.ask()))
        self.assertFalse(torch.equal(a.theta, b.theta))

    def test_solves_sphere(self):
        es = OpenAIES(torch.full((20,), 3.0), population=32, sigma=0.1, lr=0.05, seed=0)
        start = float(sphere(es.theta.detach()[None]))
        for _ in range(300):
            es.tell(sphere(es.ask()))
        end = float(sphere(es.theta.detach()[None]))
        self.assertLess(abs(end), abs(start) * 0.01)

    def test_improves_rastrigin(self):
        es = OpenAIES(torch.full((5,), 3.0), population=64, sigma=2.0, lr=0.05, seed=0)
        start = float(rastrigin(es.theta.detach()[None]))
        for _ in range(200):
            es.tell(rastrigin(es.ask()))
        end = float(rastrigin(es.theta.detach()[None]))
        self.assertLess(abs(end), abs(start) * 0.5)  # optimum is 0


class CheckpointTest(unittest.TestCase):
    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = OpenAIES(torch.full((8,), 1.0), population=8, sigma=0.1, lr=0.05, seed=5)
            for _ in range(10):
                full.tell(sphere(full.ask()))

            part = OpenAIES(torch.full((8,), 1.0), population=8, sigma=0.1, lr=0.05, seed=5)
            for _ in range(4):
                part.tell(sphere(part.ask()))
            path = save_checkpoint(Path(tmp) / "ck.pt", part.state_dict())

            resumed = OpenAIES(torch.zeros(8), population=8, sigma=0.1, lr=0.05, seed=0)
            resumed.load_state_dict(load_checkpoint(path))
            for _ in range(6):
                resumed.tell(sphere(resumed.ask()))
            self.assertTrue(torch.allclose(full.theta, resumed.theta))


class TrainToyTest(unittest.TestCase):
    def test_train_writes_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(None, [
                "es.generations=20", "es.population=8", "toy.dim=4",
                "run.device='cpu'", f"run.output_dir='{Path(tmp).as_posix()}'",
            ])
            es, run = train(config, quiet=True)
            self.assertEqual(es.generation, 20)
            self.assertTrue((run.path / "checkpoint.pt").exists())
            self.assertTrue((run.path / "metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()


class ScheduleTest(unittest.TestCase):
    def test_decay_and_floor(self):
        es = OpenAIES(torch.zeros(4), population=4, sigma=1.0, lr=0.1,
                      sigma_decay=0.5, sigma_min=0.3, lr_decay=0.5, lr_min=0.04)
        for _ in range(3):
            stats = es.tell(sphere(es.ask()))
        self.assertAlmostEqual(es.sigma, 0.3)
        self.assertAlmostEqual(stats["lr"], 0.04)

    def test_schedule_survives_checkpoint(self):
        es = OpenAIES(torch.zeros(4), population=4, sigma=1.0, lr=0.1, sigma_decay=0.9)
        other = OpenAIES(torch.zeros(4), population=4, sigma=1.0, lr=0.1)
        other.load_state_dict(es.state_dict())
        self.assertAlmostEqual(other.sigma_decay, 0.9)
