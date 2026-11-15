import csv
import json
import tempfile
import unittest
from pathlib import Path

from egg.config import load_config
from egg.run import RunDir
from egg.seeding import seed_everything
from egg.smoke import main as smoke_main


class RunDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_creates_files(self):
        config = load_config(None, ["run.name=t"])
        with RunDir(config, root=self.root) as run:
            run.metrics.log(step=0, value=1.5)
            run.metrics.log(step=1, value=2.5)
        self.assertTrue((run.path / "env.json").exists())
        saved = json.loads((run.path / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["run"]["name"], "t")
        with open(run.path / "metrics.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["value"] for r in rows], ["1.5", "2.5"])

    def test_same_name_does_not_overwrite(self):
        config = load_config(None, ["run.name=t"])
        with RunDir(config, root=self.root) as a, RunDir(config, root=self.root) as b:
            self.assertNotEqual(a.path, b.path)

    def test_new_metric_column_is_rejected(self):
        config = load_config(None)
        with RunDir(config, root=self.root) as run:
            run.metrics.log(step=0)
            with self.assertRaises(ValueError):
                run.metrics.log(step=1, surprise=3)


class ReproducibilityTest(unittest.TestCase):
    def test_same_seed_same_numbers(self):
        a = seed_everything(123).normal(size=5)
        b = seed_everything(123).normal(size=5)
        c = seed_everything(124).normal(size=5)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertNotEqual(a.tolist(), c.tolist())

    def test_smoke_runs_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--set", "run.seed=7", "--set", f"run.output_dir='{Path(tmp).as_posix()}'"]
            smoke_main(args)
            smoke_main(args)
            runs = sorted(Path(tmp).iterdir())
            self.assertEqual(len(runs), 2)
            first, second = (r / "metrics.csv" for r in runs)
            self.assertEqual(first.read_text(), second.read_text())


if __name__ == "__main__":
    unittest.main()
