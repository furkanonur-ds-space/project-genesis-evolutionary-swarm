import tempfile
import unittest
from pathlib import Path

from egg.config import ConfigError, load_config


class LoadConfigTest(unittest.TestCase):
    def _write(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
        tmp.write(text)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_defaults_without_file(self):
        config = load_config()
        self.assertEqual(config.run.seed, 0)
        self.assertEqual(config.run.device, "auto")

    def test_file_values(self):
        path = self._write('[run]\nname = "abc"\nseed = 5\n')
        config = load_config(path)
        self.assertEqual(config.run.name, "abc")
        self.assertEqual(config.run.seed, 5)

    def test_unknown_key_is_rejected(self):
        path = self._write("[run]\nsede = 5\n")
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_unknown_section_is_rejected(self):
        path = self._write("[rnu]\nseed = 5\n")
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_wrong_type_is_rejected(self):
        path = self._write('[run]\nseed = "five"\n')
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_bool_is_not_int(self):
        path = self._write("[run]\nseed = true\n")
        with self.assertRaises(ConfigError):
            load_config(path)

    def test_overrides(self):
        path = self._write('[run]\nseed = 5\nname = "a"\n')
        config = load_config(path, ["run.seed=9", "run.name=b"])
        self.assertEqual(config.run.seed, 9)
        self.assertEqual(config.run.name, "b")

    def test_bad_override_format(self):
        with self.assertRaises(ConfigError):
            load_config(None, ["run.seed"])


if __name__ == "__main__":
    unittest.main()
