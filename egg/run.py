"""Run directories and metric logging.

Every experiment run gets its own folder:

    runs/<YYYYMMDD-HHMMSS>_<name>/
        config.json   frozen copy of the config used
        env.json      code version (git commit), Python and package versions
        metrics.csv   one row per logged step

This is the flight data recorder of the project: after a run you can
always see what was run, on which code, and what happened.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from egg.config import Config

TRACKED_PACKAGES = ("numpy", "torch", "pygame-ce")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def environment_info() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


class MetricsLogger:
    """Append-only CSV logger. Columns are fixed by the first logged row."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fieldnames: list[str] | None = None
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None

    def log(self, **values: Any) -> None:
        if self._writer is None:
            self._fieldnames = list(values)
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        extra = set(values) - set(self._fieldnames or [])
        if extra:
            raise ValueError(f"unexpected metric columns {sorted(extra)}; fixed: {self._fieldnames}")
        self._writer.writerow(values)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class RunDir:
    """Creates and owns one run folder."""

    def __init__(self, config: Config, root: str | Path | None = None) -> None:
        root = Path(root if root is not None else config.run.output_dir)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = root / f"{stamp}_{config.run.name}"
        path = base
        counter = 1
        while path.exists():
            path = base.with_name(f"{base.name}_{counter}")
            counter += 1
        path.mkdir(parents=True)

        self.path = path
        self._write_json("config.json", config.to_dict())
        self._write_json("env.json", environment_info())
        self.metrics = MetricsLogger(path / "metrics.csv")

    def _write_json(self, name: str, data: dict[str, Any]) -> None:
        with open(self.path / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def close(self) -> None:
        self.metrics.close()

    def __enter__(self) -> "RunDir":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
