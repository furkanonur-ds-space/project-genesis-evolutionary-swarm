"""Fail if any non-report file contains non-ASCII characters.

Reports (docs/manual, docs/vision) and the frozen legacy folder are
exempt. Usage: python tools/check_ascii.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

EXEMPT_PREFIXES = ("docs/manual/", "docs/vision/", "legacy/")


def tracked_files() -> list[str]:
    """Files of the public repo plus the local-only markdown files.

    Markdown (except README) is ignored by the main repo and tracked in
    the local .git-notes repo, so it is collected separately here.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    )
    files = {line for line in out.stdout.splitlines() if line}
    files.update(p.as_posix() for p in Path(".").glob("*.md"))
    files.update(p.as_posix() for p in Path("docs").rglob("*.md"))
    return sorted(files)


def main() -> int:
    problems = 0
    for path in tracked_files():
        if path.startswith(EXEMPT_PREFIXES):
            continue
        try:
            data = open(path, "rb").read()
        except OSError:
            continue
        for lineno, line in enumerate(data.splitlines(), 1):
            if any(b > 127 for b in line):
                print(f"{path}:{lineno}: {line.decode('utf-8', 'replace').strip()}")
                problems += 1
    if problems:
        print(f"{problems} line(s) with non-ASCII characters")
        return 1
    print("ASCII check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
