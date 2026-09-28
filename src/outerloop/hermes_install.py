"""The pinned Hermes runtime layout shared by setup and session launch."""

import os
from pathlib import Path

HERMES_SHA = "f80f453ae0679347e38abc917c7f94f717bf96c5"


def hermes_runtime(repo: Path) -> Path:
    return Path(f"{repo.expanduser().resolve()}.runtime") / HERMES_SHA


def hermes_ready(repo: Path) -> bool:
    runtime = hermes_runtime(repo)
    try:
        return (
            (repo / "run_agent.py").is_file()
            and (runtime / ".complete").read_text().strip() == HERMES_SHA
            and (runtime / "venv/bin/python").is_file()
            and os.access(runtime / "venv/bin/python", os.X_OK)
        )
    except OSError:
        return False
