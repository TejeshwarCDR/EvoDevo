"""Per-evaluation run logging: newline-delimited JSON plus run metadata."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional


class RunLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "run_log.jsonl")
        self._fh = open(self.log_path, "w")

    def log(self, record: dict) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def write_meta(self, config_dict: dict, encoder_describe: dict) -> None:
        meta = {
            "config": config_dict,
            "encoder": encoder_describe,
            "git_commit": _git_commit_hash(),
            "library_versions": _library_versions(),
        }
        with open(os.path.join(self.output_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)


def _git_commit_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _library_versions() -> dict:
    versions = {"python": sys.version}
    for lib in ("numpy", "yaml", "pytest"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    return versions
