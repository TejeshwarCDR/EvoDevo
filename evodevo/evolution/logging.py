"""Per-evaluation run logging: newline-delimited JSON plus run metadata."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    def __init__(self, output_dir: str, heartbeat_path: Optional[str] = None,
                 encoder: Optional[str] = None, seed: Optional[int] = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "run_log.jsonl")
        self._fh = open(self.log_path, "w")
        # heartbeat_path, if given, is a single shared file (typically
        # <benchmark output_dir>/.heartbeat.json) rewritten after every
        # completed evaluation, for an unattended multi-hour run to be
        # externally monitorable without tailing per-run JSONL files.
        self._heartbeat_path = heartbeat_path
        self._encoder = encoder
        self._seed = seed
        self._evaluations_completed = 0

    def log(self, record: dict) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()
        self._evaluations_completed += 1
        self._write_heartbeat(record.get("eval_index"))

    def _write_heartbeat(self, eval_index) -> None:
        if self._heartbeat_path is None:
            return
        heartbeat = {
            "timestamp": _utc_now_iso(),
            "encoder": self._encoder,
            "seed": self._seed,
            "evaluation_index": eval_index,
            "evaluations_completed_this_run": self._evaluations_completed,
        }
        tmp_path = self._heartbeat_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(heartbeat, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self._heartbeat_path)

    def close(self) -> None:
        self._fh.close()

    def write_meta(
        self,
        config_dict: dict,
        encoder_describe: dict,
        cache_hit_rate: Optional[float] = None,
    ) -> None:
        meta = {
            "config": config_dict,
            "encoder": encoder_describe,
            "cache_hit_rate": cache_hit_rate,
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
