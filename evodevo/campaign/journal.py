"""Durable append-only PPO task journal."""
from __future__ import annotations

import json
import os
from dataclasses import asdict


class ResultJournal:
    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(self, record: dict) -> None:
        # O_APPEND plus flush/fsync makes a completed task durable before the
        # scheduler is allowed to tell its run.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append_task(self, task, result, genome_hash: str) -> None:
        self.append({"run_key": task.run_key, "generation": task.generation,
                     "genome_id": task.genome_id, "eval_index": task.eval_index,
                     "ppo_k": task.ppo_k, "genome_hash": genome_hash,
                     "result": asdict(result)})

    def records(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def indexed(self):
        return {(r["run_key"], r["eval_index"], r["ppo_k"]): r for r in self.records()}
