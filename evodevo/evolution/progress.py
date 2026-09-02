"""Live stdout progress reporting for the benchmark orchestrator.

Purely additive observability: this module never touches persisted
run_log.jsonl / run_meta.json / heartbeat content, and is wired in via an
optional callback so it can be omitted with zero behavior change.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from typing import Dict, List, Tuple


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProgressTracker:
    """Tracks progress across the full (encoder, task, seed) run matrix.

    `eval_budgets` maps (encoder, task, seed) -> that run's eval_budget, used
    to compute per-run and full-matrix ETAs. `window` bounds the rolling
    mean's trailing sample count.
    """

    def __init__(
        self,
        eval_budgets: Dict[Tuple[str, str, int], int],
        window: int = 20,
        stream=sys.stdout,
    ):
        self.eval_budgets = eval_budgets
        self.total_evals = sum(eval_budgets.values())
        self.completed_total = 0
        self.window = window
        self.stream = stream

    def for_run(self, encoder: str, task: str, seed: int) -> "_RunProgress":
        budget = self.eval_budgets[(encoder, task, seed)]
        return _RunProgress(self, encoder, task, seed, budget)


class _RunProgress:
    """Callable passed as RunLogger(on_eval=...) for a single (encoder, task,
    seed) run; records one stdout line per completed evaluation."""

    def __init__(self, tracker: ProgressTracker, encoder: str, task: str, seed: int, budget: int):
        self.tracker = tracker
        self.encoder = encoder
        self.task = task
        self.seed = seed
        self.budget = budget
        self.run_completed = 0
        self.times: deque = deque(maxlen=tracker.window)

    def __call__(self, record: dict) -> None:
        self.run_completed += 1
        self.tracker.completed_total += 1

        wall_clock_s = record.get("wall_clock_s") or 0.0
        self.times.append(wall_clock_s)
        mean = sum(self.times) / len(self.times)

        remaining_run = max(self.budget - self.run_completed, 0)
        remaining_total = max(self.tracker.total_evals - self.tracker.completed_total, 0)
        eta_run = _fmt_hms(remaining_run * mean)
        eta_total = _fmt_hms(remaining_total * mean)

        if record.get("cache_hit"):
            tag = "cache hit"
        elif not record.get("valid", True):
            tag = "validity failure"
        else:
            tag = "real-PPO"

        total_evals = self.tracker.total_evals
        pct = 100.0 * self.tracker.completed_total / total_evals if total_evals else 0.0

        line = (
            f"[{self.encoder}/{self.task}/seed_{self.seed}] "
            f"eval {self.run_completed}/{self.budget} | "
            f"{self.tracker.completed_total:,}/{total_evals:,} total evaluations, {pct:.1f}% | "
            f"{tag} | "
            f"mean {mean:.2f}s/eval (trailing {len(self.times)}) | "
            f"ETA run {eta_run} | ETA total {eta_total}"
        )
        print(line, file=self.tracker.stream, flush=True)
