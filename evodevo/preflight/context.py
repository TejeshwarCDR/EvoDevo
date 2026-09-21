"""PreflightContext: the one object every stage's run(ctx) receives.

Deliberately does NOT carry population_size, generations, or anything else
read from a B1-B7-scoped experiment config -- D15 (how big the primary EA
matrix is) is an explicitly open decision, enforced elsewhere by
OpenProfileDecisionError, and pre-flight's whole job is to inform that
decision, not assume an answer to it. `config` here is only the resolved
CLI configuration for the harness itself (which run directory, how often
to sample telemetry) -- never experiment-scoped hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from evodevo.parallel.topology import Topology, cpu_slots
from evodevo.parallel.worker_pool import EvaluationPool
from evodevo.preflight.events import EventLog
from evodevo.preflight.run_dir import RunDir
from evodevo.preflight.telemetry import TelemetrySampler


@dataclass
class PreflightContext:
    run_dir: RunDir
    events: EventLog
    telemetry: TelemetrySampler
    topology: Topology
    config: dict = field(default_factory=dict)

    def make_pool(
        self,
        n_workers: int,
        mode: str = "surrogate",
        timeout_s: Optional[float] = None,
        max_tasks_per_child: Optional[int] = None,
        max_retries: int = 1,
    ) -> EvaluationPool:
        """Builds an EvaluationPool sized/pinned against THIS machine's
        actual topology (evodevo.parallel), so a B1-B7 stage never has to
        re-derive an affinity plan itself -- it just asks for N workers."""
        plan = cpu_slots(self.topology, n_workers)
        return EvaluationPool(
            n_workers=n_workers,
            plan=plan,
            mode=mode,
            timeout_s=timeout_s,
            max_tasks_per_child=max_tasks_per_child,
            max_retries=max_retries,
        )
