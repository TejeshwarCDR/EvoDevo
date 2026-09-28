"""Crash-resumable, phase-ordered campaign execution.

THIS PACKAGE'S __init__ MUST STAY FREE OF numpy/torch/evogym/
stable_baselines3, DIRECTLY OR TRANSITIVELY. Every spawned PPO worker (see
evodevo/parallel/worker_pool.py) imports this package before its
initializer can call configure_threading(1, strict=True): under the
`spawn` start method, multiprocessing re-imports the parent's main module
(and therefore its parent packages) in each child. An eager import here
once pulled in numpy via `.plan` -> evodevo.experiments.benchmark and
crashed every worker with ThreadingConfiguredTooLateError ->
BrokenProcessPool (see CAMPAIGN_READINESS.md, "Changes since ab8f73d").
Guarded by tests/campaign/test_entry_point.py.

The public names are still re-exported, lazily, via module __getattr__.
"""

__all__ = ["CampaignPlan", "CampaignScheduler"]


def __getattr__(name):
    if name == "CampaignPlan":
        from .plan import CampaignPlan
        return CampaignPlan
    if name == "CampaignScheduler":
        from .scheduler import CampaignScheduler
        return CampaignScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
