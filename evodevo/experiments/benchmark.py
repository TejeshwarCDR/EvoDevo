"""Unified N-way benchmark orchestrator: runs every (encoder, task, seed)
combination in a benchmark config through the standard EA pipeline
(evodevo.experiments.run_experiment.run_experiment), after verifying every
resolved per-encoder config is pairwise fairness-comparable.

MUST NOT be used to patch or re-run a single arm after seeing results -- see
docs/decisions.md and CONTRIBUTING.md's scientific-integrity note. If a bug
is found partway through a benchmark, fix it and re-run every arm from
scratch, not just the one that was affected.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import yaml

from evodevo.core.config import PPO_SEED_AGGREGATIONS, SHARED_BASE_KEYS, ConfigOverrideError, ExperimentConfig
from evodevo.core.factors import ExperimentFactors, extract_factors
from evodevo.core.provenance import DirtyTreeError, check_git_state, write_factors_json
from evodevo.encoders import get_encoder
from evodevo.encoders.nca.resolved_config import PRIMARY as NCA_PRIMARY
from evodevo.encoders.rule_based.resolved_config import (
    MONOTONE_ABLATION as RULE_BASED_MONOTONE,
    PRIMARY as RULE_BASED_PRIMARY,
)
from evodevo.evaluation.ppo_runner import make_evaluator
from evodevo.evolution.progress import ProgressTracker
from evodevo.evolution.ea import resolved_policy_for
from evodevo.experiments.fairness import FairnessViolation, assert_comparable, assert_identical_policy
from evodevo.experiments.run_experiment import (
    DEFAULT_STRATEGY,
    DONE_MARKER,
    STRATEGIES,
    _combine_tag_and_strategy,
    _run_output_dir,
    run_experiment,
)

DEFAULT_CONFIGS_ROOT = "evodevo/experiments/configs"
DEFAULT_OUTPUT_DIR = "results/runs"

# Two named PROFILES, the single source of truth for the SHARED_BASE_KEYS
# fields that determine wall-clock cost (ppo_timesteps/population_size/
# eval_budget/offspring_size) -- everything else (arm/strategy/task
# definitions, encoder dynamics configuration) is resolved identically
# regardless of profile, so nothing about an encoder architecture decision
# can duplicate or drift between profiles.
#
# FAST_DEV: bit-for-bit today's values (see evodevo/experiments/configs/
# shared/base.yaml, unchanged) -- this is what CI, structural/fitness-blind
# analysis, and all N/R-series tooling continues to use. A dedicated test
# (test_fast_dev_profile_matches_shared_base_yaml) pins this equality so it
# cannot silently drift from shared/base.yaml.
#
# PRIMARY_MATRIX: ppo_timesteps=1_000_000, the explicit target this project
# has always stated (docs/decisions.md's "Direct-encoding real-PPO baseline
# budget" row already names EvoGym's own >1e6-timestep recommendation; 5000
# was always disclosed there as a "pipeline-validation-scale budget, not the
# scientific comparison budget"). population_size/eval_budget/offspring_size
# are DELIBERATELY ABSENT here -- there is no equivalent explicit recorded
# target for these anywhere in this project's decisions, and inventing one
# would misrepresent an open decision as a resolved one. See
# PRIMARY_MATRIX_OPEN_FIELDS and _check_profile_ready_for_real_run below.
FAST_DEV = "fast_dev"
PRIMARY_MATRIX = "primary_matrix"
# CAMPAIGN: the final campaign design, every value ACCEPTED in docs/
# decisions.md on 2026-09-27 (D15 population/offspring/eval_budget, D6's
# per-task ppo_timesteps, D16's per-task ppo_seeds and "max" aggregation).
# Fields listed in PER_TASK_PROFILE_FIELDS are {task: value} mappings,
# resolved per task in resolve_configs() -- D6 and D16 both permit values to
# differ BETWEEN tasks, never between arms on one task (every arm on a task
# gets the identical resolved value, so check_fairness still holds).
CAMPAIGN = "campaign"
CAMPAIGN_VALIDATION = "campaign_validation"
PER_TASK_PROFILE_FIELDS = ("ppo_timesteps", "ppo_seeds")
CAMPAIGN_REQUIRED_FIELDS = (
    "population_size", "offspring_size", "eval_budget", "ppo_timesteps", "ppo_seeds", "ppo_seed_aggregation",
)

PROFILES = {
    FAST_DEV: {
        "ppo_timesteps": 5000,
        "population_size": 15,
        "offspring_size": 15,
        "eval_budget": 300,
    },
    PRIMARY_MATRIX: {
        "ppo_timesteps": 1_000_000,
    },
    CAMPAIGN: {
        # D15: eval_budget counts GENOMES (one genome = one budget unit, however
        # many PPO seeds it is trained with) -- 300 = 15 initial + 19 x 15 offspring.
        "population_size": 15,
        "offspring_size": 15,
        "eval_budget": 300,
        # D6 application (UpStepper-v0: registered knee; Walker-v0: fallback rule).
        "ppo_timesteps": {"Walker-v0": 1_000_000, "UpStepper-v0": 1_200_000},
        # D16: K PPO trainings per genome; genome fitness = aggregation over them.
        "ppo_seeds": {"Walker-v0": 1, "UpStepper-v0": 2},
        "ppo_seed_aggregation": "max",
    },
    # Exact campaign grid, but runnable only through the campaign launcher's
    # validation mode, whose PPO workers sleep for scaled measured durations.
    # Keeping a distinct profile makes accidental scientific use fail loud.
    CAMPAIGN_VALIDATION: {
        "population_size": 15,
        "offspring_size": 15,
        "eval_budget": 300,
        "ppo_timesteps": {"Walker-v0": 1_000_000, "UpStepper-v0": 1_200_000},
        "ppo_seeds": {"Walker-v0": 1, "UpStepper-v0": 2},
        "ppo_seed_aggregation": "max",
    },
}

# population_size/offspring_size/eval_budget are jointly "how big is one
# generation, and how many generations" -- an undecided scoping question as
# a whole (docs/PRE_PARALLELIZATION_READINESS.md's "NOT covered" section),
# not three independently-guessable numbers, so all three are gated together.
PRIMARY_MATRIX_OPEN_FIELDS = ("population_size", "offspring_size", "eval_budget")


class OpenProfileDecisionError(RuntimeError):
    """Raised when a REAL (non-surrogate) run is attempted under a profile
    that does not have every SHARED_BASE_KEYS field this project tracks
    explicitly resolved -- currently only `primary_matrix`, missing
    PRIMARY_MATRIX_OPEN_FIELDS. This is a genuine open project decision
    pending the pre-flight wall-clock projection and an explicit scoping
    call, NOT a bug to route around by guessing a value."""


class CampaignProfileError(ValueError):
    """Raised when a profile's values fail schema validation: a missing
    required field, a per-task mapping with no value for one of the
    benchmark's tasks, ppo_seeds (K) < 1, or an unknown
    ppo_seed_aggregation name. Checked before any config is resolved."""


class PpoSeedsNotImplementedError(NotImplementedError):
    """Raised when a run with ppo_seeds (K) > 1 is requested before the
    evaluator supports multi-seed PPO evaluation (docs/decisions.md's D16
    row; implemented in a later step). Without this, a K=2 config would
    silently run as K=1."""


def validate_profile(profile_kwargs: dict, tasks, profile_name: str = CAMPAIGN) -> None:
    """Schema validation for a resolved profile dict (see CampaignProfileError)."""
    if profile_name in {CAMPAIGN, CAMPAIGN_VALIDATION}:
        missing = [k for k in CAMPAIGN_REQUIRED_FIELDS if k not in profile_kwargs]
        if missing:
            raise CampaignProfileError(f"profile '{profile_name}' is missing required field(s) {missing}")
    for field_name in PER_TASK_PROFILE_FIELDS:
        if field_name not in profile_kwargs:
            continue
        value = profile_kwargs[field_name]
        per_task = value if isinstance(value, dict) else {t: value for t in tasks}
        missing_tasks = [t for t in tasks if t not in per_task]
        if missing_tasks:
            raise CampaignProfileError(
                f"profile '{profile_name}' field '{field_name}' has no value for task(s) {missing_tasks}"
            )
        for task, v in per_task.items():
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise CampaignProfileError(
                    f"profile '{profile_name}' field '{field_name}' for task '{task}' must be an integer "
                    f">= 1, got {v!r}"
                )
    if "ppo_seed_aggregation" in profile_kwargs and profile_kwargs["ppo_seed_aggregation"] not in PPO_SEED_AGGREGATIONS:
        raise CampaignProfileError(
            f"profile '{profile_name}' has unknown ppo_seed_aggregation "
            f"{profile_kwargs['ppo_seed_aggregation']!r}; allowed: {sorted(PPO_SEED_AGGREGATIONS)}"
        )


def _profile_kwargs_for_task(profile_kwargs: dict, task: str) -> dict:
    """Resolves PER_TASK_PROFILE_FIELDS' {task: value} mappings to the one
    value for `task`; every other field passes through unchanged."""
    return {k: (v[task] if k in PER_TASK_PROFILE_FIELDS and isinstance(v, dict) else v)
            for k, v in profile_kwargs.items()}


def check_ppo_seeds_supported(resolved: dict) -> None:
    """Compatibility hook retained now that the evaluator supports all K >= 1."""
    return None


class FastDevRealRunGuardError(RuntimeError):
    """Raised when a REAL (non-surrogate) run is attempted under the
    `fast_dev` profile without an explicit acknowledgement -- fast_dev's
    ppo_timesteps is a pipeline-validation-scale placeholder (docs/
    decisions.md's "Direct-encoding real-PPO baseline budget" row), and
    launching a real PPO matrix under it silently would be exactly the kind
    of expensive, hard-to-notice mistake this guard exists to prevent."""
MANIFEST_NAME = "benchmark_manifest.json"

# The calibrated per-arm mutation strength (docs/decisions.md's mutation-
# tuning rows, most recently the "THIRD re-run" row) never reached the real
# benchmark.yaml-driven path -- the exact same wiring gap already found and
# fixed once for NCA's hidden_channels (see _BUILTIN_ARM_DEFAULT_KWARGS's
# own comment above), just for a different field, and never checked because
# resolved_config.py deliberately does NOT carry mutation_rate/mutation_sigma
# (its own docstring: "a solved value is itself an artifact, not a design
# choice"). The artifact IS the single source of truth here instead; read
# lazily (at resolve time, not import time) so merely importing this module
# does not require the calibration to have been run, matching
# OpenProfileDecisionError/FastDevRealRunGuardError's own "fail loudly only
# when a real resolution is actually attempted" discipline.
MUTATION_CALIBRATION_RAW = os.path.join("results", "mutation", "mutation_calibration_raw.json")


def _load_calibrated_mutation_kwargs() -> dict:
    """{registry_encoder: {param_name: param_value}} read from
    MUTATION_CALIBRATION_RAW (evodevo.analysis.mutation_calibration's own
    committed output). Raises FileNotFoundError with an actionable message
    if the artifact is missing, rather than silently falling back to each
    encoder's own uncalibrated class default (direct=0.1, nca=0.1,
    rule_based=0.2 -- all different from the calibrated values)."""
    if not os.path.exists(MUTATION_CALIBRATION_RAW):
        raise FileNotFoundError(
            f"{MUTATION_CALIBRATION_RAW} not found -- the calibrated per-arm mutation "
            "strength (docs/decisions.md's mutation-tuning rows) cannot be resolved for "
            "a real benchmark run. Run `python -m evodevo.analysis.mutation_calibration` "
            "first; do not silently fall back to each encoder's own uncalibrated default."
        )
    with open(MUTATION_CALIBRATION_RAW) as f:
        calibrated = json.load(f)["calibrated"]
    return {
        arm: {calibrated[arm]["param_name"]: calibrated[arm]["param_value"]}
        for arm in ("direct", "nca", "rule_based")
    }


def _task_to_config_stem(task: str) -> str:
    """'Walker-v0' -> 'walker', 'UpStepper-v0' -> 'upstepper' -- matches the
    naming convention of evodevo/experiments/configs/<method>/*.yaml."""
    stem = re.sub(r"-v\d+$", "", task)
    return stem.lower()


@dataclass
class ArmSpec:
    """The generic "distinct arm identity" mechanism: an arm is (encoder
    registry key, optional variant label). `variant=None` is the base/primary
    arm for that encoder -- its arm_id equals the bare encoder name, so every
    pre-existing single-string benchmark.yaml entry is unaffected. A second
    (third, fourth, ...) arm built on the SAME encoder class (e.g.
    rule_based's monotone ablation) is declared with a `variant` label, which
    both disambiguates its arm_id everywhere identity is used (config
    resolution, output dir, manifest, wall-clock projection, analysis
    grouping) AND is folded into run_experiment()'s existing `tag` parameter
    for path separation -- reusing the exact mechanism already used to
    separate the `strategy` axis, rather than inventing a second one."""

    encoder: str
    variant: Optional[str] = None
    encoder_kwargs: dict = field(default_factory=dict)

    @property
    def arm_id(self) -> str:
        return self.encoder if self.variant is None else f"{self.encoder}_{self.variant}"

    @property
    def is_variant(self) -> bool:
        return self.variant is not None

    @classmethod
    def from_yaml_entry(cls, entry: Union[str, dict]) -> "ArmSpec":
        if isinstance(entry, str):
            return cls(encoder=entry)
        return cls(
            encoder=entry["encoder"],
            variant=entry.get("variant"),
            encoder_kwargs=dict(entry.get("encoder_kwargs", {})),
        )


@dataclass
class BenchmarkConfig:
    # Each element is either a bare encoder registry name (str, e.g.
    # "rule_based" -- the base/primary arm, fully backward compatible with
    # every pre-existing benchmark.yaml) or a mapping
    # {encoder, variant, encoder_kwargs} describing an additional named arm
    # built on that same encoder class (e.g. {encoder: rule_based, variant:
    # monotone}). Use resolved_arms() to get the parsed List[ArmSpec] -- every
    # internal function below iterates that, never this raw field, so a new
    # variant only ever requires a new list entry here, never a code change.
    encoders: List[Union[str, dict]]
    tasks: List[str]
    seeds: List[int]
    configs_root: str = DEFAULT_CONFIGS_ROOT
    output_dir: str = DEFAULT_OUTPUT_DIR
    surrogate: bool = False
    # dev_steps/init_condition are required, fairness-critical experimental
    # factors (see evodevo.core.factors.ExperimentFactors) applied
    # identically to every encoder in this benchmark -- there is
    # deliberately no default here; a benchmark.yaml that omits them fails
    # to construct ExperimentFactors below rather than silently reusing a
    # per-arm default.
    factors: dict = dataclasses.field(default_factory=dict)
    # The strategy axis (P7): {ea, random_search} crossed with every encoder
    # -- random_search is the no-selection/no-mutation control that
    # separates the encoding's representational prior from the EA's search
    # contribution (see evodevo/evolution/random_search.py). Defaults to
    # ["ea"] only, so every pre-existing benchmark.yaml keeps its exact
    # current behaviour and output paths unless it opts in.
    strategies: List[str] = dataclasses.field(default_factory=lambda: [DEFAULT_STRATEGY])
    # Per-(strategy, task) seed count override: {strategy: {task: [seeds]}}.
    # Falls back to `seeds` when a strategy/task isn't listed here. Lets
    # random_search (the cheapest arm to run) match the primary comparison's
    # seed count on the primary task while using a reduced seed count
    # elsewhere, per this prompt's compute-note recommendation, WITHOUT
    # ever dropping random_search entirely.
    strategy_seeds: dict = dataclasses.field(default_factory=dict)
    # Per-arm encoder_kwargs override: {encoder_name: {kwarg: value}}, merged
    # on top of each arm's resolved default (see _resolved_arm_kwargs below)
    # and on top of the shared `factors` block, then passed through to
    # get_encoder/RuleBasedEncoder etc. as `remaining_kwargs`
    # (evodevo.core.factors.extract_factors already splits ExperimentFactors
    # fields out of this merged dict, so factor-shaped and encoder-specific
    # kwargs can coexist here without conflict). Deliberately NOT subject to
    # check_fairness/SHARED_BASE_KEYS: these are arm-specific dynamics
    # knobs (e.g. rule_based's readout_mode/readout_window), semantically
    # different from the genuinely-shared quantities that check protects
    # (same reasoning as NCA's update_rule/decay never being subject to it
    # either -- see docs/decisions.md's "Wire the resolved readout policy
    # into the primary benchmark path" row). Defaults to {} so every
    # pre-existing benchmark.yaml keeps its exact current behaviour unless a
    # config or _resolved_arm_kwargs's own built-in default opts an arm in.
    encoder_kwargs: dict = dataclasses.field(default_factory=dict)
    # Which named PROFILE's ppo_timesteps/population_size/offspring_size/
    # eval_budget to apply (see PROFILES above), overriding whatever
    # configs/shared/base.yaml says for those SAME fields for every arm
    # identically (so this can never create a fairness violation between
    # arms -- every arm gets the exact same profile-resolved values).
    # Defaults to "fast_dev" so every pre-existing benchmark.yaml/test that
    # doesn't set this keeps its EXACT current behaviour.
    profile: str = FAST_DEV
    # Explicit, caller-supplied override of PROFILES[profile], highest
    # precedence -- this is how a genuinely-made scoping decision for
    # `primary_matrix`'s population_size/eval_budget/offspring_size is
    # supplied, once made (see PRIMARY_MATRIX_OPEN_FIELDS/
    # OpenProfileDecisionError) -- NEVER a place to invent a default.
    profile_overrides: dict = dataclasses.field(default_factory=dict)
    # Explicit acknowledgement required to launch a REAL (non-surrogate) run
    # under `fast_dev` -- see FastDevRealRunGuardError. False by default, so
    # the guard is loud unless deliberately silenced.
    allow_fast_dev_real_run: bool = False
    # Campaign-extension fields (docs/campaign_prep/ext/). All default to
    # None, which reproduces the original campaign's behaviour exactly: the
    # original benchmark.yaml sets none of them, so its CampaignPlan (and
    # therefore its resume config hash) is byte-identical to before these
    # fields existed (pinned by tests/campaign/test_plan_hash_invariance.py).
    #   estimate_path:  the capacity estimate whose per-(arm|task) t_PPO priors
    #                   set each phase's PPO-task timeout and feed the
    #                   dashboard/monitor priors. None = the original
    #                   results/campaign_prep/final_estimate/final_estimate.json.
    #   decisions_path: a FINAL decision record (JSON) whose per-task
    #                   ppo_timesteps/ppo_seeds must equal this YAML's
    #                   profile_overrides exactly, or the plan refuses to build.
    #   guard_outcome:  the budget-guard outcome text recorded in the plan.
    estimate_path: Optional[str] = None
    decisions_path: Optional[str] = None
    guard_outcome: Optional[str] = None
    # Amendment 2 (docs/campaign_prep/ext/PREREGISTRATION_EXT.md): phases are
    # (task, seed block) pairs, followed by champion re-training stages. The
    # block is generated from the decision record (python -m evodevo.extension
    # amend2 derive); None reproduces every earlier campaign exactly.
    amendment2: Optional[dict] = None

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def config_path(self, encoder: str, task: str) -> str:
        return os.path.join(self.configs_root, encoder, f"{_task_to_config_stem(task)}.yaml")

    def resolved_arms(self) -> List[ArmSpec]:
        """Parses `self.encoders` (a mix of bare strings and/or
        {encoder, variant, encoder_kwargs} mappings) into the List[ArmSpec]
        every function in this module actually iterates."""
        return [ArmSpec.from_yaml_entry(entry) for entry in self.encoders]

    def resolved_profile_kwargs(self) -> dict:
        """PROFILES[self.profile], with self.profile_overrides layered on
        top (highest precedence). For `fast_dev` this is always all four
        SHARED_BASE_KEYS fields; for `primary_matrix` it is ppo_timesteps
        plus whatever of PRIMARY_MATRIX_OPEN_FIELDS profile_overrides has
        explicitly supplied -- possibly none of them, by design. This is the
        FULL declarative profile (used for display/error messages and by
        _check_profile_ready_for_real_run) -- see applied_profile_kwargs()
        for what resolve_configs() actually applies to a resolved config."""
        if self.profile not in PROFILES:
            raise ValueError(f"unknown profile '{self.profile}'; available: {sorted(PROFILES)}")
        resolved = {**PROFILES[self.profile], **self.profile_overrides}
        validate_profile(resolved, self.tasks, self.profile)
        return resolved

    def applied_profile_kwargs(self) -> dict:
        """What resolve_configs() actually overrides on top of each arm's
        own resolved (shared/base.yaml + per-method) config. `fast_dev` is
        a deliberate NO-OP here -- it is BY DEFINITION today's existing
        behaviour (whatever configs/shared/base.yaml, or a test's own
        fixture, already provides), not a value forcibly re-applied on top
        of it -- so a test using a custom, deliberately tiny eval_budget for
        fast iteration is completely unaffected by the profile mechanism.
        `primary_matrix` (and any other future non-fast_dev profile) DOES
        apply its resolved_profile_kwargs(), since changing the effective
        budget is the entire point of selecting it."""
        if self.profile == FAST_DEV:
            return dict(self.profile_overrides)
        return self.resolved_profile_kwargs()

    def seeds_for(self, strategy: str, task: str) -> List[int]:
        per_strategy = self.strategy_seeds.get(strategy)
        if not per_strategy:
            return self.seeds
        per_task = per_strategy.get(task)
        return per_task if per_task else self.seeds


# Built-in per-arm default encoder_kwargs, resolved from each arm's own
# resolved_config.py candidate rather than hardcoded literals here -- this is
# what makes rule_based's resolved readout policy (mode_of_window, W=6)
# reach the real benchmark path, mirroring how NCA's resolved readout
# reaches it via NCAEncoder's own class default (see benchmark.yaml's
# header comment and docs/decisions.md's "Wire the resolved readout policy
# into the primary benchmark path" row). Keyed by arm_id (NOT bare encoder),
# so a second arm built on the same encoder class -- e.g. "rule_based_monotone"
# -- gets its own resolved default (here, sourced from
# resolved_config.MONOTONE_ABLATION, the other PPO_MATRIX_CANDIDATES entry)
# without colliding with "rule_based"'s. An ArmSpec.encoder_kwargs entry (from
# the encoders: list itself) is layered on top of this, and
# bconfig.encoder_kwargs[arm_id] on top of that (highest precedence).
_BUILTIN_ARM_DEFAULT_KWARGS = {
    "rule_based": {
        "readout_mode": RULE_BASED_PRIMARY.readout_mode,
        "readout_window": RULE_BASED_PRIMARY.readout_window,
    },
    "rule_based_monotone": {
        "readout_mode": RULE_BASED_MONOTONE.readout_mode,
        "readout_window": RULE_BASED_MONOTONE.readout_window,
        "action_monotonicity": RULE_BASED_MONOTONE.action_monotonicity,
    },
    # hidden_channels is the ONE field of NCA's resolved architecture that
    # does NOT reach the real benchmark path via NCAEncoder's own class
    # defaults (unlike update_rule/decay/state_bound/init_gain/readout_mode/
    # readout_window, which already equal resolved_config.PRIMARY exactly as
    # NCAEncoder's own constructor defaults) -- hidden_channels instead comes
    # from the shared, cross-encoder ExperimentFactors.hidden_channels field
    # (evodevo/core/factors.py), whose own default (3) predates the N-series
    # "Hidden channel count (H) -- final re-resolution" decision (DROP H to
    # 0, docs/decisions.md) and was never wired to track it. Confirmed via
    # docs/decisions.md's row history that H=0 is the last-recorded,
    # unsuperseded decision (resolved_config.py::PRIMARY.hidden_channels is
    # already 0 and correct) -- this entry is the missing wiring, not a
    # change to the decision itself.
    "nca": {
        "hidden_channels": NCA_PRIMARY.hidden_channels,
    },
}


def _resolved_arm_kwargs(bconfig: "BenchmarkConfig", arm: ArmSpec) -> dict:
    """encoder_kwargs actually passed to run_experiment/get_encoder for
    `arm`: the shared `factors` block, layered with the calibrated mutation
    strength for this arm's BASE encoder (keyed by arm.encoder, e.g.
    "rule_based_monotone" still gets rule_based's own calibrated
    mutation_rate, since mutation strength is a property of the encoder
    class/genome representation, not of the dynamics variant), layered with
    this arm's built-in resolved dynamics default (if any, keyed by
    arm_id), layered with any encoder_kwargs declared inline on the ArmSpec
    itself, layered with any explicit bconfig.encoder_kwargs[arm_id]
    override (highest precedence)."""
    merged = dict(bconfig.factors)
    merged.update(_load_calibrated_mutation_kwargs().get(arm.encoder, {}))
    merged.update(_BUILTIN_ARM_DEFAULT_KWARGS.get(arm.arm_id, {}))
    merged.update(arm.encoder_kwargs)
    merged.update(bconfig.encoder_kwargs.get(arm.arm_id, {}))
    return merged


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
    for lib in ("numpy", "yaml", "pytest", "pandas", "matplotlib"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def resolve_configs(bconfig: BenchmarkConfig) -> dict:
    """Returns {(arm_id, task): ExperimentConfig}, using bconfig.seeds[0] as
    a representative seed (seed is exempt from the fairness protocol, so any
    seed produces identical values on every other field). Two arms sharing
    an underlying encoder (e.g. rule_based / rule_based_monotone) resolve
    their config from the SAME configs/<encoder>/<task>.yaml -- only their
    encoder_kwargs (see _resolved_arm_kwargs) differ -- but are keyed
    distinctly here by arm_id.

    bconfig.applied_profile_kwargs() (a no-op for the default `fast_dev`
    profile; ppo_timesteps=1_000_000 plus any explicit profile_overrides for
    `primary_matrix`) is applied identically to every arm's resolved config
    here, on top of whatever configs/shared/base.yaml (or a test's own
    fixture) provided for those same fields -- since it is applied
    identically across every arm, this can never introduce a fairness
    violation between arms."""
    representative_seed = bconfig.seeds[0]
    profile_kwargs = bconfig.applied_profile_kwargs()
    resolved = {}
    for arm in bconfig.resolved_arms():
        for task in bconfig.tasks:
            path = bconfig.config_path(arm.encoder, task)
            config = ExperimentConfig.from_method_yaml(path)
            # The per-method file is located by a version-stripped stem
            # (ObstacleTraverser-v0 and -v1 share obstacletraverser.yaml), and
            # that file's own `task:` is what PPO actually trains on. A
            # mismatch would silently run the wrong environment, so refuse.
            if config.task != task:
                raise ValueError(
                    f"{path} declares task {config.task!r} but the benchmark requests {task!r}; "
                    "the per-method config would train PPO on the wrong environment"
                )
            resolved[(arm.arm_id, task)] = dataclasses.replace(
                config, seed=representative_seed, **_profile_kwargs_for_task(profile_kwargs, task)
            )
    return resolved


def _check_profile_ready_for_real_run(bconfig: BenchmarkConfig) -> None:
    """Refuses to proceed with a REAL (non-surrogate) run unless: (a) the
    active profile has every SHARED_BASE_KEYS field this project tracks
    explicitly resolved (currently only `primary_matrix` can fail this, by
    design -- see PRIMARY_MATRIX_OPEN_FIELDS), and (b) a real run under
    `fast_dev` is explicitly acknowledged (fast_dev's ppo_timesteps is a
    disclosed pipeline-validation-scale placeholder, not the scientific
    comparison budget). A surrogate run is exempt from both checks -- it
    never trains real PPO, so neither guard's concern applies. Called before
    ANY run starts (including a non-surrogate dry run, since
    estimate_wall_clock's projection is meaningless without a real
    population_size/eval_budget to extrapolate from)."""
    if bconfig.surrogate:
        return

    resolved = bconfig.resolved_profile_kwargs()
    if bconfig.profile == PRIMARY_MATRIX:
        missing = [k for k in PRIMARY_MATRIX_OPEN_FIELDS if k not in resolved]
        if missing:
            raise OpenProfileDecisionError(
                f"profile='primary_matrix' has no explicit value for {missing} -- this is "
                "a genuine OPEN project decision (population_size/offspring_size/eval_budget "
                "for the primary matrix), pending the pre-flight wall-clock projection and an "
                "explicit scoping decision (see docs/PRE_PARALLELIZATION_READINESS.md's 'NOT "
                "covered' section). This is not a bug to route around -- pass "
                "profile_overrides={'population_size': ..., 'offspring_size': ..., "
                "'eval_budget': ...} explicitly once that decision is made; do not guess a value."
            )

    if bconfig.profile == FAST_DEV and not bconfig.allow_fast_dev_real_run:
        raise FastDevRealRunGuardError(
            f"refusing to launch a REAL (non-surrogate) PPO run under profile='fast_dev' "
            f"(ppo_timesteps={resolved['ppo_timesteps']}, population_size={resolved['population_size']}, "
            f"eval_budget={resolved['eval_budget']} -- a pipeline-validation-scale placeholder, "
            "NOT the scientific comparison budget; see docs/decisions.md's 'Direct-encoding "
            "real-PPO baseline budget' row). If this is genuinely intended (e.g. local "
            "smoke-testing with real PPO), pass allow_fast_dev_real_run=True "
            "(--i-know-this-is-fast-dev on the CLI) explicitly."
        )


def check_fairness(bconfig: BenchmarkConfig) -> dict:
    """Resolve every (arm, task) config and assert pairwise comparability
    within each task, BEFORE anything is run. Raises FairnessViolation on
    any mismatch. Returns the resolved configs dict."""
    unknown_strategies = [s for s in bconfig.strategies if s not in STRATEGIES]
    if unknown_strategies:
        raise ValueError(f"unknown strategy(ies) {unknown_strategies}; available: {sorted(STRATEGIES)}")

    arms = bconfig.resolved_arms()
    resolved = resolve_configs(bconfig)
    for task in bconfig.tasks:
        for arm_a, arm_b in itertools.combinations(arms, 2):
            assert_comparable(resolved[(arm_a.arm_id, task)], resolved[(arm_b.arm_id, task)])

    # Run-start assertion (see evodevo.evolution.ea.resolved_policy_for):
    # the invalid-genome/cache-charging/budget-counter policy every arm will
    # actually run under must be identical, checked here before any run
    # starts, not just implicitly true because run_ea() is shared code.
    for task in bconfig.tasks:
        policies = {arm.arm_id: resolved_policy_for(resolved[(arm.arm_id, task)]) for arm in arms}
        assert_identical_policy(policies)

    return resolved


def _shared_fields(config: ExperimentConfig) -> dict:
    d = config.to_dict()
    return {k: d[k] for k in sorted(SHARED_BASE_KEYS)}


def _run_config_for(bconfig: BenchmarkConfig, resolved_configs: dict, arm: ArmSpec, task: str, seed: int) -> ExperimentConfig:
    """The per-run config actually used to execute (arm, task, seed):
    the resolved (arm_id, task) config with output_dir forced to
    bconfig.output_dir. This makes bconfig.output_dir the single
    authoritative root for both the manifest and every run's output
    directory -- a per-method configs/<method>/*.yaml's own `output_dir`
    field is NOT used for benchmark-orchestrated runs (it still is when
    that config is run directly via run_experiment.py outside a
    benchmark)."""
    return dataclasses.replace(resolved_configs[(arm.arm_id, task)], seed=seed, output_dir=bconfig.output_dir)


def _output_dir_for(bconfig: BenchmarkConfig, resolved_configs: dict, arm: ArmSpec, task: str, seed: int,
                     strategy: str) -> str:
    """Folds both the arm's variant label (if any -- e.g. "monotone") and
    the strategy into run_experiment.py's `tag`, via the exact same
    _combine_tag_and_strategy() helper run_experiment() itself uses -- so a
    benchmark-orchestrated run and a directly-invoked run_experiment.py call
    with the same (arm, task, seed, strategy) always land at the same
    path, for any number of variant arms, not just the strategy axis."""
    effective_tag = _combine_tag_and_strategy(arm.variant, strategy)
    run_config = _run_config_for(bconfig, resolved_configs, arm, task, seed)
    return _run_output_dir(run_config, seed, tag=effective_tag)


def _resolved_config_path(bconfig: BenchmarkConfig, arm: ArmSpec, task: str, seed: int) -> str:
    """Where a fully-resolved, self-contained per-run config is written so
    run_experiment.py (which always reloads its config from a file path)
    executes against bconfig.output_dir rather than whatever `output_dir`
    the source configs/<method>/*.yaml happens to contain. Keyed by arm_id
    (not bare encoder), so two arms sharing an encoder class never overwrite
    each other's resolved config file."""
    directory = os.path.join(bconfig.output_dir, "_resolved_configs", arm.arm_id, _task_to_config_stem(task))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"seed_{seed}.yaml")


def build_manifest(bconfig: BenchmarkConfig, resolved_configs: dict) -> dict:
    """The resolved shared config (proven identical across every arm by
    check_fairness), each arm's describe(), the git commit, library
    versions, and the full planned (arm, task, seed) run matrix -- written
    BEFORE any run starts. Every arm-identity-bearing entry here is keyed by
    arm_id, not bare encoder, so two arms sharing an encoder class (e.g.
    rule_based / rule_based_monotone) each get their own manifest entry
    rather than colliding into one."""
    any_config = next(iter(resolved_configs.values()))
    shared_config = _shared_fields(any_config)

    arms = bconfig.resolved_arms()
    encoder_describe = {}
    for arm in arms:
        any_arm_config = next(cfg for (aid, _task), cfg in resolved_configs.items() if aid == arm.arm_id)
        factors, remaining_kwargs = extract_factors(_resolved_arm_kwargs(bconfig, arm), seed=any_arm_config.seed)
        encoder_describe[arm.arm_id] = get_encoder(
            arm.encoder, factors=factors, grid_shape=any_arm_config.grid_shape, **remaining_kwargs
        ).describe()

    run_matrix = [
        {
            "arm_id": arm.arm_id,
            "encoder": arm.encoder,
            "task": task,
            "seed": seed,
            "strategy": strategy,
            "output_dir": _output_dir_for(bconfig, resolved_configs, arm, task, seed, strategy),
        }
        for arm in arms
        for task in bconfig.tasks
        for strategy in bconfig.strategies
        for seed in bconfig.seeds_for(strategy, task)
    ]

    return {
        "shared_config": shared_config,
        "encoders": encoder_describe,
        "git_commit": _git_commit_hash(),
        "library_versions": _library_versions(),
        "surrogate": bconfig.surrogate,
        "run_matrix": run_matrix,
    }


def _is_run_complete(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, DONE_MARKER))


def estimate_wall_clock(bconfig: BenchmarkConfig, resolved_configs: dict) -> dict:
    """Measures ONE real evaluation's wall-clock cost per arm (using that
    arm's own resolved hyperparameters and evaluator mode) and extrapolates
    the whole benchmark's estimated total wall-clock from it. Keyed by
    arm_id -- two arms sharing an encoder class (e.g. rule_based's PRIMARY
    and monotone variants) each get their own measured cost rather than one
    overwriting the other."""
    evaluator = make_evaluator("surrogate" if bconfig.surrogate else "real")
    arms = bconfig.resolved_arms()
    per_encoder_cost = {}

    for arm in arms:
        any_task = bconfig.tasks[0]
        config = resolved_configs[(arm.arm_id, any_task)]
        factors, remaining_kwargs = extract_factors(_resolved_arm_kwargs(bconfig, arm), seed=config.seed)
        encoder_instance = get_encoder(arm.encoder, factors=factors, grid_shape=config.grid_shape, **remaining_kwargs)
        rng = np.random.default_rng(config.seed)
        genome = encoder_instance.random_genome(rng)
        body = encoder_instance.decode(genome)

        start = time.time()
        evaluator(
            body,
            task=config.task,
            ppo_timesteps=config.ppo_timesteps,
            eval_episodes=config.ppo_eval_episodes,
            seed=config.seed,
        )
        per_encoder_cost[arm.arm_id] = time.time() - start

    total_estimate = 0.0
    n_runs_per_encoder = len(bconfig.tasks) * len(bconfig.seeds)
    for arm in arms:
        any_task = bconfig.tasks[0]
        eval_budget = resolved_configs[(arm.arm_id, any_task)].eval_budget
        total_estimate += per_encoder_cost[arm.arm_id] * eval_budget * n_runs_per_encoder

    return {
        "per_encoder_measured_cost_s": per_encoder_cost,
        "estimated_total_wall_clock_s": total_estimate,
    }


def _write_manifest(bconfig: BenchmarkConfig, manifest: dict) -> str:
    os.makedirs(bconfig.output_dir, exist_ok=True)
    path = os.path.join(bconfig.output_dir, MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    return path


def run_benchmark(bconfig: BenchmarkConfig, dry_run: bool = False, allow_dirty: bool = False) -> dict:
    # -1. Announce the active profile and (for a real run) refuse to
    # proceed under an unready/unacknowledged one, BEFORE anything else --
    # cheapest possible check, and the one this project's own history shows
    # is worth failing loudly rather than silently (see PROFILES above).
    print(f"[profile] {bconfig.profile} (surrogate={bconfig.surrogate})")
    _check_profile_ready_for_real_run(bconfig)

    # 0. Provenance gate FIRST, before anything else -- a standalone check
    # (evodevo.core.provenance), independent of ExperimentConfig/lock_hash.
    # Computed EXACTLY ONCE here, before any of this function's own writes
    # (write_factors_json/_write_manifest below, or each run_experiment()
    # call's own factors.json), and threaded through to every
    # run_experiment() call via precomputed_git_state. Without this, each
    # run_experiment() call would re-derive git state independently and see
    # this function's own prior bookkeeping writes as tree dirtiness --
    # spuriously raising DirtyTreeError on a later (genuinely new) run in
    # the same invocation, even though the tree was clean when the
    # invocation started.
    commit_hash, dirty_files = check_git_state(allow_dirty)

    # 1. Fairness check FIRST, before anything is run or written.
    resolved_configs = check_fairness(bconfig)
    # 1b. D16's K > 1 is not consumed by the evaluator yet: refuse a real
    # execution (a dry run evaluates nothing, so it may still project costs).
    if not dry_run:
        check_ppo_seeds_supported(resolved_configs)

    factors = ExperimentFactors(seed=0, **bconfig.factors)
    write_factors_json(bconfig.output_dir, factors, commit_hash, dirty_files)

    # 2. Manifest written before any run starts.
    manifest = build_manifest(bconfig, resolved_configs)
    manifest_path = _write_manifest(bconfig, manifest)

    if dry_run:
        timing = estimate_wall_clock(bconfig, resolved_configs)
        manifest["dry_run_timing_estimate"] = timing
        _write_manifest(bconfig, manifest)
        print(json.dumps(manifest["run_matrix"], indent=2))
        print(
            f"Estimated total wall-clock: {timing['estimated_total_wall_clock_s']:.1f}s "
            f"(measured per-encoder costs: {timing['per_encoder_measured_cost_s']})"
        )
        return manifest

    # 3. Execute, resuming (skipping) any already-completed run.
    # eval_budgets drives ProgressTracker's ETA math -- purely additive stdout
    # reporting, wired in via run_experiment's optional progress_callback.
    # Keyed by arm_id (not bare encoder) so two arms sharing an encoder class
    # never collide in the budget lookup.
    arms_by_id = {arm.arm_id: arm for arm in bconfig.resolved_arms()}
    eval_budgets = {
        (entry["arm_id"], entry["task"], entry["seed"]): resolved_configs[(entry["arm_id"], entry["task"])].eval_budget
        for entry in manifest["run_matrix"]
    }
    tracker = ProgressTracker(eval_budgets)

    completed, skipped = [], []
    for entry in manifest["run_matrix"]:
        arm_id, task, seed = entry["arm_id"], entry["task"], entry["seed"]
        arm = arms_by_id[arm_id]
        strategy = entry["strategy"]
        output_dir = entry["output_dir"]
        label = f"{arm_id}/{task}/{strategy}/seed_{seed}"

        if _is_run_complete(output_dir):
            print(f"[skip] {label} already complete at {output_dir} -- resuming")
            skipped.append(entry)
            continue

        run_config = _run_config_for(bconfig, resolved_configs, arm, task, seed)
        resolved_path = _resolved_config_path(bconfig, arm, task, seed)
        run_config.to_yaml(resolved_path)

        print(f"[run]  {label}")
        run_progress = tracker.for_run(arm_id, task, seed)
        run_experiment(resolved_path, seed, surrogate=bconfig.surrogate, overwrite=False,
                        encoder_kwargs=_resolved_arm_kwargs(bconfig, arm),
                        progress_callback=run_progress,
                        allow_dirty=allow_dirty,
                        strategy=strategy,
                        tag=arm.variant,
                        arm_id=arm.arm_id,
                        precomputed_git_state=(commit_hash, dirty_files))
        completed.append(entry)

    manifest["completed_this_invocation"] = completed
    manifest["skipped_already_done"] = skipped
    _write_manifest(bconfig, manifest)
    print(f"manifest written to {manifest_path}")
    print(f"completed {len(completed)} run(s), skipped {len(skipped)} already-done run(s)")

    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a benchmark YAML config")
    parser.add_argument("--dry-run", action="store_true", help="print the run matrix and a wall-clock estimate; run nothing")
    parser.add_argument("--surrogate", action="store_true", help="override the config's surrogate flag to True")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow running with an uncommitted working tree (local iteration only -- "
        "never for a run whose output is cited in a report or a docs/decisions.md row)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=None,
        help="override the config's profile (default: whatever the config's own `profile` field "
        "says, or 'fast_dev' if unset)",
    )
    parser.add_argument(
        "--i-know-this-is-fast-dev",
        action="store_true",
        dest="allow_fast_dev_real_run",
        help="required to launch a REAL (non-surrogate) run under the 'fast_dev' profile -- "
        "see FastDevRealRunGuardError",
    )
    args = parser.parse_args(argv)

    bconfig = BenchmarkConfig.from_yaml(args.config)
    if args.surrogate:
        bconfig = dataclasses.replace(bconfig, surrogate=True)
    if args.profile is not None:
        bconfig = dataclasses.replace(bconfig, profile=args.profile)
    if args.allow_fast_dev_real_run:
        bconfig = dataclasses.replace(bconfig, allow_fast_dev_real_run=True)

    try:
        run_benchmark(bconfig, dry_run=args.dry_run, allow_dirty=args.allow_dirty)
    except (FairnessViolation, ConfigOverrideError) as e:
        # Both are "abort before anything runs" outcomes: ConfigOverrideError
        # fires earlier (at config-load time, if a per-method config tries to
        # override a shared-base key); FairnessViolation fires afterward (if
        # two already-loaded configs still disagree on a required-equal
        # field). Either way, nothing has been written to output_dir yet.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except (OpenProfileDecisionError, FastDevRealRunGuardError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except DirtyTreeError:
        sys.exit(1)


if __name__ == "__main__":
    main()
