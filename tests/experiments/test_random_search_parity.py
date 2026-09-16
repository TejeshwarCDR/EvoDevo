"""Part 4: random_search must be a first-class arm for EVERY encoder, with
EXACTLY the same resolved encoder_kwargs (readout policy, dynamics
parameters) as its EA counterpart -- the only thing permitted to differ
between an arm's `ea` and `random_search` runs is the search strategy
itself, never the encoder configuration.

evodevo/experiments/run_experiment.py::run_experiment() constructs its
encoder exactly once, before branching into STRATEGIES[strategy] (see its
own source) -- so this is structurally guaranteed already, for every arm,
including rule_based_monotone once Part 1 registers it as an ArmSpec
flowing through the same bconfig.strategies x resolved_arms() cross product.
This test makes that guarantee explicit and permanent (not just
structurally true) via the REAL, production benchmark.yaml config path --
mirroring test_benchmark.py::test_real_benchmark_yaml_wires_rule_based_resolved_readout_policy's
style -- rather than relying on code inspection alone.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile

from evodevo.experiments.benchmark import BenchmarkConfig, run_benchmark

# Dynamics/readout fields that must be identical between an arm's ea and
# random_search encoder -- deliberately excludes fields that are expected to
# vary for unrelated reasons (e.g. none currently, but kept as an explicit
# allowlist rather than comparing the whole describe() dict, since e.g. a
# future encoder-level RNG-derived field would be a false-positive mismatch
# unrelated to this test's actual concern).
DYNAMICS_FIELDS_BY_ENCODER = {
    "direct": ("mutation_rate",),
    "nca": ("update_rule", "decay", "state_bound", "hidden_channels", "spectral_normalise",
             "readout_mode", "readout_window"),
    "rule_based": ("readout_mode", "readout_window", "conflict_resolution", "position_predicate",
                    "boundary", "neighbourhood", "action_monotonicity", "mutation_rate"),
}


def test_random_search_resolved_kwargs_match_ea_for_every_real_benchmark_arm():
    with tempfile.TemporaryDirectory() as output_dir:
        real_bconfig = BenchmarkConfig.from_yaml("evodevo/experiments/configs/benchmark.yaml")
        bconfig = dataclasses.replace(
            real_bconfig,
            seeds=[0],
            strategies=["ea", "random_search"],
            output_dir=output_dir,
            surrogate=True,
            factors={"dev_steps": 5, "init_condition": "center_seed"},
        )

        manifest = run_benchmark(bconfig, dry_run=False, allow_dirty=True)

        by_arm_strategy = {(e["arm_id"], e["strategy"]): e for e in manifest["run_matrix"]}
        arm_ids = sorted({arm_id for arm_id, _strategy in by_arm_strategy})
        assert arm_ids == sorted(a.arm_id for a in bconfig.resolved_arms())
        assert len(arm_ids) == 4  # direct, nca, rule_based, rule_based_monotone

        for arm_id in arm_ids:
            ea_entry = by_arm_strategy[(arm_id, "ea")]
            rs_entry = by_arm_strategy[(arm_id, "random_search")]
            assert ea_entry["output_dir"] != rs_entry["output_dir"]

            with open(os.path.join(ea_entry["output_dir"], "run_meta.json")) as f:
                ea_meta = json.load(f)
            with open(os.path.join(rs_entry["output_dir"], "run_meta.json")) as f:
                rs_meta = json.load(f)

            registry_encoder = ea_meta["config"]["encoder"]
            assert rs_meta["config"]["encoder"] == registry_encoder

            fields = DYNAMICS_FIELDS_BY_ENCODER[registry_encoder]
            ea_describe = ea_meta["encoder"]
            rs_describe = rs_meta["encoder"]
            for field in fields:
                assert ea_describe[field] == rs_describe[field], (
                    f"arm={arm_id} field={field}: ea={ea_describe[field]!r} != "
                    f"random_search={rs_describe[field]!r} -- random_search must resolve the "
                    "SAME encoder configuration as its ea counterpart, only the search strategy "
                    "may differ"
                )

            # And the encoder_kwargs actually passed through are identical too
            # (not just the encoder's own describe() -- the recorded input).
            assert ea_meta["config"]["encoder_kwargs"] == rs_meta["config"]["encoder_kwargs"]
