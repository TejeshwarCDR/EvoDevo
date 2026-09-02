"""Reusable Encoder-contract assertion functions.

Each `check_*` function takes an Encoder instance and raises AssertionError
with a message prefixed by a stable `[contract:<clause>]` tag identifying
which clause of the contract was violated, if it was. These are shared by:
  - tests/encoders/test_contract.py, which runs every check against every
    encoder currently in the registry (evodevo.encoders.list_encoders());
  - tests/encoders/test_contract_self_check.py, which proves the mechanism
    itself works by running these checks against a deliberately broken
    dummy encoder and asserting they fail.
"""
from __future__ import annotations

import ast
import inspect
import json
import warnings

import numpy as np

from evodevo.core.materials import N_MATERIALS
from evodevo.morphology.validate import validate_body

FORBIDDEN_MODULE_ROOTS = {"evogym", "stable_baselines3", "gym", "torch"}


def _genomes_equal(a, b) -> bool:
    """Generic genome equality: numpy array_equal for array-like genomes,
    plain == otherwise (e.g. a variable-length list of rule objects)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def check_decode_returns_valid_5x5_int_array(encoder) -> None:
    rng = np.random.default_rng(0)
    genome = encoder.random_genome(rng)
    body = encoder.decode(genome)
    assert isinstance(body, np.ndarray), (
        f"[contract:decode_shape] {encoder.name}.decode() must return a numpy array, got {type(body)!r}"
    )
    assert body.shape == (5, 5), (
        f"[contract:decode_shape] {encoder.name}.decode() must return a (5, 5) array, got shape {body.shape}"
    )
    assert np.issubdtype(body.dtype, np.integer), (
        f"[contract:decode_shape] {encoder.name}.decode() must return an integer-dtype array, got {body.dtype}"
    )
    values = set(np.unique(body).tolist())
    assert values <= set(range(N_MATERIALS)), (
        f"[contract:decode_values] {encoder.name}.decode() returned values outside 0..{N_MATERIALS - 1}: "
        f"{sorted(values - set(range(N_MATERIALS)))} (material id 5 = EvoGym terrain, never a valid robot voxel)"
    )


def check_decode_is_deterministic(encoder) -> None:
    rng = np.random.default_rng(1)
    genome = encoder.random_genome(rng)
    body1 = encoder.decode(genome)
    body2 = encoder.decode(genome)
    assert np.array_equal(body1, body2), (
        f"[contract:decode_deterministic] {encoder.name}.decode() returned different results "
        "for the same genome across two calls"
    )


def check_random_genome_reproducible_and_seed_sensitive(encoder) -> None:
    rng_a1 = np.random.default_rng(7)
    rng_a2 = np.random.default_rng(7)
    genome_a1 = encoder.random_genome(rng_a1)
    genome_a2 = encoder.random_genome(rng_a2)
    assert _genomes_equal(genome_a1, genome_a2), (
        f"[contract:random_genome_reproducible] {encoder.name}.random_genome() gave different "
        "genomes for two rng.Generator instances seeded identically"
    )

    rng_b = np.random.default_rng(8)
    genome_b = encoder.random_genome(rng_b)
    assert not _genomes_equal(genome_a1, genome_b), (
        f"[contract:random_genome_seed_sensitive] {encoder.name}.random_genome() gave the same "
        "genome for two different rng seeds (7 and 8) -- it does not appear to consume the rng"
    )


def check_mutate_is_reproducible(encoder) -> None:
    rng_seed = np.random.default_rng(2)
    genome = encoder.random_genome(rng_seed)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    mutated1 = encoder.mutate(genome, rng1)
    mutated2 = encoder.mutate(genome, rng2)
    assert _genomes_equal(mutated1, mutated2), (
        f"[contract:mutate_reproducible] {encoder.name}.mutate() gave different results for the "
        "same genome and two rng.Generator instances seeded identically"
    )


def check_mutate_output_roundtrips_through_serialize(encoder) -> None:
    rng_seed = np.random.default_rng(3)
    genome = encoder.random_genome(rng_seed)
    mutated = encoder.mutate(genome, np.random.default_rng(11))

    roundtripped = encoder.deserialize_genome(encoder.serialize_genome(mutated))
    assert _genomes_equal(mutated, roundtripped), (
        f"[contract:mutate_serialize_roundtrip] {encoder.name}.mutate() produced a genome that does "
        "not survive deserialize_genome(serialize_genome(g)) unchanged"
    )


def check_serialize_deserialize_roundtrip(encoder) -> None:
    rng = np.random.default_rng(4)
    genome = encoder.random_genome(rng)
    serialized = encoder.serialize_genome(genome)

    json.dumps(serialized)  # must be JSON-serialisable

    roundtripped = encoder.deserialize_genome(serialized)
    assert _genomes_equal(genome, roundtripped), (
        f"[contract:serialize_roundtrip] {encoder.name}: deserialize_genome(serialize_genome(g)) "
        "!= g for a freshly sampled random genome"
    )


def check_random_genome_sometimes_produces_invalid_bodies(encoder, n: int = 500) -> None:
    """An encoder must never secretly resample/repair until a body is valid.
    Over N random genomes, at least one must decode to an invalid body --
    if every single one is valid, that is a strong signal of hidden repair
    logic hiding inside random_genome()/decode(), which violates the
    contract (base.py explicitly forbids this)."""
    rng = np.random.default_rng(20260902)
    invalid_found = False
    for _ in range(n):
        genome = encoder.random_genome(rng)
        body = encoder.decode(genome)
        if not validate_body(body).valid:
            invalid_found = True
            break

    if not invalid_found:
        warnings.warn(
            f"[contract:no_resample_until_valid] encoder '{encoder.name}' produced {n}/{n} valid "
            "random bodies. This usually means decode()/random_genome() contains hidden "
            "repair-until-valid logic, which the Encoder contract explicitly forbids "
            "(invalid-genome handling must be left entirely to the pipeline).",
            stacklevel=2,
        )
    assert invalid_found, (
        f"[contract:no_resample_until_valid] encoder '{encoder.name}' never produced an invalid "
        f"body across {n} random genomes -- see the warning above"
    )


def check_describe_is_json_serialisable_and_covers_constructor_params(encoder) -> None:
    described = encoder.describe()
    json.dumps(described)  # must not raise

    sig = inspect.signature(type(encoder).__init__)
    ctor_params = [p for p in sig.parameters if p != "self"]
    missing = [p for p in ctor_params if p not in described]
    assert not missing, (
        f"[contract:describe_completeness] {encoder.name}.describe() is missing constructor "
        f"parameters {missing}; describe() must include every hyperparameter that can affect "
        "decode() so a run's configuration is fully reconstructable from its log"
    )


def check_module_is_physics_and_rl_free(encoder) -> None:
    module_file = inspect.getsourcefile(type(encoder))
    assert module_file is not None, f"could not locate source file for {type(encoder)!r}"
    source = open(module_file).read()
    tree = ast.parse(source, filename=module_file)

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])

    forbidden_hit = imported_roots & FORBIDDEN_MODULE_ROOTS
    assert not forbidden_hit, (
        f"[contract:physics_rl_free] {encoder.name}'s module ({module_file}) imports "
        f"{sorted(forbidden_hit)} -- encoders must stay physics-free and RL-free; only "
        "evodevo/evaluation/ppo_runner.py may import these"
    )


ALL_CHECKS = [
    check_decode_returns_valid_5x5_int_array,
    check_decode_is_deterministic,
    check_random_genome_reproducible_and_seed_sensitive,
    check_mutate_is_reproducible,
    check_mutate_output_roundtrips_through_serialize,
    check_serialize_deserialize_roundtrip,
    check_random_genome_sometimes_produces_invalid_bodies,
    check_describe_is_json_serialisable_and_covers_constructor_params,
    check_module_is_physics_and_rl_free,
]
