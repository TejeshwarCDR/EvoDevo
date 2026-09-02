# Adding a new encoder

This project treats each genome-to-body representation (direct, NCA, and
eventually rule-based) as an isolated plugin behind one frozen contract:
`evodevo.encoders.base.Encoder`. The contract is central and stable so that
team members can work on different encodings without colliding — you should
never need to touch shared pipeline code to make your encoder work.

## The rule

**A contributor must not modify `evodevo/encoders/base.py`,
`evodevo/encoders/registry.py`, `evodevo/evolution/ea.py`,
`evodevo/evaluation/ppo_runner.py`, or anything under `evodevo/morphology/`
to make their encoder work.**

If you find yourself needing to change one of those files so your encoder
can be used, that is a signal the contract itself is incomplete or wrong —
open a discussion about extending the contract for everyone, rather than
special-casing your encoder in shared code. The whole point of the plugin
architecture is that shared code has no knowledge of which encodings exist.

## Steps

1. **Create a package under `evodevo/encoders/`.**

   ```
   evodevo/encoders/my_encoding/
     __init__.py    # re-exports your Encoder subclass
     encoder.py     # the implementation
   ```

   Look at `evodevo/encoders/direct/` or `evodevo/encoders/nca/` for the
   expected layout. `__init__.py` should just be:

   ```python
   from evodevo.encoders.my_encoding.encoder import MyEncoding

   __all__ = ["MyEncoding"]
   ```

2. **Implement the `Encoder` ABC** (`evodevo/encoders/base.py`). You must
   implement:

   - `name` (property) — a short, stable string identifier.
   - `genome_length()` — only meaningful for fixed-length genomes; for a
     variable-length genome (e.g. a list of rule objects) return whatever
     size makes sense for logging (e.g. the current number of rules).
   - `genome_dtype` (property) — a numpy dtype for array-like genomes, or
     the Python type `object` for anything else (e.g. a list of non-numeric
     objects).
   - `random_genome(rng)` — sample a genome using only the given
     `np.random.Generator`. Never use global RNG state.
   - `decode(genome)` — return a `(5, 5)` **integer** numpy array with
     values in `0..4` (material id `5` is EvoGym's reserved terrain
     material and must never appear here).
   - `mutate(genome, rng)` — return a mutated *copy*; never mutate the
     input in place.
   - `serialize_genome(genome)` / `deserialize_genome(obj)` — convert to and
     from a JSON-serialisable object. Required so your genome can be
     written to a run log or a saved-genome file even if it isn't a numpy
     array. Must round-trip exactly:
     `deserialize_genome(serialize_genome(g)) == g`.
   - `describe()` — a JSON-serialisable dict of every hyperparameter that
     can affect `decode()`. The generic conformance suite (see below)
     checks that this includes every constructor parameter.

   Optionally, if your encoding has a meaningful recombination operator,
   override:

   - `supports_crossover()` — return `True`.
   - `crossover(parent_a, parent_b, rng)` — return a child genome.

   If you don't implement crossover, leave both at their base-class
   defaults (`supports_crossover()` returns `False`; `crossover()` raises
   `NotImplementedError`).

   **What you must NOT do:** loop, resample, or repair inside
   `random_genome()`/`decode()`/`mutate()` to guarantee a valid body.
   Invalid-genome handling is deliberately a pipeline concern (see
   `evodevo/morphology/validate.py` and `evodevo/evaluation/ppo_runner.py`),
   applied identically to every encoding. An encoder that secretly filters
   out invalid bodies breaks the fairness of the cross-encoding comparison
   and will fail the conformance suite (see below).

   **Your module must stay physics- and RL-free**: do not import `evogym`,
   `stable_baselines3`, `gym`, or `torch` from your encoder module. Only
   `evodevo/evaluation/ppo_runner.py` is allowed to import those.

3. **Register your encoder** in `evodevo/encoders/__init__.py`:

   ```python
   from evodevo.encoders.my_encoding import MyEncoding

   register_encoder("my_encoding", MyEncoding)
   ```

   Registration is always an explicit import + `register_encoder()` call —
   never dynamic filesystem scanning — so a broken import fails loudly at
   import time instead of your encoder silently not existing.

4. **Add a config directory**: `evodevo/experiments/configs/my_encoding/`,
   mirroring the shared fields in the existing `direct/` and `nca/` configs
   (`population_size`, `offspring_size`, `eval_budget`, `ppo_timesteps`,
   `ppo_eval_episodes`, `grid_shape`, ...). Only `encoder` and
   encoder-specific hyperparameters should differ from the other arms —
   see `evodevo/experiments/fairness.py::assert_comparable`.

5. **Run the conformance suite.** The moment your encoder registers, it is
   automatically picked up and checked by
   `tests/encoders/test_contract.py`, parametrized over every name in
   `evodevo.encoders.list_encoders()`. Run just your encoder's checks with:

   ```
   pytest tests/encoders/test_contract.py -k my_encoding
   ```

   The suite checks (see `tests/encoders/_contract_checks.py` for exact
   assertions): `decode()` shape/dtype/value range; `decode()` determinism;
   `random_genome()` reproducibility and seed-sensitivity; `mutate()`
   reproducibility; that mutated genomes survive a serialize/deserialize
   round trip; that `serialize_genome`/`deserialize_genome` round-trip
   exactly; that `random_genome()` produces at least one invalid body over
   500 samples (i.e. you are not secretly repairing); that `describe()` is
   JSON-serialisable and covers every constructor parameter; and that your
   encoder's module doesn't import a physics/RL package.

   If a check fails, its message is prefixed `[contract:<clause>]`, naming
   exactly which part of the contract was violated.

6. **Add encoder-specific tests** in `tests/encoders/test_my_encoding.py`
   for anything unique to your representation (e.g. NCA's synchronous
   update rule, or a rule-based encoder's interpreter semantics) — don't
   duplicate what the contract suite already covers generically.

## Why this split exists

- `tests/encoders/test_contract.py` + `_contract_checks.py` catch
  structural violations of the shared contract for every encoder, present
  and future, with zero per-encoder test-writing effort.
- `tests/encoders/test_direct.py` / `test_nca.py` (and your
  `test_my_encoding.py`) hold only what's actually specific to that one
  representation.
- `tests/encoders/test_contract_self_check.py` proves the enforcement
  mechanism itself works, by running the same checks against deliberately
  broken dummy encoders and asserting they fail with the right clause
  named — useful reading if you want to see exactly what a violation looks
  like before you hit one yourself.
