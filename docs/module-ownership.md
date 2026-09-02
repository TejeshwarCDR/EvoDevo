# Module ownership

This project's directory structure exists specifically so that team members
can iterate on their own encoding without running or breaking the others
(see `docs/adding-an-encoder.md` for the contract that makes this possible).
Ownership maps directly onto that structure.

| Directory | Owner | Affects |
|---|---|---|
| `evodevo/encoders/nca/*` | Tejeshwar | Only the NCA arm's results |
| `evodevo/encoders/rule_based/*` | teammate | Only the Rule-Based arm's results |
| `evodevo/encoders/direct/*` | shared / joint-review | Only the Direct arm's results, but treated as shared since it is the reference baseline both other arms are compared against |
| `evodevo/core/*` | shared / joint-review | Every arm |
| `evodevo/morphology/*` | shared / joint-review | Every arm (validation and metrics are applied identically to all three) |
| `evodevo/evaluation/*` | shared / joint-review | Every arm (the one place PPO/evogym may be imported) |
| `evodevo/evolution/*` | shared / joint-review | Every arm (the EA loop and run logging) |
| `evodevo/analysis/*` | shared / joint-review | Every arm's post-hoc comparison and reporting |
| `evodevo/visualization/*` | shared / joint-review | Every arm's rendering (not yet populated -- see the finalization prompt sequence) |
| `evodevo/encoders/base.py`, `evodevo/encoders/registry.py` | shared / joint-review | The plugin contract itself -- changing it can silently change what "conforming" means for every encoder |

## The rule

**Changes to a shared module require agreement from both owners**, because
they alter every arm's results, not just one. A change to `evodevo/core/materials.py`,
`evodevo/morphology/validate.py`, `evodevo/evaluation/ppo_runner.py`, or
`evodevo/evolution/ea.py` changes what every encoding is evaluated against --
even a change that looks purely like a bugfix from inside one arm's
perspective can shift every arm's fitness numbers, so it needs sign-off from
whoever owns the arm you are not currently working on.

**Changes inside your own encoder package affect only that arm.** You are
free to edit `evodevo/encoders/<your-method>/*` and its config
(`evodevo/experiments/configs/<your-method>/*.yaml`, restricted to
encoder-specific keys -- see `evodevo/experiments/configs/shared/base.yaml`)
and method-specific tests (`tests/encoders/test_<your-method>.py`) without
needing anyone else's approval, as long as:

- `make test-<your-method>` passes (contract suite + your method-specific
  tests + the shared-core tests your encoder depends on), and
- you have not touched `base.py`, `registry.py`, `ea.py`, `ppo_runner.py`,
  or anything under `morphology/` to make it work (see
  `docs/adding-an-encoder.md`'s explicit rule about this -- if you needed
  to, that is a signal the contract itself needs to change, which routes
  back to the shared-module rule above).

## Direct as a special case

`encoders/direct/` is technically a single-owner encoder package like the
other two, but it is listed as shared/joint-review here because it is the
project's reference baseline: every fairness/regression claim about NCA or
Rule-Based is stated relative to Direct's numbers, so an undiscussed change
to Direct's behaviour would retroactively change the meaning of every
already-reported NCA/Rule-Based comparison.
