# Discarded: progress-reporting cutover

Archived 2026-09-02.

This directory holds the state of `results/runs/` from the primary
three-way comparison (direct/nca/rule_based × Walker-v0 × seeds 0-4,
locked config hash `7e93f59a82ba780e2ea4229a3d60681d7e98c08cfc33e3479a63af56d88c24ce`)
at the moment the live run (PID 58253) was stopped to add live stdout
progress reporting (`evodevo/evolution/progress.py`).

**Contents at archive time:**
- `direct/Walker-v0/seed_0` .. `seed_4`: all five seeds show a `run_meta.json`
  and 300/300 evaluations in `run_log.jsonl` -- i.e. the direct arm had
  actually finished all five seeds under the pre-progress-reporting code path.
- `nca/Walker-v0/seed_0`: partial, 94/300 evaluations, no `run_meta.json`
  (in flight when the process was stopped).
- `_resolved_configs/`, `benchmark_manifest.json`, `primary_run.log`,
  `.heartbeat.json`: supporting artifacts from that invocation.
- `rule_based` had not started.

**Why discarded, not resumed:** this data is not invalid or wrong -- the
direct arm's completed seeds used real PPO evaluation exactly as the locked
config specifies. It is discarded so that the full primary comparison comes
from a single, continuously-logged run under one code path (with live
progress/ETA reporting from evaluation 1 onward) rather than a run stitched
together from a resume. The eval_budget/ppo_timesteps/population_size/
offspring_size/sentinel_fitness/min_voxels/min_actuators/use_crossover/NCA
hyperparameters and the config lock hash are unchanged by this cutover --
only additive stdout observability was added.
