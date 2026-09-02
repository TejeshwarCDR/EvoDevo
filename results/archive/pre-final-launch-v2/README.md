# Pre-final-launch archive v2: results/runs/ contents cleared before the primary run

This directory holds everything that was present in `results/runs/`
immediately before it was cleared to launch the final, locked 11-hour
direct/nca/rule_based primary comparison: an incomplete `direct`/Walker-v0
run (`direct/Walker-v0/seed_0/run_log.jsonl` only -- no `run_meta.json` or
`best_body.npz`, i.e. the run was stopped before completing), plus
`primary_run.log`, `.heartbeat.json`, `_resolved_configs/`, and
`benchmark_manifest.json` from that same stopped attempt.

Why it is archived rather than merged or deleted:

- This is the data from the run that was stopped for the live-display
  investigation, and left in place through the subsequent rule_based
  modularization work (freezing the port-equivalence fixture, removing
  `external/rule-based-reference/`, restructuring `encoders/rule_based/`
  into parity with `encoders/nca/`).
- It **predates** the v2 config lock (tag `pre-primary-run-lock-v2`, hash
  `56b3ed67ca5e74e6303e3917921be7f347c6957b44880092afe711c2377b53c8`,
  computed by `evodevo/experiments/lock_hash.py`) and predates the final
  pre-launch smoke test (`docs/FINAL_SMOKE_TEST.md`). Mixing it into the
  final comparison would let the `direct` arm's data originate from a run
  that was not gated by the same pre-launch checks (v2 lock, smoke test)
  the other arms' eventual data will be.
- It is archived **for provenance only** -- it was not found invalid,
  corrupted, or wrong in any way; it is simply out of scope for the final
  locked comparison because it predates the checks that comparison depends
  on, and because it is incomplete (stopped mid-run, never reached
  `run_meta.json`/`DONE_MARKER`).

This data is preserved for reference only and must not be merged with
`results/runs/` or `results/BENCHMARK_REPORT.md` for the final three-way
comparison. Nothing here was deleted.
