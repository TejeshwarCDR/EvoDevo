# Incident: apparent all-validity-failure stretch in the direct/Walker-v0/seed_0 run

Diagnosis only, per explicit instruction — no code changed, benchmark not
restarted, `results/runs/` and `results/archive/` untouched by this
investigation.

## Trigger

The live progress stream was observed showing 208/208 evaluations tagged
`validity failure` at `0.00s/eval`, on the `direct`/`Walker-v0`/seed 0 run.
Direct's own previously-measured random-genome validity rate is ~2.25%, so
zero successes in 208 evaluations looked like a structural break (e.g. a
masked exception) rather than encoder-normal invalidity. The run was stopped
before this investigation began.

## Part A — exact failure reason(s)

The persisted log (`results/runs/direct/Walker-v0/seed_0/run_log.jsonl`,
237 records — the run had progressed past the 208 the user observed live,
per `.heartbeat.json`: `evaluation_index: 236`, `evaluations_completed_this_run: 237`)
was inspected directly:

```
valid counts: {False: 229, True: 8}
reason counts:
  229  'not a single connected component under cardinal adjacency'
    8  None   (the 8 valid=True records; `reason` is None on success)
```

**All 229 failing records carry the identical, single reason string** `not
a single connected component under cardinal adjacency`. Cross-referenced
against `evodevo/morphology/validate.py:66`, this is the exact literal
string `validate_body()` appends for a genuine connectivity failure — not a
caught-exception message. It contains no exception type name, no
traceback, no file path, no `NoneType`/`shape mismatch`/`unexpected keyword
argument` text. **No record's `reason` field carries the
`ppo_runner.py` exception-path prefix** (`simulation_instability: ...`,
see Part B) anywhere in the 237-record log.

## Part B — masked-exception hypothesis: RULED OUT

**B.1 — the exact try/except in `evodevo/evaluation/ppo_runner.py` (lines 73–116):**

```python
try:
    ...
    return EvalResult(fitness=fitness, valid=True, reason=None, ...)
except Exception as exc:  # noqa: BLE001 -- intentionally broad
    return EvalResult(
        fitness=sentinel_fitness,
        valid=False,
        reason=f"simulation_instability: {type(exc).__name__}: {exc}",
        ...
    )
```

It catches `Exception` (broad, but not bare `except:`), scoped only around
the PPO train/eval block (after `validate_body()` has already returned
`valid=True` and connectivity has been computed) — and on catch it records
a clearly-prefixed, non-generic reason string (`simulation_instability:
<ExceptionType>: <message>`), not a truncated/opaque one. This prefix
is grep-distinguishable from a genuine validation reason and — per Part
A — **does not appear once** in the 237-record log. The structural-validity
short-circuit (`if not validation.valid: return ...`) happens **before**
any PPO/evogym code runs, at `wall_clock_s` on the order of 1e-4s, which
matches the `0.00s/eval` the user observed live (rounded to 2 decimals by
the progress-line formatter) — that number is expected and correct for a
validation-only rejection, not evidence of a bug.

**B.2 — direct reproduction, bypassing `evaluate_body`'s own try/except:**

A throwaway script (`/private/tmp/.../scratchpad/direct_probe.py`, not
committed, not touching any project file) built a hand-verified valid 5x5
body (`validate_body()` confirmed `valid=True, reasons=[]`), then:

1. Called `evaluate_body(body, task="Walker-v0", ppo_timesteps=5000,
   eval_episodes=1, seed=0)` (current code, current config) — **succeeded**:
   `EvalResult(fitness=-0.00024517764459369706, valid=True, reason=None,
   wall_clock_s=8.875695943832397, ...)`.
2. Called the identical underlying `make_vec_env` / `PPO(...).learn(...)`
   / manual rollout logic a second time with **no try/except at all**
   (bypassing `evaluate_body`'s error handling entirely) — **also
   succeeded**, same fitness, no exception raised, no traceback to report.

Both wall-clock figures (~8.8s) match the previously-measured real-PPO cost
for this config. **No hidden exception exists** — real PPO training and
evaluation work correctly end to end on the current code path.

**Conclusion: masked exception hypothesis ruled out.** The 229 failures are
genuine structural-validity rejections, occurring before PPO is ever
invoked, exactly as `validate_body()` and `ppo_runner.py` are designed to
produce.

## Part C — config lock re-verification: MATCH

```
computed hash: 7e93f59a82ba780e2ea4229a3d60681d7e98c08cfc33e3479a63af56d88c24ce
locked hash:   7e93f59a82ba780e2ea4229a3d60681d7e98c08cfc33e3479a63af56d88c24ce
MATCH
```

Resolved values actually in force for this run, from both a fresh
`resolve_configs()` call and the `_resolved_configs/direct/walker/seed_0.yaml`
actually written to disk for this run:

| field | locked | actual |
|---|---|---|
| grid_shape | (5, 5) | (5, 5) |
| min_voxels | 3 | 3 |
| min_actuators | 1 | 1 |
| eval_budget | 300 | 300 |
| ppo_timesteps | 5000 | 5000 |
| sentinel_fitness | -10.0 | -10.0 |
| use_crossover | False | False |

No config drift of any kind occurred across the restart.

## Statistical assessment (why 208/208-looking is not actually anomalous)

Over the full 237-record log, 8 evaluations were valid (all real-PPO,
7.7–9.2s each) — an observed rate of 3.4%, against the ~2.25% prior. For
n=237, p=0.0225, the binomial standard deviation is ≈2.3 successes; the
observed count is ~1.2 standard deviations above the mean — ordinary
sampling noise, not a deviation requiring explanation. Runs of consecutive
failures are expected at this validity rate: the probability of a
209-evaluation valid-free stretch happening at some point in a run this
length is unremarkable at p≈0.0225 (expected inter-success gap ≈44 evals,
but gap length is geometrically distributed with high variance — several
observed successes are in fact clustered late, e.g. eval_index 219, 224,
231, 233, 236). **The live view the user saw was a real, unlucky-but-normal
failure streak that happened to be showing at the moment of inspection**,
not a frozen or broken state — the log shows the run continuing to produce
genuine real-PPO successes both before (index 135, 153) and after (index
208+) the point of concern.

## Root cause

None found. No fix is proposed because none is needed: the observed
behavior (a long run of connectivity-validation failures, each at ~0.0001s,
interspersed with real-PPO evaluations at ~8–9s) is exactly what the
direct encoder's known ~2–3% validity rate predicts, reproduced directly
against current code with no exception surfacing under any wrapping.

## Addendum: live-display reconciliation

Follow-up check, still diagnosis only: does the progress-reporting code
itself explain a human watching live seeing a continuous wall of "validity
failure" through ~eval 208, given the persisted log shows real=True
successes at eval_index 135 and 153 (both well before 208)?

### 1. Exact tag-selection logic

`evodevo/evolution/progress.py`, `_RunProgress.__call__` (lines 73–78):

```python
if record.get("cache_hit"):
    tag = "cache hit"
elif not record.get("valid", True):
    tag = "validity failure"
else:
    tag = "real-PPO"
```

It reads exactly two fields off the per-evaluation `record` dict: `cache_hit`
and `valid`. Both are used nowhere else in the module.

### 2. Compared against the persisted records for eval_index 135 and 153

Critically, `evodevo/evolution/logging.py`'s `RunLogger.log(record)` writes
`record` to the JSONL file and then invokes `self._on_eval(record)` — **the
same object**, no copy, no field mutation, no re-derivation, in between:

```python
def log(self, record: dict) -> None:
    self._fh.write(json.dumps(record, sort_keys=True) + "\n")
    self._fh.flush()
    self._evaluations_completed += 1
    self._write_heartbeat(record.get("eval_index"))
    if self._on_eval is not None:
        self._on_eval(record)
```

So whatever `valid`/`cache_hit` values end up in the JSONL line are, by
construction, the identical values the tag logic saw. Running the tag
function directly against the two persisted records confirms this:

```
record[135]: valid=True, cache_hit=False, reason=None, wall_clock_s=9.176s
  tag_for(record[135]) -> "real-PPO"

record[153]: valid=True, cache_hit=False, reason=None, wall_clock_s=8.459s
  tag_for(record[153]) -> "real-PPO"
```

**No display-logic bug exists.** The progress line printed for both
evaluations would correctly have read `real-PPO`, not `validity failure`.
This also rules out the weaker version of the same concern (a stale/cached
`tag` variable, an off-by-one in which record is passed, etc.) — there is
only one code path from "value written to JSONL" to "value read for the
tag", and it is the same object reference.

### 3. Buffering / drop / scroll-past check

- `print(line, file=self.tracker.stream, flush=True)` — every progress line
  is explicitly flushed (`progress.py:91`); there is no Python-level stdout
  buffering that could delay or coalesce lines, and no dropped writes are
  possible at that layer.
- Line **rate**, however, is extremely uneven and explains the miss without
  needing a bug: the 10 evaluations immediately before eval_index 135 and
  the 10 immediately after together took **0.00037s and 0.00043s of total
  wall-clock**, respectively (i.e., each burst of ~10 validity-failure lines
  printed in under half a millisecond), sandwiching a single `real-PPO` line
  that itself took ~9.2s (the PPO training/eval call) before it appeared.
  Across eval_index 100–208 as a whole: 109 lines were printed, all but 3
  (`135, 153, 208`) essentially instantaneous, for a total of only 26.3s of
  genuine eval wall-clock spread across that whole span — meaning each
  `real-PPO` line is visually preceded and followed by a burst of dozens of
  failure lines arriving faster than a terminal can render distinct frames,
  let alone faster than a human can read. A viewer glancing at the terminal
  during one of these bursts sees what looks like solid, unbroken
  `validity failure` output; the two `real-PPO` lines are real, correctly
  printed, and immediately flushed, but trivially easy to miss between two
  near-instantaneous floods of dozens of other lines each.

### 4. Definitive conclusion

**(b) — the display logic is correct.** No bug exists in `progress.py` or
its wiring through `logging.py`: the tag is computed from, and only from,
the exact same `record` object that is persisted to `run_log.jsonl`, so the
two can never disagree by construction, and this was directly verified for
both eval_index 135 and 153. The discrepancy is fully explained by a
human-observation gap, with specific supporting evidence (not just
plausibility): the two genuine `real-PPO` lines are each bracketed by
bursts of ~10+ `validity failure` lines printed in well under a
millisecond, against a backdrop where a `real-PPO` line only appears after
an ~8–9s pause — an output pattern in which a human watching a scrolling
terminal is highly likely to perceive "continuous failures" and not
register the rare, correctly-printed success line. No further action is
needed on the progress-reporting code.

## Disposition of the previously-archived direct seeds 0-1 data

`results/archive/discarded-progress-reporting-cutover/direct/` (and the
earlier `results/archive/pre-final-run-direct/`) remain **valid reference
data**. This investigation found no bug in `ppo_runner.py`, `ea.py`,
`validate.py`, or the config resolution path — the exact same code path
those archived seeds ran under is confirmed correct here. Nothing found in
this incident calls that data into question; it was archived for the
data-hygiene/clean-restart reason already documented in each archive
directory's own README, not because it was invalid.
