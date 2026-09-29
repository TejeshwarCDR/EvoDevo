# Revised CPU thermal/runtime optimization — 2026-09-29

> **Superseded on 2026-09-30.** Administrator-authorized guarded real-PPO
> measurements found a qualifying configuration: `p8e10` (P CPUs
> 0,2,4,6,8,10,12,14 and E CPUs 16–25) with P policies capped at 4.2 GHz,
> RAPL PL1=125 W and PL2=160 W, turbo otherwise enabled.  Its 960-genome
> test ran 22.3 minutes at 42.96 genomes/min, reached 84 C maximum (77 C
> p95), had no throttle-counter increment or worker failures, verified
> affinity, and projects to 18.1 days.  The complete current matrix and its
> strict 88 C selection gate are in `FINAL_SELECTION_20260930.md`; do not use
> the earlier "no safe final launch" conclusion below.

## Outcome

**No safe final launch configuration exists with the controls currently
available to the unprivileged campaign account.**  The campaign must not be
started with the present 10-E-core default: its measured projection is about
51.2 days, outside the 15–20 day requirement.  Conversely, every tested
mixed-P/E arrangement that could plausibly approach the required throughput
reached the thermal safety limit or thermal-throttled.

The fastest prior configuration, 8 P-core primaries plus 12 E-cores, was the
stopped campaign that sustained 98–100 C and throttled.  Its historical
46.27 genomes/min corresponds to 16.8 days.  Thus the requested 15-day end
requires a further 12% throughput improvement over an already unsafe run;
20 days requires at least 38.87 genomes/min.

## Method and safeguards

- Real EvoGym/PPO evaluation, the same deterministic workload in every
  comparison (20,000 PPO timesteps/genome, one thread per worker), never the
  actual campaign.
- One explicitly pinned CPU per spawned worker; parent/scheduler affinity is
  kept separate from worker placement.
- One-second package temperature, per-selected-CPU frequency, aggregate CPU
  utilization, kernel thermal-throttle counter, worker utilization, and
  failure telemetry.
- Fresh cooldown to at most 55 C before each benchmark.  The watchdog kills
  the complete spawned process group at the configured package-temperature
  limit, so aborted rows intentionally have no throughput result.
- The viable E10 control was a 240-genome / 748-second (12.5-minute) real-PPO
  run, long enough to observe its loaded temperature plateau.  Unsafe mixed
  profiles were automatically aborted as soon as their heat buildup reached
  the safety boundary; extending those tests would not provide unattended-run
  evidence.

## Measurements

The throughput target is calculated by scaling the accepted 20-worker capacity
model: `46.27 genomes/min × 16.8 days / target days`.

| Configuration / software control | Workers and affinity | Real-PPO result | Throughput (genomes/min) | Thermal result | Throttle / failures | Projected campaign runtime |
|---|---:|---|---:|---|---|---:|
| Historical standard | 8 P (0,2,…,14) + 12 E (16–27), 20 | stopped real campaign | 46.27 | **98–100 C sustained** | thermal-throttled | 16.8 d |
| P8 | P0,2,…,14, 8 | safety abort | historical 30.42 | **99 C** | incremented | 27.6 d |
| E8 | E16–23, 8 | completed | 13.24 | 75 C peak, about 73 C sustained | no increment / 0 | 64.0 d |
| E10 (initial) | E16–25, 10 | completed | 16.09 | 86 C peak | no increment / 0 | 51.2 d model |
| E10 (long confirmation) | E16–25, 10 | 240 genomes, 748 s | 19.25 | 88 C peak; 79–82 C over 11 loaded minutes | no increment / 0 | 40.4 d optimistic scaling; 51.2 d accepted model |
| E12 | E16–27, 12 | completed | 18.03 | 91 C peak; 81–86 C loaded minute means | no increment / 0 | 42.7 d model |
| P1+E12 | P0 + E16–27, 13 | completed | 21.33 | **95 C peak** | no increment / 0 | 35.8 d model |
| P2+E12 | P0,2 + E16–27, 14 | safety abort | — | **96 C** | no increment before abort | rejected |
| P3+E12 | P0,2,4 + E16–27, 15 | safety abort | — | **99 C** | incremented | rejected |
| P4+E8 | P0,2,4,6 + E16–23, 12 | safety abort | — | **100 C** | incremented | rejected |
| P4+E12 | P0,2,4,6 + E16–27, 16 | safety abort | — | **97 C** | no increment before abort | rejected |
| P6+E6 | P0,2,…,10 + E16–21, 12 | safety abort | — | **99 C** | incremented | rejected |
| P6+E10 | P0,2,…,10 + E16–25, 16 | safety abort | — | **96 C** | incremented | rejected |
| P8+E12, CPUQuota=1700% | all 20 cores, cgroup quota | safety abort at 11 s | — | 46→90 C in 1 s; **93 C** at 11 s | **+4** counter / no worker failure recorded | ideal ceiling 19.8 d, but unsafe |
| P8+E12, CPUQuota=1400% | all 20 cores, cgroup quota | safety abort at 9 s | — | 50→82 C in 2 s; **93 C** at 9 s | no increment before abort | ideal ceiling 24.0 d; also unsafe |
| P8+E12, CPUQuota=1000% | all 20 cores, cgroup quota | safety abort at 2 s | — | 50→84 C in 1 s; **90 C** at 2 s | no increment before abort | ideal ceiling 33.6 d; also unsafe |

`CPUQuota` was tested in a transient user systemd scope and therefore did not
alter persistent host policy.  It limits aggregate scheduling time, not
hardware turbo: telemetry shows P-core samples around 5.1–5.5 GHz and E-core
samples around 4.1–4.3 GHz in all three quota tests.  It is not a safe thermal
control for this workload.  The 1700% result is the only quota whose *ideal*
linear upper bound could reach 20 days, and it throttled.

## Control availability

The machine is using active `intel_pstate`, boost enabled (`no_turbo=0`),
`max_perf_pct=100`, and `performance` energy preference on both P and E cores.
Changing `intel_pstate` limits, EPP/governor, RAPL package limits, or MSR
turbo ratios requires administrator authority on this host.  `powerprofilesctl`
offers power-saver, but activation fails because the CPU driver's EPP file is
busy; it did not change the active performance EPP.  No non-privileged
frequency/power control remains after the cgroup-quota tests.

## Required next decision

Obtain administrator access to apply a **reversible hardware power/frequency
cap** (for example a package RAPL PL1/PL2 cap and/or P-core maximum-performance
limit), then repeat the same benchmark matrix.  Start conservatively, retain
at least several degrees below the 96 C safety abort during a 20–30 minute
run, and only accept a profile after its measured throughput projects to
15–20 days.  The exact cap must be benchmarked rather than assumed; a
particular wattage or percentage cannot be inferred safely from this host's
current thermal behavior.

`scripts/campaign/with_cpu_power_cap.sh` is the root-only, fail-safe wrapper
prepared for those tests.  It snapshots each changed `intel_pstate`/RAPL value
and restores them in reverse order on exit, error, or interruption.  Its
`--dry-run` path and its non-root refusal path were validated without changing
the host.  It must wrap only `benchmark_thermal_affinity.py`, never the real
campaign.

The first authorized matrix should use the 18–20-worker profiles that can
actually satisfy the target if the cap is effective: `p6e12`, `p7e12`,
`p8e10`, and `p8e12`.  Each profile is explicitly pinned to P-core primaries
and E-cores, and a 960-genome run is at least about twenty minutes at the
historical full-speed rate (longer under a cap).  For example, an operator can
run the following *benchmark only* command, then repeat at further cap levels
only when the preceding level has adequate thermal margin:

```bash
sudo scripts/campaign/with_cpu_power_cap.sh --no-turbo --pl1-w 125 --pl2-w 160 -- \
  .venv/bin/python scripts/campaign/benchmark_thermal_affinity.py \
  --profiles p6e12,p7e12,p8e10,p8e12 --genomes 960 --timesteps 20000 \
  --sample-interval 1 --abort-temp-c 90 --cooldown-temp-c 55 \
  --output results/thermal_worker_study/powercap_125w.json --control-label no_turbo_pl1_125w_pl2_160w
```

This is a starting measurement, not a recommended final cap or launch
configuration.  Accept no row merely because it completes: it must have no
thermal-counter increase, credible temperature headroom throughout the full
run, zero worker failures, verified affinity, and a measured throughput that
projects to 15–20 days.

After one or more capped result files are produced, run
`scripts/campaign/analyze_thermal_matrix.py <result.json> --output <report.md>`.
It uses the accepted 46.27 genomes/min / 16.8-day baseline to calculate each
projection and selects only the fastest row meeting all of these explicit
gates: at least 20 minutes of real PPO, peak at most 90 C, no thermal-counter
increase, zero worker failures, verified affinity, and a 15–20-day projection.
The analyzer was checked against the existing E10 and CPU-quota evidence; it
correctly selected no configuration under the actual gates.

Raw evidence: `affinity_benchmark*.json`, `p8e12_cpuquota{1000,1400,1700}.json`,
and the benchmark harness `scripts/campaign/benchmark_thermal_affinity.py`.
