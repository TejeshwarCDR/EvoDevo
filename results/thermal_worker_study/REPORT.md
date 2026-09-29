# EvoDevo thermal worker study — 2026-09-29

## Decision

Use **10 E-core workers on logical CPUs 16–25**, with the scheduler and
resource-tracker restricted to E-cores 26–27. Leave every P-core idle.

This is the highest-throughput configuration tested with credible thermal
headroom for a long unattended run. It does not make the campaign fast: the
accepted discrete-event model projects 1,228.6 h mean (1,220.6–1,237.2 h
p5–p95), or 51.2 days. The reduced-cost benchmark ratio against the historical
20-worker sweep gives a less conservative 969 h (40.4 days). Plan for
**approximately 40–52 days**, with 51.2 days as the capacity model's estimate.

No tested configuration both preserved the earlier ~27-day runtime and stayed
thermally safe. Restoring that runtime requires fixing the machine's cooling or
obtaining an administrator-controlled frequency/power cap, then repeating this
study; worker count alone cannot do it.

## Method

- Real EvoGym/PPO work, not surrogate load.
- Identical deterministic 120-genome workload for the main comparisons,
  20,000 PPO timesteps per genome, one thread per worker.
- Explicit single-CPU affinity per worker.
- One-second package-temperature, total-CPU and kernel throttle-counter samples.
- Fresh cooldown to <=55 C before each profile.
- Automatic process-group termination at >=96 C.
- E10 was repeated with 240 genomes for a longer 748 s confirmation.
- No scientific campaign was launched.

Raw data:

- `affinity_benchmark.json`
- `affinity_benchmark_low_p.json`
- `affinity_benchmark_e_knee.json`
- `affinity_benchmark_e10_long.json`
- aborted standard-P8 traces under `20260929T112638Z_82110a1f/telemetry.jsonl`

## Comparison

| Profile | Worker CPUs | Result | Throughput (genomes/min) | Worker util. | Temperature | Throttle counter | Projected total |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard 20 | 8 P + 12 E | prior real campaign stopped | historical 46.27 | — | **98–100 C sustained** | thermal throttle | 403.2 h / 16.8 d |
| Standard P8 | P primaries 0,2,…,14 | **aborted** | historical 30.42 | — | **99 C** | incremented | 662.7 h / 27.6 d |
| E8 | E 16–23 | completed | 13.24 | 95.1% | 75 C peak; ~73 C sustained | no increase | 1,535.5 h / 64.0 d |
| **E10** | **E 16–25** | **completed twice** | 16.09 (120); 19.25 (240) | 91.3%; 95.5% | **88 C peak; 79–82 C sustained for 11 min** | **no increase** | **1,228.6 h / 51.2 d** |
| E12 | E 16–27 | completed | 18.03 | 87.2% | 91 C peak; 81–86 C loaded minute means | no increase | 1,024.0 h / 42.7 d |
| P1+E12 | P0 + E16–27 | completed | 21.33 | 85.9% | **95 C peak** | no increase | 858.5 h / 35.8 d |
| P2+E12 | P0,2 + all E | **aborted** | — | — | **96 C** | no increase before abort | — |
| P3+E12 | P0,2,4 + all E | **aborted** | — | — | **99 C** | incremented | — |
| P4+E8 | P0,2,4,6 + E16–23 | **aborted** | — | — | **100 C** | incremented | — |
| P4+E12 | P0,2,4,6 + all E | **aborted** | — | — | **97 C** | no increase before abort | — |
| P6+E6 | six P + six E | **aborted** | — | — | **99 C** | incremented | — |
| P6+E10 | six P + ten E | **aborted** | — | — | **96 C** | incremented | — |

## Why E10, not E12

E12 was 12% faster than E10 on the identical 120-genome workload, but had a
5 C higher peak, a rising first four-minute temperature profile, and consumed
every E-core. A real E12 campaign would have to place orchestration on a P-core,
which reintroduces the observed turbo-temperature spike. E10 leaves 12 C to
Tjmax at its measured peak and keeps the complete campaign process tree off the
P-cores. Its longer confirmation showed no sustained rise and no throttle event.

The evidence is still a controlled short benchmark, not a guarantee over 40–52
days. Runtime monitoring and thermal alerts remain required. A later cooling,
ambient-temperature, BIOS, or power-policy change invalidates this result and
requires re-running the benchmark.
