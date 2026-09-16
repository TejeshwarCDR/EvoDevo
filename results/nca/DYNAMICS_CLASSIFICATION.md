# NCA dynamics classification: widened period search + perturbation-sensitivity probe

**Non-gating scientific context** for the paper's limitations/discussion section. This does NOT gate results/nca/READOUT_POLICY_SELECTION.md's Parts 2-4 (the T-shift operational sensitivity test), which stand on their own regardless of the classification below.

Fitness-blind (only decode()/develop()/NCAEncoder._step()/random_genome() are used -- NO PPO, NO task, NO reward, NO EA). N=2000 genomes, the SAME seeded sample as horizon_extension.py (seed=0), extended to T_max=2000. 931/2000 (0.4655) genomes are never-settled at T_max=2000 (HORIZON_EXTENSION.md's residual population).

**Cost**: period-search pass 54.81s; sensitivity-probe pass 4.84s (total 59.65s) -- negligible relative to any PPO training cost.

## Part (a): widened period search (window=100, was 10)

| period | count | fraction of never-settled-at-T_max genomes |
| --- | --- | --- |
| p=35 | 1 | 0.0011 |
| p=38 | 1 | 0.0011 |
| p=40 | 1 | 0.0011 |
| p=44 | 2 | 0.0021 |
| p=45 | 1 | 0.0011 |
| p=57 | 1 | 0.0011 |
| p=70 | 1 | 0.0011 |
| p=71 | 1 | 0.0011 |
| p=74 | 1 | 0.0011 |
| p=77 | 1 | 0.0011 |
| p=85 | 1 | 0.0011 |
| p=86 | 1 | 0.0011 |
| p=87 | 1 | 0.0011 |
| p=89 | 1 | 0.0011 |
| p=92 | 2 | 0.0021 |
| p=93 | 1 | 0.0011 |
| p=94 | 1 | 0.0011 |
| p=96 | 3 | 0.0032 |
| p=98 | 1 | 0.0011 |
| p=100 | 10 | 0.0107 |
| no period <=100 detected | 898 | 0.9646 |

**Widening the window DOES reveal periods the window=10 search missed**: 33 genomes (0.0354 of never-settled, 0.0165 of the full N=2000 population) have a detected period in (10, 100] -- the earlier 'no cycle <=10' conclusion was correct as stated (no period <=10 exists) but understated the cyclic fraction of this population once longer periods are considered.

## Part (b): perturbation-sensitivity probe on genomes with no period <=100

898 genomes show no detected period <=100 and were probed: genome perturbed by eps=1e-06 in a random z-space direction, both original and perturbed genomes developed to T=200 from the same initial condition, ||s_t - s_t'|| measured at each step.

**Methodology correction**: the ORIGINAL fit (least-squares slope of log(distance) vs t over a FIXED first 50 steps) reported 898/898 genomes as chaotic-like with slopes clustered in a suspiciously narrow band (1.6032-1.7439) despite widely varying per-genome K(theta) -- a near-universal slope across ~900 structurally different genomes was the signal that the measurement was dominated by shared constants, not genome-specific dynamics. **Corrected method**: for each genome, t* = the LAST step such that ||s_t - s_t'|| <= 0.1 * state_bound (still well inside the pre-saturation regime); the slope is fit ONLY over [t_start, t*], where t_start additionally excludes a leading run of EXACT-ZERO distances (see ZERO_DISTANCE_FLOOR in the source: both trajectories start from the IDENTICAL initial state -- only the genome's WEIGHTS differ -- so it takes a few steps for the perturbation to propagate into a numerically nonzero state difference at all; fitting through log(0 + tiny) at t=0 was a SECOND, previously-unnoticed artifact, inflating every genome's slope by a similar amount regardless of its own dynamics, on top of the saturation confound). Genomes with fewer than 5 usable points in [t_start, t*] are reported as UNMEASURABLE rather than force-fit through too few points.

**The original 'saturates within ~8-9 steps' premise (implied by the uncorrected ~1.65/step slope) turns out to be WRONG once measured directly, not just imprecise**: under the corrected fit, t* (the pre-saturation window length) has median 200 -- i.e. essentially the ENTIRE 200-step probe horizon, for almost every probed genome. The real perturbation growth is far slower than the uncorrected slope implied; most probed genomes never approach state-bound saturation within this probe's horizon at all. This is reported directly rather than silently reconciled with the original hypothesis.

### Old (window=50, uncorrected) vs new (per-genome pre-saturation) slope distributions

| | n | min | q25 | median | q75 | max | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLD (fixed 50-step window) | 898 | 1.6032 | 1.6414 | 1.6554 | 1.6683 | 1.7439 | 0.0206 |
| NEW (pre-saturation, measurable only) | 898 | -0.0045 | 0.0151 | 0.0247 | 0.0361 | 0.0865 | 0.0150 |

**Spread**: OLD std=0.0206 vs NEW std=0.0150 -- the corrected fit does NOT show materially more spread than the original -- the narrow OLD band was not simply a saturation artifact masking wide genuine variation.

t* (pre-saturation window length) stats: min=159, median=200.0, max=200. **0/898 genomes (0.0000) are UNMEASURABLE** (t* < 5, saturating almost immediately) -- a small fraction, not materially affecting the classification below.

### Old vs new classification

| | chaotic-like | power-law-creep-like | unmeasurable |
| --- | --- | --- | --- |
| OLD (uncorrected) | 898 (1.0000) | 0 (0.0000) | 0 (0.0000) |
| NEW (corrected) | 891 (0.9922) | 7 (0.0078) | 0 (0.0000) |

### K(theta)-vs-slope correlation (the decisive check)

Pearson r between each measurable genome's corrected slope and its own K(theta) (`evodevo.encoders.nca.certificate.contraction_constant`): **r = 0.1248**.

**|r|=0.1248 <= 0.3 -- weak or no correlation with K(theta)**. Removing the saturation confound does NOT reveal a clean genome-dependent chaos signal tied to the certificate's own local-expansion quantity; the original '44.9% chaotic' framing should NOT be read as 'these genomes have high K(theta)-driven local expansion' -- whatever is driving the measured slopes is not well explained by K(theta) alone (possible causes include nonlinearity beyond the local linearisation K(theta) measures, or the decay/state_bound constants still dominating even within the pre-saturation window at this eps).

## Summary for the paper's limitations/discussion section

Of the full N=2000 PRIMARY population, at T_max=2000: **0.0165 show a period in (10, 100]**. Of the population probed for sensitivity (never-settled, no period<=100), under the CORRECTED (pre-saturation) fit: **0.9922 of probed genomes (0.4455 of the full population) are chaotic-like**, **0.0078 of probed genomes (0.0035 of the full population) are power-law-creep-like**, and **0.0000 of probed genomes (0.0000 of the full population) are UNMEASURABLE** at this eps. The corrected K(theta)-vs-slope correlation is r=0.1248. **This supersedes the original '44.9% chaotic' figure computed under the uncorrected fixed-window fit, which was confounded by shared post-saturation plateau and should no longer be cited as the characterisation of this population's dynamics.**

