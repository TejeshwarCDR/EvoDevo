> **PARTIALLY SUPERSEDED as of 2026-09-19.** The "`status: not yet run`" claim
> for every fitness column remains true today — no A1/A2/E3 real-PPO fitness
> run has ever been executed, per the same disclosure in
> `docs/decisions.md`'s "Ablation sweep (A1/A2/E3) real-PPO execution scope"
> row. However, Section A2's framing "`hidden_channels=3` is the NCA default"
> (and the `prior_*` structural statistics throughout, all measured at
> `dev_steps<=20` under the pre-redesign unbounded-residual NCA) is stale:
> the current resolved PRIMARY is `hidden_channels=0`
> (`evodevo/encoders/nca/resolved_config.py::PRIMARY`, per
> `docs/decisions.md`'s "Hidden channel count (H) -- final re-resolution"
> row). If/when this sweep is actually run, it should be re-measured against
> the current resolved architecture, not the priors tabulated here.

# Ablation Report

3 seeds per arm, identical PPO/EA budget to the main comparison (`population_size`, `offspring_size`, `eval_budget`, `ppo_timesteps` unchanged across arms within each axis). "Prior" statistics are computed on random genomes only (no PPO, no evolution; N=500) and characterise the encoding's raw morphology-generation behaviour under each hyperparameter setting, independent of fitness.

Any arm marked `not yet run` has no completed EA+PPO runs on disk yet (run `scripts/run_ablations.sh` to produce them); its prior statistics are still reported since they require no PPO compute.

## A1: developmental steps (`dev_steps`)

T=1 is the key control: it isolates whether *iterated* development matters, or whether the network's expressiveness alone (a single perception+update pass) explains any observed effect.

| arm | status | n_seeds | final_best_fitness_mean | final_best_fitness_std | evolved_validity_rate | unique_bodies_explored | prior_validity_rate | prior_degenerate_fraction | prior_never_settled_fraction |
|---|---|---|---|---|---|---|---|---|---|
| dev_steps=1 | not yet run | 0 | nan | nan | nan | 0 | 0.6300 | 0.0960 | 1.0000 |
| dev_steps=5 | not yet run | 0 | nan | nan | nan | 0 | 0.6400 | 0.0340 | 0.9180 |
| dev_steps=10 | not yet run | 0 | nan | nan | nan | 0 | 0.6700 | 0.0140 | 0.6620 |
| dev_steps=20 | not yet run | 0 | nan | nan | nan | 0 | 0.6840 | 0.0120 | 0.4220 |

*(figure omitted: no A1 arm has completed PPO results yet)*

## A2: hidden channels (`hidden_channels`)

`hidden_channels=3` is the NCA default and is identical to A1's `dev_steps=10` cell; its numbers are reused here rather than re-run.

| arm | status | n_seeds | final_best_fitness_mean | final_best_fitness_std | evolved_validity_rate | unique_bodies_explored | prior_validity_rate | prior_degenerate_fraction | prior_never_settled_fraction |
|---|---|---|---|---|---|---|---|---|---|
| hidden_channels=0 | not yet run | 0 | nan | nan | nan | 0 | 0.6940 | 0.0260 | 0.7040 |
| hidden_channels=3 | not yet run | 0 | nan | nan | nan | 0 | 0.6700 | 0.0140 | 0.6620 |

*(figure omitted: no A2 arm has completed PPO results yet)*

## E3: second task (UpStepper-v0)

Tests whether any effect seen on Walker-v0 is task-specific.

| arm | status | n_seeds | final_best_fitness_mean | final_best_fitness_std | evolved_validity_rate | unique_bodies_explored | prior_validity_rate | prior_degenerate_fraction | prior_never_settled_fraction |
|---|---|---|---|---|---|---|---|---|---|
| direct (UpStepper-v0) | not yet run | 0 | nan | nan | nan | 0 |  |  |  |
| nca (UpStepper-v0) | not yet run | 0 | nan | nan | nan | 0 | 0.6700 | 0.0140 | 0.6620 |

*(figure omitted: no E3 arm has completed PPO results yet)*

## What the data shows, what remains a hypothesis, what a larger study needs

- **What the data shows:** whichever `final_best_fitness_mean`/`std` and prior-statistic differences are populated above, from actually-executed arms, at 3 seeds each. With only 3 seeds per cell, treat any ordering between adjacent settings (e.g. `dev_steps=5` vs `dev_steps=10`) as indicative, not conclusive -- across-seed std columns should be read alongside the means, not ignored.
- **What remains a hypothesis:** any causal story connecting a prior-statistic difference (e.g. validity rate or never-settled fraction) to a fitness difference. The prior statistics and the fitness results are both real, but this report does not run a statistical test tying them together; a correlation across ablation cells is suggestive, not proof, at n=4 (A1) or n=2 (A2) points.
- **What a larger study would need:** more seeds per cell (5+, matching the main comparison), a formal significance test (e.g. a seed-level bootstrap or Mann-Whitney U across arms) rather than eyeballing mean/std, and -- if `dev_steps=1` turns out to perform comparably to `dev_steps=10` -- a targeted follow-up isolating network expressiveness from iteration count more directly (e.g. matching parameter count across T while ablating recurrence).
