# NCA mutation_sigma re-solve under the mode_of_window readout policy (N6 readout cascade)

Fitness-blind (only decode()/mutate()/validate_body are used -- NO PPO, NO task, NO reward, NO EA). Re-solves NCA's `mutation_sigma` ONLY, holding D* FIXED at the value the original joint N6 calibration already committed to (D*=4.9915, `results/mutation/mutation_calibration_raw.json`, UNCHANGED). direct's and rule_based's own solved parameters are NOT re-derived here and remain exactly as committed.

## Before / after (readout-level, NOT the N10 architecture-level table)

| | mutation_sigma | verify_stats.mean | within_tolerance |
| --- | --- | --- | --- |
| fixed-T-argmax-at-T=20 (previous, `readout_mode=fixed_t`) | 0.0201 | 5.1220 | True |
| mode-of-window-at-T=20,W=50 (new, `readout_mode=mode_of_window`) | 0.0140 | 5.7075 | False |

delta mutation_sigma = -0.0061 (-30.39% relative to the old value). direct's mutation_rate and rule_based's mutation_rate are UNCHANGED (not re-run) -- this table is readout-level for nca only, dev_steps held at T_PRIMARY=20 exactly as the original N6 module's own controlled-comparison convention specifies (see mutation_calibration.py's T_PRIMARY docstring).

