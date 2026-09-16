# rule_based mutation-displacement: does D* move under mode_of_window?

Fitness-blind (only decode()/mutate()/random_genome() are used -- NO PPO, NO task, NO reward, NO EA). N=2000, dev_steps=20 (mutation_calibration.py's own frozen T_PRIMARY, unchanged), W=6, mutation_rate held at rule_based's already-committed solved value (0.1707) so this ISOLATES the readout-mode effect on displacement, not a re-solve of the rate itself.

| readout | mean | median | std | iqr |
| --- | --- | --- | --- | --- |
| fixed_t | 5.3100 | 0.0000 | 7.5218 | 9.0000 |
| mode_of_window (W=6) | 5.3280 | 0.0000 | 7.5410 | 9.0000 |

**mean displacement delta = 0.0180 cells** (within the calibration's own 0.5-cell tolerance).

**Verdict: no full re-calibration needed.** The measured movement is within the calibration's own tolerance band -- D*=4.9915 and rule_based's solved `mutation_rate=0.1707` remain valid under mode_of_window without re-solving; the 11.55% cycling population's displacement contribution is not large enough to shift the mean meaningfully at N=2000.

