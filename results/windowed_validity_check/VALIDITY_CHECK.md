# windowed readout validity check: closing the not-re-measured gap

N=2000, seed=0, T_primary=100 (the corrected shared T_primary, results/T_PRIMARY_UNIFIED.md). Reuses evodevo.analysis.structural.validity_rate/diversity directly against fixed_t- and mode_of_window-configured encoders at IDENTICAL (T_primary, other settings) -- isolating the readout-mode effect.

| arm | readout | valid_rate | unique_valid_body_fraction | material_entropy_bits |
| --- | --- | --- | --- | --- |
| nca | fixed_t | 0.6790 | 0.6755 | 2.3209 |
| nca | mode_of_window (W=50) | 0.6705 | 0.6660 | 2.3193 |
| rule_based | fixed_t | 0.5470 | 0.3315 | 2.3043 |
| rule_based | mode_of_window (W=6) | 0.5750 | 0.3470 | 2.3095 |

