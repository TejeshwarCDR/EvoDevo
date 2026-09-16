# Unified T_primary: replacing the censored-percentile derivation with T-shift stability

**The old derivation was ILL-POSED, not merely biased, for BOTH arms.** q95(settle_step) requires >=95% of genomes to have a known, finite settle_step; rule_based's own never_settled_fraction at T_max=100 is 0.1155 and NCA's is 0.40-0.64 depending on the architecture point -- both far above the 5% censoring threshold. This retroactively explains results/nca/HORIZON_EXTENSION.md's finding that q95 'tracks whatever ceiling is tested': the expected behaviour of a censored percentile past 5% censoring, not a heavy tail worth chasing with ever-larger T_max.

**OLD (superseded) T_primary = 39**, computed via `depth_study.py::_select_t_primary` (retained in code, now explicitly labelled SUPERSEDED).

## New criterion: per-arm T-shift-stability sweep

Per arm, using its own resolved mode_of_window readout and W, sweep candidate T and measure the same fraction_changed-under-T-shift metric already used in both arms' own READOUT_POLICY_SELECTION.md reports; select the smallest T at which fraction_changed falls below a data-justified threshold.

| arm | resolved W | selected T | threshold used | recommended <2% reachable? |
| --- | --- | --- | --- | --- |
| nca | 50 | 100 | 0.1735 | False |
| rule_based | 6 | 40 | 0.0055 | False (not applicable -- see below) |

rule_based's own selection additionally corrected the originating task's odd-W assumption: direct measurement showed EVEN W gives perfect T-shift stability for a pure period-2 cycle (its dominant detected period, results/rule_based/SETTLING_AUDIT.md) while ODD W gives the worst possible instability -- see results/rule_based/READOUT_POLICY_SELECTION.md's own 'Odd-vs-even W tie behaviour' section for the full derivation.

## Final shared T_primary

`T_primary = max(nca_T=100, rb_T=40) = **100**` -- the same asymmetric-loss logic as the original P4 argument (extra steps are cheap for an arm whose readout is already stable; insufficient steps are not), now maximising a well-posed stability-threshold crossing rather than a censored percentile.

**IMPORTANT: rule_based's own selected T (40) is SMALLER than the shared T_primary (100)** -- since the shared benchmark actually runs rule_based at `dev_steps=100`, not at its own smaller optimum, its T-shift-stability figure AT the value it is actually deployed at is the one that matters for the real comparison, not the smaller-T optimum found during the per-arm sweep. Reported explicitly, not conflated:
rule_based's own optimum (T=40, W=6): 0.0055. rule_based's ACTUAL deployed figure at the shared T_primary (T=100, W=6): **0.0070** (shift=-5: 0.0070, shift=+5: 0.0035) -- slightly worse than its own optimum but still ~16x more stable than rule_based's own fixed_t readout (0.1150-0.1155) at a comparable T, and far more stable than NCA's own figure at the shared T_primary (0.1735).

## Equivalence confirmation at the FINAL shared T_primary

NCA, at T_primary=100, W=50: n_settled=52, n_mismatch=0 -- EXACT match, as expected.

rule_based, at T_primary=100, W=6: n_settled=1769, n_mismatch=0 -- EXACT match, as expected (the 88.45% clean-settling population).

