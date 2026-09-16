# rule_based architecture before/after: the R2/R3 evidence table

Fitness-blind throughout (decode()/develop_rules()/random_genome()/validate_body()/mutate() plus evodevo.analysis.structural.* only -- NO PPO, NO task, NO reward, NO EA). N=2000, T_primary=100 (the shared, unified T-shift-stability-selected value, results/T_PRIMARY_UNIFIED.md), init_condition=`center_seed` (P3), seed=0, both configurations measured at IDENTICAL settings.

**OLD** = the pre-R2/R3 defaults: `conflict_resolution="row_major_last_wins"` (order-DEPENDENT tie-break -- 12.95% of genomes affected by rule ORDER alone, results/rule_based/ORDER_INDEPENDENCE_AUDIT.md) and `position_predicate="enabled"` (ConditionType.POSITION generated in random genomes, later found to break translation-equivariance -- 0.2140 fraction of genomes violate D(translate(s))==translate(D(s)) on a torus, results/rule_based/LOCALITY_STUDY.md).

**NEW** = the FINAL resolved primary (evodevo/encoders/rule_based/resolved_config.py::PRIMARY): `conflict_resolution="first_rule_index"` (the only order-independent candidate with EQUAL-OR-BETTER validity than the historical baseline, docs/decisions.md's "Make the rule_based interpreter provably order-independent by default" row) and `position_predicate="disabled"` (R3 restored, 0.0000 violations on the same torus test, docs/decisions.md's "Restore translation-equivariance (Proposition R3)..." row). `boundary="no_op"` and `neighbourhood="von_neumann"` are held fixed at the interpreter's entire historical behaviour in BOTH columns -- neither R2 nor R3 changed those axes, so this table isolates exactly the two axes that did change.

## The table

| quantity | OLD (row_major_last_wins, position enabled) | NEW (first_rule_index, position disabled) | direction |
| --- | --- | --- | --- |
| validity rate | 0.5695 | 0.5470 | DOWN |
| degenerate fraction | 0.1870 | 0.2380 | UP |
| unique-valid-body fraction | 0.3875 | 0.3315 | DOWN |
| material entropy (bits) | 2.3032 | 2.3043 | UP |
| genomes per unique phenotype (redundancy) | 1.8484 | 2.1645 | UP |
| largest neutral-class fraction | 0.0620 | 0.0760 | UP |
| never_settled_fraction | 0.1155 | 0.1080 | DOWN |
| mean settle_step | 2.1995 | 2.1754 | DOWN |
| mean mutation displacement (Hamming) | 5.6615 | 5.8280 | UP |
| median mutation displacement (Hamming) | 1.0000 | 0.0000 | DOWN |
| fraction zero displacement | 0.4915 | 0.5230 | UP |

## What changed and why

**Validity**: 0.5695 -> 0.5470. R2's own four-way evidence (results/rule_based/CONFLICT_POLICY.md) already showed `first_rule_index` beats the historical `row_major_last_wins` baseline on validity alone; R3's own cost disclosure (results/rule_based/LOCALITY_STUDY.md) showed disabling `position_predicate` COSTS 0.0250 validity at THOSE settings (N=2000, T=100, conflict_resolution already first_rule_index). The net row above is the two effects composed together, measured fresh rather than summed from the two separate reports, since those reports used different intermediate baselines.
**Diversity** (unique-valid-body fraction): 0.3875 -> 0.3315, the same composition of R2's improvement and R3's cost.
**never_settled_fraction**: 0.1155 -> 0.1080. Per docs/decisions.md's R2 row, conflict resolution changes WHICH body a cycling/growing genome ends up with, not WHETHER it settles, and R3's own cost disclosure states never_settled_fraction is essentially unchanged by disabling POSITION -- so this row is expected to move little, and any large observed shift here would be a genuine, unexpected finding worth flagging.
**Mutation displacement** (mean Hamming): 5.6615 -> 5.8280. The mutation operator itself (`mutate_developmental_rules`) is untouched by R2/R3 -- only DECODE's tie-break and predicate-generation changed -- so any shift here is purely a downstream consequence of which body a given (parent, mutated-child) genome pair now decodes to, not a change in mutation strength itself.

## Notes

This module never computes or consults a fitness, reward, or task-performance quantity: every number above is derived from decode()/develop_rules()/random_genome()/validate_body()/mutate() or evodevo.analysis.structural.* alone.

