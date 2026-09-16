# rule_based growth-conflict resolution policy: four-way comparison

Fitness-blind (only decode()/develop_rules()/random_genome()/validate_body() are used -- NO PPO, NO task, NO reward, NO EA). N=2000, seed=0, T=100. Compares the historical order-DEPENDENT `row_major_last_wins` (baseline, results/rule_based/ORDER_INDEPENDENCE_AUDIT.md's own subject) against the three order-INDEPENDENT alternatives.

## Four-way comparison

| policy | valid_rate | unique_valid_body_fraction | uniform_body_fraction | never_settled_fraction | mean_occupied_cells |
| --- | --- | --- | --- | --- | --- |
| row_major_last_wins (baseline) | 0.5695 | 0.3875 | 0.1870 | 0.1155 | 19.12 |
| no_op | 0.5420 | 0.3740 | 0.1845 | 0.1155 | 18.65 |
| lowest_material_id | 0.5615 | 0.3805 | 0.1870 | 0.1155 | 19.13 |
| first_rule_index | 0.5720 | 0.3940 | 0.1875 | 0.1155 | 19.12 |

**Does `no_op` stall growth, as its 'contact inhibition' framing might predict?** mean_occupied_cells: 18.65 vs baseline's 19.12 (ratio 0.975). No meaningful stalling observed at this population/T -- growth conflicts are not frequent enough per-genome for cancellation to visibly suppress total occupied area, though the PER-CONFLICT semantics still differ from the other policies.

## Selection

Selection rule: among the three order-independent candidates, reject any whose mean_occupied_cells collapses to less than half the baseline's (growth-stalling disqualifier), then pick the highest valid_rate among the survivors -- applied to the actual numbers above, not decided in advance.

**Adopted default: `conflict_resolution="first_rule_index"`.**

At the adopted policy: valid_rate=0.5720 (baseline 0.5695, delta 0.0025), unique_valid_body_fraction=0.3940 (baseline 0.3875), mean_occupied_cells=19.12 (baseline 19.12).

## Permutation-test re-confirmation under the adopted default

Re-running results/rule_based/ORDER_INDEPENDENCE_AUDIT.md's own permutation test (N=2000, T=100, same fixed seed=12345 permutation), now under `conflict_resolution="first_rule_index"`:

**Fraction of genomes whose decoded body differs under permutation: 0/2000.**

**0/2000 -- the adopted default is confirmed order-independent, exactly as required.** The restructure (Pass 2b collects growth candidates, groups by target, then resolves per-target as a pure function of the competing-write SET rather than enumeration order) achieves genuine order-independence, not merely a claim.

