# rule_based exact context analysis: reachability, coverage, redundancy

Fitness-blind (only decode()/develop_rules()/random_genome() are used -- NO PPO, NO task, NO reward, NO EA). N=2000 random genomes (position_predicate="disabled", the encoder's own default), seed=0, dynamic horizon T=100 (matches the shared resolved T_primary, results/T_PRIMARY_UNIFIED.md).

**Exhaustive enumeration verified complete**: `enumerate_contexts()` produces exactly 6480 unique contexts (5 * 6^4, OFF_GRID distinguished per docs/rule-language-spec.md section D), confirmed programmatically in `tests/encoders/test_context_analysis.py::test_enumerate_contexts_is_exhaustive_and_unique`.

**EVOLVED GENOMES note**: no evolved-genome archives exist anywhere in this project's committed run artifacts to analyze -- `results/runs/**/run_log.jsonl` and `generation_log.jsonl` record only fitness/validity metadata and decoded body hashes, never the genome (rule list) itself, and `best_body.npz` stores only the decoded phenotype array. This is disclosed honestly rather than fabricating evolved genomes: this report covers RANDOM genomes only.

## Static vs dynamic dead-rule counts

STATIC dead = provably unreachable (some earlier rule's condition strictly dominates it over ALL of C). DYNAMIC dead = never actually fired across the full T-step development from the project's own center-seed initial condition, for this specific genome -- a WEAKER, sample-dependent notion; every statically dead rule is also dynamically dead, but not vice versa (see the worked examples' own `dynamic_dead` column for genomes where this diverges).

### static dead-rule count per genome distribution

| value | count | fraction |
| --- | --- | --- |
| 0 | 592 | 0.2960 |
| 1 | 459 | 0.2295 |
| 2 | 338 | 0.1690 |
| 3 | 223 | 0.1115 |
| 4 | 168 | 0.0840 |
| 5 | 151 | 0.0755 |
| 6 | 69 | 0.0345 |

mean=1.8225, median=1.0000, max=6

### dynamic dead-rule count per genome distribution

| value | count | fraction |
| --- | --- | --- |
| 0 | 291 | 0.1455 |
| 1 | 480 | 0.2400 |
| 2 | 437 | 0.2185 |
| 3 | 335 | 0.1675 |
| 4 | 216 | 0.1080 |
| 5 | 160 | 0.0800 |
| 6 | 81 | 0.0405 |

mean=2.2545, median=2.0000, max=6

**690/2000 genomes (0.3450) have MORE dynamically-dead rules than statically-dead rules** -- i.e. at least one rule that is provably reachable in principle but never actually fires during this genome's own real development from the center-seed condition within T=100 steps. This is the concrete gap between "could this rule ever matter" and "did development actually encounter the input this rule needs."

### effective_rule_count (static) per genome distribution

| value | count | fraction |
| --- | --- | --- |
| 1 | 299 | 0.1495 |
| 2 | 310 | 0.1550 |
| 3 | 538 | 0.2690 |
| 4 | 492 | 0.2460 |
| 5 | 263 | 0.1315 |
| 6 | 83 | 0.0415 |
| 7 | 15 | 0.0075 |

mean=3.2095, median=3.0000, max=7

## Coverage-share distribution (does one broad early rule dominate?)

For each genome, the LARGEST single rule's static coverage as a fraction of the full context space: mean=0.6035, median=0.6420, max=1.0000, min=0.1667.

**0.6280 of genomes have a single rule covering MORE THAN HALF of the entire context space** -- confirming a strong 'one broad early rule dominates' pattern (typically an early ALWAYS or wide-threshold NEIGHBOR_COUNT rule absorbs the large majority of possible inputs, leaving later rules to handle only a narrow residual).

## Action-redundancy rates, per action-type pair

For every pair of rules in every genome, the fraction of the full context space where their two actions would produce the SAME resulting value for the acting cell (see `context_analysis.py::_action_effect_on_self`'s own docstring for the exact semantics used, including the GROW_DIRECTIONAL/COPY_NEIGHBOR-off-grid conventions), aggregated by the pair's (action_type, action_type) label.

**CAVEAT on `grow_directional`/`grow_directional` pairs**: GROW_DIRECTIONAL never writes the ACTING cell at all (it writes a NEIGHBOR) -- so under this metric ("effect on the acting cell"), ANY two GROW_DIRECTIONAL rules trivially show 100% agreement (both leave the actor unchanged), REGARDLESS of whether they grow in different directions or different materials. This is a limitation of comparing target-affecting actions via an actor-centric metric, not a genuine behavioural-redundancy finding -- reported here explicitly rather than left to be misread as "all growth-growth rule pairs are interchangeable", which this metric cannot actually establish.

| action-type pair | n pairs observed | mean redundancy fraction | max |
| --- | --- | --- | --- |
| grow_directional / grow_directional | 7501 | 1.0000 | 1.0000 |
| copy_neighbor / grow_directional | 3109 | 0.3333 | 0.3333 |
| grow_directional / set_empty | 1741 | 0.2000 | 0.2000 |
| grow_directional / set_soft | 1679 | 0.2000 | 0.2000 |
| grow_directional / set_hard | 1636 | 0.2000 | 0.2000 |
| grow_directional / set_actuator_vertical | 1633 | 0.2000 | 0.2000 |
| grow_directional / set_actuator_horizontal | 1630 | 0.2000 | 0.2000 |
| copy_neighbor / set_empty | 295 | 0.2000 | 0.2000 |
| copy_neighbor / set_actuator_vertical | 274 | 0.2000 | 0.2000 |
| copy_neighbor / set_actuator_horizontal | 262 | 0.2000 | 0.2000 |
| copy_neighbor / set_soft | 255 | 0.2000 | 0.2000 |
| copy_neighbor / copy_neighbor | 251 | 0.3958 | 1.0000 |
| copy_neighbor / set_hard | 233 | 0.2000 | 0.2000 |
| set_actuator_vertical / set_empty | 174 | 0.0000 | 0.0000 |
| set_actuator_vertical / set_hard | 149 | 0.0000 | 0.0000 |
| set_actuator_horizontal / set_empty | 148 | 0.0000 | 0.0000 |
| set_actuator_horizontal / set_actuator_vertical | 141 | 0.0000 | 0.0000 |
| set_actuator_horizontal / set_soft | 139 | 0.0000 | 0.0000 |
| set_actuator_vertical / set_soft | 132 | 0.0000 | 0.0000 |
| set_empty / set_hard | 129 | 0.0000 | 0.0000 |
| set_hard / set_soft | 128 | 0.0000 | 0.0000 |
| set_actuator_horizontal / set_hard | 127 | 0.0000 | 0.0000 |
| set_empty / set_soft | 120 | 0.0000 | 0.0000 |
| set_empty / set_empty | 75 | 1.0000 | 1.0000 |
| set_soft / set_soft | 60 | 1.0000 | 1.0000 |
| set_actuator_horizontal / set_actuator_horizontal | 59 | 1.0000 | 1.0000 |
| set_hard / set_hard | 58 | 1.0000 | 1.0000 |
| set_actuator_vertical / set_actuator_vertical | 56 | 1.0000 | 1.0000 |

## Worked examples: genomes with a statically dead rule

### Genome index 3

Rule list (in evaluation order):

0. DevelopmentalRule(cell_type{'state': <VoxelState.SOFT: 'soft'>} -> grow_directional{'direction': <Direction.LEFT: 'left'>, 'voxel_type': <VoxelState.SOFT: 'soft'>})
1. DevelopmentalRule(empty_neighbor{'direction': <Direction.DOWN: 'down'>} -> copy_neighbor{'direction': <Direction.DOWN: 'down'>})
2. DevelopmentalRule(empty_neighbor{'direction': <Direction.LEFT: 'left'>} -> set_hard)
3. DevelopmentalRule(cell_type{'state': <VoxelState.EMPTY: 'empty'>} -> set_empty)
4. DevelopmentalRule(empty_neighbor{'direction': <Direction.DOWN: 'down'>} -> grow_directional{'direction': <Direction.DOWN: 'down'>, 'voxel_type': <VoxelState.ACTUATOR_VERTICAL: 'actuator_vertical'>}) **<- DEAD (static coverage 0)**
5. DevelopmentalRule(empty_neighbor{'direction': <Direction.LEFT: 'left'>} -> set_hard)
6. DevelopmentalRule(cell_type{'state': <VoxelState.HARD: 'hard'>} -> copy_neighbor{'direction': <Direction.UP: 'up'>})

Static coverage counts (out of 6480): [1296, 864, 720, 900, 0, 0, 900]

Rule 4 would match 1080 contexts on its own condition alone, but rule 0 (earlier in evaluation order) already matches every one of them -- rule 4 is PROVABLY unreachable, not merely unobserved: `DevelopmentalRule(cell_type{'state': <VoxelState.SOFT: 'soft'>} -> grow_directional{'direction': <Direction.LEFT: 'left'>, 'voxel_type': <VoxelState.SOFT: 'soft'>})`.

Dynamically-dead rules for this genome (T=100): [4, 5]

### Genome index 5

Rule list (in evaluation order):

0. DevelopmentalRule(neighbor_count{'threshold': 4} -> set_actuator_vertical)
1. DevelopmentalRule(neighbor_count{'threshold': 4} -> grow_directional{'direction': <Direction.UP: 'up'>, 'voxel_type': <VoxelState.HARD: 'hard'>}) **<- DEAD (static coverage 0)**
2. DevelopmentalRule(cell_type{'state': <VoxelState.SOFT: 'soft'>} -> grow_directional{'direction': <Direction.UP: 'up'>, 'voxel_type': <VoxelState.ACTUATOR_VERTICAL: 'actuator_vertical'>})
3. DevelopmentalRule(empty_neighbor{'direction': <Direction.LEFT: 'left'>} -> grow_directional{'direction': <Direction.LEFT: 'left'>, 'voxel_type': <VoxelState.ACTUATOR_VERTICAL: 'actuator_vertical'>})
4. DevelopmentalRule(neighbor_count{'threshold': 1} -> set_soft)
5. DevelopmentalRule(empty_neighbor{'direction': <Direction.UP: 'up'>} -> grow_directional{'direction': <Direction.RIGHT: 'right'>, 'voxel_type': <VoxelState.HARD: 'hard'>})

Static coverage counts (out of 6480): [1280, 0, 1040, 864, 3264, 16]

Rule 1 would match 1280 contexts on its own condition alone, but rule 0 (earlier in evaluation order) already matches every one of them -- rule 1 is PROVABLY unreachable, not merely unobserved: `DevelopmentalRule(neighbor_count{'threshold': 4} -> set_actuator_vertical)`.

Dynamically-dead rules for this genome (T=100): [1]

### Genome index 6

Rule list (in evaluation order):

0. DevelopmentalRule(empty_neighbor{'direction': <Direction.DOWN: 'down'>} -> grow_directional{'direction': <Direction.DOWN: 'down'>, 'voxel_type': <VoxelState.HARD: 'hard'>})
1. DevelopmentalRule(neighbor_count{'threshold': 4} -> copy_neighbor{'direction': <Direction.LEFT: 'left'>})
2. DevelopmentalRule(always -> grow_directional{'direction': <Direction.UP: 'up'>, 'voxel_type': <VoxelState.HARD: 'hard'>})
3. DevelopmentalRule(cell_type{'state': <VoxelState.SOFT: 'soft'>} -> grow_directional{'direction': <Direction.LEFT: 'left'>, 'voxel_type': <VoxelState.ACTUATOR_VERTICAL: 'actuator_vertical'>}) **<- DEAD (static coverage 0)**
4. DevelopmentalRule(cell_type{'state': <VoxelState.ACTUATOR_HORIZONTAL: 'actuator_horizontal'>} -> copy_neighbor{'direction': <Direction.RIGHT: 'right'>})

Static coverage counts (out of 6480): [1080, 1280, 4120, 0, 0]

Rule 3 would match 1296 contexts on its own condition alone, but rule 0 (earlier in evaluation order) already matches every one of them -- rule 3 is PROVABLY unreachable, not merely unobserved: `DevelopmentalRule(empty_neighbor{'direction': <Direction.DOWN: 'down'>} -> grow_directional{'direction': <Direction.DOWN: 'down'>, 'voxel_type': <VoxelState.HARD: 'hard'>})`.

Dynamically-dead rules for this genome (T=100): [3, 4]

