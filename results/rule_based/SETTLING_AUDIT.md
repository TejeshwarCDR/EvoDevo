# rule_based settling audit

Fitness-blind (only decode()/develop_trajectory()/random_genome() are used -- NO PPO, NO task, NO reward, NO EA). Does NOT modify RuleBasedEncoder, the rule vocabulary, or the interpreter -- diagnosis only.

## Step 1: implementation-bug check (ruled out first, as required)

N=20 genomes cross-checked: `decode()` at T=39/T=40 (two separately-constructed encoders) against an INDEPENDENT code path, `develop_trajectory()` (which re-derives the grid at every iteration count 0..T via its own loop, not by calling decode()). **0/20 cross-check mismatches** -- 0, confirming decode() is NOT the source of the discrepancy: it agrees exactly with the independently-computed trajectory at both T=39 and T=40 for every probed genome.

2/20 probed genomes have a decoded body that genuinely differs between T=39 and T=40 (confirmed correct by the cross-check above, not a comparison-code artifact). Example (first differing genome, if any):

genome index 7:
```
T=39:
[[0 0 0 0 0]
 [0 0 1 1 1]
 [0 0 2 1 0]
 [0 0 1 1 1]
 [4 0 4 0 0]]
T=40:
[[0 0 0 0 0]
 [0 0 1 1 1]
 [0 0 2 1 0]
 [0 0 1 1 1]
 [0 4 0 4 0]]
```

**Verdict on (1) vs (2): NOT an implementation bug.** decode() is confirmed correct against an independent code path; the discrepancy reflects genuine ongoing behaviour in rule_based's own development. Proceeding to (2).

## Step 2: rule_based's own never_settled_fraction at T_max=100

N=2000 genomes (seed=0), fitness-blind, full trajectory recorded via `develop_trajectory()`. **This number has never been explicitly reported on its own before** -- only rule_based's median/q95 settle_step AMONG SETTLED GENOMES has been discussed in prior reports.

**never_settled_fraction = 0.1155** (231/2000 genomes), at T_max=100 -- well beyond rule_based's expected fast settling (median settle_step~2). settle_steps_median (among SETTLED genomes only) = 2.0000, settle_steps_q95 (settled only) = 5.0000.

## Period-cycle breakdown of the never-settled population

| period | count | fraction of never-settled genomes |
| --- | --- | --- |
| p=2 | 169 | 0.7316 |
| p=3 | 25 | 0.1082 |
| p=4 | 9 | 0.0390 |
| p=5 | 1 | 0.0043 |
| p=6 | 20 | 0.0866 |
| p=8 | 4 | 0.0173 |
| p=9 | 1 | 0.0043 |
| p=10 | 1 | 0.0043 |
| no period <=20 detected | 1 | 0.0043 |

**0.9957 of never-settled genomes have a detected period <= 20** -- i.e. genuine rule-cycle behaviour, not merely slow transient convergence. As a fraction of the FULL N=2000 population: 0.1150.

## Traced example cycling genomes (rule lists + hand-traceable mechanism)

### Genome index 16 (period 2)

Rule list (in evaluation order -- first matching rule wins each cell each step):

0. IF cell_type{'state': <VoxelState.SOFT: 'soft'>} THEN grow_directional{'direction': <Direction.RIGHT: 'right'>, 'voxel_type': <VoxelState.ACTUATOR_VERTICAL: 'actuator_vertical'>}
1. IF always THEN copy_neighbor{'direction': <Direction.DOWN: 'down'>}
2. IF position{'position': <PositionType.BOUNDARY: 'boundary'>} THEN set_soft
3. IF empty_neighbor{'direction': <Direction.LEFT: 'left'>} THEN grow_directional{'direction': <Direction.LEFT: 'left'>, 'voxel_type': <VoxelState.HARD: 'hard'>}
4. IF empty_neighbor{'direction': <Direction.LEFT: 'left'>} THEN set_hard

Last 4 grid snapshots (showing the period-2 cycle):

t=-4:
```
[[0 0 2 4 0]
 [0 0 2 0 0]
 [0 0 2 4 0]
 [0 0 0 0 0]
 [0 0 0 0 0]]
```
t=-3:
```
[[0 0 2 0 0]
 [0 0 2 4 0]
 [0 0 2 0 0]
 [0 0 0 0 0]
 [0 0 0 0 0]]
```
t=-2:
```
[[0 0 2 4 0]
 [0 0 2 0 0]
 [0 0 2 4 0]
 [0 0 0 0 0]
 [0 0 0 0 0]]
```
t=-1:
```
[[0 0 2 0 0]
 [0 0 2 4 0]
 [0 0 2 0 0]
 [0 0 0 0 0]
 [0 0 0 0 0]]
```

### Genome index 19 (period 3)

Rule list (in evaluation order -- first matching rule wins each cell each step):

0. IF empty_neighbor{'direction': <Direction.RIGHT: 'right'>} THEN grow_directional{'direction': <Direction.RIGHT: 'right'>, 'voxel_type': <VoxelState.SOFT: 'soft'>}
1. IF neighbor_count{'threshold': 1} THEN copy_neighbor{'direction': <Direction.LEFT: 'left'>}
2. IF cell_type{'state': <VoxelState.SOFT: 'soft'>} THEN grow_directional{'direction': <Direction.UP: 'up'>, 'voxel_type': <VoxelState.ACTUATOR_VERTICAL: 'actuator_vertical'>}
3. IF position{'position': <PositionType.RIGHT: 'right'>} THEN set_actuator_vertical

Last 5 grid snapshots (showing the period-3 cycle):

t=-5:
```
[[0 2 0 2 2]
 [0 2 0 2 2]
 [0 2 0 2 2]
 [0 2 0 2 2]
 [0 2 0 2 2]]
```
t=-4:
```
[[0 2 2 0 2]
 [0 2 2 0 2]
 [0 2 2 0 2]
 [0 2 2 0 2]
 [0 2 2 0 2]]
```
t=-3:
```
[[0 0 2 2 0]
 [0 0 2 2 0]
 [0 0 2 2 0]
 [0 0 2 2 0]
 [0 0 2 2 0]]
```
t=-2:
```
[[0 2 0 2 2]
 [0 2 0 2 2]
 [0 2 0 2 2]
 [0 2 0 2 2]
 [0 2 0 2 2]]
```
t=-1:
```
[[0 2 2 0 2]
 [0 2 2 0 2]
 [0 2 2 0 2]
 [0 2 2 0 2]
 [0 2 2 0 2]]
```

### Genome index 21 (period 2)

Rule list (in evaluation order -- first matching rule wins each cell each step):

0. IF position{'position': <PositionType.INTERIOR: 'interior'>} THEN set_empty
1. IF empty_neighbor{'direction': <Direction.RIGHT: 'right'>} THEN grow_directional{'direction': <Direction.DOWN: 'down'>, 'voxel_type': <VoxelState.SOFT: 'soft'>}
2. IF cell_type{'state': <VoxelState.SOFT: 'soft'>} THEN grow_directional{'direction': <Direction.LEFT: 'left'>, 'voxel_type': <VoxelState.SOFT: 'soft'>}
3. IF empty_neighbor{'direction': <Direction.RIGHT: 'right'>} THEN grow_directional{'direction': <Direction.RIGHT: 'right'>, 'voxel_type': <VoxelState.HARD: 'hard'>}
4. IF position{'position': <PositionType.BOTTOM: 'bottom'>} THEN set_actuator_vertical

Last 4 grid snapshots (showing the period-2 cycle):

t=-4:
```
[[0 0 0 0 0]
 [2 2 2 2 0]
 [2 0 0 0 0]
 [2 0 0 0 0]
 [2 4 4 4 4]]
```
t=-3:
```
[[0 0 0 0 0]
 [2 0 0 0 0]
 [2 0 0 0 0]
 [2 0 0 0 0]
 [2 4 4 4 4]]
```
t=-2:
```
[[0 0 0 0 0]
 [2 2 2 2 0]
 [2 0 0 0 0]
 [2 0 0 0 0]
 [2 4 4 4 4]]
```
t=-1:
```
[[0 0 0 0 0]
 [2 0 0 0 0]
 [2 0 0 0 0]
 [2 0 0 0 0]
 [2 4 4 4 4]]
```


## Does this change T_primary's validity?

**Yes, partially -- a real gap in the original P4 derivation, not just an NCA-side issue.** `evodevo/analysis/depth_study.py::run_layer2` computes `median_settle_step`/`p95_settle_step` (the quantities `_select_t_primary` maxes over) from `settle_steps`, a list that ONLY ever receives an entry inside the `if settled:` branch -- never-settled genomes are never appended. So rule_based's own q95 in the committed P4 derivation was **conditioned on settled genomes only**, silently excluding its own non-settling tail -- exactly the same structural gap this project already found and corrected for NCA (results/nca/HORIZON_EXTENSION.md), just never checked on rule_based's side until now. At the ORIGINAL T_max=40 measurement, rule_based's own `never_settled_fraction` was already recorded (results/depth/depth_study_raw.json, `layer2.rule_based.never_settled_fraction = 0.1155`) but not reported as a headline finding anywhere, and never fed into the T_primary derivation's own never-settled accounting the way NCA's was.

This does NOT immediately overturn T_primary=39 as a NUMBER (it was derived as `max(q95(settle_step))` over both arms, and rule_based's own q95=5 was never the binding term -- NCA's q95=39 dominated), but it DOES mean the derivation's underlying premise ("q95 among settled genomes represents this arm's convergence horizon") was quietly false for rule_based too, not just for NCA. Since rule_based's non-settling fraction was never separately gated against the `NEVER_SETTLED_PROVISIONAL_THRESHOLD=0.5` fallback check the way NCA's is, this is a genuine, previously-uncaught gap in the derivation's completeness -- disclosed here as a finding for `docs/decisions.md`, with the fix (extending the provisional-fallback / non-settling check to BOTH arms, not just NCA) deferred to a follow-up rather than silently patched in this diagnosis-only task.

## Does rule_based need the same readout-policy treatment as NCA?

**Yes, for the same reason.** 0.1150 of the full rule_based population has a confirmed genuine rule-cycle (not a transient), i.e. an UNDEFINED phenotype under any point-in-time (fixed-T) readout for those genomes specifically -- structurally the same problem NCA's `mode_of_window` readout (results/nca/READOUT_POLICY_SELECTION.md) was built to address, and if anything a CLEANER case here: these are exact, provably periodic grid cycles (not chaotic-like sensitivity), so a windowed/majority readout is a natural, well-motivated fix. Implementing it for `RuleBasedEncoder` is explicitly OUT OF SCOPE for this diagnosis-only task (per its own MUST NOT constraint: no changes to rule_based's vocabulary/interpreter/genome representation) and is deferred to a follow-up.

**Cost**: 384.19s wall-clock for N=2000 genomes x T_max=100 steps -- negligible relative to any PPO training cost.

