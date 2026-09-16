# rule_based capacity study: justifying MIN_RULES/MAX_RULES on evidence

Fitness-blind (only decode()/develop_rules()/random_genome()/validate_body() are used -- NO PPO, NO task, NO reward, NO EA).

## Part 1: description length, three arms side by side

Rule space: |Cond|=20 (auto-derived from the actual `types.py` vocabulary, not hardcoded: always(1) + cell_type(5) + neighbor_count(4 threshold values) + empty_neighbor(4) + position(6)) x |Act|=25 (5 parameterless SET_* + grow_directional(4x4) + copy_neighbor(4)) = 500 (condition, action) pairs -> **8.966 bits/rule**.

A genome's description length = log2(rule-count choices) + n * bits/rule -- at MIN_RULES=2: **21.39 bits**; at MAX_RULES=12: **111.05 bits**.

**direct**: 25 cells x log2(5 materials) = **58.05 bits** (~58 bits, matching the task's own citation).

**NCA**: the task's own citation of "792 float64 parameters" refers to `hidden_channels=3` -- the PRE-Outcome-B architecture point (docs/decisions.md's "NCA joint stability-vs-expressivity trade-off (Outcome B)" row), NOT the CURRENT resolved primary. The CURRENT primary (`resolved_config.py::PRIMARY`, `hidden_channels=0`) has **501 parameters**, not 792. Both are reported here, clearly labelled, rather than silently substituting one for the other: at a STATED, deliberately rough assumption of 32-bit effective precision per parameter (chosen as a representative middle ground between float32 storage and the much lower ACTUAL functional precision any single weight plausibly carries after training/mutation noise -- this comparison is necessarily rough, as the task itself notes), CURRENT primary = **16032 bits**, H=3 (historical) = **25344 bits**. At full float64 storage precision (64 bits/param, an upper bound, not a claim about USABLE information): CURRENT primary = 32064 bits, H=3 = 50688 bits.

| arm | description length (bits) |
| --- | --- |
| direct | 58.0 |
| rule_based, MIN_RULES=2 | 21.4 |
| rule_based, MAX_RULES=12 | 111.0 |
| NCA, CURRENT primary (H=0), 32-bit effective | 16032 |
| NCA, H=3 (historical, task's own citation), 32-bit effective | 25344 |

**The asymmetry is real and large (roughly 2-3 orders of magnitude), and this is characterised honestly rather than papered over**: direct and rule_based sit within roughly one order of magnitude of each other (tens to low hundreds of bits), while NCA sits several orders of magnitude higher regardless of which precision assumption or hidden-channel point is used. The encodings cannot be made equal in capacity without fundamentally changing what each one IS (a fixed-length float vector vs. a small ordered symbolic rule list vs. a per-cell material grid) -- this asymmetry is a structural property of the comparison, not a bug or an oversight to fix.

## Part 2: effective_rule_count distributions (does MAX_RULES bind?)

**No evolved-genome archives exist** anywhere in this project's committed run artifacts to analyze (re-confirmed from results/rule_based/CONTEXT_ANALYSIS.md's own earlier finding: `results/runs/**` stores only fitness/validity metadata and decoded-body hashes, never the genome/rule-list itself) -- disclosed honestly; this report covers random genomes only, at two different sampling regimes.

### (a) as actually sampled (sample_developmental_rules(), [3,7] rules)

| value | count | fraction |
| --- | --- | --- |
| 1 | 291 | 0.1455 |
| 2 | 324 | 0.1620 |
| 3 | 582 | 0.2910 |
| 4 | 432 | 0.2160 |
| 5 | 265 | 0.1325 |
| 6 | 93 | 0.0465 |
| 7 | 13 | 0.0065 |

mean=3.1935, median=3.0000, max=7

### (b) explicitly given the FULL MAX_RULES=12-rule budget

| value | count | fraction |
| --- | --- | --- |
| 1 | 499 | 0.2495 |
| 2 | 420 | 0.2100 |
| 3 | 332 | 0.1660 |
| 4 | 255 | 0.1275 |
| 5 | 172 | 0.0860 |
| 6 | 135 | 0.0675 |
| 7 | 116 | 0.0580 |
| 8 | 51 | 0.0255 |
| 9 | 15 | 0.0075 |
| 10 | 5 | 0.0025 |

mean=3.2150, median=3.0000, max=10 (out of 12 available)

**0.0000 of genomes explicitly given all 12 rules have ALL 12 statically effective** (no dead rules at all), vs **1.0000 showing at least one dead rule even at the full budget**. Effective rule count DOES saturate below 12 for the large majority of genomes -- evidence FOR 12 being a generous, non-binding cap, not an arbitrary choice.

## Part 3: max_rules sweep -- does expressivity keep increasing past 12?

N=2000 genomes per cell, each drawn with EXACTLY the stated rule count.

| max_rules | valid_rate | unique_valid_body_fraction | material_entropy_bits |
| --- | --- | --- | --- |
| 4 | 0.4295 | 0.2170 | 2.2200 |
| 8 | 0.4600 | 0.2450 | 2.2675 |
| 12 | 0.4580 | 0.2415 | 2.2700 |
| 24 | 0.4580 | 0.2410 | 2.2700 |

unique_valid_body_fraction from 12->24 rules: 0.2415 -> 0.2410 (delta -0.0005). **Expressivity has PLATEAUED by 12** -- little to no further gain from doubling the rule budget to 24, evidence FOR keeping MAX_RULES=12 as currently set.

## Part 4: MIN_RULES=1 -- what breaks?

| n_rules | valid_rate | degenerate_fraction |
| --- | --- | --- |
| n=1 | 0.2150 | 0.2705 |
| n=2 (MIN_RULES) | 0.3395 | 0.3710 |

At n=1: valid_rate=0.2150 vs n=2: valid_rate=0.3395. A single rule cannot express BOTH a growth-triggering condition AND a fallback/default behaviour for cells that don't match it -- with exactly one rule, every non-matching cell is frozen (no rule ever applies to it, so it can never change from EMPTY), which the measured validity collapse at n=1 confirms directly rather than merely asserts: MIN_RULES=2 is justified as the smallest budget that can express both "a triggering condition" and "a fallback" (even a trivial ALWAYS-based one), which single-rule genomes structurally cannot.

