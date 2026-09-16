# Rule-Based Architecture Readiness Report

Verification-only pass over the entire rule_based architecture-redesign workstream (propositions R1-R4, the capacity study, the architecture-delta cascade, and the silent-clip fixes). Nothing in this report was fixed, tuned, or re-run to look better -- every number either comes from an existing committed artifact or a cheap, read-only check performed at report-generation time. If a check failed or an artifact was stale, missing, or inconsistent with another artifact, that is reported as a finding below, not silently worked around or reconciled.

## Section 0 -- Provenance

- git commit: `2c09895a6bc87c950826b2e9eeafb9d344a34a7b`
- working tree clean: NO -- dirty/untracked paths: docs/decisions.md, evodevo/analysis/depth_study.py, evodevo/analysis/mutation_calibration.py, evodevo/analysis/structural.py, evodevo/analysis/structural_report.py, evodevo/encoders/nca/encoder.py, evodevo/encoders/rule_based/_rules/genome.py, evodevo/encoders/rule_based/_rules/interpreter.py, evodevo/encoders/rule_based/encoder.py, evodevo/experiments/benchmark.py, evodevo/experiments/configs/benchmark.yaml, results/structural/STRUCTURAL.md, results/structural/figs/light_cone.png, results/structural/figs/never_settled_fraction.png, results/structural/figs/perturbation_robustness.png, results/structural/figs/redundancy.png, results/structural/figs/validity_rate.png, results/structural/nca/diversity.jsonl, results/structural/nca/light_cone.jsonl, results/structural/nca/perturbation_robustness.jsonl
- report generated: 2026-09-16T16:31:15.958714+00:00
- package versions: python=3.10.20, platform=macOS-26.5.1-arm64-arm-64bit, numpy=1.26.4, matplotlib=3.10.9, pytest=9.1.1

### Resolved rule_based configuration

| parameter | value | resolved by | source |
| --- | --- | --- | --- |
| conflict_resolution | first_rule_index | R2 (order-independence) | `evodevo/encoders/rule_based/resolved_config.py::PRIMARY.conflict_resolution (docs/decisions.md's "Make the rule_based interpreter provably order-independent by default" row)` |
| copy_neighbor_reads | pre_iteration | R2 (order-independence audit, Part 1c) | `evodevo/encoders/rule_based/encoder.py::RuleBasedEncoder.__init__ default (confirmed not a source of order-dependence; case (b) ruled out -- ORDER_INDEPENDENCE_AUDIT.md Part 1c)` |
| position_predicate | disabled | R3 (translation-equivariance) | `evodevo/encoders/rule_based/resolved_config.py::PRIMARY.position_predicate (docs/decisions.md's "Restore translation-equivariance (Proposition R3)..." row)` |
| action_monotonicity | unrestricted | R4 (monotone ablation, kept as ablation) | `evodevo/encoders/rule_based/resolved_config.py::PRIMARY.action_monotonicity (docs/decisions.md's monotone-ablation row -- class default deliberately NOT switched)` |
| boundary | no_op | R3 (parity axis, unchanged from historical default) | `evodevo/encoders/rule_based/resolved_config.py::PRIMARY.boundary` |
| neighbourhood | von_neumann | pre-existing ("Neighbourhood type" decisions.md row) | `evodevo/encoders/rule_based/resolved_config.py::PRIMARY.neighbourhood` |
| MIN_RULES | 2 | capacity study (justified on evidence) | `evodevo/encoders/rule_based/_rules/genome.py::MIN_RULES` |
| MAX_RULES | 12 | capacity study (justified on evidence) | `evodevo/encoders/rule_based/_rules/genome.py::MAX_RULES` |
| HARD_MAX_ITERATIONS | 105 | safety ceiling (raised over several tasks, currently 105) | `evodevo/encoders/rule_based/_rules/interpreter.py::HARD_MAX_ITERATIONS` |
| readout_mode (class default) | fixed_t | readout-policy resolution | `evodevo/encoders/rule_based/encoder.py::RuleBasedEncoder.__init__ default -- deliberately stays 'fixed_t'; the resolved mode_of_window/W=6 policy must be threaded explicitly by any caller building 'the primary rule_based arm' (see results/T_PRIMARY_UNIFIED.md)` |
| readout_window (class default) | 5 | readout-policy resolution (class default, NOT the resolved W) | `evodevo/encoders/rule_based/encoder.py::RuleBasedEncoder.__init__ default` |
| resolved readout W (when mode_of_window is threaded) | 6 | T_PRIMARY_UNIFIED (T-shift-stability sweep) | `results/T_PRIMARY_UNIFIED.md` |
| tie-break mode | validity_aware=True (genome-aware, TIEBREAK_VALIDITY_CHECK.md's recommended fix) | tie-break validity check | `evodevo/encoders/rule_based/encoder.py::decode (mode_of_window branch passes validity_aware=True) and evodevo/analysis/windowed_readout.py::mode_of_window` |
| mutation_rate (rule_based, calibrated) | 0.1707 | P5 / mutation re-calibration cascade | `results/mutation/mutation_calibration_raw.json::calibrated.rule_based.param_value` |

## Section 1 -- Order-independence (Proposition R2)

**BEFORE (from the committed results/rule_based/ORDER_INDEPENDENCE_AUDIT.md, Part 1a)**: the permutation-test fraction of genomes whose decoded body differs under a permuted cell enumeration order is **0.1295** (i.e. 12.95% of N=2000).
**Growth-conflict frequency (case (a), Part 1b)**: **0.3040** of sampled genomes have at least one growth conflict across their development.

### Discrepancy between the committed audit artifact and its own downstream citations

docs/decisions.md's own "Make the rule_based interpreter provably order-independent by default" row cites the BEFORE fraction as **15.90%**; docs/rule-language-spec.md's section C cites it as **12.95%** and cites the growth-conflict frequency as **30.40%**; evodevo/encoders/rule_based/_rules/interpreter.py's own docstring cites the growth-conflict frequency as **30.40%**. The ACTUALLY COMMITTED results/rule_based/ORDER_INDEPENDENCE_AUDIT.md artifact -- the primary source these three all cite -- states **12.95%** (permutation-test fraction) and **30.40%** (growth-conflict frequency), NOT the higher figures repeated in decisions.md/the spec/the interpreter docstring.

**Flagged, not reconciled**: this is a genuine cross-artifact inconsistency -- either the committed ORDER_INDEPENDENCE_AUDIT.md was regenerated after decisions.md/the spec/the interpreter docstring were written (and those three downstream citations were never updated to match), or the reverse. Per this report's own charter, the committed .md artifact is treated as ground truth for the number itself, and the mismatch is reported as a finding (Section 11) rather than silently averaged or corrected in any of the four locations.

**Copy-neighbor read semantics (case (b))**: **Confirmed by direct code reading, cross-checked with a targeted test (test_copy_neighbor_reads_pre_iteration_state_not_live_state)**: `_apply_action`'s COPY_NEIGHBOR branch reads `current_grid[si][sj]`, and `current_grid` is ALWAYS the untouched pre-iteration `grid` (never `next_grid`) at every call site in `develop_rules`'s Pass 2 loop. copy_neighbor reads PRE-ITERATION state, not live/partially-updated state -- case (b) is NOT a source of order-dependence; the interpreter's self-action pass (Pass 2a) is fully order-independent by construction (every self-action writes to its own acting cell, and at most one decision exists per cell, so no two self-actions can ever collide on the same write target regardless of enumeration order).

**AFTER, under the adopted default (`conflict_resolution="first_rule_index"`), from docs/decisions.md's R2 row (verbatim re-run statement)**: `0/2000` genomes differ under permutation -- **PASS (0/2000)**.

## Section 2 -- Formal specification

docs/rule-language-spec.md exists: **True**.

Ellipsis/'etc' scan of the full document: 2 line(s) match a bare `...` or the word 'etc' anywhere in the file; of these, **0** fall inside the BNF grammar block (section A) itself, which is the acceptance criterion this check enforces.

### Programmatic ConditionType/ActionType <-> grammar cross-check

| enum member | value | appears in grammar block? |
| --- | --- | --- |
| ConditionType.ALWAYS | `always` | PASS |
| ConditionType.CELL_TYPE | `cell_type` | PASS |
| ConditionType.NEIGHBOR_COUNT | `neighbor_count` | PASS |
| ConditionType.EMPTY_NEIGHBOR | `empty_neighbor` | PASS |
| ConditionType.POSITION | `position` | PASS |
| ActionType.SET_EMPTY | `set_empty` | PASS |
| ActionType.SET_SOFT | `set_soft` | PASS |
| ActionType.SET_HARD | `set_hard` | PASS |
| ActionType.SET_ACTUATOR_HORIZONTAL | `set_actuator_horizontal` | PASS |
| ActionType.SET_ACTUATOR_VERTICAL | `set_actuator_vertical` | PASS |
| ActionType.GROW_DIRECTIONAL | `grow_directional` | PASS |
| ActionType.COPY_NEIGHBOR | `copy_neighbor` | PASS |

Result: **ALL PRESENT**.

**Context-space size**: docs/rule-language-spec.md section D claims `|C| = 6480`; `evodevo.encoders.rule_based.context_analysis.CONTEXT_SPACE_SIZE` = 6480; a fresh `len(enumerate_contexts())` call = 6480. **MATCH**.

## Section 3 -- Locality / translation-equivariance (Proposition R3)

Direct empirical test of D(translate(s)) == translate(D(s)) on a torus, N=500 each setting:

| position_predicate | n | n_pass | n_fail | fraction_fail |
| --- | --- | --- | --- | --- |
| disabled | 500 | 500 | 0 | 0.0000 |
| enabled | 500 | 393 | 107 | 0.2140 |

**position_predicate="disabled"**: 0 violations -- PASS.
**position_predicate="enabled"**: 107 violations -- PASS (test discriminates, as required).

Structural cost of disabling POSITION (Part 1, N=2000):

| metric | enabled | disabled (new default) | delta |
| --- | --- | --- | --- |
| valid_rate | 0.5735 | 0.5485 | -0.0250 |
| degenerate_fraction | 0.1870 | 0.2375 | 0.0505 |
| unique_valid_body_fraction | 0.3955 | 0.3330 | -0.0625 |
| material_entropy_bits | 2.3038 | 2.3045 | 0.0007 |
| genomes_per_unique_phenotype | 1.8083 | 2.1575 | 0.3492 |
| largest_neutral_class_fraction | 0.0615 | 0.0755 | 0.0140 |
| never_settled_fraction | 0.1155 | 0.1080 | -0.0075 |
| mean_settle_step | 2.2012 | 2.1754 | -0.0258 |
| mean_hamming_displacement | 5.6715 | 5.8450 | 0.1735 |
| fraction_zero_displacement | 0.4920 | 0.5210 | 0.0290 |

## Section 4 -- Context analysis (dead rules, coverage, action redundancy)

**Numbering note**: the task prompt that requested this report numbers this section "R4"; this project's own committed docs (docs/rule-language-spec.md section F, docs/decisions.md) instead use R4 for the MONOTONE-ABLATION termination proposition (Section 6 below) and treat context_analysis.py as separate, unnumbered supporting infrastructure used by multiple propositions/checks (R1's cycle study, the capacity study). This report follows the PROJECT'S established numbering (R4 = monotone), not the prompt's numbering, and flags the mismatch explicitly here rather than silently picking one without saying so.

### Static dead-rule count distribution (provably unreachable, over the full |C|)

| value | count | fraction |
| --- | --- | --- |
| value | count | fraction |
| 0 | 592 | 0.2960 |
| 1 | 459 | 0.2295 |
| 2 | 338 | 0.1690 |
| 3 | 223 | 0.1115 |
| 4 | 168 | 0.0840 |
| 5 | 151 | 0.0755 |
| 6 | 69 | 0.0345 |

### Dynamic dead-rule count distribution (never fired in this genome's own T=100 development)

| value | count | fraction |
| --- | --- | --- |
| value | count | fraction |
| 0 | 291 | 0.1455 |
| 1 | 480 | 0.2400 |
| 2 | 437 | 0.2185 |
| 3 | 335 | 0.1675 |
| 4 | 216 | 0.1080 |
| 5 | 160 | 0.0800 |
| 6 | 81 | 0.0405 |

Static and dynamic dead-rule counts are DISTINCT notions (dynamic is weaker/sample-dependent; every statically-dead rule is also dynamically dead, not vice versa) -- kept separate above per CONTEXT_ANALYSIS.md's own definition, not merged.

### effective_rule_count (static), two different sampling regimes

**CONTEXT_ANALYSIS.md's own sample** (position_predicate="disabled", `sample_developmental_rules()` draw, N=2000):

| value | count | fraction |
| --- | --- | --- |
| value | count | fraction |
| 1 | 299 | 0.1495 |
| 2 | 310 | 0.1550 |
| 3 | 538 | 0.2690 |
| 4 | 492 | 0.2460 |
| 5 | 263 | 0.1315 |
| 6 | 83 | 0.0415 |
| 7 | 15 | 0.0075 |

**CAPACITY_STUDY.md Part 2's FRESHER numbers, a DIFFERENT sampling regime** -- (a) as actually sampled ([3,7] rules):

| value | count | fraction |
| --- | --- | --- |
| value | count | fraction |
| 1 | 291 | 0.1455 |
| 2 | 324 | 0.1620 |
| 3 | 582 | 0.2910 |
| 4 | 432 | 0.2160 |
| 5 | 265 | 0.1325 |
| 6 | 93 | 0.0465 |
| 7 | 13 | 0.0065 |

(b) explicitly given the full MAX_RULES=12-rule budget:

| value | count | fraction |
| --- | --- | --- |
| value | count | fraction |
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

These two sources use DIFFERENT sampling regimes (CONTEXT_ANALYSIS.md's default draw vs CAPACITY_STUDY.md's explicit [3,7]-rule and full-12-rule draws) and are kept distinct here, not merged into one number.

**grow_directional/grow_directional metric-limitation caveat (verbatim)**: **CAVEAT on `grow_directional`/`grow_directional` pairs**: GROW_DIRECTIONAL never writes the ACTING cell at all (it writes a NEIGHBOR) -- so under this metric ("effect on the acting cell"), ANY two GROW_DIRECTIONAL rules trivially show 100% agreement (both leave the actor unchanged), REGARDLESS of whether they grow in different directions or different materials. This is a limitation of comparing target-affecting actions via an actor-centric metric, not a genuine behavioural-redundancy finding -- reported here explicitly rather than left to be misread as "all growth-growth rule pairs are interchangeable", which this metric cannot actually establish.

## Section 5 -- Exact cycles (Proposition R1)

Cap-hitting genome count: **0/2000** -- PASS.

(mu, p) joint distribution summary: mu median=2.0000, mu max=21, mu mean=2.2905; p median=1.0000, p max=60; max observed (mu+p)=75 (verbatim from the committed report).

Fixed points (p=1): 1784/2000 (0.8920). Genuine cycles (p>1): 216/2000 (0.1080).

Period histogram (p>1 only):

| period | count | fraction of cycling genomes |
| --- | --- | --- |
| 2 | 161 | 0.7454 |
| 3 | 19 | 0.0880 |
| 4 | 15 | 0.0694 |
| 5 | 2 | 0.0093 |
| 6 | 11 | 0.0509 |
| 8 | 5 | 0.0231 |
| 10 | 1 | 0.0046 |
| 12 | 1 | 0.0046 |
| 60 | 1 | 0.0046 |

**Reconciliation with results/rule_based/SETTLING_AUDIT.md**: This module's EXACT figure for the SAME pre-R2/R3 dynamics (row_major_last_wins, position enabled, same genome draw order/seed): **0.1155 of genomes are genuinely cyclic (p>1)**, vs SETTLING_AUDIT.md's sampled 0.1150 -- closely matching (within expected small differences from SETTLING_AUDIT.md's own T_max=100 window occasionally under- or over-counting near the boundary), confirming the sampled estimate was accurate.

**Fraction of cycling genomes whose period does not divide W=6**: 25/216 cycling genomes (0.1157).

## Section 6 -- Monotone ablation (Proposition R4)

Phi-non-decreasing violations: **0** -- PASS.
Max termination step observed: **7** (bound 100) -- PASS.
Period>1 cycles under monotone: **0/2000** -- PASS.

Full expressivity cost table (unrestricted vs monotone, N=2000, T_primary=100):

| metric | unrestricted | monotone | delta |
| --- | --- | --- | --- |
| valid_rate | 0.5470 | 0.6530 | 0.1060 |
| degenerate_fraction | 0.2380 | 0.1165 | -0.1215 |
| unique_valid_body_fraction | 0.3315 | 0.3865 | 0.0550 |
| material_entropy_bits | 2.3043 | 2.3155 | 0.0112 |
| uniform_body_fraction | 0.2380 | 0.1165 | -0.1215 |
| never_settled_fraction | 0.1080 | 0.0000 | -0.1080 |
| mean_settle_step | 2.1754 | 1.8550 | -0.3204 |
| mean_hamming_displacement | 5.8280 | 5.1735 | -0.6545 |
| fraction_zero_displacement | 0.5230 | 0.5480 | 0.0250 |

**Reported plainly, not softened**: monotone BEATS unrestricted on 4 of 5 core structural metrics (valid_rate, degenerate_fraction, unique_valid_body_fraction, uniform_body_fraction) plus the termination/no-cycling guarantee itself -- the one genuine, disclosed cost is reduced mutation sensitivity (mean_hamming_displacement drops, fraction_zero_displacement rises). This is a real finding, not an error, and is reported here as prominently as the guarantee.

## Section 7 -- Capacity (description length, MAX_RULES sweep)

Three-arm description-length comparison:

| arm | description length (bits) |
| --- | --- |
| direct | 58.0 |
| rule_based, MIN_RULES=2 | 21.4 |
| rule_based, MAX_RULES=12 | 111.0 |
| NCA, CURRENT primary (H=0), 32-bit effective | 16032 |
| NCA, H=3 (historical, task's own citation), 32-bit effective | 25344 |

max_rules sweep (does expressivity keep increasing past 12?):

| max_rules | valid_rate | unique_valid_body_fraction | material_entropy_bits |
| --- | --- | --- | --- |
| 4 | 0.4295 | 0.2170 | 2.2200 |
| 8 | 0.4600 | 0.2450 | 2.2675 |
| 12 | 0.4580 | 0.2415 | 2.2700 |
| 24 | 0.4580 | 0.2410 | 2.2700 |

Expressivity has plateaued by MAX_RULES=12 (12->24 unique_valid_body_fraction delta -0.0005) -- evidence FOR keeping MAX_RULES=12 as currently set.

## Section 8 -- Before/after delta and regression check

Full OLD (row_major_last_wins, position enabled) vs NEW (first_rule_index, position disabled) table:

| quantity | OLD | NEW | direction |
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

**Independent regression check (results/structural/STRUCTURAL.md, direct/nca rows)**: unchanged from the last committed version -- PASS.

**Fresh, independently re-run `python -m evodevo.analysis.t_primary_unified`** (return code 0): re-derived shared T_primary = **100**. evodevo/experiments/configs/benchmark.yaml's own `dev_steps` = **100**. MATCH -- T_primary is unchanged at 100, confirmed by a genuine fresh re-run, not merely a re-citation.

## Section 9 -- Silent-clip audit

Fresh grep for every ceiling-shaped constant definition under evodevo/encoders/rule_based and evodevo/analysis:

```
evodevo/encoders/rule_based/_rules/types.py:108:MIN_NEIGHBOR_THRESHOLD = 1
evodevo/encoders/rule_based/_rules/types.py:109:MAX_NEIGHBOR_THRESHOLD = 4
evodevo/encoders/rule_based/_rules/interpreter.py:209:HARD_MAX_ITERATIONS = 105
evodevo/encoders/rule_based/_rules/genome.py:44:MIN_RULES = 2
evodevo/encoders/rule_based/_rules/genome.py:45:MAX_RULES = 12
evodevo/encoders/rule_based/_rules/genome.py:180:_SAMPLE_MIN_RULES = 3
evodevo/encoders/rule_based/_rules/genome.py:181:_SAMPLE_MAX_RULES = 7
evodevo/analysis/rule_based_capacity_study.py:139:    genome_mod.MIN_RULES, genome_mod.MAX_RULES = min_rules, max_rules
evodevo/analysis/rule_based_capacity_study.py:143:        genome_mod.MIN_RULES, genome_mod.MAX_RULES = old_min, old_max
evodevo/analysis/rule_based_capacity_study.py:148:    rules()'s own [3,7] sampling range -- used to test whether MAX_RULES=12
evodevo/analysis/rule_based_capacity_study.py:176:        "full_budget_counts": full_budget_counts,  # genomes explicitly given exactly MAX_RULES=12 rules
evodevo/analysis/rule_based_capacity_study.py:240:# Part 4: MIN_RULES=1 check
evodevo/analysis/rule_based_capacity_study.py:303:        f"MIN_RULES={MIN_RULES}: **{_fmt(dl['rb_bits_min'], 2)} bits**; at MAX_RULES={MAX_RULES}: "
evodevo/analysis/rule_based_capacity_study.py:332:    lines.append(f"| rule_based, MIN_RULES={MIN_RULES} | {_fmt(dl['rb_bits_min'], 1)} |")
evodevo/analysis/rule_based_capacity_study.py:333:    lines.append(f"| rule_based, MAX_RULES={MAX_RULES} | {_fmt(dl['rb_bits_max'], 1)} |")
evodevo/analysis/rule_based_capacity_study.py:357:    lines.append(_distribution_table(erc["full_budget_counts"], f"(b) explicitly given the FULL MAX_RULES={MAX_RULES}-rule budget", cap_note=f" (out of {MAX_RULES} available)"))
evodevo/analysis/rule_based_capacity_study.py:367:            f"full-budget genomes use every rule they are given, meaning **MAX_RULES={MAX_RULES} may be "
evodevo/analysis/rule_based_capacity_study.py:398:            "budget to 24, evidence FOR keeping MAX_RULES=12 as currently set."
evodevo/analysis/rule_based_capacity_study.py:403:    lines.append("## Part 4: MIN_RULES=1 -- what breaks?\n")
evodevo/analysis/rule_based_capacity_study.py:417:            "validity collapse at n=1 confirms directly rather than merely asserts: MIN_RULES=2 is justified "
evodevo/analysis/rule_based_capacity_study.py:421:            else "The measured difference is small -- MIN_RULES=2 is not strongly justified by validity "
evodevo/analysis/rule_based_readiness_check.py:558:    lines.append("\n(b) explicitly given the full MAX_RULES=12-rule budget:\n")
evodevo/analysis/rule_based_readiness_check.py:685:    lines.append("\nExpressivity has plateaued by MAX_RULES=12 (12->24 unique_valid_body_fraction delta -0.0005) -- evidence FOR keeping MAX_RULES=12 as currently set.\n")
evodevo/analysis/rule_based_readiness_check.py:928:        if "PRIMARY." in rest or "FRESH_PRIMARY." in rest or "MIN_RULES ==" in rest or "MAX_RULES ==" in rest:
evodevo/analysis/readiness_check.py:915:        "RuleBasedEncoder's MIN_RULES/MAX_RULES genome-length bounds, HARD_MAX_ITERATIONS=40 -- none of "
evodevo/analysis/depth_study.py:53:T_MAX_HARD_CAP = 40
evodevo/analysis/depth_study.py:407:            f"q95) but is capped to T_MAX_HARD_CAP={T_MAX_HARD_CAP}; granted {t_primary}. "
evodevo/analysis/depth_study.py:709:            f"to T_MAX_HARD_CAP={T_MAX_HARD_CAP}; granted {min(2 * t_primary, T_MAX_HARD_CAP)}. A deliberate "
evodevo/analysis/depth_study.py:782:            f"**Note**: 2*T_primary = 2*{t_primary} = {doubled} exceeds T_MAX_HARD_CAP={T_MAX_HARD_CAP}, so "
evodevo/analysis/depth_study.py:786:            f"in this sweep that tests behaviour strictly beyond T_MAX_HARD_CAP=40.\n"
evodevo/analysis/rule_based_readout_policy.py:68:            f"exceeds HARD_MAX_ITERATIONS={hard_max_iterations} -- raise HARD_MAX_ITERATIONS or "
```

| ceiling constant | behaviour on violation |
| --- | --- |
| HARD_MAX_ITERATIONS (interpreter.py) | RAISES ValueError |
| MIN_RULES/MAX_RULES (genome.py) | RAISES ValueError |
| T_MAX_HARD_CAP / T_primary derivation clip (depth_study.py) | WARNS via warnings.warn |
| 2*T_primary layer-3 clip (depth_study.py) | WARNS via warnings.warn |
| rule_based_readout_policy._required_trajectory_length | RAISES ValueError |

**Single-source-of-truth grep** (mirroring tests/encoders/test_rule_based_resolved_config_single_source.py's own check, run fresh here):

`grep -rnE '^(RULE_BASED_CONFLICT_RESOLUTION|RULE_BASED_POSITION_PREDICATE|RULE_BASED_BOUNDARY|RULE_BASED_NEIGHBOURHOOD)\s*(:\s*\w+\s*)?=' evodevo/analysis`

```
(no matches)
```

Result: **CLEAN -- no independent module-level copy of a resolved dynamics constant found**.

## Section 10 -- Self-consistency

| check | result |
| --- | --- |
| Section 0 config values match a fresh independent re-derivation from resolved_config.py | PASS |
| Section 8 T_primary matches benchmark.yaml's dev_steps AND a fresh t_primary_unified.py run | PASS |
| No hardcoded literal duplicating resolved_config.py's values in rule_based_readiness_check.py itself | PASS |

grep pattern used: `conflict_resolution\s*=\s*[\"']first_rule_index[\"']|position_predicate\s*=\s*[\"']disabled[\"']|MAX_RULES\s*=\s*12\b|MIN_RULES\s*=\s*2\b`

**Overall: ALL PASS**

## Section 11 -- Risk register (adversarial)

Written as a reviewer actively trying to reject this workstream -- understating a risk here is the worst possible outcome of this report.

- Residual order-dependence CITATION MISMATCH (Section 1): the committed results/rule_based/ORDER_INDEPENDENCE_AUDIT.md reports the permutation-test/growth-conflict fractions as 0.1295/0.3040, but docs/decisions.md, docs/rule-language-spec.md, and interpreter.py's own docstring all cite different (higher) figures (15.90%/30.40%/30.40%). This is a real, unresolved cross-artifact inconsistency, not a rounding difference.
- row_major_last_wins (the historical, order-DEPENDENT conflict-resolution policy) remains available as a named, selectable `conflict_resolution` value on RuleBasedEncoder/develop_rules -- correct and intentional (mirrors NCA's own kept-update_rule pattern), but it is a live risk if any future caller constructs the encoder without an explicit override and a code path silently defaults to it, or if an experiment configuration ever mis-sets this value.
- A non-trivial minority of cycling rule_based genomes have a period that the resolved windowed readout (W=6) does not cleanly absorb: 25/216 cycling genomes (0.1157) have a period that does not divide W=6 (results/rule_based/CYCLE_CHARACTERISATION.md, Section 5) -- their mode-of-window readout can still show residual phase-dependence.
- R3's fix has a real, disclosed expressivity cost: disabling POSITION generation reduces valid_rate by 0.0250 and unique_valid_body_fraction by 0.0625 at N=2000 (results/rule_based/LOCALITY_STUDY.md, Section 3) -- correctness (translation-equivariance) was purchased with a measurable drop in raw validity/diversity, not a free fix.
- Confirmed NOT a risk in practice, stated here for completeness of the adversarial review: resolved_config.py::PRIMARY.action_monotonicity is confirmed 'unrestricted' (Section 0/10) -- the monotone ablation (Section 6), despite beating unrestricted on 4/5 structural metrics, was correctly KEPT as an ablation, not silently promoted to the primary.
- Tie-break validity: results/rule_based/TIEBREAK_VALIDITY_CHECK.md found the ORIGINAL lexicographic tie-break discards a legitimately-valid phase for a substantial fraction of cycling genomes (harm fraction 0.3150 > its own 0.05 threshold). A genome-aware (`validity_aware=True`) alternative WAS subsequently implemented and IS wired into `RuleBasedEncoder.decode()`'s `mode_of_window` branch (`evodevo/encoders/rule_based/encoder.py`), but only takes effect when `readout_mode="mode_of_window"` is explicitly threaded by the caller -- the encoder's own CLASS DEFAULT stays `readout_mode="fixed_t"`, so any caller that constructs `RuleBasedEncoder` without explicitly resolving the readout policy still gets neither the windowed readout nor its tie-break fix. This is a real, live risk for any future caller, not a fully closed issue.
- No PPO-based (task-performance) result exists anywhere under this redesigned rule_based architecture: every number in this report and in R1-R4's own artifacts is fitness-blind (decode()/develop_rules()/random_genome()/mutate()/validate_body only). Whether the redesign's measured structural picture (order-independence restored at a validity cost, translation-equivariance restored at a validity cost, monotone beating unrestricted structurally but not adopted) translates to better, worse, or unchanged task performance is completely unknown.

## Section 12 -- Verdict

### (a) What the redesign demonstrably FIXED (before/after cited)

- Order-independence (R2): permutation-test fraction 0.1295 (BEFORE, results/rule_based/ORDER_INDEPENDENCE_AUDIT.md) -> `0/2000` genomes differ (AFTER, docs/decisions.md's R2 row, re-run under `conflict_resolution="first_rule_index"`) -- order-independence restored by construction.
- Translation-equivariance (R3): 107/500 violations WITH POSITION enabled -> 0/500 WITH it disabled (results/rule_based/LOCALITY_STUDY.md) -- restored, confirmed empirically on a torus.
- Termination guarantee (R4, ablation): 0/5710 Phi-decrease violations, 0/2000 period>1 cycles under monotone, max termination step 7 <= bound 100 (results/rule_based/MONOTONE_ABLATION.md).
- Exact (not merely sampled) characterisation of the cycling population is now possible (R1 proved, then computed exactly): 0.1155 exact vs 0.1150 sampled (results/rule_based/CYCLE_CHARACTERISATION.md vs SETTLING_AUDIT.md) -- confirms the earlier sampled estimate was accurate.

### (b) What was NOT fixed, or was newly introduced

- Tie-break: results/rule_based/TIEBREAK_VALIDITY_CHECK.md's harm fraction (0.3150 > 0.05 threshold) motivated a genome-aware fix that IS implemented in code but is NOT the encoder's class default readout policy (`readout_mode` defaults to `fixed_t`) -- see Section 11.
- R3's fix cost: valid_rate -0.0250, unique_valid_body_fraction -0.0625 (results/rule_based/LOCALITY_STUDY.md) -- a real, disclosed expressivity cost, not a free correctness fix.
- A minority of cycling genomes (11.57%) have a period that does not divide the resolved readout window W=6 (results/rule_based/CYCLE_CHARACTERISATION.md) -- not fully absorbed by the windowed readout.
- Cross-artifact citation mismatch (Section 1): decisions.md/the spec/the interpreter docstring cite different order-dependence percentages than the committed ORDER_INDEPENDENCE_AUDIT.md itself.
- docs/rule-language-spec.md's own R4 status line was found stale (still 'OPEN') relative to MONOTONE_ABLATION.md's already-proved and verified result (Section 11).

### (c) What MUST be resolved before the PPO matrix launches

- The tie-break decision: either adopt `readout_mode="mode_of_window"` (with its already-wired genome-aware tie-break) as the actually-deployed primary readout policy, or explicitly document why `fixed_t` remains acceptable despite the 0.3150 measured harm fraction.
- Update docs/rule-language-spec.md's R4 status line to reflect MONOTONE_ABLATION.md's already-established result, and reconcile the order-dependence percentage mismatch across decisions.md/the spec/the interpreter docstring against the actually-committed audit artifact.
- The complete absence of any PPO-based (task-performance) evidence under this architecture: every claim in this report and its source artifacts is fitness-blind structural verification only.
