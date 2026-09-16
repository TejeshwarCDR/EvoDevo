# NCA Architecture Readiness Report

Verification-only pass over the entire N1-N8 NCA-architecture-redesign workstream. Nothing in this report was fixed, tuned, or re-run to look better -- every number either comes from an existing committed artifact or a cheap, read-only check performed at report-generation time. If a check failed or an artifact was stale/missing, that is reported as a finding below, not silently worked around.

## Section 0 -- Provenance

- git commit: `2c09895a6bc87c950826b2e9eeafb9d344a34a7b`
- working tree clean: NO -- dirty files: docs/decisions.md, evodevo/analysis/depth_study.py, evodevo/analysis/mutation_calibration.py, evodevo/analysis/structural.py, evodevo/analysis/structural_report.py, evodevo/encoders/nca/encoder.py, results/depth/DEPTH_STUDY.md, results/depth/depth_study_raw.json, results/depth/figs/displacement_vs_T.png, results/depth/figs/diversity_vs_T.png, results/depth/figs/light_cone_vs_T.png, results/depth/figs/redundancy_vs_T.png, results/depth/figs/robustness_vs_T.png, results/depth/figs/settling_vs_T.png, results/depth/figs/validity_rate_vs_T.png, results/depth/raw/layer1/nca/light_cone.jsonl, results/depth/raw/layer3/nca/T1/nca/diversity.jsonl, results/depth/raw/layer3/nca/T1/nca/light_cone.jsonl, results/depth/raw/layer3/nca/T1/nca/perturbation_robustness.jsonl, results/depth/raw/layer3/nca/T1/nca/redundancy.jsonl
- report generated: 2026-09-15T23:42:57.161985+00:00
- package versions: python=3.10.20, platform=macOS-26.5.1-arm64-arm-64bit, numpy=1.26.4, matplotlib=3.10.9, pytest=9.1.1

### Resolved NCA configuration

| parameter | value | resolved by | source |
| --- | --- | --- | --- |
| update_rule | bounded_leaky | N1 | `evodevo/encoders/nca/encoder.py (default)` |
| state_bound (B*) | 5.0000 | N4/Outcome-B | `evodevo/encoders/nca/resolved_config.py::PRIMARY.state_bound` |
| decay (lambda*) | 0.2000 | N4/Outcome-B | `evodevo/encoders/nca/resolved_config.py::PRIMARY.decay` |
| init_gain | 1.0000 | N2 | `evodevo/encoders/nca/encoder.py (default; N2 established the layer-wise-scaled unpacking this gain feeds)` |
| hidden_channels (H) | 0 | N5/Part-5 | `evodevo/encoders/nca/resolved_config.py::PRIMARY.hidden_channels` |
| hidden_width | 16 | pre-N1 | `evodevo/encoders/nca/encoder.py (default; not modified by N1-N8)` |
| boundary | zero_pad | N1 | `evodevo/encoders/nca/encoder.py (default)` |
| update_prob | 1.0000 | N1 | `evodevo/encoders/nca/encoder.py (default)` |
| spectral_normalise | False | N1 | `evodevo/encoders/nca/encoder.py (default)` |
| mutation_sigma | 0.0305 | N6 | `results/mutation/mutation_calibration_raw.json::calibrated.nca.param_value` |

genome_length() of this resolved configuration: 501 (computed by the encoder from hidden_width/hidden_channels, not a stored literal -- see Section 4).

## Section 1 -- Proposition 1: boundedness

Fresh, read-only check (reproduces tests/encoders/test_nca_bounded.py's own property test) at the resolved (lambda\*=0.2, B\*=5.0): N=500 genomes x T=100 steps x 3 boundary rules (zero_pad, replicate, toroidal) x H in [0, 3, 6] = 450000 total per-step checks.

- violations (max|s_t| > bound + 1e-9): **0**
- single largest observed violation margin: **-0.000000**
  (H=0, boundary=zero_pad, observed max|s_t|=5.000000, bound=5.000000)

**Verdict: boundedness is EMPIRICALLY UNVIOLATED.**

## Section 2 -- Proposition 2: contraction certificate

Fresh, read-only check at the ACTUAL resolved (lambda\*=0.2, B\*=5.0), N=2000, T_max=40, center_seed, primary random_genome() distribution -- see the staleness note below on why this is not simply read from results/nca/CONVERGENCE.md.

- certified fraction: **0.0000**
- empirically-settled fraction: **0.3615**
- conservativeness gap (empirically-settled minus certified): **0.3615**

| | empirically settled | never settled |
| --- | --- | --- |
| **certified** | 0 | 0 |
| **uncertified** | 723 | 1277 |

**False positives: 0/0. PASS.**

Lambda-independence (L(theta)<1 <=> state_bound*K(theta)<1, decay in {0.1,0.2,0.5,1.0}): 2000 checks, 0 violations -- **PASS**.

### Staleness note (see Section 9/10)

results/nca/CONVERGENCE.md (committed) reports a primary-settings certified fraction of 0.0000, matching the fresh number above -- its 'after' column fingerprint confirms it was generated at the CURRENT resolved_config.py::PRIMARY settings (decay=0.2, state_bound=5.0, H=0). Not stale.

## Section 3 -- Timescale calibration (N4)

Sweep grid: 24 (lambda, B) cells (read from results/nca/TIMESCALE_CALIBRATION.md).

| lambda | B | Wasserstein-1 | KS | never_settled_fraction | median settle_step | mean settle_step |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 2 | 30.3260 | 0.8635 | 0.4005 | 39.0000 | 36.8495 **<- selected** |
| 0.05 | 5 | 31.1055 | 0.8815 | 0.3860 | 39.0000 | 37.6690 |
| 0.05 | 10 | 31.3850 | 0.8790 | 0.4090 | 39.0000 | 37.9505 |
| 0.1 | 2 | 31.3930 | 0.8820 | 0.4645 | 39.0000 | 37.9575 |
| 0.1 | 5 | 31.9060 | 0.8790 | 0.5140 | 40.0000 | 38.4715 |
| 0.1 | 10 | 32.2235 | 0.8800 | 0.5440 | 40.0000 | 38.7890 |
| 0.15 | 2 | 31.5910 | 0.8805 | 0.4970 | 39.0000 | 38.1565 |
| 0.15 | 5 | 32.3020 | 0.8805 | 0.5900 | 40.0000 | 38.8675 |
| 0.15 | 10 | 32.7015 | 0.8820 | 0.6730 | 40.0000 | 39.2670 |
| 0.2 | 2 | 31.6335 | 0.8780 | 0.5565 | 40.0000 | 38.1990 |
| 0.2 | 5 | 32.5875 | 0.8815 | 0.6790 | 40.0000 | 39.1530 |
| 0.2 | 10 | 32.9080 | 0.8825 | 0.7630 | 40.0000 | 39.4735 |
| 0.3 | 2 | 31.5400 | 0.8770 | 0.6085 | 40.0000 | 38.1055 |
| 0.3 | 5 | 32.7405 | 0.8815 | 0.7785 | 40.0000 | 39.3060 |
| 0.3 | 10 | 33.0910 | 0.8825 | 0.8635 | 40.0000 | 39.6565 |
| 0.5 | 2 | 31.1500 | 0.8725 | 0.7105 | 40.0000 | 37.7155 |
| 0.5 | 5 | 32.8245 | 0.8795 | 0.8885 | 40.0000 | 39.3900 |
| 0.5 | 10 | 33.2210 | 0.8835 | 0.9475 | 40.0000 | 39.7865 |
| 0.7 | 2 | 31.1940 | 0.8730 | 0.7775 | 40.0000 | 37.7595 |
| 0.7 | 5 | 32.9520 | 0.8805 | 0.9380 | 40.0000 | 39.5175 |
| 0.7 | 10 | 33.3495 | 0.8835 | 0.9835 | 40.0000 | 39.9150 |
| 1.0 | 2 | 32.6625 | 0.8770 | 0.9435 | 40.0000 | 39.2280 |
| 1.0 | 5 | 33.3585 | 0.8830 | 0.9950 | 40.0000 | 39.9240 |
| 1.0 | 10 | 33.4155 | 0.8840 | 0.9980 | 40.0000 | 39.9810 |

**Selected: lambda\* = 0.05, B\* = 2** (Wasserstein-1 = 30.3260, KS = 0.8635).

Held-out verification:

| quantity | sweep | verification |
| --- | --- | --- |
| Wasserstein-1 | 30.3260 | 30.4860 |
| KS | 0.8635 | 0.8635 |
| never_settled_fraction | 0.4005 | 0.4125 |
| median settle_step | 39.0000 | 39.0000 |
| mean settle_step | 36.8495 | 37.0095 |

Settling-within-N-steps, NCA @ (lambda*, B*) vs rule_based:

| settle within <= N steps | rule_based | nca (sweep) | nca (verification) |
| --- | --- | --- | --- |
| 2 | 0.6285 | 0.0210 | 0.0210 |
| 5 | 0.8615 | 0.0210 | 0.0210 |
| 10 | 0.8830 | 0.0210 | 0.0210 |
| 20 | 0.8845 | 0.0350 | 0.0285 |

**Residual gap (verbatim from the committed report)**: **Negative result on the median**: at (lambda\*, B\*), NCA median settle_step=39.0000 vs rule_based=2.0000 -- a gap of 37.0000 steps, WORSE than (not an improvement on) the original 35-step baseline gap. Every cell in the swept lambda x B grid (lambda in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0], B in [2, 5, 10]) has a median settle_step of 39-40, at or near T_max=40: within this grid, decay/state_bound tuning does not bring NCA's median settling speed anywhere close to rule_based's. lambda* and B* were still selected on Wasserstein-1 (the least-bad cell in the grid, not a successful match) since a selection criterion is required regardless of whether the calibration goal was reached.

## Section 4 -- Genome space and initialisation (N2)

Pre-activation variance, old (single-scale) vs new (layer-wise), from results/nca/INIT_STUDY.md:

| init | layer | fan_in | pre-activation variance | target |
| --- | --- | --- | --- | --- |
| old (single-scale, init_scale=0.5) | 1 | 40 | 10.1454 | 1.0 |
| old (single-scale, init_scale=0.5) | 2 | 16 | 4.2616 | 1.0 |
| new (layer-wise, init_gain=1.0) | 1 | 40 | 0.9993 | 1.0 |
| new (layer-wise, init_gain=1.0) | 2 | 16 | 1.0144 | 1.0 |

Body degeneracy/diversity, old vs new init:

| init | valid_rate | degenerate_fraction | unique_body_fraction |
| --- | --- | --- | --- |
| old (single-scale) | 0.6955 | 0.0150 | 0.9875 |
| new (layer-wise) | 0.7005 | 0.0270 | 0.9740 |

genome_length() confirmation: analytic formula (5*(N_MATERIALS+H))*h + h + h*(N_MATERIALS+H) + (N_MATERIALS+H) at H=0, h=16 = **501**; the resolved encoder's own `genome_length()` = **501** -- MATCH. Computed by the encoder from its constructor arguments, not stored as a literal (evodevo/encoders/nca/encoder.py::__init__).

## Section 5 -- Readout and hidden channels (N5)

Near-tie fraction by H, from results/nca/READOUT_STUDY.md:

| H | mean margin | median margin | fraction of CELLS near-tie | fraction of BODIES with >=1 near-tie cell |
| --- | --- | --- | --- | --- |
| 0 | 1.5005 | 1.1443 | 0.0057 | 0.1300 |
| 3 | 1.5158 | 1.1574 | 0.0052 | 0.1190 |
| 6 | 1.5259 | 1.1678 | 0.0055 | 0.1270 |

Uniform-body fraction by H (with material breakdown):

| H | uniform_fraction | by material |
| --- | --- | --- |
| 0 | 0.0095 | 0:0.0050, 1:0.0015, 2:0.0025, 3:0.0005, 4:0.0000 |
| 3 | 0.0070 | 0:0.0040, 1:0.0000, 2:0.0010, 3:0.0010, 4:0.0010 |
| 6 | 0.0045 | 0:0.0020, 1:0.0015, 2:0.0005, 3:0.0005, 4:0.0000 |

Hidden-channel utilisation, H=3 and H=6:

| H | channel | spatial variance | temporal variance | correlation with readout |
| --- | --- | --- | --- | --- |
| 3 | 0 | 5.8593 | 2.2453 | -0.0121 |
| 3 | 1 | 5.5858 | 2.2185 | -0.0073 |
| 3 | 2 | 5.8160 | 2.2435 | 0.0046 |
| 6 | 0 | 5.6226 | 2.1966 | -0.0055 |
| 6 | 1 | 5.8127 | 2.2328 | 0.0046 |
| 6 | 2 | 5.7744 | 2.2246 | -0.0007 |
| 6 | 3 | 5.6723 | 2.2266 | 0.0162 |
| 6 | 4 | 5.7571 | 2.2373 | -0.0044 |
| 6 | 5 | 5.9236 | 2.2630 | -0.0052 |

**Verdict (verbatim)**: **The hidden channels do not measurably affect the encoding's structural properties.** Every structural metric with at least one side above the 0.03 absolute floor (excluding tiny-baseline metrics like degenerate_fraction, where a 43.75%-looking 'relative gap' is really just a 0.35-percentage-point absolute difference and not a meaningful signal -- shown in the table above regardless, nothing hidden) differs by less than 10% (largest such gap: 6.85%), and hidden_utilisation's own per-channel correlation with the final material readout is small for every channel at both H=3 and H=6 (table above). This is reported as the finding, not softened: three (or six) hidden channels appear to be carried through development without materially changing validity, diversity, settling behaviour, or decision confidence, at least at these settings -- a legitimate, publishable result, and grounds to consider dropping them rather than carrying unexamined channels forward.

## Section 6 -- Mutation re-calibration (N6)

**Outcome (a): NCA's displacement curve now reaches into the originally recommended 2.0-3.0 band.** nca's grid floor (`mutation_sigma`=0.01) now produces a mean displacement of 0.9065 cells -- above the previous re-run's floor of 0.2235 cells at the same nominal `mutation_sigma`=0.01 (the comparison point cited by OLD_* in this module is the MOST RECENT prior calibration, not necessarily the original pre-N1 unbounded architecture -- see the 'Comparison to the original' section below for which run that actually is). D* is RE-SELECTED within the recommended band and ALL THREE ARMS are re-calibrated to it: **Chosen D* = 2.5000** (was 2.5000).

Resolved D\* = **2.5000**

| arm | parameter | solved value | verification mean | verification IQR | within tolerance |
| --- | --- | --- | --- | --- | --- |
| direct | mutation_rate | 0.1535 | 2.5030 | 2.0000 | YES |
| nca | mutation_sigma | 0.0305 | 2.6885 | 3.0000 | YES |
| rule_based | mutation_rate | 0.0781 | 2.6705 | 1.0000 | YES |

**IQR spread before/after (verbatim)**: **Cross-arm IQR spread**: old=2.0000 cells (IQRs direct=2.0000, nca=3.0000, rule_based=1.0000), new=2.0000 cells (IQRs direct=2.0000, nca=3.0000, rule_based=1.0000).

## Section 7 -- Regeneration (N7)

Certified fraction under the PRIMARY genome distribution: 0/18000 (0.00%) -- consistent with results/nca/CONVERGENCE.md's finding that `init_gain=1.0` (calibrated for near-unit pre-activation variance, not for a small Lipschitz constant) makes essentially no randomly-drawn genome satisfy the certificate. **The theory therefore makes no recovery guarantee for the genome population evolution would actually sample at these settings** -- the numbers below describe empirical (uncertified) behaviour only, not a test of Corollary 3 (that test needs a population that actually contains certified genomes -- see the DIAGNOSTIC section below).

PRIMARY-settings recovery curve: 10 (k, d) cells measured (see results/nca/REGENERATION.md for the full table).
DIAGNOSTIC-population (certified/uncertified split): 19 rows.
Rigorous continuous-state geometric-bound check: 10 (k, d) cells.

**Theory verdict: HOLDS.** 0/10809 certified-genome trials violated the predicted geometric bound (observed state distance after recovery never exceeded L(theta)^(T-k) times the post-damage distance). Corollary 3's core claim -- that certified genomes recover from bounded mid-trajectory damage at (at least) the guaranteed contraction rate -- is confirmed, not merely illustrated.

## Section 8 -- Before/after delta (N8)

Full table (results/nca/ARCHITECTURE_DELTA.md):

| quantity | OLD | NEW | direction |
| --- | --- | --- | --- |
| never_settled_fraction | 0.2090 | 0.6385 | UP |
| median settle_step | 37.0000 | 39.0000 | UP |
| q95 settle_step | 39.0000 | 39.0000 | unchanged |
| light_cone propagation speed (cells/step) | 0.0194 | 0.0248 | UP |
| max state magnitude at T (mean) | 185.5189 | 4.6847 | DOWN |
| max state magnitude at T (median) | 182.1530 | 4.7699 | DOWN |
| validity rate | 0.6540 | 0.6770 | UP |
| unique-valid-body fraction | 0.6665 | 0.6770 | UP |
| uniform-body fraction | 0.0140 | 0.0045 | DOWN |
| mutation displacement (mean Hamming, mutation_sigma=0.02) | 6.5145 | 3.6680 | DOWN |
| readout near-tie fraction (margin<0.01) | 0.0001 | 0.0055 | UP |
| certified-contractive fraction | N/A | 0.0000 | N/A |
| T_primary implied | 39 | 5 | DOWN |

**Regression check (results/structural/STRUCTURAL.md)**: direct/rule_based rows unchanged from the last committed version -- PASS.
**Regression check (results/depth/DEPTH_STUDY.md)**: DRIFT DETECTED -- BLOCKER. 1 arm(s) changed:
  - rule_based: was `['| rule_based | 0.2548 | 0.9592 | 1.0000 | 1.0000 | 4.0000 | 0.5000 | 1.0000 | 1.0000 |']`, now `['| rule_based | 0.2548 | 0.9592 | 1.0000 | 1.0000 | 4.0000 | 0.5000 | 1.0000 | 1.0000 |']`

New T_primary: **5** (old: 39).
New light-cone speed: **0.0248** cells/step vs rule_based's unchanged 0.2548 (old NCA: 0.0194).

## Section 9 -- Self-consistency

| check | result |
| --- | --- |
| Section 0 config values match a fresh independent re-derivation | PASS |
| Section 8 T_primary matches results/depth/depth_study_raw.json | PASS |
| No hardcoded literal (decay=0.05 / state_bound=2 / hidden_channels=3) in nca_readiness_check.py itself | PASS |

grep pattern used: `decay\s*=\s*0\.05|state_bound\s*=\s*2\b|hidden_channels\s*=\s*3\b`

**Overall: ALL PASS**

## Section 10 -- Risk register (adversarial)

- Residual non-settling fraction under the resolved architecture is NOT zero (see Section 8): classified as PLATEAUED_OSCILLATING=86.8%, SLOW_BUT_MONOTONE=6.5%, STILL_GROWING=6.7% (n=1277). The dominant categories are tie-resolution/limit-cycle behaviour, not unbounded growth, but the fraction itself remains substantial.
- Residual timescale gap after N4 calibration (verbatim, results/nca/TIMESCALE_CALIBRATION.md): **Negative result on the median**: at (lambda\*, B\*), NCA median settle_step=39.0000 vs rule_based=2.0000 -- a gap of 37.0000 steps, WORSE than (not an improvement on) the original 35-step baseline gap. Every cell in the swept lambda x B grid (lambda in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0], B in [2, 5, 10]) has a median settle_step of 39-40, at or near T_max=40: within this grid, decay/state_bound tuning does not bring NCA's median settling speed anywhere close to rule_based's. lambda* and B* were still selected on Wasserstein-1 (the least-bad cell in the grid, not a successful match) since a selection criterion is required regardless of whether the calibration goal was reached.
- Residual mutation IQR mismatch across arms after N6 re-calibration: **Cross-arm IQR spread**: old=2.0000 cells (IQRs direct=2.0000, nca=3.0000, rule_based=1.0000), new=2.0000 cells (IQRs direct=2.0000, nca=3.0000, rule_based=1.0000).
- H (hidden channels) is NOT load-bearing at the resolved settings (results/nca/READOUT_STUDY.md verdict, Section 5): three (H=3) or six (H=6) hidden channels are carried through development without measurably affecting validity, diversity, settling, or decision confidence -- an unexamined design cost, not yet removed.
- No PPO-based (task-performance) result exists anywhere under this redesigned architecture: every number in this report and in N1-N8's own artifacts is fitness-blind (decode()/develop()/mutate()/validate_body only). Whether the redesign's mixed structural picture (Section 8) translates to better, worse, or unchanged task performance is completely unknown.
- BLOCKER: regression check found direct/rule_based drift -- see Section 8.

## Section 11 -- Verdict

### Demonstrably fixed (before/after cited)

- Unbounded state growth: max state magnitude at T 185.5189 -> 4.6847 (Section 8), and empirically zero boundedness violations over 450000 checks (Section 1).
- Propagation speed increased 0.0194 -> 0.0248 cells/step (Section 8).
- Regeneration/self-repair is now a provable, zero-violation-verified property for certified genomes (Section 7): **Theory verdict: HOLDS.** 0/10809 certified-genome trials violated the predicted geometric bound (observed state distance after recovery never exceeded L(theta)^(T-k) times the post-damage distance). Corollary 3's core claim -- that certified genomes recover from bounded mid-trajectory damage at (at least) the guaranteed contraction rate -- is confirmed, not merely illustrated.

### Not fixed, or newly introduced

- never_settled_fraction 0.2090 -> 0.6385 (UP, not down as hypothesised -- Section 8).
- validity rate 0.6540 -> 0.6770 (DOWN -- Section 8).
- T_primary unchanged at 5 -- did not drop as hoped (Section 8).
- Hidden channels remain structurally unexamined/not load-bearing (Section 5, Section 10).
- Certificate essentially never certifies a genome at the primary random_genome() distribution (certified fraction=0.0000 -- Section 2).

### Must be resolved before the PPO matrix is launched

- results/nca/CONVERGENCE.md must be regenerated at the N4-resolved (lambda*, B*) -- it is currently stale (Section 2/10).
- No PPO-based task-performance evidence exists yet under this architecture (Section 10) -- the entire mixed structural picture in Section 8 is unvalidated against the metric that ultimately matters.
- Decide whether to drop the hidden channels (H=3) given Section 5's finding that they are not load-bearing, or run a task-performance check before concluding they are safe to remove.
