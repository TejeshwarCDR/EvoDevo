# Final consolidated before/after table: both arms, every structural metric used throughout this workstream

This is the table the paper's comparison section is built from. It consolidates results already independently
measured and committed across this workstream -- it does not re-derive them, to avoid a second,
possibly-inconsistent re-implementation of any single methodology. Citations are given per row group.

**Superseded-value notice**: an earlier version of this table cited NCA's shared T_primary as 40 and left
validity_rate/unique_valid_body_fraction unmeasured under windowed readout for both arms. Both are now
corrected: (1) T_primary=40 for NCA was internally inconsistent with its own resolved W=50 (a 50-step
window cannot exist within a 40-step trajectory -- `windowed_readout.py` silently clips rather than
raising), corrected to **T_primary=100** (see docs/decisions.md's "NCA T_primary/W consistency fix" row);
(2) rule_based's even-W tie-break was found to discard a legitimately-valid phase for 30.5% of its cycling
population and was replaced with a genome-aware tie-break (`validity_aware=True`, see docs/decisions.md's
"rule_based tie-break validity check" row) -- the validity numbers below reflect this fix; (3) rule_based's
own T-shift-stability figure was mislabeled as "0.0055 at T=100" when 0.0055 was actually its T=40 own-optimum
value, and the T=100 measurement itself was corrupted by a trajectory-length clipping artifact -- both are
now fixed and clearly distinguished (see the "Readout-policy T-shift stability" section's own correction
note below).

Four columns are compared:
- **pre-redesign NCA**: `update_rule="unbounded_residual"`, pre-N2 raw-reshape genome (results/nca/ARCHITECTURE_DELTA.md's OLD column)
- **final NCA**: `bounded_leaky` (decay=0.2, state_bound=5.0, init_gain=1.0), N2's layer-wise-scaled genome, H=0,
  readout_mode=mode_of_window (W=50, `validity_aware=False` -- unchanged from N13) -- `evodevo/encoders/nca/resolved_config.py::PRIMARY`
- **pre-audit rule_based**: the rule vocabulary/interpreter/genome representation, UNCHANGED throughout this
  entire workstream (the settling audit and this unification never modified them) -- fixed_t readout, the
  historical default
- **final rule_based**: SAME rule vocabulary/interpreter/genome representation (untouched), readout_mode
  STILL DEFAULTS to fixed_t (per the "fixed_t default preserved" requirement) -- mode_of_window (W=6,
  `validity_aware=True`) now available and selectable, and IS the readout actually reported in the
  "windowed" sub-columns below. Since the underlying rule_based encoder/interpreter never changed,
  "pre-audit" and "final" are IDENTICAL for every metric below except where a windowed-readout value is
  shown explicitly -- this is stated plainly rather than fabricating a spurious difference.

All windowed-readout measurements below are taken at the CORRECTED shared T_primary=100
(results/T_PRIMARY_UNIFIED.md, post-consistency-fix).

## Settling / validity / structural metrics

| quantity | pre-redesign NCA | final NCA (fixed_t, T=40) | final NCA (mode_of_window, W=50, T=100) | pre-audit == final rule_based (fixed_t, T=40) | final rule_based (mode_of_window, W=6, T=100, validity_aware) |
| --- | --- | --- | --- | --- | --- |
| never_settled_fraction | 0.2090 | 0.6385 | n/a (readout does not change whether the underlying trajectory settles) | 0.1155 | n/a (same reason) |
| median settle_step | 37.0000 | 39.0000 | n/a | 2.0000 | n/a |
| q95 settle_step (SUPERSEDED statistic -- see below) | 39.0000 | 39.0000 | n/a | 5.0000 | n/a |
| light_cone propagation speed (cells/step) | 0.0194 | 0.0248 | n/a | (not separately measured for rule_based in this workstream) | n/a |
| max state magnitude at T (mean) | 185.5189 | 4.6847 | n/a | n/a (discrete symbolic state, no continuous magnitude) | n/a |
| **valid_rate (N=2000, seed=0, T_primary=100)** | 0.6540 | 0.6790 | **0.6705** | 0.5695 | **0.6005** |
| **unique_valid_body_fraction (N=2000, seed=0, T_primary=100)** | 0.6665 | 0.6755 | **0.6660** | 0.3875 | **0.4105** |
| material_entropy_bits (N=2000, seed=0, T_primary=100) | (not measured) | 2.3209 | 2.3193 | 2.3032 | 2.3076 |
| mutation displacement (mean Hamming, matched calibration point) | 6.5145 (sigma=0.02, un-recalibrated) | 3.6680 (sigma=0.02, un-recalibrated); 5.1220 (N6-recalibrated, sigma=0.0201, fixed_t) | 5.7075 (readout-recalibrated, sigma=0.0140) -- NOT within original D* tolerance, see results/nca/MUTATION_CALIBRATION_READOUT_CASCADE.md | 5.3100 (N6-committed, mutation_rate=0.1707, fixed_t) | 5.3280 (SAME mutation_rate=0.1707, mode_of_window W=6) -- delta=0.018 cells, WITHIN tolerance, no re-solve needed (results/rule_based/MUTATION_DISPLACEMENT_CHECK.md; measured before the validity_aware tie-break fix, unaffected since it only changes WHICH tied body is returned, not the displacement-sampling procedure) |

**Validity findings**: NCA's windowed readout is very slightly less valid than fixed_t (0.6705 vs 0.6790,
-0.0085) -- a small, expected cost of trading a point-in-time snapshot for a stability-oriented vote across
a still-partly-unsettled population, not a regression worth reverting for. rule_based's windowed readout
with the validity-aware tie-break is MORE valid than fixed_t (0.6005 vs 0.5695, +0.0310) and more diverse
(0.4105 vs 0.3875 unique-valid-body fraction) -- the tie-break fix does not merely avoid harm, it nets a
real improvement over the historical fixed_t baseline, because it recovers a valid phase whenever the
genome's cycle contains one, on top of gaining T-shift stability.

## Readout-policy T-shift stability (the operational metric this whole redesign optimises)

| quantity | NCA fixed_t (T=1000, old pragmatic cap) | NCA mode_of_window (T=100, W=50) | rule_based fixed_t (T=39/40) | rule_based mode_of_window, AS ACTUALLY DEPLOYED (T=100, W=6) | rule_based mode_of_window, its OWN per-arm optimum (T=40, W=6) |
| --- | --- | --- | --- | --- | --- |
| fraction_changed under +-5 T-shift | 0.7020-0.7305 | 0.1705-0.1735 | 0.1150-0.1155 | 0.0070 / 0.0035 | 0.0055 / 0.0040 |
| cyclic/never-settling population this readout must handle | ~44.9% chaotic-like + ~1.65% period(10,100] (results/nca/DYNAMICS_CLASSIFICATION.md) | (same population, now readout-stabilised) | 11.55% genuine rule-cycles, 99.6% exact periods, dominant period 2 (results/rule_based/SETTLING_AUDIT.md) | (same population, now readout-stabilised, ~16x more stable than fixed_t) | (same population; slightly better than the deployed figure, since T=40 is rule_based's own smaller optimum) |

**Correction (2026-09-16, "T=40-staleness audit")**: an earlier version of this table cited rule_based's
mode_of_window T-shift figure as "0.0055 at T=100, W=6" -- this was WRONG on two counts, both now fixed: (1)
0.0055 is rule_based's OWN per-arm optimum, measured at T=40, not T=100; the actual figure AT the shared
T_primary=100 (where rule_based is actually deployed in the real benchmark) is 0.0070/0.0035, freshly
measured, not 0.0055. (2) The ORIGINAL T=100 measurement itself was corrupted by a trajectory-length ceiling
artifact (`RuleBasedEncoder`'s `HARD_MAX_ITERATIONS` was 100, so a T=100+5=105 T-shift query got silently
clipped back to T=100 itself, making shift=+1/+5 spuriously read 0.0 -- not genuine stability). Both are
fixed: `HARD_MAX_ITERATIONS` raised to 105 (same safety-ceiling-only precedent as its two prior raises), and
the sweep re-run; both columns above are correct and distinct from each other. Full trace in
`results/T_PRIMARY_UNIFIED.md`'s "Final shared T_primary" section.

**Headline**: mode_of_window reduces T-shift sensitivity by ~4-5x for NCA and ~16-21x for rule_based (at its
deployed T=100 vs its own T=40 optimum respectively), relative to each arm's own fixed_t readout at a
comparable T (NCA's reduction is smaller than the previously-reported ~9x because the earlier "0.1055 at
T=40, W=50" figure was invalid -- W=50 exceeds T=40, so it was never measuring a genuine 50-step window; the
corrected, consistent figure is 0.1735 at T=100, W=50). rule_based's absolute stability under mode_of_window
(0.0055-0.0070) remains far better than NCA's (0.1705-0.1735) because rule_based's residual non-settling
population is a SMALL (11.55%), SHORT-PERIOD (dominant period 2) genuinely-cyclic population that a small
window fully absorbs, whereas NCA's residual is a LARGE (~45-64% depending on T_max), effectively-chaotic
population that no window size can fully suppress (results/nca/DYNAMICS_CLASSIFICATION.md's corrected
sensitivity-probe finding). The validity-aware tie-break fix does not change these T-shift-stability numbers
(verified: it is T-shift stable by construction, same as the original tie-break, since it depends only on
the genome's fixed cycle content) -- only WHICH body is returned changed, not whether the choice is stable
under a shift.

## T_primary derivation: old (superseded) vs new

| quantity | OLD (q95-based, censored, ill-posed) | NEW (T-shift-stability criterion, corrected) |
| --- | --- | --- |
| NCA | 39 (results/depth/DEPTH_STUDY.md, `_select_t_primary`, now labelled SUPERSEDED) | 100 (results/nca/READOUT_POLICY_SELECTION.md, W<=T_primary constraint enforced) |
| rule_based | 39 (never separately checked against its own censoring) | 40 (results/rule_based/READOUT_POLICY_SELECTION.md) |
| shared T_primary (`max` over arms) | 39 | **100** (results/T_PRIMARY_UNIFIED.md) |

See `results/T_PRIMARY_UNIFIED.md` for the full derivation and `docs/decisions.md`'s "unified T_primary
(T-shift-stability criterion, both arms)" and "NCA T_primary/W consistency fix" rows for the decision
record.

## direct: explicitly verified unaffected

`direct`'s encoder (`evodevo/encoders/direct/encoder.py`) and its calibration
(`results/mutation/mutation_calibration_raw.json`'s `direct` entry, `mutation_rate=0.3057`) are untouched by
this entire workstream -- `git status` confirms no changes to `evodevo/encoders/direct/` at any point across
this workstream, and `direct` never appears in `depth_study.py`'s `ARMS` list or either arm's readout-policy
modules.
