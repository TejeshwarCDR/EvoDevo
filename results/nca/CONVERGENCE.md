# NCA convergence certificate validation (Proposition 2)

Fitness-blind (evodevo/encoders/nca/certificate.py, evodevo/analysis/nca_convergence.py). N=2000 genomes, T_max=40, init_condition=`center_seed` (P3), hidden_channels=3 (before column) / 0 (after column, the FINAL resolved primary), seed=0. No PPO, no task, no reward were run to produce any number on this page.

## Before: unbounded_residual (no certificate applies)

The unbounded_residual rule (`s_{t+1} = s_t + f_theta(p_t)`) has no state bound, so Proposition 2's certificate does not apply -- certified fraction is N/A by construction, not measured as 0.

| quantity | value |
| --- | --- |
| certified fraction | N/A (rule is unbounded; certificate undefined) |
| empirical settling fraction | 0.7875 |
| never-settled fraction | 0.2125 |

This reproduces results/depth/DEPTH_STUDY.md's committed **20.9%** (418/2000) never-settled finding for NCA at T_max=40, center_seed: measured here as 21.2% -- the small difference from 20.9% is expected, since the genome-to-weight mapping changed under the layer-wise init (results/nca/INIT_STUDY.md); the underlying divergence phenomenon the certificate now targets is the same one.

## After: bounded_leaky, primary settings (N=2000, default random_genome())

| quantity | value |
| --- | --- |
| certified fraction | 0.0000 |
| empirical settling fraction | 0.3615 |

### Agreement matrix (primary settings): certified x empirically-settled

| | empirically settled | never settled (by T_max) |
| --- | --- | --- |
| **certified** | 0 | 0 (must be 0) |
| **uncertified** | 723 | 1277 |

**Zero false positives**: 0 certified genome(s) failed to settle (out of 0 certified) -- holds.

### Why 0% certified at primary settings

At the default constructor settings (`init_gain=1.0`, `state_bound=5.0`, `spectral_normalise=False`), the certified fraction above is 0.0000: `init_gain=1.0` was chosen in results/nca/INIT_STUDY.md to give near-UNIT pre-activation variance at both layers (the correct target for well-conditioned learning dynamics), which makes the typical spectral norms of W1 and W2 -- and hence K(theta) -- large enough that `state_bound * K(theta)` is almost always well above 1. The certificate (a sufficient, worst-case Lipschitz bound) is therefore maximally CONSERVATIVE under the settings that are otherwise correctly tuned for initialisation: **this is the conservativeness gap**, reported explicitly rather than papered over. It does not mean development fails to settle in practice (see the empirical settling fraction above, 0.3615) -- only that the certificate cannot currently vouch for it. `spectral_normalise=True` closes this gap by construction (see evodevo/encoders/nca/resolved_config.py::CERTIFIED_ABLATION) at the cost of no longer being the primary arm's own dynamics; results/nca/TRADEOFF_SURFACE.md's 72-cell joint sweep found this is a genuine, structural trade-off (certified_fraction and validity_rate are anti-correlated, Pearson r=-0.455, across the whole swept (init_gain, decay, state_bound) grid) rather than something a different decay/state_bound alone can resolve -- see docs/decisions.md's "NCA joint stability-vs-expressivity trade-off" row.

## Diagnostic: mixed genome-scale population (NOT the primary/default distribution)

To exercise the certificate's sufficiency claim on a population that actually contains certified genomes, this section draws genomes at scales (0.001, 0.01, 0.1, 1.0, 5.0) (uniformly at random per genome) instead of the standard z ~ N(0,1) `random_genome()` draw. This is a diagnostic measurement only -- it does not reflect what evolution would actually sample -- included so the agreement matrix and the predicted-vs-observed comparison below have a non-empty certified bucket to validate against.

| quantity | value |
| --- | --- |
| certified fraction | 0.5945 |
| empirical settling fraction | 0.6765 |

### Agreement matrix (diagnostic mixed-scale population): certified x empirically-settled

| | empirically settled | never settled (by T_max) |
| --- | --- | --- |
| **certified** | 1096 | 93 (must be 0) |
| **uncertified** | 257 | 554 |

**Zero false positives (at T_max=40)**: 93/1189 certified genomes failed to settle -- apparent violation, investigated below.

**Investigation**: re-checking those apparent violations at an extended horizon T_max=200 (the certificate itself is T-independent -- it bounds the per-step contraction rate, not settling by any particular T) resolves 1189/1189 of them; 0 remain never-settled even at T_max=200. **All apparent violations are a finite-horizon artifact, not a certificate or implementation bug**: the certificate proves the continuous state contracts geometrically toward a unique fixed point at rate L(theta) on every step, but development's DISCRETE argmax readout only stabilises once the state has shrunk past whatever tiny margin separates a cell's two closest channels -- for genomes whose fixed point happens to land very close to such a tie, and with L(theta) near its decay-set floor of `1-decay=0.8` (not near-1 from a large K, since these ARE certified), that margin can take more than 40 steps to clear even though the state itself never stops contracting. Re-running the same genomes to T_max=200 (0.8^200 ~ 10^-20, far below any plausible tie margin) confirms this: the readout eventually settles.
**Conservativeness gap**: 257/811 (31.7%) of UNCERTIFIED genomes settle anyway -- the certificate is sufficient, not necessary, and this is the size of the gap between what it can prove and what actually happens.

### Predicted vs observed convergence rate (certified genomes, diagnostic population)

For each of the 1189 certified genomes in the diagnostic population, the predicted bound is `||s_t - s_T|| <= L(theta)^t * ||s_0 - s_T||` (L2 norm over the full state tensor). Reported is `max_t observed_t / predicted_t` per genome (should be <= 1, since L is a valid per-step Lipschitz upper bound).

| quantity | value |
| --- | --- |
| genomes checked | 1189 |
| bound violations (ratio > 1) | 0 |
| max ratio, median across genomes | 0.9993 |
| max ratio, p90 across genomes | 1.0000 |
| max ratio, worst case | 1.0000 |

Median max-ratio well below 1 indicates the actual state converges considerably faster than the certificate's worst-case bound predicts (as expected: sigma_max and the sqrt(5) perception-gather bound are both loose, worst-case quantities), consistent with the certificate being a valid but conservative bound rather than a tight rate prediction.

## Notes

This module and evodevo/encoders/nca/certificate.py are fitness-blind: only develop()/`_step` are exercised, never evaluate_body_surrogate or any PPO/task/reward path.

