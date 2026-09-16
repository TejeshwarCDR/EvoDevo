"""Single source of truth for NCA's resolved architecture configuration.

Before this module existed, the resolved (lambda*, B*, init_gain, H) values
were copied as independent local module-level constants in at least six
files (evodevo/analysis/{architecture_delta,regeneration,readout_study,
depth_study,structural_report,mutation_calibration}.py) -- flagged as a
live instance of the project's stale-literal-copy risk pattern in
results/NCA_READINESS.md Section 10. All six now import PRIMARY (and
CERTIFIED_ABLATION where relevant) from here instead of retyping the
numbers.

See docs/decisions.md's "NCA joint stability-vs-expressivity trade-off
(Outcome B)" row for the evidence and reasoning behind these values, and
results/nca/TRADEOFF_SURFACE.md for the full 72-cell grid this was decided
from.

PRIMARY: the settings the primary NCA arm actually uses. Proposition 1's
unconditional boundedness guarantee holds for PRIMARY regardless of the
contraction certificate (Proposition 2) being satisfied -- PRIMARY is
NOT expected to be certificate-contractive for most genomes, and its
no-divergence claim does not rely on the certificate at all.

CERTIFIED_ABLATION: PRIMARY's own decay/state_bound/init_gain, with
spectral_normalise=True added -- this is the one change that forces
(by construction) essentially every genome to satisfy the certificate,
without altering the developmental dynamics decay/state_bound/init_gain
otherwise control. This is a SEPARATE, explicitly labelled ablation
regime, used specifically to test Corollary 3 (see
results/nca/REGENERATION.md and results/nca/CONVERGENCE.md) -- Corollary
3's "HOLDS, 0 violations" result is a property of THIS population, not of
PRIMARY, and must never be cited as if it described the primary arm.

readout_mode/readout_window: the readout-policy resolution (see
results/nca/READOUT_POLICY_SELECTION.md and docs/decisions.md's "NCA
readout policy: fixed-T argmax vs mode-of-window" row) -- PRIMARY now
reads out via mode_of_window(W=50) rather than the older fixed-T argmax,
since the T-shift operational sensitivity test found mode_of_window
reduces sensitivity by roughly an order of magnitude at matched T_primary,
even though neither policy reaches a small (<2%) absolute sensitivity for
this population (a substantial genuinely-chaotic subfraction, see
results/nca/DYNAMICS_CLASSIFICATION.md, cannot be fully suppressed by any
window or horizon). The default field values (fixed_t, window=10) preserve
the OLD behavior for any config that does not explicitly opt in, matching
NCAEncoder's own constructor defaults.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NCAResolvedConfig:
    label: str
    update_rule: str
    init_gain: float
    decay: float
    state_bound: float
    hidden_channels: int
    spectral_normalise: bool = False
    mutation_sigma: float = None  # None where not yet re-solved for this config
    readout_mode: str = "fixed_t"
    readout_window: int = 10


PRIMARY = NCAResolvedConfig(
    label="primary",
    update_rule="bounded_leaky",
    init_gain=1.0,
    decay=0.2,
    state_bound=5.0,
    hidden_channels=0,
    spectral_normalise=False,
    mutation_sigma=None,  # solved fresh by the N6 re-run; read from
    # results/mutation/mutation_calibration_raw.json rather than duplicated
    # here, since a solved value is itself an artifact, not a design choice.
    readout_mode="mode_of_window",
    readout_window=50,
)

CERTIFIED_ABLATION = NCAResolvedConfig(
    label="certified_ablation",
    update_rule=PRIMARY.update_rule,
    init_gain=PRIMARY.init_gain,
    decay=PRIMARY.decay,
    state_bound=PRIMARY.state_bound,
    hidden_channels=PRIMARY.hidden_channels,
    spectral_normalise=True,
    mutation_sigma=None,
)
