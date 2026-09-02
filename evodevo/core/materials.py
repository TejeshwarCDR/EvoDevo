"""Single authoritative definition of the voxel-material vocabulary.

Every module that needs a material integer literal, the material count, or
material names for logging/plotting must import from here instead of
redefining any of this locally. Before this module existed, the vocabulary
was defined independently in at least three places (evodevo's validate.py
MATERIALS dict, direct encoder's N_MATERIALS, NCA encoder's N_MATERIALS) --
this file replaces all of those.

Deliberately excludes FIXED=5: EvoGym reserves material id 5 for terrain
voxels, not robot bodies. FIXED must never appear in, or be treated as
valid for, an evolved robot body produced by any encoder in this project.
"""
from __future__ import annotations

EMPTY = 0
RIGID = 1
SOFT = 2
H_ACT = 3
V_ACT = 4

N_MATERIALS = 5  # EMPTY, RIGID, SOFT, H_ACT, V_ACT

MATERIAL_NAMES = ("EMPTY", "RIGID", "SOFT", "H_ACT", "V_ACT")

ACTUATORS = frozenset({H_ACT, V_ACT})

# Backwards-compatible mapping form, kept for callers that prefer name -> id.
MATERIALS = {
    "EMPTY": EMPTY,
    "RIGID": RIGID,
    "SOFT": SOFT,
    "H_ACT": H_ACT,
    "V_ACT": V_ACT,
}
