"""Measured-capacity calendar for the thermal-validated production profile."""
from __future__ import annotations

# The benchmark measured 42.96 genomes/min for P8+E10 under the selected
# controls.  Applying that measured rate to the accepted 46.27 genomes/min /
# 16.8-day campaign model yields about 18.1 days.  These rounded values are
# deliberately not presented as a new Monte-Carlo confidence interval.
MEASURED_GENOMES_PER_MINUTE = 42.96
PROJECTED_TOTAL_H = 18.1 * 24
WALKER_FRACTION = 0.305
PROJECTED_WALKER_H = PROJECTED_TOTAL_H * WALKER_FRACTION
PROJECTED_UPSTEPPER_H = PROJECTED_TOTAL_H - PROJECTED_WALKER_H
SOURCE = "results/thermal_worker_study/FINAL_SELECTION_20260930.md"


def projection() -> dict:
    return {
        "source": SOURCE,
        "measured_genomes_per_minute": MEASURED_GENOMES_PER_MINUTE,
        "total_h": PROJECTED_TOTAL_H,
        "walker_h": PROJECTED_WALKER_H,
        "upstepper_h": PROJECTED_UPSTEPPER_H,
        "walker_fraction": WALKER_FRACTION,
    }
