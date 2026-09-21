"""python -m evodevo.preflight.report -- regenerates every figure and
report/PREFLIGHT_REPORT.md for a pre-flight run FROM STAGE JSON ONLY.

Never re-runs a measurement, never touches evodevo.parallel.worker_pool,
never imports numpy/torch/evogym itself (the per-stage figure functions
this module calls into do, lazily, exactly as they already do when called
from a live run -- see e.g. evodevo/preflight/stages/b2_sweep.py's own
_make_figure). Safe to run at any time, including against a run directory
that is still in progress or was killed partway through: every section is
built from whatever stages/*.json files exist right now, and D9 (report
must label every artifact with how complete the run behind it is) is
handled by reading manifest.json's own status per stage rather than
assuming everything finished.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from evodevo.preflight.run_dir import RunDir, RunDirNotFoundError, read_json
from evodevo.preflight.stages import load_manifest

_STAGE_ORDER = (
    "machine_profile",
    "concurrency_sweep",
    "evaluation_cost",
    "learning_curves",
    "evaluation_noise",
    "projection",
    "reliability",
)


def _stage_json(run_dir: RunDir, name: str) -> Optional[dict]:
    path = run_dir.stage_output_path(name)
    if not os.path.exists(path):
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _regenerate_b2_figure(run_dir: RunDir, data: dict) -> Optional[str]:
    from evodevo.preflight.stages.b2_sweep import _make_figure

    try:
        machine = read_json(run_dir.machine_path)
        detection_method = machine.get("topology", {}).get("detection_method", "unknown")
    except (OSError, ValueError):
        detection_method = "unknown"

    figure_path = os.path.join(run_dir.report_dir, "b2_concurrency.png")
    _make_figure(
        figure_path, data["levels"], data["boundaries"], data["ppo_timesteps"], detection_method,
        self_check_stamp=data.get("self_check_stamp"),
    )
    return figure_path


def _regenerate_b4_figure(run_dir: RunDir, data: dict) -> Optional[str]:
    from evodevo.preflight.stages.b4_learning_curves import _make_figure

    figure_path = os.path.join(run_dir.report_dir, "b4_learning_curves.png")
    _make_figure(figure_path, data["tasks"], data["knee_rule_text"], self_check_stamp=data.get("self_check_stamp"))
    return figure_path


def _regenerate_b5_figure(run_dir: RunDir, data: dict) -> Optional[str]:
    from evodevo.preflight.stages.b5_noise import _make_figure

    figure_path = os.path.join(run_dir.report_dir, "b5_noise.png")
    _make_figure(figure_path, data["tasks"], self_check_stamp=data.get("self_check_stamp"))
    return figure_path


def _regenerate_b6_artifacts(run_dir: RunDir, data: dict) -> dict:
    from evodevo.preflight.stages.b6_projection import _make_figure, _write_csv, _write_markdown

    md_path = os.path.join(run_dir.report_dir, "projection.md")
    csv_path = os.path.join(run_dir.report_dir, "projection.csv")
    figure_path = os.path.join(run_dir.report_dir, "b6_projection.png")

    rows = data.get("rows", [])
    baseline_row = rows[0] if rows else {}
    _write_markdown(
        md_path, rows, data.get("validity_sources", {}), data.get("overhead", {}),
        data.get("concurrency_used"), data.get("d14_deltas", []), baseline_row,
        self_check_stamp=data.get("self_check_stamp"),
    )
    _write_csv(csv_path, rows)

    cost_ranges = {}
    for key, value in data.get("cost_per_eval_ranges", {}).items():
        arm, _, task = key.partition("::")
        cost_ranges[(arm, task)] = value
    if cost_ranges:
        _make_figure(figure_path, cost_ranges, self_check_stamp=data.get("self_check_stamp"))
    else:
        figure_path = None

    return {"markdown": md_path, "csv": csv_path, "figure": figure_path}


def regenerate_figures(run_dir: RunDir) -> dict:
    """Regenerates every figure/markdown/csv artifact this package's stages
    produce, from each stage's own already-written JSON -- skipping any
    stage that has not produced output yet (no JSON on disk). Returns a
    dict of stage_name -> list of paths (re)written."""
    written: dict = {}

    b2 = _stage_json(run_dir, "concurrency_sweep")
    if b2 and b2.get("levels"):
        written["concurrency_sweep"] = [_regenerate_b2_figure(run_dir, b2)]

    b4 = _stage_json(run_dir, "learning_curves")
    if b4 and b4.get("tasks"):
        written["learning_curves"] = [_regenerate_b4_figure(run_dir, b4)]

    b5 = _stage_json(run_dir, "evaluation_noise")
    if b5 and b5.get("tasks"):
        written["evaluation_noise"] = [_regenerate_b5_figure(run_dir, b5)]

    b6 = _stage_json(run_dir, "projection")
    if b6 is not None:
        paths = _regenerate_b6_artifacts(run_dir, b6)
        written["projection"] = [p for p in paths.values() if p]

    return written


def _completion_fraction(manifest: dict) -> tuple:
    stages = manifest.get("stages", {})
    n_done = sum(1 for e in stages.values() if e.get("status") == "done")
    n_total = len(_STAGE_ORDER)
    return n_done, n_total


def generate_report_markdown(run_dir: RunDir) -> str:
    manifest = load_manifest(run_dir)
    n_done, n_total = _completion_fraction(manifest)

    lines = []
    lines.append(f"# EvoDevo Pre-Flight Report -- run `{run_dir.run_id}`")
    lines.append("")
    lines.append(
        f"**Completion: {n_done}/{n_total} stages done.** This report is regenerated from "
        f"`stages/*.json` only and is safe to run against a still-in-progress or interrupted "
        f"run (D9) -- every section below states its own stage's status explicitly rather than "
        f"assuming the whole run finished."
    )
    lines.append("")

    for name in _STAGE_ORDER:
        entry = manifest.get("stages", {}).get(name, {"status": "pending"})
        status = entry.get("status", "pending")
        data = _stage_json(run_dir, name)
        lines.append(f"## {name} -- status: `{status}`")
        lines.append("")
        if status != "done" or data is None:
            lines.append(
                f"_Not yet complete (status=`{status}`) -- no measured output to report. "
                f"This section will populate once this stage finishes and report.py is re-run._"
            )
            lines.append("")
            continue

        stamp = data.get("self_check_stamp")
        if stamp:
            lines.append(f"> **{stamp}** -- this stage's numbers below are from a self-check run, not a real measurement.")
            lines.append("")

        partial = data.get("partial")
        if partial:
            lines.append("**partial: true** -- this stage's own output reports incomplete coverage; see its JSON for detail.")
            lines.append("")

        lines.append(f"Duration: {entry.get('duration_s')}s. Raw output: `stages/{name}.json`.")
        report_paths = data.get("report_paths") or {}
        figure_path = data.get("figure_path") or report_paths.get("figure")
        if figure_path and os.path.exists(figure_path):
            rel = os.path.relpath(figure_path, run_dir.report_dir)
            lines.append("")
            lines.append(f"![{name}]({rel})")
        markdown_path = report_paths.get("markdown")
        if markdown_path and os.path.exists(markdown_path):
            rel = os.path.relpath(markdown_path, run_dir.report_dir)
            lines.append("")
            lines.append(f"See also: [{os.path.basename(markdown_path)}]({rel})")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_report(run_dir: RunDir) -> str:
    regenerate_figures(run_dir)
    markdown = generate_report_markdown(run_dir)
    out_path = os.path.join(run_dir.report_dir, "PREFLIGHT_REPORT.md")
    with open(out_path, "w") as f:
        f.write(markdown)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evodevo.preflight.report")
    parser.add_argument("--base-dir", default="results/preflight")
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dir = RunDir(os.path.join(args.base_dir, args.run_id))
    if not os.path.isdir(run_dir.path):
        print(f"error: run directory {run_dir.path!r} does not exist", file=sys.stderr)
        return 1
    if not os.path.exists(run_dir.manifest_path):
        raise RunDirNotFoundError(f"{run_dir.path!r} has no manifest.json")

    out_path = generate_report(run_dir)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
