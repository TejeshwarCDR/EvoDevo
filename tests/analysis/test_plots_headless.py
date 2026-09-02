import os
import subprocess
import sys
import tempfile
import textwrap


def test_plots_module_imports_and_saves_a_figure_with_no_display():
    """Regression guard for the "works only by accident" bug: analysis/plots.py
    used to rely on nca_sanity.py's import-time `matplotlib.use("Agg")` side
    effect rather than setting its own backend. Importing plots.py directly,
    in a subprocess with DISPLAY unset, must still be able to save a figure."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "fig.png")
        script = textwrap.dedent(
            f"""
            import pandas as pd
            from evodevo.analysis.plots import plot_fitness_curves
            import matplotlib.pyplot as plt

            df = pd.DataFrame(
                {{
                    "encoder": ["direct", "direct", "nca", "nca"],
                    "seed": [0, 0, 0, 0],
                    "eval_index": [0, 1, 0, 1],
                    "fitness": [0.1, 0.2, 0.3, 0.4],
                }}
            )
            fig, ax = plt.subplots()
            plot_fitness_curves(df, ax=ax)
            fig.savefig({out_path!r})
            """
        )
        env = dict(os.environ)
        env.pop("DISPLAY", None)
        env["MPLBACKEND"] = ""  # ensure our own matplotlib.use("Agg") is what wins
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0


def test_ablation_report_module_also_forces_agg_backend():
    """analysis/ablation_report.py must not depend on plots.py (or anything
    else) having already set the backend -- every rendering module sets its
    own."""
    script = textwrap.dedent(
        """
        import os
        os.environ.pop("DISPLAY", None)
        import matplotlib
        import evodevo.analysis.ablation_report  # noqa: F401
        assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()
        print("OK")
        """
    )
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
