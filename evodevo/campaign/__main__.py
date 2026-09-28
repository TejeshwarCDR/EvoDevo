"""python -m evodevo.campaign -- the ONE entry point for the campaign scheduler.

IMPORT ORDER IS LOAD-BEARING -- READ BEFORE TOUCHING THIS FILE'S IMPORTS.
(Same pattern as evodevo/preflight/__main__.py.)

configure_threading(1, strict=True), just below, MUST run before numpy,
torch, evogym or stable_baselines3 is imported anywhere in this process.
OMP_NUM_THREADS/MKL_NUM_THREADS/etc. are read ONCE, by OpenMP/MKL's own C
initialization, at the library's first import -- see
evodevo/parallel/threading_control.py. The only thing imported before the
call is threading_control itself (numpy/torch-free), and the packages it
lives in (evodevo, evodevo.parallel) plus THIS package's __init__
(evodevo.campaign), all of which are kept numpy/torch-free and guarded by
tests/campaign/test_entry_point.py.

Why this module exists: the scheduler used to be launched as
`python -m evodevo.campaign.scheduler`. Under the `spawn` start method the
worker pool uses (evodevo/parallel/worker_pool.py), every worker re-imports
the parent's main module as __mp_main__ before its initializer runs; that
re-import dragged in evodevo.campaign/__init__ -> plan -> benchmark ->
numpy, so every worker's configure_threading(strict=True) raised and the
pool died with BrokenProcessPool. Here, ALL real logic lives in main(),
which imports the scheduler lazily and runs only under
`if __name__ == "__main__"`, so any process that re-imports this module
(CPython's spawn in fact skips re-importing a `pkg.__main__` main module
altogether) at most sets thread limits.

Never import this module from other code or in-process from tests; invoke
it as `python -m evodevo.campaign ...` in a fresh process.
"""
from evodevo.parallel.threading_control import configure_threading

_THREADING_REPORT = configure_threading(1, strict=True)

# Everything below is stdlib-only at module scope. Heavy modules are
# imported lazily inside main(), after the call above.
import sys  # noqa: E402

USAGE = """usage:
  python -m evodevo.campaign [scheduler options]        run / resume a campaign
  python -m evodevo.campaign status --campaign ID [--output-root DIR]
run `python -m evodevo.campaign --help` for scheduler options."""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--dry-run" in argv:
        sys.dont_write_bytecode = True  # a dry run writes NOTHING, not even __pycache__
    if argv and argv[0] == "status":
        from evodevo.campaign.dashboard import main as dashboard_main
        return dashboard_main(argv[1:])
    from evodevo.campaign.scheduler import cli_main
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
