"""python -m evodevo.extension -- the ONE entry point for extension tooling.

Import order is load-bearing, exactly as in evodevo/campaign/__main__.py:
thread limits are pinned before numpy/torch/evogym can be imported, and all
real logic is imported lazily inside main(). Under the `spawn` start method
CPython does not re-import a package's __main__ in workers, so PPO workers
started by the preflight pool only ever run their own initializer.

Commands:
  preflight bodies|curves|noise|status  run the extension preflight (lab host)
  derive                               apply the pre-registered rules and write
                                       the decision record, estimate and YAML
  check                                verify committed decision artifacts
  amend2 derive|check                  Amendment 2: derive / verify the production decision
                                       record (A2_DECISIONS.*, final_estimate_a2.json, campaign_a2.yaml)
"""
from evodevo.parallel.threading_control import configure_threading

_THREADING_REPORT = configure_threading(1, strict=True)

import sys  # noqa: E402

USAGE = __doc__


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "preflight":
        from evodevo.extension.preflight import main as preflight_main
        return preflight_main(rest)
    if command == "amend2":
        from evodevo.extension.amend2 import main as amend2_main
        return amend2_main(rest)
    if command in {"derive", "check"}:
        from evodevo.extension.derive import main as derive_main
        return derive_main([command, *rest])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
