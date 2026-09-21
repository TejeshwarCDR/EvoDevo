"""Explicit, deliberate thread-count control for the parallel campaign.

CONTEXT (why this module exists): PyTorch, NumPy's BLAS backend (OpenBLAS or
MKL) and NumExpr all default to spawning one internal thread per available
logical CPU. That is the right default for a single process running alone.
It is badly wrong for this project's execution model, where N worker
processes each run one PPO evaluation concurrently: N workers x N internal
threads each contends for the same physical cores, and the failure mode is
never a crash -- it is silent slowdown that looks like "the machine is just
slow today" unless thread counts are made explicit up front.

The one thing that makes this hard: OpenMP/MKL/OpenBLAS all read their
thread-count environment variables ONCE, at the C/C++ runtime's own
initialization time, which happens as a side effect of the *first* `import
numpy` or `import torch` in the process. Setting `OMP_NUM_THREADS` after
that import is a no-op for OpenMP's own thread pool (torch's Python-level
`torch.set_num_threads()` is a separate, narrower knob that does NOT
reach OpenBLAS/MKL's own internal pools). `configure_threading()` therefore
refuses (in `strict` mode) to proceed if numpy or torch is already imported
-- that refusal is the entire reason this module exists, not an
inconvenience to work around.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

# Numeric thread-count variables: each set to str(n_threads).
_NUMERIC_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Dynamic-adjustment flags: each pinned to "FALSE" so the runtime cannot
# silently override the fixed count above under its own load heuristics.
_DYNAMIC_FLAG_VARS = (
    "MKL_DYNAMIC",
    "OMP_DYNAMIC",
)

# The complete set of variables this module pins. Order matches the numeric
# thread-count vars first, then the dynamic-adjustment flags, per the spec.
THREAD_ENV_VARS = _NUMERIC_THREAD_VARS + _DYNAMIC_FLAG_VARS

# Modules whose presence in sys.modules means the env vars below are too
# late to affect that library's own internal (OpenMP/MKL) thread pool.
_IMPORT_ORDER_SENSITIVE_MODULES = ("numpy", "torch")


class ThreadingConfiguredTooLateError(RuntimeError):
    """Raised by configure_threading(strict=True) when numpy or torch has
    already been imported: the thread-count env vars this function sets
    cannot retroactively change that library's already-initialized OpenMP/
    MKL thread pool, so proceeding would silently produce the exact
    contention this module exists to prevent."""


def _env_value_for(var: str, n_threads: int) -> str:
    if var in _DYNAMIC_FLAG_VARS:
        return "FALSE"
    return str(n_threads)


@dataclass(frozen=True)
class ThreadingReport:
    """Record of one configure_threading() call, for logging/provenance."""

    n_threads: int
    strict: bool
    env_before: dict[str, Optional[str]] = field(default_factory=dict)
    env_after: dict[str, str] = field(default_factory=dict)
    torch_importable: bool = False
    torch_num_threads: Optional[int] = None
    guard_passed: bool = True
    already_imported: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_threads": self.n_threads,
            "strict": self.strict,
            "env_before": dict(self.env_before),
            "env_after": dict(self.env_after),
            "torch_importable": self.torch_importable,
            "torch_num_threads": self.torch_num_threads,
            "guard_passed": self.guard_passed,
            "already_imported": list(self.already_imported),
        }


def configure_threading(n_threads: int = 1, strict: bool = True) -> ThreadingReport:
    """Pin every variable in THREAD_ENV_VARS to n_threads (or "FALSE" for
    the *_DYNAMIC flags), then best-effort-align torch's own Python-level
    thread settings if torch happens to already be imported.

    Call this ONCE, as early as possible in a worker process -- before any
    `import numpy` or `import torch`, including indirect ones (e.g. `import
    stable_baselines3`) -- so the env vars are in place before OpenMP/MKL
    read them at library-load time. See the module docstring for why.

    strict=True (the default) makes that ordering requirement a hard
    failure rather than a silent no-op: if numpy or torch is already
    present in sys.modules, this raises ThreadingConfiguredTooLateError
    instead of setting env vars that can no longer do their job. Do not
    catch this and retry with strict=False as a matter of course -- fix the
    import order instead. strict=False exists for callers (e.g. a REPL, a
    notebook, or a test) that knowingly accept the reduced effect.
    """
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")

    already_imported = tuple(m for m in _IMPORT_ORDER_SENSITIVE_MODULES if m in sys.modules)
    guard_passed = not already_imported

    if strict and not guard_passed:
        raise ThreadingConfiguredTooLateError(
            f"configure_threading(strict=True) was called after {', '.join(already_imported)} "
            f"had already been imported. The env vars in THREAD_ENV_VARS "
            f"({', '.join(THREAD_ENV_VARS)}) are read once, at that library's own OpenMP/MKL "
            f"initialization time, which already happened -- setting them now cannot retroactively "
            f"change its internal thread pool. Call configure_threading() before importing numpy or "
            f"torch (directly or indirectly, e.g. via stable_baselines3), or pass strict=False if you "
            f"understand and accept that this call will only affect libraries imported after it."
        )

    env_before: dict[str, Optional[str]] = {var: os.environ.get(var) for var in THREAD_ENV_VARS}
    env_after: dict[str, str] = {}
    for var in THREAD_ENV_VARS:
        value = _env_value_for(var, n_threads)
        os.environ[var] = value
        env_after[var] = value

    torch_importable = importlib.util.find_spec("torch") is not None
    torch_num_threads: Optional[int] = None
    if "torch" in sys.modules:
        import torch  # noqa: PLC0415 -- intentionally lazy; see module docstring/MUST NOT rule.

        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            # torch raises if interop threads are set after parallel work
            # has already begun in this process -- best-effort only, the
            # intra-op thread count above is still applied.
            pass
        torch_num_threads = torch.get_num_threads()

    return ThreadingReport(
        n_threads=n_threads,
        strict=strict,
        env_before=env_before,
        env_after=env_after,
        torch_importable=torch_importable,
        torch_num_threads=torch_num_threads,
        guard_passed=guard_passed,
        already_imported=already_imported,
    )


def _numpy_backend_name(numpy_module: Any) -> Optional[str]:
    """Best-effort extraction of the BLAS backend name numpy was built
    against (e.g. "openblas64", "mkl"). Never raises -- returns None if the
    installed numpy's introspection API doesn't support this."""
    try:
        config = numpy_module.show_config(mode="dicts")
        return config["Build Dependencies"]["blas"]["name"]
    except Exception:
        return None


def describe_threading_environment() -> dict[str, Any]:
    """Read-only snapshot of the current threading configuration, for
    logging/provenance. Never imports numpy or torch itself (doing so would
    be a mutation of process state, and could itself trip up a LATER
    configure_threading(strict=True) call elsewhere in the same process) --
    if either is already imported, its current settings are reported; if
    not, the corresponding fields are None."""
    info: dict[str, Any] = {
        "env": {var: os.environ.get(var) for var in THREAD_ENV_VARS},
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "numpy_backend": None,
    }

    if "torch" in sys.modules:
        import torch  # noqa: PLC0415 -- lazy; torch is already loaded, this is a no-op import.

        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_num_interop_threads"] = torch.get_num_interop_threads()

    if "numpy" in sys.modules:
        import numpy  # noqa: PLC0415 -- lazy; numpy is already loaded, this is a no-op import.

        info["numpy_backend"] = _numpy_backend_name(numpy)

    return info
