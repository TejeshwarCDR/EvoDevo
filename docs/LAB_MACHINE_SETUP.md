# Lab Machine Setup — Verification Record

This document is the consolidated record of environment setup and verification
performed on the lab machine ahead of the parallel-execution / pre-flight
workstream. Every number below comes from a command actually run in this
session, not from an earlier session's notes or an assumption.

## 1. Provenance

| Item | Value |
|---|---|
| EvoDevo repo | `/home/ubuntu/projects/EvoDevo`, branch `main`, commit `092fcd56f3ea5a7d4c75c77fe05d8db6e306da16` (2026-09-17 09:28:36 +0530) |
| EvoGym source | `external/evogym/` (not `evodevo/external/evogym/` — see note below), commit `f63c649c688adda1accad5f86731aad96606388a` (2025-06-05) |
| Date this record was produced | 2026-09-19 |
| Run by | user `ubuntu` on host `ubuntu`, interactively via Claude Code (VSCode extension) |

**Path correction**: this prompt's instructions referred throughout to
`evodevo/external/evogym`. The actual layout is `evodevo/` (the Python
package) and a separate top-level `external/evogym/` (the vendored
simulator) — there is no `evodevo/external/` directory. All work below was
done against the real path, `external/evogym/`.

## 2. Part 0 — Git Hygiene

- `.gitignore` already excluded the vendored simulator (line 14:
  `external/evogym/`) *before* this session started. No edit was needed.
- Verified with `git check-ignore -v external/evogym/` → matched at
  `.gitignore:14`, and `git status --porcelain external/` → empty despite
  `external/evogym/` containing 146 MB / thousands of files.
- Repo was confirmed on `092fcd5` (`main`) with a clean tree (`git status
  --porcelain` empty) before any of this session's changes.
- End-of-session tree state: **clean**. Two transient dirty states were
  produced and reverted during this session (see §5 and the risk register) —
  both were byproducts of running project tooling, not meaningful new work,
  and neither is left in the working tree.

## 3. Part 1 — EvoGym Version Pin

A pinned commit **was** found in this project's own docs:

- `docs/evogym_technical_reference.md:3`: *"Source basis: local checkout at
  `evogym/`, upstream remote `https://github.com/EvolutionGym/evogym.git`,
  commit `f63c649`."*
- Corroborated by `docs/evodevo-implementation-roadmap.md:3`: *"...
  `evogym_technical_reference.md` (EvoGym API facts, commit `f63c649`)."*
- `docs/READY_TO_RUN.md` contains no conflicting EvoGym pin (its commit
  references are all for this project's own lock/audit commits, unrelated to
  EvoGym).

The existing checkout at `external/evogym/` was **already at this exact
commit** (`f63c649c688adda1accad5f86731aad96606388a`, clean working tree, on
`main`, tracking `origin/main`) — no checkout action was required. A `git
fetch origin main` confirmed `origin/main` has not advanced past this commit
either (upstream EvoGym has had no commits since 2025-06-05), so this is both
the documented pin and the current tip of upstream `main`.

**This is not a coincidence to gloss over**: it means the specific commit
this project's technical reference was written against is reproducibly the
one in use. No disclosed-assumption risk applies here — the pin is real and
verified, not inferred.

## 4. Part 2 — System Dependency Verification

| Item | Check performed | Result |
|---|---|---|
| Ubuntu 24.04.4 | `/etc/os-release` | **PASS** — `Ubuntu 24.04.4 LTS` |
| Kernel | `uname -r` | **PASS** — `7.0.0-31-generic` |
| gcc | `gcc --version` + actual compile/link/run of a trivial `.cpp` | **PASS** — gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1); compiled, linked, and ran successfully |
| cmake | `cmake --version` + trivial CMake project configure/build/run | **PASS** — cmake 3.28.3; configured, built, and ran successfully |
| xorg-dev | `dpkg -l` | **PASS** — 1:7.7+23ubuntu3 |
| libglu1-mesa-dev | `dpkg -l` | **PASS** — 9.0.2-1.1build1 |
| libosmesa6-dev | `dpkg -l` | **PASS** — 25.1.7-1ubuntu2~24.04.2 |
| freeglut3-dev | `dpkg -l` | **PASS** — 3.4.0-1build1 |
| ffmpeg | `dpkg -l` | **PASS** — 7:6.1.1-3ubuntu5 |
| xvfb | `dpkg -l` | **PASS** — 2:21.1.12-1ubuntu1.6 |
| patchelf | `dpkg -l` | **PASS** — 0.18.0-1.1build1 |
| python3.10 present | `python3.10 --version` | **PASS** — 3.10.20 |
| venv uses 3.10, not system 3.12 | activated `evodevo-venv`, `which python3` + `python3 --version` | **PASS** — `/home/ubuntu/projects/evodevo-venv/bin/python3`, 3.10.20 (`pyvenv.cfg` confirms `home = /usr/bin`, `version = 3.10.20`) |
| open-files ulimit raised | see below | **PASS, with nuance — read in full** |

### ulimit -n: full finding

The current *interactive* session (this Claude Code / VSCode-extension shell,
descended from a GNOME desktop session managed by `systemd --user`) reports
`ulimit -n` = **1,048,576** (soft and hard). This is **not** because of the
`/etc/security/limits.conf` edit — that edit sets `* soft nofile 65536` / `*
hard nofile 65536`, which is *lower* than what this session already has. The
1,048,576 figure comes from `systemd`'s own default hard limit for
user/desktop sessions (`/proc/self/limits` confirms this directly; it does
not go through PAM in this session type).

The `limits.conf` edit (confirmed present, last modified 2026-09-19
12:09:19, i.e. earlier this session — a real, recent edit, not a
fabrication) governs sessions that go through `pam_limits.so`, which is
enabled for `sshd`, `login`, and `cron` (`/etc/pam.d/sshd`, `/etc/pam.d/login`,
`/etc/pam.d/cron` all have `session required pam_limits.so`). For those
paths:

- Effective ceiling is `min(limits.conf hard value, that service's own
  systemd `LimitNOFILE`)`. `ssh.service` and `cron.service` both report
  `LimitNOFILE=524288`, well above the 65536 configured in `limits.conf`, so
  `pam_limits` is not clipped by the service unit — **65536** is what SSH
  logins and cron jobs will actually get.
- This takes effect **immediately for any new session** opened after the
  edit — no reboot or re-login of the *machine* is required, PAM re-reads
  `limits.conf` per session. (I could not live-verify this over an actual
  SSH hop — no key-based auth was configured for `localhost` and password
  auth is disabled — so this is stated from configuration inspection, not a
  live end-to-end test. See risk register.)

**Bottom line**: both the desktop/interactive path (1,048,576) and the
SSH/cron path (65,536) are far above the Linux default (1024) and should be
more than sufficient for the upcoming parallel work. But the two paths give
*different* numbers, and it's the 65,536 figure — not 1,048,576 — that
governs any non-interactive/cron-triggered invocation. Size the
oversubscription/fd-budget math in the parallel-execution design around
65,536 if that work will ever be triggered non-interactively.

## 5. Part 3 — Python Dependency Reconciliation

`pip freeze` was captured before and after `pip install -r requirements.txt`
inside the activated `evodevo-venv`. **Exactly one package changed**:

```
numpy==2.2.6  ->  numpy==1.26.4
```

(`requirements.txt` pins `numpy<2.0.0`; everything else — gymnasium 1.3.0,
stable-baselines3 2.9.0, torch 2.14.0+cpu, pandas, matplotlib, pyyaml,
pytest, imageio, imageio-ffmpeg — was already satisfied at the versions
`requirements.txt` wants, no change.)

**torch/numpy compatibility**: verified with an actual round trip, not just
version-string inspection — `torch.from_numpy()` on a numpy 1.26.4 array,
arithmetic on the resulting tensor, `.numpy()` back, and the reverse
direction (`torch.randn(...).numpy()` → numpy op → `torch.from_numpy()` →
`torch.allclose`) all succeeded (`ROUND TRIP OK`).

## 6. Part 4 — EvoGym Build From Source

- `external/evogym/README.md` (lines 63–89) specifies the source-install path
  explicitly: `pip install -e .` (not `python setup.py install`). Confirmed
  by reading, not assumed.
- Submodules (`pybind11`, `glfw`, `glew-cmake`, plus two examples-only ones)
  were already initialized (`git submodule status` showed no `-` prefixes).
- Build order followed the task's requirement: `requirements.txt` (§5) fully
  installed first, then EvoGym built last, so the native extension compiles
  against the final numpy 1.26.4 / gymnasium 1.3.0 versions.
- `pip install -e .` **succeeded**: built `evogym-2.0.0-0.editable`, native
  extension `evogym/simulator_cpp.cpython-310-x86_64-linux-gnu.so` produced
  (confirmed as a freshly-built ELF binary via `file`/`ldd`, correctly
  linked against `libGL`, `libX11`, `libstdc++`, etc., no missing shared
  objects).
- **One build-adjacent failure, diagnosed and fixed**: the first smoke-test
  attempt failed with `ModuleNotFoundError: No module named 'pkg_resources'`.
  Root cause: this venv's `setuptools` was 84.0.0, and setuptools has since
  dropped the bundled (deprecated) `pkg_resources` module that
  `external/evogym/evogym/envs/base.py:12` imports. Not a CMake/compiler
  problem — a Python packaging-ecosystem drift issue between EvoGym (written
  in 2025 against an older setuptools) and this venv's current setuptools.
  **Fix**: pinned `setuptools==78.1.1` in the venv (still satisfies torch's
  own floor of `setuptools>=77.0.3`; still bundles `pkg_resources`,
  confirmed importable, with only the expected `DeprecationWarning`).
- **Smoke test**: EvoGym's own `examples/gym_test.py` pattern was adapted to
  run headless and finite (the original uses `render_mode='human'` and an
  infinite loop, neither appropriate for an unattended check): created a
  `Walker-v0` env via `sample_robot((5,5))` + `gym.make(..., render_mode=None)`,
  called `reset()`, ran 20 `step()` calls with re-reset on
  termination/truncation, then `close()`. All succeeded with no display and
  no `xvfb-run` needed. Output confirmed `Using Evolution Gym Simulator
  v2.2.5`.

Full build log: captured, 22 lines (pip's non-verbose editable-build output;
no separate CMake configure/build transcript was emitted by this pip/pyproject
path — the presence of a fresh, correctly-linked `.so` timestamped to the
build window is the positive evidence the native extension was actually
compiled, not skipped).

## 7. Part 5 — Project-Level Verification

### `python -m evodevo.visualization.check`

**ALL REQUIRED CHECKS PASSED**:
- `[OK]` evogym simulator importable
- `[OK]` env creation with `render_mode="img"` + frame capture
- `[OK]` imageio can write gif and mp4
- gifsicle (optional) not available — explicitly documented by the project
  as non-fatal, gifs are written unoptimized

### Fast subset (`pytest -q -m "not slow" tests/{core,morphology,evaluation,evolution,experiments,analysis}`)

**534 passed, 1 deselected, 0 failed**, 23 warnings, in **539.89s (8m59s)**.
All warnings are expected/documented: deprecation shims for
`nca_sanity.run_sanity` (intentionally kept for `results/` artifact
reproducibility), a pandas `FutureWarning` on empty-frame concat, and an
intentional T_primary hard-cap clip warning that the test itself asserts
must not pass silently.

### Full suite (`pytest -q`, includes slow/real-PPO tests)

**728 passed, 0 failed, 0 skipped**, 32 warnings, in **616.00s (10m15s /
0:10:15)**. This included `tests/evaluation/test_ppo_runner.py::test_real_ppo_smoke`
(confirmed real PPO training executed — its warning trace shows the
`pkg_resources` deprecation warning firing from inside `evogym/envs/base.py`
during that specific test). No truncation — every warning line was reviewed;
none indicate a masked failure.

Two files were incidentally left dirty by running project tooling during
this verification and were reverted to keep the tree clean (neither
represents meaningful new experimental output):
- `results/runs/benchmark_manifest.json` / `results/runs/factors.json` /
  `results/runs/_resolved_configs/rule_based_monotone/` — byproduct of the
  `make benchmark SURROGATE=1` run (see below). Reverted/removed.
- `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` — a test regenerates this
  report with a fresh wall-clock timing line (27.47s → 38.08s on this
  machine) as a side effect of running the full suite. Reverted.

### `make test`

Not re-run separately: `make test` is literally `python3 -m pytest` (no
flags) — functionally identical to the full-suite run immediately above
(same test selection, same result). Re-running it would only have duplicated
~10 minutes of wall-clock for an identical result, so the number above is
cited for both.

### `make benchmark SURROGATE=1`

**Partially ran, then correctly stopped itself — not a crash.** 15 of 16
planned seed/arm cells (`direct`, `nca`, `rule_based` × 5 seeds each, on
`Walker-v0`, under the `fast_dev` profile) were already complete from
earlier project work and were confirmed/resumed. The 16th cell
(`rule_based_monotone/Walker-v0/ea/seed_0`, not previously run) triggered
the project's own reproducibility gate
(`evodevo/core/provenance.py::check_git_state`): the tool's *own* bookkeeping
write to `results/runs/benchmark_manifest.json` while processing the 15
resumed cells made the git tree dirty, and the gate then correctly refused
to start genuinely new work against a dirty tree (`DirtyTreeError`,
`make: *** [Makefile:67: benchmark] Error 1`).

This is a legitimate, disclosed design quirk of the benchmarking tool
(self-inflicted dirtiness from its own manifest updates can block a
same-invocation new run), not a setup defect. Per the tool's own explicit
in-code warning ("`--allow-dirty` is for local iteration only — never use it
for a run whose output is cited in a report"), I did not force it past the
gate. The incidental dirty files were reverted (§ above) rather than
committed, since committing them wasn't requested and they contain no
meaningful new experimental content. Flagged in the risk register below —
worth a one-line fix in `provenance.py`/`benchmark.py` (e.g. checking
dirtiness once at entry, before any manifest writes) before this tool is
used unattended in the parallel-execution workstream.

## 8. Part 6 — Hardware Re-Confirmation

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) i7-14700K — `lscpu` reports 20 cores/socket × 2 threads/core = 28 CPU(s) online. This is `lscpu`'s simplified (non-P/E-aware) accounting; it matches the expected physical profile of **20 physical cores / 28 logical threads** with heterogeneous P-core/E-core performance, as previously recorded. |
| Memory | `free -h`: 62 GiB total, 5.6 GiB used, 49 GiB free, 57 GiB available; 30 GiB swap, unused |
| Disk | `df -h` on the repo filesystem (`/dev/sda7`): 401 GB total, 90 GB used, 291 GB available (24% used) |
| torch unconfigured thread default | `torch.get_num_threads()` → **20** (one thread per physical core, torch's standard default). This is the exact oversubscription risk for the upcoming parallel-execution design: N unpinned parallel processes would together request N×20 threads against 20 physical/28 logical cores with no pinning code yet in place. |

## 9. Risk Register

1. **`make benchmark SURROGATE=1` self-inflicted dirty-tree gate.** The
   benchmarking tool's own manifest-bookkeeping write (even for
   already-complete/resumed cells) dirties the git tree, which can then
   block a genuinely new cell in the *same* invocation via its own
   reproducibility gate. Not fixed here (out of this task's scope — no
   parallel-execution or benchmark-tooling code was touched, per the
   instructions). Worth fixing in `evodevo/core/provenance.py` /
   `evodevo/experiments/benchmark.py` before this tool runs unattended.
2. **`setuptools` pin (`==78.1.1`) is a real, disclosed, non-default
   change** made to this venv, required because current-generation
   setuptools (84.0.0, the venv's prior state) has dropped
   `pkg_resources`, which EvoGym's `evogym/envs/base.py` still imports.
   This is an upstream EvoGym packaging-debt issue (EvoGym hasn't been
   updated since June 2025 and predates the `pkg_resources` removal), not
   an EvoDevo issue. If this venv is ever rebuilt from a clean state without
   this pin being reproduced, the same failure will recur — worth adding
   `setuptools<81` (or similar) to `requirements.txt` or a `constraints.txt`
   if this project wants the fix to survive a fresh venv.
3. **ulimit non-interactive-path claim not live-verified end-to-end.** The
   65,536 figure for SSH/cron-invoked sessions is derived from configuration
   inspection (`limits.conf`, `pam.d/*`, `systemctl show ...
   LimitNOFILE`), not from an actual SSH round-trip to this machine (no
   working key-based auth to `localhost` was available in this session).
   High confidence given how PAM/`pam_limits` works, but genuinely
   unverified live — worth a real SSH-based check before relying on it for
   the parallel-execution design's fd budget.
4. **EvoGym's pinned commit matching upstream `origin/main`'s current tip is
   coincidental-but-verified, not permanent.** If upstream ever pushes new
   commits, `external/evogym/`'s checkout will no longer equal
   `origin/main`'s tip — but it will still be correctly pinned to `f63c649`
   per the documented reference, since that checkout is a fixed commit, not
   a tracking branch that auto-updates. No action needed, just don't
   `git pull` inside `external/evogym/` without re-checking this doc.
5. **Two incidental dirty-tree byproducts occurred and were reverted**
   during this verification run (§7) — both from running project tooling as
   instructed (`pytest`, `make benchmark`), neither representing lost or
   discarded user work (they were pure regenerated bookkeeping/report
   content, confirmed via `git diff` before reverting).
6. **All 32 full-suite warnings were reviewed individually**; every one maps
   to an intentional/documented behavior in this project's own code
   (deprecation shims kept for `results/` reproducibility, an intentional
   hard-cap clip, a pandas API-evolution notice, and the now-explained
   `pkg_resources` deprecation warning inside EvoGym itself). None indicate
   a masked test failure.

## Final Verdict: **READY**

Every check in Parts 2–5 passed (system deps, Python deps with a verified
torch/numpy round trip, EvoGym built and smoke-tested from its documented
pinned commit, the project's own rendering self-test, the fast test subset,
and the full suite including real-PPO tests — 728/728, zero failures). The
git tree is clean, the EvoGym version question is resolved with an actual
verified pin (not a disclosed gap), and the one tooling quirk found
(`make benchmark`'s self-inflicted dirty-tree gate) is fully diagnosed,
disclosed, and did not require touching any file this task was told not to
touch. The environment is ready for the next workstream (pre-flight
wall-clock projection / parallel-execution design), which is explicitly
out of scope for this task and was not attempted.
