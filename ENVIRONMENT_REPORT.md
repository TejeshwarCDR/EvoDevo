# Environment Report — Lab Machine Setup (Step 0)

Date: 2026-09-19
Repo commit at start of this session: `0a630cab8537fd42cdab1f8d25c9a8a6a31504cb` (main)
This report covers Step 0 only (interpreter, dependencies, system limits,
storage/accelerator decisions, parity gate). No work from Prompts 1-7 was
started, per this task's explicit instruction.

## 0. A discrepancy in the task brief, disclosed up front

This task's brief asserted this machine's "verified" starting state included
"no conda, no virtualenv, torch not installed" and `ulimit -n 1024`. Live
inspection at the start of this session found both claims stale relative to
actual machine state:

- A working venv already existed at `/home/ubuntu/projects/evodevo-venv`
  (outside this repo, at a sibling path) with `torch==2.14.0+cpu`,
  `numpy==1.26.4`, EvoGym built, and a recorded 728/728-passing full test
  suite — the product of a prior Claude Code session on this same machine,
  documented in `docs/LAB_MACHINE_SETUP.md` (committed at `23ade98`, which
  is an ancestor of the commit `0a630ca` this session started from —
  confirmed via `git merge-base --is-ancestor`).
- This interactive session's own `ulimit -n` was already 1,048,576 (systemd
  desktop-session default), not 1024 — also independently confirmed by that
  same prior document.

Neither the brief's claims nor the prior document's claims were taken on
trust: every fact below was re-verified live in this session. This session
built its own fresh `.venv/` at the exact path Step 0.1 specifies (not the
old sibling path) and re-ran the full parity gate against it, rather than
reusing the old venv directly — so the results below are this session's own
independent verification, not a copy of the prior session's report.

## 1. Interpreter decision (Step 0.1)

**Decided: Python 3.10.20**, not the system default (3.12.3). Full
rationale and evidence recorded in `docs/decisions.md` (2026-09-19,
"Interpreter version for the campaign `.venv/`, decided not defaulted").

Summary: no `pyproject.toml`/`setup.cfg`/`.python-version` exists in this
repo. Every committed `results/runs/*/run_meta.json` records
`library_versions.python == "3.10.20 (main, Mar 3 2026, 00:49:35) [Clang
21.0.0 ...]"` (the original macOS/Clang development machine), independently
corroborated by the prior session's venv on this machine also being built
against `/usr/bin/python3.10`. `/usr/bin/python3.10` was already present on
this machine (Ubuntu 24.04's default repos or a prior install — not
re-investigated further since it was already there), so no deadsnakes
PPA/pyenv/uv step was needed.

```
$ /usr/bin/python3.10 -m venv .venv
$ .venv/bin/python3 --version
Python 3.10.20
```

Note on `evodevo` itself: this repo has no `setup.py`/`pyproject.toml` for
the `evodevo` package — it is not pip-installed (editable or otherwise)
anywhere, including in the prior session's venv. It is imported via the
repo root being on `sys.path` (true for any invocation from the repo root,
including `python -m evodevo...` and `python -c "import evodevo"`). This
was verified, not assumed — see §5's acceptance-criterion command below.

## 2. Dependencies (Step 0.2)

Installed into `.venv/` from `requirements.txt` (all existing pins honoured,
including `numpy<2.0.0` and `setuptools<81`), plus `psutil` (for Prompt 3's
telemetry sampler, not yet implemented — this only pre-installs the
dependency).

Full frozen package list (`.venv/bin/pip list --format=freeze`):

```
cloudpickle==3.1.2
contourpy==1.3.2
cycler==0.12.1
evogym==2.0.0
exceptiongroup==1.3.1
Farama-Notifications==0.0.6
filelock==4.0.1
fonttools==4.65.0
fsspec==2026.9.0
gymnasium==1.3.0
ImageIO==2.37.4
imageio-ffmpeg==0.6.0
iniconfig==2.3.0
Jinja2==3.1.6
kiwisolver==1.5.1
MarkupSafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
networkx==3.4.2
numpy==1.26.4
packaging==26.3
pandas==2.3.3
pillow==12.3.0
pip==26.2.1
pluggy==1.6.0
psutil==7.2.2
Pygments==2.21.0
pyparsing==3.3.2
pytest==9.1.1
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.3
setuptools==80.10.2
six==1.17.0
stable_baselines3==2.9.0
sympy==1.14.0
tomli==2.4.1
torch==2.14.0+cpu
typing_extensions==4.16.0
tzdata==2026.4
wheel==0.48.0
```

**A real problem found and fixed, not just installed-and-moved-on**:
`stable-baselines3` pulls `torch` in transitively (no explicit `torch` line
in `requirements.txt`), and the plain `pip install` initially resolved the
default PyPI wheel, `torch==2.14.0+cu130` — pulling ~5.6GB of unused
`nvidia-*`/`triton`/`cuda-toolkit` packages and emitting a `CUDA
initialization: The NVIDIA driver on your system is too old` warning on
every `import torch` (a stale/partial bundled CUDA userspace library, no
driver present). This was caught by checking `torch.__version__` /
`torch.cuda.is_available()` after install rather than assuming correctness,
then fixed: reinstalled explicitly via
`pip install "torch==2.14.0+cpu" --index-url https://download.pytorch.org/whl/cpu`
and uninstalled the now-unused CUDA packages. `.venv/` shrank from 5.6GB to
**1.2GB**. See the accelerator decision in `docs/decisions.md` for the full
rationale (§6 below).

### EvoGym build

- Source: `external/evogym/`, commit `f63c649c688adda1accad5f86731aad96606388a`
  (2025-06-05), matching the pin documented in
  `docs/evogym_technical_reference.md:3` and independently confirmed as
  still being upstream `origin/main`'s tip.
- Submodules (`pybind11`, `glfw`, `glew-cmake`) already initialized.
- Built via `pip install -e .` from `external/evogym/` into `.venv/`.
  Produced a freshly-timestamped
  `evogym/simulator_cpp.cpython-310-x86_64-linux-gnu.so` (confirmed via
  `file`: ELF 64-bit, dynamically linked).
- Verified: `.venv/bin/python -c "import evogym.simulator_cpp"` succeeds.
  Simulator reports "Using Evolution Gym Simulator v2.2.5" / "Using GLEW
  2.2.0" on first use.
- All system build dependencies (`xorg-dev`, `libglu1-mesa-dev`,
  `libosmesa6-dev`, `freeglut3-dev`, `patchelf`, `xvfb`) were already
  present (`dpkg -l`, independently re-checked this session, not assumed
  from the prior document).

### Thermal monitoring (lm-sensors)

`coretemp` kernel module is already loaded (`lsmod | grep coretemp`), and
`libsensors5`/`libsensors-config` are already installed, but the `sensors`
CLI itself (package `lm-sensors`) is **not** installed, and this session
cannot install it: `sudo apt-get install -y lm-sensors` fails
non-interactively —

```
sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper
sudo: a password is required
```

— this session has no passwordless sudo and no askpass helper, so even with
your approval (given) to install it, the command must be run by you
directly in a real terminal:

```bash
sudo apt-get install -y lm-sensors
sensors   # should show coretemp immediately, no sensors-detect needed
```

**Until this is run, CPU package temperature is not readable — this is a
genuine D13 gap, disclosed rather than silently absent.**

## 3. System limits and power (Step 0.3)

- **Open-file limit**: hard limit is 1,048,576 (`ulimit -Hn`), far above the
  65,535 target. Added `ulimit -n 65535` (soft-limit change, no sudo needed)
  to all 13 launcher scripts under `scripts/`:
  `scripts/{benchmark/run_ablations,benchmark/render_results,benchmark/run_comparison,direct/run,direct/test,direct/run_baseline,direct/inspect,rule_based/run,rule_based/test,rule_based/inspect,nca/run,nca/test,nca/inspect}.sh`.
  All verified with `bash -n` after editing (no syntax errors introduced).
- **Core dumps**: `ulimit -c` is 0. Per your instruction, left disabled —
  not enabled unilaterally. To enable later if an EvoGym segfault needs
  post-mortem debugging: `ulimit -c unlimited` (soft-limit change, no sudo).
- **Sleep/suspend prevention**: checked and found **already satisfied**, no
  action needed or taken.
  - `systemctl status sleep.target suspend.target hibernate.target
    hybrid-sleep.target` → all four report `static` (not something a
    service can trigger unprompted).
  - GNOME's own idle-suspend is already off:
    `org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type` =
    `'nothing'`.
  - GNOME's own screen-blank is already off:
    `org.gnome.desktop.session idle-delay` = `0`.
  - `/etc/systemd/logind.conf` has no `HandleLidSwitch`/`IdleAction`
    override (defaults apply, no idle-triggered suspend via logind either).
  - Per your answer, no sudo commands (e.g. masking the sleep targets) were
    run, since there's nothing here for them to guard against.
- **Automatic updates / unattended-upgrades**:
  `unattended-upgrades.service` is active and enabled;
  `apt-daily-upgrade.timer` is scheduled to fire in the next several hours
  (checked via `systemctl list-timers`). `Unattended-Upgrade::Automatic-Reboot`
  is **not** set in `/etc/apt/apt.conf.d/50unattended-upgrades` (commented
  out / default false), so it will not reboot the machine — but it can still
  install security updates and (via `needrestart`, if installed) restart
  services mid-run. To defer for the duration of a campaign run (not run
  here — requires sudo, and no sleep/reboot risk was found that would make
  this urgent):
  ```bash
  sudo systemctl stop apt-daily-upgrade.timer apt-daily.timer
  sudo systemctl mask apt-daily-upgrade.timer apt-daily.timer   # to prevent re-arming
  # re-enable after the campaign:
  sudo systemctl unmask apt-daily-upgrade.timer apt-daily.timer
  sudo systemctl start apt-daily.timer
  ```

## 4. Storage decision — prepared, not taken (Step 0.4)

| Item | Value |
|---|---|
| Free space on `/` (`/dev/sda7`) | 291 GB available (401 GB total, 22% used) |
| `/dev/sdb1` | **not mounted** |
| `/dev/sdb1` size | 6,001,173,463,040 bytes ≈ 5.46 TiB (`lsblk -b`) |
| `/dev/sdb1` filesystem | `ntfs` (label "New Volume"), per `lsblk -f` |
| What mounting would require | The `ntfs-3g` driver is already installed (`ntfs-3g`, `mount.ntfs-3g` both present), so no new package is needed. Mounting is a single `sudo mount -t ntfs-3g /dev/sdb1 <mountpoint>` (or an `/etc/fstab` entry for persistence) — **not run**, since this is presented as a decision, not taken. NTFS also has known caveats for a Linux workload (no POSIX permissions/ownership by default, weaker crash-consistency under heavy concurrent small-file writes than ext4 — worth weighing against reformatting if this disk is dedicated to run output, which was not asked and not decided here). |

This is left as an open decision for you, to be taken against B1's measured
disk-rate and projected log volume (not yet measured — that's Prompt-3/B1
territory, out of Step 0's scope). Current `results/` on `/` is 154 MB, for
reference scale only.

## 5. Accelerator decision — recorded (Step 0.5)

**CPU-only**, driver not installed. Full rationale in `docs/decisions.md`
(2026-09-19, "Accelerator: CPU-only campaign"). Confirmed:

```
$ nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver...
$ .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.14.0+cpu False
```

No driver was installed; no CUDA code path exists in this codebase (grep
confirms zero references to `.cuda()`/`device="cuda"` anywhere in
`evodevo/`).

## 6. Parity gate results (Step 0.6)

All items **PASS**. Details:

**(a) Full test suite** — `.venv/bin/python -m pytest -q`:

```
744 passed, 33 warnings in 629.36s (0:10:29)
```

This is **not** the same raw number as the 728-passed baseline recorded in
`docs/LAB_MACHINE_SETUP.md` (written at commit `23ade98`), so the delta was
run to ground rather than waved off:

- `git worktree add` at `23ade98` + `pytest --collect-only` there: **729
  tests collected** at that commit (LAB_MACHINE_SETUP.md's "728 passed"
  plus this session's own re-verification both land within 1 of each other
  — consistent, not identical, and not investigated further since the
  question that matters is the delta to HEAD, answered next).
- Diffing collected test IDs between `23ade98` and `HEAD` (`0a630ca`):
  **zero tests removed, 15 tests added**, all traceable to two specific
  commits after the baseline was recorded (`163a7f9` "close the
  silent-clip bug class", `0a630ca` "wire calibrated mutation") — new
  regression tests for real bugs those commits fixed (silent
  mutation-rate-overflow clipping, calibrated-mutation-strength wiring),
  documented in `docs/decisions.md` and `docs/PROJECT_LOOSE_ENDS_RESOLVED.md`.
  Full added-test list available in this session's scratch log if needed.
- The 33-vs-32 warning count is explained the same way: some of the 15 new
  tests (`test_mutation_rate_overflow_warns_loudly_instead_of_silently_clipping`,
  ×2) assert that a warning IS raised — an intentional, test-asserted
  warning, the same pattern as the already-documented T_primary hard-cap
  clip warning.

**744/744 passed, 0 failed, 0 skipped.** No test was newly skipped.

**(b) Rule-based port-equivalence test**:

```
tests/encoders/test_rule_based.py::test_ported_interpreter_matches_frozen_port_equivalence_fixture PASSED
tests/encoders/test_rule_based.py::test_ported_mutation_matches_frozen_port_equivalence_fixture PASSED
```

Passed, not skipped.

**(c) Encoder regression tests (pinned seed → body)**:

```
tests/encoders/test_nca.py::test_regression_decode_pinned_for_seed_5 PASSED
tests/encoders/test_nca.py::test_regression_mutate_pinned_for_seed_9 PASSED
tests/encoders/test_nca_bounded.py::test_unbounded_residual_reproduces_pinned_decode_seed_5 PASSED
tests/encoders/test_direct.py::test_regression_random_genome_pinned_for_seed_0 PASSED
tests/encoders/test_direct.py::test_regression_random_genome_pinned_for_seed_123 PASSED
tests/encoders/test_direct.py::test_regression_mutate_pinned_for_seed_99 PASSED
```

Bit-identical to the pinned fixtures — interpreter/numpy change did not
alter decode behaviour.

**(d) Config lock hash**:

```
$ .venv/bin/python -m evodevo.experiments.lock_hash --config evodevo/experiments/configs/benchmark.yaml
c1685531d6e6543215b012b204d820c15e1a9284bbd6b0097cdc01aa1224f7ed
```

Matches the value recorded in `docs/READY_TO_RUN.md` ("Current value
(`fast_dev` profile, as of 2026-09-19, commit `bbe54c6`)") exactly. (No
literal `--verify` flag exists on this module — it prints the hash; this
report does the comparison, which is the equivalent the task asked for.)

**(e) Visualization check**:

```
$ .venv/bin/python -m evodevo.visualization.check
[OK] evogym simulator importable
[OK] env creation with render_mode="img" + frame capture
[OK] imageio can write gif and mp4
[OK] gifsicle (optional gif optimization) -- not available, gifs written unoptimized (documented non-fatal)
ALL REQUIRED CHECKS PASSED.
```

Green. No headless-OpenGL or ffmpeg gap — EvoGym's simulator doesn't need a
display for `render_mode="img"`, and `imageio-ffmpeg` bundles its own
portable ffmpeg binary (no system `ffmpeg` dependency).

**(f) Real-PPO smoke evaluation** (liveness check, NOT a timing measurement
— never to be cited as `t_PPO`):

```python
evaluate_body(GOOD_WALKER, task="Walker-v0", ppo_timesteps=2000, eval_episodes=1, seed=0)
```

```
valid: True
fitness: 0.00038149723658209567
wall_clock_seconds: 5.55
```

Single process, single known-valid body. Confirms the pipeline runs
end-to-end, nothing more.

## 7. Acceptance criteria

1. `.venv/bin/python -c "import evodevo, evogym.simulator_cpp, torch, stable_baselines3"` → **succeeds** (run from repo root; `evodevo` has no
   formal install, see §1).
2. Parity gate items a-d: **green**, discrepancies (test count, warning
   count) diagnosed and explained above, not worked around or hidden.
3. This file states the interpreter decision explicitly (§1): **3.10.20**.
4. **No sudo command has been executed and no driver installed.** One sudo
   command (`lm-sensors` install) was attempted per your approval but
   failed for lack of a password/askpass helper in this non-interactive
   session — it was not retried via any workaround (no `--break-system-packages`-style
   shortcut exists for sudo; this is disclosed in §2, not silently skipped).

## 8. Plain list of what did not reproduce byte-for-byte

- Test count: 744 vs the previously-recorded 728 (§6a) — explained (15
  legitimately added tests, 0 removed, codebase moved forward by two
  commits between when that baseline was recorded and this session's start).
- Warning count: 33 vs 32 (§6a) — explained (1 new intentional,
  test-asserted warning from the same 15-test delta).
- `torch` initially resolved to the CUDA build, not `+cpu` (§2) — caught
  and corrected within this session, not a leftover.
- `sensors`/`lm-sensors` — **still absent**, blocked on you running the
  install command yourself (§2). This is the one genuinely open gap.
- `docs/PROJECT_LOOSE_ENDS_RESOLVED.md` remains untracked (pre-existing,
  from before this session; not touched, per the MUST-NOT list's "delete or
  archive existing results" and general scope discipline).
- Running the full suite regenerated `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md`'s
  wall-clock line (27.47s → 38.82s), a documented side effect of one test
  (same behavior LAB_MACHINE_SETUP.md recorded on this machine previously).
  No content changed besides the timing number; reverted with `git checkout`
  to keep the tree clean, consistent with the MUST-NOT list (not treated as
  a new result to keep).

Everything else in the parity gate reproduced exactly.
