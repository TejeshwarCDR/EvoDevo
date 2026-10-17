# Commit hash map (history rewrite of 2026-10-17)

On 2026-10-17 the history of this repository was rewritten so that every commit is attributed to a single author identity (`TejeshwarCDR <tejeshwarcdr@gmail.com>`) and the attribution trailers (`Co-Authored-By`, `Claude-Session`) were removed from commit messages.

## What changed and what did not

- **Unchanged for all 162 commits:** the file contents of every commit (identical tree hashes, listed below), the author dates, the committer dates, and the commit order.
- **Changed:** author and committer name/email, the removed trailers, and any commit hash quoted inside a commit message (updated to the new hash).
- **Consequence:** a commit hash covers the author, the date and the message, so every commit hash changed. The table below maps each pre-rewrite hash to its current hash.

## Using this map for provenance

Records written before the rewrite deliberately keep the hash that was true when they were produced: run metadata (`run_meta.json`, `factors.json`, `benchmark_manifest.json`), provenance and readiness reports, documents under `docs/`, and result directory names such as `results/preflight/<timestamp>_<hash8>`. To resolve one of those hashes in the current history, look it up in the `old` column below (or in `commit_hash_map.tsv`, which has the full 40-character hashes). The `tree` column is the same before and after the rewrite, so it can be used to confirm that a commit holds exactly the same files.

Tests that pin a commit (`tests/evaluation/test_ppo_runner.py`, `tests/evolution/test_propose_tell.py`) were repointed to the new hashes in a separate commit.

## Map (oldest first)

| # | old | new | tree | author date | subject |
|---|-----|-----|------|-------------|---------|
| 1 | `826b81785a00` | `99d900259251` | `50b1c0d7ac80` | 2026-09-02T09:15:02+05:30 | Initial commit: pre-restructure snapshot of evodevo pipeline |
| 2 | `7c7784be3fec` | `68d09aa6f436` | `a9f8236373e0` | 2026-09-02T09:18:00+05:30 | Restructure workspace into modular layout (moves/renames only) |
| 3 | `e33c0b4205a7` | `2a2b8cd6d3f5` | `8f477159aaf6` | 2026-09-02T09:26:11+05:30 | Harden shared core: single materials source, deduped connectivity, wired fitness cache, headless plotting, parameterized validation |
| 4 | `5b18b4464819` | `84f454e803e3` | `b33d584f7c3a` | 2026-09-02T09:33:56+05:30 | Encoder plugin architecture: extended contract, explicit registry, shared conformance suite |
| 5 | `14a30d4aba92` | `6e433916fb29` | `14ddf096559b` | 2026-09-02T12:33:12+05:30 | Port the Rule-Based encoding as a third arm, conforming to the Encoder contract |
| 6 | `731871f12e1b` | `aaa3dce13c74` | `0d29a53b6e6c` | 2026-09-02T12:48:01+05:30 | Isolated per-method pipelines: scripts, Makefile, ownership docs, config inheritance |
| 7 | `ae620979c73d` | `76d260a03da8` | `3044638771fa` | 2026-09-02T13:02:32+05:30 | Unified three-way benchmark pipeline: orchestrator, aggregate/statistics/ plots extensions, report generator |
| 8 | `2844b3e1da17` | `aefe9a1d5172` | `2deb0952bbc7` | 2026-09-02T13:32:21+05:30 | Visualization layer: rendering, morphology/development figures, and a self-test diagnostic |
| 9 | `1e46c3cdebd4` | `f2bf4ff66ac6` | `4e9becce14fe` | 2026-09-02T14:17:25+05:30 | Fix encoder isolation violation, freeze NCA calibration, lock primary run config |
| 10 | `33d9d855b49b` | `838d2bf3b877` | `0d0d87354842` | 2026-09-02T16:53:09+05:30 | Add benchmark progress and ETA reporting |
| 11 | `273f89c33121` | `fabaf06b99ff` | `194da8c78a24` | 2026-09-02T16:53:09+05:30 | Archive previous benchmark results |
| 12 | `8399751f633c` | `e3450077ac36` | `0dd8d5781a12` | 2026-09-02T17:00:47+05:30 | Update project README |
| 13 | `50b63d77a335` | `21a19bc4bbbe` | `57b85787fbba` | 2026-09-02T20:36:56+05:30 | Freeze rule-based port equivalence, remove external checkout, re-lock config hash |
| 14 | `9e5cef87f577` | `8c0093bc6e7f` | `c79c00596f87` | 2026-09-02T21:00:49+05:30 | Archive stale results/runs/ contents, clearing it for the final locked primary run |
| 15 | `0a9a61b16694` | `ad274af4781d` | `add9fae9f2ee` | 2026-09-15T20:32:27+05:30 | Add git-provenance guard for experiment runs; commit pre-existing project docs |
| 16 | `9a0458b11d09` | `2aec2205339e` | `a786352f16dc` | 2026-09-15T20:32:35+05:30 | P1: crossed ExperimentFactors and named initial conditions |
| 17 | `95aeac4fc22d` | `6cf50422373b` | `3ba645046d26` | 2026-09-15T20:32:40+05:30 | P2: fitness-blind structural measurement substrate |
| 18 | `e7362fd0839d` | `0bab2fb1203b` | `7433ae813a98` | 2026-09-15T20:32:46+05:30 | P3: initial-condition crossed study |
| 19 | `6af3709c65f1` | `0445646fb4be` | `8986c03b44f5` | 2026-09-15T20:32:55+05:30 | P4: developmental-depth study (T_primary derivation, corrected) |
| 20 | `293e1fb3ba29` | `29af32a812d6` | `1d24e719bf6b` | 2026-09-15T20:33:02+05:30 | P5: mutation calibration by phenotypic displacement |
| 21 | `b8d2e5e8d609` | `194b230db42f` | `246f01cb391f` | 2026-09-15T20:33:10+05:30 | P6: dual budget counters and two-panel comparison figures |
| 22 | `690f7690a1c4` | `1fde4a7a7bfc` | `10820b1fd77e` | 2026-09-15T20:33:25+05:30 | P7: random-search control strategy and six-cell benchmark |
| 23 | `01459759237c` | `e8a7af5d3425` | `1d1fae3c0098` | 2026-09-15T20:33:33+05:30 | Commit pre-workstream primary benchmark results (stale, for provenance only) |
| 24 | `7675f0d4ed90` | `918756537d36` | `2589876737a5` | 2026-09-15T20:33:47+05:30 | Commit pre-existing visualization/report artifacts |
| 25 | `1fb8b565df3e` | `f300bd79b492` | `67157522ee0b` | 2026-09-15T20:34:29+05:30 | This fix: remove unjustified T_primary ceiling, diagnose NCA non-settling, close remaining decisions.md OPEN rows, add readiness_check.py |
| 26 | `55c15acd3d31` | `500f881574af` | `be280cf0efcc` | 2026-09-15T20:34:44+05:30 | Regenerate COMPARISON_READINESS.md against the clean, committed tree |
| 27 | `2c09895a6bc8` | `5a8bbf4d47a3` | `3631926bdb7f` | 2026-09-15T20:44:11+05:30 | Fix Section 0/Section 3 T_primary divergence and OPEN-row false positive; explain P7 zero-variance result |
| 28 | `370ed12cfd29` | `84c37e91978d` | `32b22531c00d` | 2026-09-17T04:29:37+05:30 | Register rule_based_monotone as a distinct, non-colliding benchmark arm |
| 29 | `c44f16fd4c3a` | `f0426970f726` | `503960b1b5b4` | 2026-09-17T04:29:54+05:30 | Lock the statistical analysis protocol before any real PPO result exists |
| 30 | `990bc13e2881` | `e376a0a61458` | `4dc6678475d7` | 2026-09-17T04:30:05+05:30 | Add per-generation diversity/validity logging; produce the figure/table manifest |
| 31 | `79bb191a9e26` | `ac580b893499` | `76d75d518db1` | 2026-09-17T04:30:14+05:30 | Add explicit random_search/EA resolved-config parity test for every arm |
| 32 | `217326e240c4` | `2f0d0b9ed2d1` | `cf38015a5fe7` | 2026-09-17T04:30:32+05:30 | Commit rule_based order-independence fixes, rule-language spec, resolved config |
| 33 | `0af2992933d5` | `9cf3ce051db7` | `b9b001e4b055` | 2026-09-17T04:30:50+05:30 | Commit rule_based analysis/calibration/readiness suite |
| 34 | `a5abe8459f1c` | `37a744f04431` | `1dc067fa032b` | 2026-09-17T04:31:08+05:30 | Commit NCA architecture redesign, convergence certificate, and readiness suite |
| 35 | `011e71fa19fe` | `1d388e2b1d07` | `420f69d08dce` | 2026-09-17T04:31:21+05:30 | Regenerate structural/depth/mutation-calibration artifacts under corrected settings |
| 36 | `787aa78336bf` | `85859179ee36` | `b36fdcfd7222` | 2026-09-17T04:31:29+05:30 | Add PRE_PARALLELIZATION_READINESS.md: the consolidated gate document |
| 37 | `c4d3f5f855fb` | `15138f750454` | `0974995aea08` | 2026-09-17T09:28:06+05:30 | Introduce fast_dev/primary_matrix config profiles; fix NCA hidden_channels wiring gap |
| 38 | `35298f8df9bb` | `0c7f638b764b` | `663b551cc6fd` | 2026-09-17T09:28:25+05:30 | Distinguish ablation-within-arm from cross-arm comparisons in the stats protocol |
| 39 | `092fcd56f3ea` | `ae3374716786` | `b95ab23bdb06` | 2026-09-17T09:28:36+05:30 | Regenerate PRE_PARALLELIZATION_READINESS.md: profiles, open decision, hidden_channels, ablation labels |
| 40 | `ff5905b37f91` | `8b89c7085acf` | `c8cdc73d6963` | 2026-09-19T14:58:11+05:30 | Pin setuptools<81 to keep pkg_resources available for EvoGym |
| 41 | `bbe54c6bc069` | `79dbde4b0f8b` | `56b746ee5f7e` | 2026-09-19T14:58:23+05:30 | Check git cleanliness exactly once per benchmark invocation |
| 42 | `23ade98b16c3` | `49e461d14843` | `d91c56d698f3` | 2026-09-19T16:55:54+05:30 | Add lab machine setup verification record |
| 43 | `a89cf1a6b7ec` | `88397bbe43b5` | `55d2cf32261d` | 2026-09-19T16:56:19+05:30 | Audit docs/ and results/ for staleness against the current resolved config |
| 44 | `e869ecfc9772` | `6450c6b477a8` | `3120e40fae69` | 2026-09-19T16:56:55+05:30 | Confirm zero OPEN rows and resolve 5 internal contradictions in decisions.md |
| 45 | `163a7f9f4cbc` | `bcec7df2de25` | `51e0ae79b7c6` | 2026-09-19T16:57:26+05:30 | Close the silent-clip bug class project-wide; dedupe rule_based literals |
| 46 | `0a630cab8537` | `f15c52b9b418` | `060651612ba9` | 2026-09-19T16:57:50+05:30 | Rewrite README/READY_TO_RUN for current state; wire calibrated mutation |
| 47 | `9e877489a101` | `12fed7a9775c` | `c0e8a4f8c5b8` | 2026-09-21T07:48:25+05:30 | Set up lab machine environment and prove pipeline parity (Step 0) |
| 48 | `131350c98d80` | `b6a71741af6e` | `4475ae5d6dcf` | 2026-09-21T07:48:35+05:30 | Add evaluate_body_with_curve (B4's periodic-eval curve training) |
| 49 | `f4f14080dd2e` | `2246f88d91d1` | `703700af5fd0` | 2026-09-21T07:48:52+05:30 | Add threading control, CPU topology, and the evaluation worker pool |
| 50 | `a0ba88c1cac5` | `a71a25dcc55a` | `f31d5cf80336` | 2026-09-21T07:49:03+05:30 | Add the D5 pre-flight measurement harness and its operator surface |
| 51 | `0dbcb50566ca` | `80555d898d30` | `8fb81b49aaa5` | 2026-09-21T08:15:55+05:30 | readiness.py: add "changes since previous report" section |
| 52 | `09018dfa4cbf` | `00c52c371997` | `8f65eded1be5` | 2026-09-21T08:26:48+05:30 | Add PREFLIGHT_READINESS.md, generated from the clean tree |
| 53 | `b9b6b8bd169d` | `874a117cfb0b` | `2140de9cd5f8` | 2026-09-21T18:00:09+05:30 | Repair the ppo_runner.py parity test: pin to pre-refactor commit |
| 54 | `71d946c45675` | `309e41c47083` | `0f9429f0530d` | 2026-09-21T18:00:28+05:30 | Add docs/final_decisions.md, the single source of truth for the campaign |
| 55 | `9b3e089cf1ee` | `8b6a2396d58b` | `8790a2762069` | 2026-09-21T18:00:42+05:30 | Add docs/PROJECT_LOOSE_ENDS_RESOLVED.md |
| 56 | `4be8402b9609` | `9d3da4024084` | `3f19cecb073f` | 2026-09-22T09:26:03+05:30 | Fix run_preflight.sh: tmux split-window -p is broken in tmux 3.4 |
| 57 | `294ee551628e` | `0f8c7bcf3949` | `48766871bd75` | 2026-09-27T17:07:14+05:30 | ADR: interleaved launcher scheduling approved |
| 58 | `f2689b1dbdcb` | `6c8fa959885b` | `5d0723820fac` | 2026-09-27T17:16:48+05:30 | Pre-register the campaign analysis plan in decisions.md |
| 59 | `1e79a5652be0` | `bd440941a9cc` | `71503edf70ea` | 2026-09-27T17:17:35+05:30 | D6: per-task ppo_timesteps (UpStepper-v0 1.2M, Walker-v0 1.0M via fallback) |
| 60 | `a27c61a70d8b` | `b2a206f2a422` | `1f38b252df32` | 2026-09-27T20:57:57+05:30 | Lock the campaign decisions: MONO, D14, D15, D16, concurrency, budget guard, infra failures |
| 61 | `c7172d1e4d8c` | `683ddbccde38` | `aca057ed7c3c` | 2026-09-27T20:58:19+05:30 | Encode the campaign profile: per-task budgets, D16 ppo_seeds, K>1 guard |
| 62 | `feed8ec35443` | `357c8540ebdd` | `7d545c0af87a` | 2026-09-27T20:58:36+05:30 | Re-lock the config hash for the campaign profile |
| 63 | `77bc84229162` | `e14b6f262489` | `11e6675e6718` | 2026-09-27T20:58:47+05:30 | Capacity simulator: final campaign estimate (4 arms, per-task budgets, K=2 seeds as pool tasks) |
| 64 | `bf754a980c7a` | `53faafcb30dc` | `ab7874004045` | 2026-09-28T00:52:22+05:30 | Preflight B2: correlate the throttle counter with temperature before reporting throttled |
| 65 | `a0b1ca563c5d` | `841634256ccd` | `4275afa98308` | 2026-09-28T00:54:16+05:30 | Preflight B2: 120-genome workload and a discarded warm-up level |
| 66 | `593282d40522` | `ff7b1c18801f` | `26b8e9b1660c` | 2026-09-28T00:56:26+05:30 | Preflight B6: P/E-core mixed-rate wall-clock model |
| 67 | `f24eb2615c66` | `b7f26d7a186e` | `77d67b27790b` | 2026-09-28T00:58:23+05:30 | Preflight B6: model each D14 option by what it changes |
| 68 | `114b4eea2a67` | `1da897031e83` | `082a4d31d1c5` | 2026-09-28T01:00:18+05:30 | Commit the capacity-planning analysis scripts behind every reported round |
| 69 | `7d4a74b6942a` | `fd2a08ac2d2f` | `1d7ba64235e4` | 2026-09-28T01:00:18+05:30 | Commit the campaign-prep measurement scripts behind P1/P2's numbers |
| 70 | `fd64b6498f25` | `dc802b44e22f` | `f0ca897886ad` | 2026-09-28T01:00:50+05:30 | Version-control the reports that justify the campaign design |
| 71 | `bd088f20cdc1` | `99c373104a50` | `4c0ec04b42ea` | 2026-09-28T01:00:50+05:30 | Ignore results/campaign_prep/ (generated campaign-prep measurement output) |
| 72 | `9b4d8b0064f7` | `c05546a34f47` | `f65dac75abf7` | 2026-09-28T01:01:28+05:30 | decisions.md: provenance of pre-flight run 20260922T122730Z_4be8402b |
| 73 | `bfb913917b26` | `716c984d31ff` | `8c8bfe3d456d` | 2026-09-28T01:01:28+05:30 | decisions.md: save every PPO policy; disclose the marginal Walker-v0 D6 budget |
| 74 | `c55944f6aaa5` | `03c4bab3a88b` | `370bebe83333` | 2026-09-28T01:37:25+05:30 | Refactor EA into propose tell state machine |
| 75 | `0873e4934be1` | `8fcc88f67765` | `24b0c3569869` | 2026-09-28T06:20:24+05:30 | Implement multi-seed PPO evaluation artifacts |
| 76 | `437a2df830ee` | `96fde1c99789` | `b4dc6bec00e6` | 2026-09-28T06:36:42+05:30 | Add campaign plan journal checkpoints |
| 77 | `1c99e29b504c` | `4a7d87e7de72` | `3e535ddbc58d` | 2026-09-28T07:04:47+05:30 | Add campaign observability alerts milestones and mirror |
| 78 | `fce536e51a41` | `62adde635708` | `02eb73924719` | 2026-09-28T07:29:10+05:30 | Make campaign phase dispatch resumable and journal-aware |
| 79 | `148531bc649e` | `d4f727834fb4` | `126e446be55e` | 2026-09-28T07:29:17+05:30 | Add campaign launcher and live operational state |
| 80 | `95f5dfeda85f` | `9a2718a0da67` | `b6b7b0686df3` | 2026-09-28T07:32:20+05:30 | Preserve monitor cadence across sidecar restarts |
| 81 | `ff0f3981f63e` | `b0197aa63f6c` | `bfd65c08de8a` | 2026-09-28T07:35:23+05:30 | Prove stop script kills a live campaign sidecar |
| 82 | `d87346f51ac6` | `dfa6eefdbdb9` | `9f0b072a5d59` | 2026-09-28T07:38:58+05:30 | Gate trained-controller milestone videos on rendering self-test |
| 83 | `e96812e39e61` | `fe24ca08949d` | `4673caacd761` | 2026-09-28T07:40:11+05:30 | Resolve milestone paths before EvoGym rendering |
| 84 | `bcd7acc77b73` | `2cdfa82e064f` | `e6788ac66768` | 2026-09-28T07:43:28+05:30 | Resolve latest campaign by manifest recency |
| 85 | `112799bef29a` | `b5daf4c29fcf` | `1ec8f19083f8` | 2026-09-28T07:54:23+05:30 | Clean tmux session even after scheduler process exits |
| 86 | `fc7db75ff513` | `c5440c839a59` | `1ca2fbd46ac9` | 2026-09-28T07:55:22+05:30 | Smoke test real tmux launch and cleanup with surrogate campaign |
| 87 | `8d4c5d337da0` | `26c50b262a63` | `6435e8819c42` | 2026-09-28T07:56:19+05:30 | Include launcher transcript on smoke timeout |
| 88 | `6325e429540b` | `1f6405ea9be6` | `cb4abf982e0c` | 2026-09-28T08:21:08+05:30 | Use per-arm core-class rates for campaign ETA and tighten alert cadence |
| 89 | `ffec6d023413` | `f733d058909a` | `61dd4ff3a95c` | 2026-09-28T08:24:50+05:30 | Harden live mirror snapshots and retry failed milestone snapshots |
| 90 | `b7dd95061d33` | `1557fdf7c0fc` | `ffdbaa98703f` | 2026-09-28T08:26:21+05:30 | Verify partial-stall persistence throughput and config-hash refusal |
| 91 | `c643c5d3cc35` | `bbc9ac89cdee` | `a186cb770de1` | 2026-09-28T11:22:16+05:30 | Report proposed run generations and actual PPO in-flight counts |
| 92 | `5f3f648c4ec3` | `455dd7f185d9` | `98e33e3d1dfe` | 2026-09-28T11:25:00+05:30 | Require fresh complete milestone package after subprocess run |
| 93 | `5900baaeb55d` | `1ae934e354f7` | `acecb31ff3d8` | 2026-09-28T11:25:24+05:30 | Assert automatic Walker milestone in tmux launch smoke |
| 94 | `563dbf59ea90` | `a17b938d2188` | `92ee2740aecf` | 2026-09-28T11:25:45+05:30 | Pin Tjmax-gated thermal throttle alert behavior |
| 95 | `b29e4d76a8d4` | `1d51b11ac218` | `d0617cb24229` | 2026-09-28T11:30:02+05:30 | Guard mirror choice and stop only the targeted campaign session |
| 96 | `ed59fbf44b25` | `7ae7ac702bea` | `52d15dc9ca53` | 2026-09-28T11:35:57+05:30 | Audit cross-process alerts and P2 prior ETA provenance |
| 97 | `0badecbcfa31` | `540f27ff8c95` | `932dd7e6e5b4` | 2026-09-28T11:39:48+05:30 | Cover paper-ready milestones SSH status and projection guards |
| 98 | `7cdc89526c0d` | `6fda5ae47c45` | `aee460cf6b9a` | 2026-09-28T11:50:28+05:30 | Complete preregistered milestone analyses and D9 snapshots |
| 99 | `1a506e9fbc89` | `d9e05626f082` | `c7f51627e979` | 2026-09-28T11:51:53+05:30 | Label legacy snapshot summary and manifest as partial |
| 100 | `ab8f73da2400` | `8998999cbcb4` | `92a4c6c38b91` | 2026-09-28T12:40:44+05:30 | Add pre-launch campaign readiness validation report |
| 101 | `179155ffda09` | `967f1fa396a0` | `2f0027c2e8e8` | 2026-09-28T14:25:23+05:30 | Fix BrokenProcessPool: campaign package imported numpy before worker thread pinning |
| 102 | `5f95a8dbfea5` | `851f3a8ca14b` | `9638c105ed0b` | 2026-09-28T14:30:08+05:30 | Make notifier delivery truthful and refuse unnotified real launches |
| 103 | `3a3c539d200e` | `60c45e9dcfae` | `33021b1580b9` | 2026-09-28T14:33:40+05:30 | Require the mirror to be a dedicated mount on a separate physical disk |
| 104 | `2fbd136ebb86` | `9df0f66c80cd` | `ea3d00b61753` | 2026-09-28T14:36:22+05:30 | Add a write-nothing --dry-run to the campaign launcher |
| 105 | `6aa13b3dbbd1` | `313db788deef` | `72eb95655f9a` | 2026-09-28T14:37:50+05:30 | Hold a systemd sleep/idle/lid inhibitor lock around the campaign scheduler |
| 106 | `39d1426965da` | `4e06853ce781` | `9ce9ad17baa2` | 2026-09-28T14:39:11+05:30 | Show the reached Walker-milestone time once the Walker phase ends |
| 107 | `2af77670c1cc` | `76a8a7bad600` | `c062c3fac280` | 2026-09-28T14:41:36+05:30 | Add an explicit artifact stamp for validation campaigns |
| 108 | `cb7180ca6f89` | `98fafa3c8da1` | `f655abab9f86` | 2026-09-28T14:58:26+05:30 | Fix a launcher race that could leave a "started" campaign not running |
| 109 | `d89f624230c7` | `d93e4be666fd` | `61bb7729ea88` | 2026-09-28T14:58:26+05:30 | Assert per-slot CPU pinning in the CLI smoke; guard setsid --wait |
| 110 | `179ae008cc3c` | `4e4a283e68ea` | `355c35185794` | 2026-09-28T15:19:47+05:30 | Never "attach" to a stale campaign session whose scheduler is dead |
| 111 | `26faf941a147` | `e8e6d6e2ebb4` | `ebe5b7644b94` | 2026-09-28T15:37:53+05:30 | Stop gracefully: campaign workers leave SIGINT to the scheduler |
| 112 | `867d62c90c63` | `00873f74ae8f` | `49a53ad18c45` | 2026-09-28T16:08:57+05:30 | Regenerate CAMPAIGN_READINESS.md after the launcher fixes: NOT READY |
| 113 | `5452cfbf835f` | `0c02d8f0e02a` | `def168a01595` | 2026-09-28T16:39:54+05:30 | docs: lock continuous dispatch policy |
| 114 | `f779833d69fb` | `b08c73123798` | `d31da91edc04` | 2026-09-28T16:45:58+05:30 | test: define continuous dispatch policy core |
| 115 | `f3d4e0e07dba` | `26ad145774de` | `1ab99f2b5e91` | 2026-09-28T16:53:02+05:30 | feat: continuously dispatch campaign evaluations |
| 116 | `2cee9669a5f6` | `b8f6b4f43cc0` | `ff4eb29b8069` | 2026-09-28T17:11:49+05:30 | analysis: rederive campaign estimate from dispatch core |
| 117 | `91ecd87128f9` | `c48c932004e1` | `7d288a6ebe9c` | 2026-09-28T17:23:26+05:30 | validation: add timed campaign conformance profile |
| 118 | `22dd0dda3a39` | `fdffe8d4973f` | `11116320f0c2` | 2026-09-28T18:05:26+05:30 | validation: make idle gate remote-aware |
| 119 | `7b47d673a089` | `d2056f7b72fc` | `f923475db0ac` | 2026-09-28T19:33:57+05:30 | fix: reset PPO timeout between phases |
| 120 | `3fc4e4cff3b9` | `04e90f13d5c4` | `61fd5a2664b8` | 2026-09-29T05:37:20+05:30 | validation: record scheduler conformance gate |
| 121 | `eac2186fd14a` | `dd0646b9abe1` | `cf29276098ed` | 2026-09-29T05:45:19+05:30 | validation: add campaign equivalence proof harness |
| 122 | `133e4b06061d` | `6a06e73fdd0d` | `2181117b30e4` | 2026-09-29T05:58:02+05:30 | fix: honor graceful stop in surrogate phases |
| 123 | `234ee166cd09` | `e36fdaf4666c` | `59e74cee44e2` | 2026-09-29T06:07:16+05:30 | fix: compare resumed equivalence checkpoints |
| 124 | `cec5dde9d528` | `04fe92874311` | `a27e24c097ae` | 2026-09-29T06:10:43+05:30 | test: prove resume hash mismatch refusal |
| 125 | `b9aab7dab2ec` | `754b194f487a` | `4cbaf1dbba09` | 2026-09-29T06:14:08+05:30 | validation: add repeated resume and stop proof |
| 126 | `dbfd137a9506` | `ae3cbcb14805` | `b5d8efd4f0ca` | 2026-09-29T06:36:50+05:30 | validation: pin serial equivalence threads |
| 127 | `673b9d033698` | `2692444d10d2` | `f313e90989ae` | 2026-09-29T10:36:19+05:30 | validation: add real PPO resume proof |
| 128 | `725de9755949` | `51541c086951` | `60f8dcd2fbc4` | 2026-09-29T10:49:19+05:30 | validation: record scheduler equivalence proof |
| 129 | `2aecf60b4c26` | `376ece9fefe6` | `f23fad7e821d` | 2026-09-29T11:49:11+05:30 | validation: record campaign suite flake investigation |
| 130 | `ab3622163966` | `4900aed908e5` | `8a3385ca0c05` | 2026-09-29T11:53:25+05:30 | feat: harden campaign operations |
| 131 | `f0d1cd099678` | `9b7b26b18c80` | `0b2d61a6d86e` | 2026-09-29T11:55:39+05:30 | docs: record operational hardening evidence |
| 132 | `fcdaae29eb9b` | `4cbe00137a73` | `5de4fc7a574c` | 2026-09-29T12:17:44+05:30 | fix: preserve notifier status call compatibility |
| 133 | `d455a83beb7c` | `da6957c83aa8` | `21e53aa0f366` | 2026-09-29T12:35:05+05:30 | validation: record real launcher mini campaign |
| 134 | `2b5e6241dfc1` | `1d2821da3312` | `ebbb605f2452` | 2026-09-29T12:36:10+05:30 | docs: regenerate campaign readiness report |
| 135 | `ca6923bd88d3` | `317e44c38a51` | `98ceb7da253d` | 2026-09-29T13:48:25+05:30 | Add campaign notification and dashboard status layer |
| 136 | `2ef7ae67df28` | `8f37c8658004` | `7bed6a2df66a` | 2026-09-29T14:25:53+05:30 | Preserve clean state around campaign launcher tests |
| 137 | `47ce0408be29` | `59905f4893ab` | `d7fdd9b59a71` | 2026-09-29T14:26:24+05:30 | Keep campaign completion marker out of test artifacts |
| 138 | `f44061de597e` | `bf9738641e6f` | `aa8c59b1a175` | 2026-09-29T14:47:55+05:30 | Record final comprehensive verification |
| 139 | `e82f1df1e0f1` | `a283ec605276` | `4c937bd086b9` | 2026-09-29T15:04:16+05:30 | docs: reconcile campaign readiness with dispatch implementation |
| 140 | `82110a1f5877` | `928ec9013f03` | `d2301f396729` | 2026-09-29T15:06:54+05:30 | style: remove capacity planner trailing whitespace |
| 141 | `8d0312c7e32a` | `553d1b23018d` | `db2169a791bb` | 2026-09-30T02:24:48+05:30 | feat(campaign): enforce validated thermal production profile |
| 142 | `24e2de6a8fa8` | `9860e7babe9d` | `37df98a94543` | 2026-09-30T02:27:59+05:30 | test(campaign): make generic real launch profile explicit |
| 143 | `6de63cc9ba99` | `61f08d730a5e` | `9eec5a4609aa` | 2026-09-30T02:30:43+05:30 | fix(campaign): scope production attestation to thermal profile |
| 144 | `69b785671c46` | `0d9c2b659ee4` | `0f4e75e03a67` | 2026-09-30T02:32:56+05:30 | docs(campaign): clarify power supervisor scope |
| 145 | `92b28a858ef3` | `80728acba723` | `f8df5bbcb680` | 2026-09-30T07:50:08+05:30 | docs(campaign): finalize authoritative launch readiness |
| 146 | `a69f4b5e3802` | `6469e3665d17` | `b59963ed339a` | 2026-09-30T09:34:29+05:30 | Fix production campaign launcher lifecycle |
| 147 | `f68e2b492238` | `22d520add8f4` | `3be3e7005310` | 2026-09-30T09:58:02+05:30 | Fix real-host campaign lifecycle cleanup |
| 148 | `7774f76c1473` | `d51138e2694b` | `7e57f0f08a7d` | 2026-09-30T10:03:44+05:30 | Record passing bounded host lifecycle validation |
| 149 | `f7d2be56cae3` | `334cbaecdaff` | `6b4d01fa2733` | 2026-10-15T07:12:08+00:00 | Pre-register the campaign extension (ObstacleTraverser-v1, Climber-v0, CaveCrawler-v0) |
| 150 | `f40cbced41be` | `40ec31c5f0dd` | `ee312709965f` | 2026-10-15T07:12:08+00:00 | Make the campaign machinery task-generic without changing the original campaign |
| 151 | `780894132646` | `3a3d133a9474` | `f472d00e8b54` | 2026-10-15T07:12:08+00:00 | Add the extension preflight, rules, derivation and configs |
| 152 | `df485d474470` | `dcf8589a148b` | `cc6365c34e97` | 2026-10-15T07:46:36+00:00 | Extension preflight: stop cleanly on SIGINT/SIGTERM and test body equivalence |
| 153 | `238b32de2df9` | `61f3887b0cd9` | `1265209e84e0` | 2026-10-15T07:46:36+00:00 | Extension runbook and lab prompts |
| 154 | `377c8e98e6ea` | `002a9e93906a` | `47e75e2cb494` | 2026-10-15T07:51:33+00:00 | derive: take the reference budgets from the preflight's recorded parameters |
| 155 | `068959bb2e48` | `42e8fb77b5d9` | `10411ebe2918` | 2026-10-15T07:53:30+00:00 | Dashboard: show the campaign's own estimate line for v2-estimate campaigns |
| 156 | `b112a93f30ae` | `3471d19726ab` | `3d7e5195af72` | 2026-10-15T08:20:31+00:00 | Lab prompts: create the rehearsal mirror folder first |
| 157 | `4db3ecadd0a2` | `67ce2c2dadd5` | `383af8a1f1d0` | 2026-10-15T14:08:38+05:30 | Pre-registration Amendment 1: 4 repeats, unbiased max-of-2 noise, refuse partial preflights |
| 158 | `0255dd2a140f` | `a0202c8b09f9` | `b0b64777ba8b` | 2026-10-15T14:09:34+05:30 | D16-ext: 4 repeats and unbiased max-of-2 within-body variance |
| 159 | `8de85a54c3ac` | `de3eaaeda13e` | `dab198c2c2cb` | 2026-10-15T14:11:31+05:30 | Extension preflight/derive: provenance label, crash-safe resume, completeness checks, byte-for-byte check |
| 160 | `5f38c35291e3` | `c74aac16deb8` | `88c59d535eb7` | 2026-10-15T14:12:24+05:30 | Campaign: single legacy-estimate path, no silent v2 fallback bands, extension-only floor check, dry-run cleanup |
| 161 | `33b69c6b44df` | `dd2168be8e30` | `256e03685d1e` | 2026-10-15T16:52:50+05:30 | Refuse TEST-labelled decision records at plan build and in check; lab prompts and runbook for the main-branch flow |
| 162 | `73429534e554` | `cf3e04507139` | `d421f3d520cb` | 2026-10-15T18:53:02+05:30 | Lab prompt 1: fast-forward pre-check for lab-only commits |
