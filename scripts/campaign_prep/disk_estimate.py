"""P2 Step 5 disk estimate. Every per-item size is MEASURED (source noted) and then given an
explicit conservative margin; every count is an upper bound (all genomes treated as valid, so
every genome costs its full K PPO trainings and a saved policy)."""
import json, shutil

MB, GB = 1e6, 1e9
est = json.load(open("results/campaign_prep/final_estimate/final_estimate.json"))
fin = est["guard_sequence"][-1]
hours = fin["result"]["total_h"]["p95"]
probe = json.load(open("results/campaign_prep/disk/policy_size_probe.json"))["tasks"]

ARMS, SEEDS, GENOMES = 4, 5, 300                       # D15; guard outcome "none" -> 4 arms both tasks
K = {"Walker-v0": 1, "UpStepper-v0": 2}                 # D16
genomes = {t: ARMS * SEEDS * GENOMES for t in K}        # 6000 per task
ppo = {t: genomes[t] * K[t] for t in K}                 # upper bound: every genome valid
runs = ARMS * SEEDS * len(K)                            # 40

items = {}
# per-evaluation JSONL: measured 516-541 B/record (results/runs/*/Walker-v0/seed_*/run_log.jsonl,
# 1500 records per arm); D10 adds fields (genome hash, worker id, timestamp, timesteps) -> 2 KB (~4x).
items["per_evaluation_jsonl"] = sum(genomes.values()) * 2_000
# per-PPO-seed journal: one record per PPO training, same 2 KB bound.
items["per_ppo_seed_journal"] = sum(ppo.values()) * 2_000
# one saved SB3 policy per valid PPO training: probe zip of the LARGEST 5x5 body per task (size is
# set by obs/action dims, not timesteps -- measured at 2048 steps), x2 margin.
items["sb3_policies"] = sum(ppo[t] * probe[t]["zip_bytes"] * 2 for t in K)
# checkpoints: every generation kept (never overwritten), 30 genomes (pop+offspring) at the LARGEST
# arm's serialized genome (nca, max 10,384 B JSON, measured) + 5 KB RNG/fitness state, x2 margin.
items["checkpoints"] = runs * 20 * (30 * 10_384 + 5_000) * 2
# telemetry + dashboard transcript: measured 3.42 MB + 10.34 MB over the 6.08 h pre-flight run
# (results/preflight/20260922T040932Z_4be8402b) = 2.26 MB/h, x2 margin, over the p95 duration.
items["telemetry_and_transcript"] = (3_423_147 + 10_340_561) / 6.084 * hours * 2
# milestone/campaign snapshot packages (D9: figures, tables, best-body renders): no measured
# precedent -> assumed 50 MB per package, one per hour of the p95 duration (a deliberately
# generous bound; the pre-flight report directory with all its figures is < 1 MB).
items["milestone_packages"] = 50 * MB * hours

total = sum(items.values())
free = shutil.disk_usage("/").free
out = dict(p95_hours=hours, genomes=genomes, ppo_trainings=ppo, runs=runs, items_bytes=items, total_bytes=total,
           three_x_total_bytes=3 * total, free_bytes_root=free, three_x_rule_satisfied=3 * total <= free)
json.dump(out, open("results/campaign_prep/disk/disk_estimate.json", "w"), indent=1)
for k, v in items.items():
    print(f"{k:28} {v / GB:8.3f} GB")
print(f"{'TOTAL':28} {total / GB:8.3f} GB   3x = {3 * total / GB:.1f} GB   free on / = {free / GB:.1f} GB   "
      f"3x rule satisfied: {3 * total <= free}")
