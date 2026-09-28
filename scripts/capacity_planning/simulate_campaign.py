"""Capacity-planning Monte Carlo discrete-event simulator (standalone analysis script,
NOT a preflight stage). Read-only over every existing preflight artifact.

Inputs (all measured, all cited in the output JSON):
  evaluation_cost.json   per-(arm,task) t_PPO mean/sd (n=3 raw obs -> too few to resample,
                         so a lognormal is fitted to mean/sd), per-arm t_sentinel mean/sd
  results/inspection/*   per-arm validity_rate
  pe_core_ratio_measurement_production_scale.json   E/P ratio 2.3172
  evodevo.parallel.topology.cpu_slots()/allocation_order()   worker -> P/E placement
  orchestration_overhead.json   Step 2 measured per-generation overhead

Two scheduling modes over the D15 grid (pop 15, 20 gens, 5 seeds, 3 arms, 2 tasks):
  SERIAL_PER_RUN  one replicate at a time, batch width min(pop,20)=15 workers
  INTERLEAVED     all replicates of all arms in a D7 task-phase pooled on 20 workers,
                  each run's own generation g+1 gated on its own generation g
D7: the Walker-v0 phase fully completes before the UpStepper-v0 phase starts.
Common random numbers: both modes consume the SAME per-trial duration draws.
"""
import heapq, itertools, json, math, os, sys, time
from collections import deque
import numpy as np

sys.path.insert(0, os.getcwd())
from evodevo.parallel.topology import detect_topology, cpu_slots
from evodevo.campaign.dispatch import DispatchCore, DispatchItem

RUN = "results/preflight/20260922T122730Z_4be8402b"
OUT = f"{RUN}/report/capacity_planning"
ARMS = ("direct", "nca", "rule_based")
TASKS = ("Walker-v0", "UpStepper-v0")
POP, GENS, SEEDS = 15, 20, 5
N_WORKERS = 20


def lognormal_params(mean, sd):
    s2 = math.log(1 + (sd / mean) ** 2)
    return math.log(mean) - s2 / 2, math.sqrt(s2)


def load_inputs():
    ec = json.load(open(f"{RUN}/stages/evaluation_cost.json"))
    ratio = json.load(open(f"{RUN}/pe_core_ratio_measurement_production_scale.json"))["ratio_e_over_p"]
    oh = json.load(open(f"{OUT}/orchestration_overhead.json"))
    valid = {a: json.load(open(f"results/inspection/{a}/{a}_sanity.json"))[a]["valid_rate"] for a in ARMS}
    ppo = {(a, t): (ec["arms"][a]["tasks"][t]["t_ppo"]["mean"], ec["arms"][a]["tasks"][t]["t_ppo"]["sd"])
           for a in ARMS for t in TASKS}
    sent = {a: (ec["arms"][a]["t_sentinel"]["mean"], ec["arms"][a]["t_sentinel"]["sd"]) for a in ARMS}
    tdev = {a: ec["arms"][a]["t_develop"]["mean"] for a in ARMS}
    n_obs = {(a, t): ec["arms"][a]["tasks"][t]["t_ppo"]["n"] for a in ARMS for t in TASKS}
    return dict(ppo=ppo, sent=sent, valid=valid, ratio=ratio, tdev=tdev, n_obs=n_obs,
                orch={a: oh["arms"][a]["per_generation_s"]["mean"] for a in ARMS})


def worker_speeds(n):
    topo = detect_topology()
    plan = cpu_slots(topo, n)
    e = set(topo.e_core_ids)
    return [1 if s[0] not in e else None for s in plan.slots]  # None => E-core (scaled later)


class Model:
    def __init__(self, inp, cv_floor=0.0, later_gen_valid=None, decode_in_launcher=True, validity_curves=None, validity_trajs=None):
        self.inp = inp
        self.cv_floor = cv_floor
        self.later_gen_valid = later_gen_valid  # None => same validity rate every generation (B6)
        self.decode = decode_in_launcher
        # empirical per-generation validity from the historical run (validity_by_generation.json):
        #   validity_curves[arm] -> (GENS,) probability per generation (pooled/fitted curve)
        #   validity_trajs[arm]  -> (n_runs, GENS): each simulated run resamples ONE historical run's trajectory
        self.validity_curves = validity_curves
        self.validity_trajs = validity_trajs
        self.ratio = inp["ratio"]
        self.ppo_ln = {}
        for k, (m, s) in inp["ppo"].items():
            s = max(s, cv_floor * m)
            self.ppo_ln[k] = lognormal_params(m, s)
        self.sent_ln = {a: lognormal_params(m, s) for a, (m, s) in inp["sent"].items()}

    def overhead(self, arm, width=POP):
        o = self.inp["orch"][arm]
        if self.decode:
            o += width * self.inp["tdev"][arm]
        return o

    def expected_cost(self, arm, task):
        v = self.inp["valid"][arm]
        return v * self.inp["ppo"][(arm, task)][0] + (1 - v) * self.inp["sent"][arm][0]

    def draw_run(self, rng, arm, task):
        return self.draw_run_with_mask(rng, arm, task)[0]

    def draw_run_with_mask(self, rng, arm, task):
        """(GENS, POP) base (P-core-speed) durations. gen 0 uses the base validity rate."""
        v = self.inp["valid"][arm]
        vmat = np.full((GENS, POP), v)
        if self.later_gen_valid is not None:
            vmat[1:, :] = self.later_gen_valid
        if self.validity_curves is not None and arm in self.validity_curves:
            vmat = np.repeat(np.asarray(self.validity_curves[arm], float)[:, None], POP, axis=1)
        if self.validity_trajs is not None and arm in self.validity_trajs:
            tr = np.asarray(self.validity_trajs[arm], float)
            vmat = np.repeat(tr[rng.integers(len(tr))][:, None], POP, axis=1)
        is_valid = rng.random((GENS, POP)) < vmat
        mu, sg = self.ppo_ln[(arm, task)]
        smu, ssg = self.sent_ln[arm]
        d_ppo = rng.lognormal(mu, sg, (GENS, POP))
        d_sen = rng.lognormal(smu, ssg, (GENS, POP))
        return np.where(is_valid, d_ppo, d_sen), is_valid


def make_speed_vec(n):
    spec = worker_speeds(n)
    return spec  # list of 1 / None


def speed_factors(spec, ratio):
    return [1.0 if s == 1 else ratio for s in spec]


def run_serial(durs, oh, factors_w15):
    """One run alone at batch width POP on the first POP workers of allocation_order.
    Returns (run_time, per-batch eval makespans). Task i -> worker i (tasks are iid)."""
    f = np.asarray(factors_w15)
    spans = (durs * f[None, :]).max(axis=1)
    return float(spans.sum() + oh * len(spans)), spans


def _run_interleaved_fifo(run_durs, run_oh, factors, ready0=0.0, trace=None, inline_masks=None):
    """Greedy list-scheduling of many runs on len(factors) workers.
    run_durs: list of (GENS,POP) arrays; run_oh: per-run inter-generation overhead.
    Returns (makespan, per-run finish times). Idle workers: fastest (lowest index) first;
    ready tasks: FIFO by (ready time, run id, task idx)."""
    nw = len(factors)
    idle = list(range(nw))
    heapq.heapify(idle)
    events = []  # (time, seq, kind, a, b)   kind 0: batch ready (run,gen)   1: task done (run, worker)
    seq = 0
    n_runs = len(run_durs)
    outstanding = [0] * n_runs
    inline_pending = [0.0] * n_runs
    next_gen = [0] * n_runs
    finish = [0.0] * n_runs
    ready = []  # heap (ready_time, run, gen, idx)
    for r in range(n_runs):
        heapq.heappush(events, (ready0, seq, 0, r, 0)); seq += 1
    t = ready0
    last = ready0
    while events or ready:
        # pull all events at the earliest time
        if events:
            t = events[0][0]
        while events and events[0][0] <= t:
            _, _, kind, a, b = heapq.heappop(events)
            if kind == 0:
                r, g = a, b
                inline_cost = 0.0
                if inline_masks is not None:
                    # inline-sentinel variant: invalid genomes (sentinel evaluations) are resolved in the
                    # launcher process (decode+validate only), take NO pool slot; their measured
                    # sentinel time is charged serially to the run instead of to a worker.
                    m = inline_masks[r][g]
                    inline_cost = float(run_durs[r][g][~m].sum())
                    idxs = [i for i in range(len(m)) if m[i]]
                else:
                    # width = tasks in this generation: POP for one PPO seed per genome, POP*K
                    # when each of a genome's K PPO seeds is its own pool task (final_estimate.py)
                    idxs = list(range(np.shape(run_durs[r])[1]))
                outstanding[r] = len(idxs)
                if not idxs:   # nothing needs a worker: generation completes after the inline work
                    outstanding[r] = 1
                    heapq.heappush(events, (t, seq, 2, r, 0)); seq += 1
                inline_pending[r] = inline_cost
                for i in idxs:
                    heapq.heappush(ready, (t, r, g, i))
            else:
                r, w = a, b
                if kind == 1:
                    heapq.heappush(idle, w)
                outstanding[r] -= 1
                if outstanding[r] == 0:
                    g_done = next_gen[r]
                    if trace is not None: trace.append((r, g_done, t))
                    next_gen[r] += 1
                    if next_gen[r] < GENS:
                        heapq.heappush(events, (t + run_oh[r] + (inline_pending[r] if inline_masks is not None else 0.0), seq, 0, r, next_gen[r])); seq += 1
                    else:
                        tf = t + (inline_pending[r] if inline_masks is not None else 0.0)
                        finish[r] = tf
                        last = max(last, tf)
        while idle and ready:
            w = heapq.heappop(idle)
            it = heapq.heappop(ready)
            r, g, i = it[1], it[2], it[3]
            dt = run_durs[r][g, i] * factors[w]
            heapq.heappush(events, (t + dt, seq, 1, r, w)); seq += 1
    return last - ready0, finish


def _run_interleaved_lgf(run_durs, run_oh, factors, ready0=0.0, trace=None,
                         inline_masks=None):
    """Virtual-clock execution of the production DispatchCore policy."""
    n_runs = len(run_durs)
    core = DispatchCore(slots=len(factors))
    run_keys = tuple(str(r) for r in range(n_runs))
    core.begin_phase("simulation", run_keys)
    idle = list(range(len(factors)))
    heapq.heapify(idle)
    events = []  # (time, sequence, kind, payload); kind 0 ready, kind 1 task complete
    sequence = itertools.count()
    inline_pending = [0.0] * n_runs
    finish = [0.0] * n_runs
    last = ready0

    def push_ready(at, run, generation):
        heapq.heappush(events, (at, next(sequence), 0, (run, generation)))

    def generation_complete(at, run, generation):
        nonlocal last
        if trace is not None:
            trace.append((run, generation, at))
        next_generation = generation + 1
        after_serial = at + run_oh[run] + inline_pending[run]
        if next_generation < GENS:
            push_ready(after_serial, run, next_generation)
        else:
            finish[run] = after_serial
            last = max(last, after_serial)
            core.mark_run_complete(str(run))

    for run in range(n_runs):
        push_ready(ready0, run, 0)

    now = ready0
    while events or core.ready_count or core.inflight_count:
        if events:
            now = events[0][0]
        while events and events[0][0] <= now:
            _at, _seq, kind, payload = heapq.heappop(events)
            if kind == 0:
                run, generation = payload
                if inline_masks is None:
                    indexes = list(range(np.shape(run_durs[run])[1]))
                    inline_pending[run] = 0.0
                else:
                    mask = inline_masks[run][generation]
                    indexes = [index for index in range(len(mask)) if mask[index]]
                    inline_pending[run] = float(run_durs[run][generation][~mask].sum())
                tasks = [DispatchItem(
                    task_id=(run, generation, index), phase="simulation", run_key=str(run),
                    generation=generation, tie_key=(run, index), payload=(run, generation, index),
                ) for index in indexes]
                barrier = core.report_ready(str(run), generation, tasks)
                if barrier is not None:
                    generation_complete(now, run, generation)
            else:
                worker, dispatch_item = payload
                heapq.heappush(idle, worker)
                barrier = core.complete(dispatch_item.task_id)
                if barrier is not None:
                    generation_complete(now, int(barrier.run_key), barrier.generation)

        while idle:
            dispatch_item = core.next_task()
            if dispatch_item is None:
                break
            worker = heapq.heappop(idle)
            run, generation, index = dispatch_item.payload
            duration = run_durs[run][generation, index] * factors[worker]
            heapq.heappush(events, (now + duration, next(sequence), 1,
                                    (worker, dispatch_item)))

    if not core.phase_complete:
        raise AssertionError("simulated dispatch ended before its phase barrier")
    return last - ready0, finish


def run_interleaved(run_durs, run_oh, factors, ready0=0.0, policy="fifo", trace=None,
                    inline_masks=None):
    """Run a virtual interleaved phase under a named dispatch policy.

    ``lgf`` is the production policy and uses the shared DispatchCore. ``fifo``
    remains solely to reproduce the historical comparison artifacts.
    """
    if policy == "lgf":
        return _run_interleaved_lgf(run_durs, run_oh, factors, ready0, trace, inline_masks)
    if policy == "fifo":
        return _run_interleaved_fifo(run_durs, run_oh, factors, ready0, trace, inline_masks)
    raise ValueError(f"unknown dispatch policy {policy!r}")


def run_continuous_dispatch(run_durs, factors, serial_costs, *, ready0=0.0,
                            prefetch=4, trace=None, inline_masks=None):
    """Production DispatchCore policy with a single blocking scheduler server.

    ``serial_costs`` contains per-run ``prepare_s`` (propose/decode/validate),
    per-run ``finish_s`` (tell/log/checkpoint), and per-PPO-task ``journal_s``.
    Prefetched tasks are already in ProcessPoolExecutor's call queue, so an
    idle worker can take one while the scheduler is busy handling a result.
    """
    n_runs = len(run_durs)
    n_workers = len(factors)
    prepare = list(serial_costs["prepare_s"])
    finish_cost = list(serial_costs["finish_s"])
    journal_cost = float(serial_costs["journal_s"])
    if len(prepare) != n_runs or len(finish_cost) != n_runs:
        raise ValueError("serial cost vectors must match run_durs")
    core = DispatchCore(n_workers, prefetch=prefetch)
    run_keys = tuple(str(run) for run in range(n_runs))
    core.begin_phase("simulation", run_keys)
    idle = list(range(n_workers))
    heapq.heapify(idle)
    submitted = deque()
    events = []
    sequence = itertools.count()
    finish = [0.0] * n_runs
    busy_worker_s = 0.0
    scheduler_busy_s = 0.0
    server_free = ready0

    def push(at, kind, payload):
        heapq.heappush(events, (at, next(sequence), kind, payload))

    def request_server(at, cost, kind, payload):
        nonlocal server_free, scheduler_busy_s
        start = max(at, server_free)
        end = start + cost
        server_free = end
        scheduler_busy_s += cost
        push(end, kind, payload)

    def start_submitted(at):
        nonlocal busy_worker_s
        while idle and submitted:
            worker = heapq.heappop(idle)
            dispatch_item = submitted.popleft()
            run, generation, index = dispatch_item.payload
            duration = float(run_durs[run][generation, index]) * factors[worker]
            busy_worker_s += duration
            push(at + duration, "worker_done", (worker, dispatch_item))

    def refill(at):
        while (dispatch_item := core.next_task()) is not None:
            submitted.append(dispatch_item)
        start_submitted(at)

    def report_batch(at, run, generation):
        if inline_masks is None:
            indexes = range(np.shape(run_durs[run])[1])
        else:
            indexes = np.flatnonzero(inline_masks[run][generation]).tolist()
        tasks = [DispatchItem(
            task_id=(run, generation, index), phase="simulation", run_key=str(run),
            generation=generation, tie_key=(run, index), payload=(run, generation, index),
        ) for index in indexes]
        barrier = core.report_ready(str(run), generation, tasks)
        if barrier is not None:
            transition(at, run, generation)
        else:
            refill(at)

    def transition(at, run, generation):
        next_generation = generation + 1
        cost = finish_cost[run]
        if next_generation < GENS:
            cost += prepare[run]
            request_server(at, cost, "batch_ready", (run, next_generation))
        else:
            request_server(at, cost, "run_done", (run, generation))

    # The real scheduler prepares initial runs in plan order and refills after
    # each one, without polling completions until all initial runs are ready.
    for run in range(n_runs):
        request_server(ready0, prepare[run], "batch_ready", (run, 0))

    now = ready0
    while events:
        now, _seq, kind, payload = heapq.heappop(events)
        if kind == "batch_ready":
            report_batch(now, *payload)
        elif kind == "worker_done":
            worker, dispatch_item = payload
            heapq.heappush(idle, worker)
            # A queued future starts without scheduler participation.
            start_submitted(now)
            request_server(now, journal_cost, "journal_done", dispatch_item)
        elif kind == "journal_done":
            dispatch_item = payload
            barrier = core.complete(dispatch_item.task_id)
            if barrier is not None:
                transition(now, int(barrier.run_key), barrier.generation)
            else:
                refill(now)
        elif kind == "run_done":
            run, generation = payload
            if trace is not None:
                trace.append((run, generation, now))
            finish[run] = now
            core.mark_run_complete(str(run))
            refill(now)
        else:
            raise AssertionError(f"unknown virtual event {kind!r}")
    if not core.phase_complete:
        raise AssertionError("continuous simulation ended before its phase barrier")
    makespan = max(finish, default=ready0) - ready0
    return makespan, finish, {
        "worker_utilization": busy_worker_s / (n_workers * makespan) if makespan else 0.0,
        "scheduler_busy_fraction": scheduler_busy_s / makespan if makespan else 0.0,
        "worker_busy_s": busy_worker_s,
        "scheduler_busy_s": scheduler_busy_s,
    }


def run_global_wave_dispatch(run_durs, factors, serial_costs, *, ready0=0.0,
                             inline_masks=None, assignment_rng=None):
    """Historical global generation barrier with fully drained waves."""
    n_runs = len(run_durs)
    n_workers = len(factors)
    prepare = list(serial_costs["prepare_s"])
    finish_cost = list(serial_costs["finish_s"])
    journal_cost = float(serial_costs["journal_s"])
    now = ready0
    busy_worker_s = 0.0
    scheduler_busy_s = 0.0
    for generation in range(GENS):
        preparation = sum(prepare)
        now += preparation
        scheduler_busy_s += preparation
        work = []
        for run in range(n_runs):
            indexes = (range(np.shape(run_durs[run])[1]) if inline_masks is None else
                       np.flatnonzero(inline_masks[run][generation]).tolist())
            work.extend((run, index) for index in indexes)
        for offset in range(0, len(work), n_workers):
            wave = work[offset:offset + n_workers]
            worker_order = np.arange(n_workers)
            if assignment_rng is not None:
                assignment_rng.shuffle(worker_order)
            completions = []
            for worker, (run, index) in enumerate(wave):
                duration = (float(run_durs[run][generation, index])
                            * factors[int(worker_order[worker])])
                busy_worker_s += duration
                completions.append(now + duration)
            journal_free = now
            for completed_at in sorted(completions):
                journal_free = max(journal_free, completed_at) + journal_cost
                scheduler_busy_s += journal_cost
            now = max(completions, default=now, )
            now = max(now, journal_free)
        finishing = sum(finish_cost)
        now += finishing
        scheduler_busy_s += finishing
    makespan = now - ready0
    return makespan, [now] * n_runs, {
        "worker_utilization": busy_worker_s / (n_workers * makespan) if makespan else 0.0,
        "scheduler_busy_fraction": scheduler_busy_s / makespan if makespan else 0.0,
        "worker_busy_s": busy_worker_s,
        "scheduler_busy_s": scheduler_busy_s,
    }


def one_trial(model, rng, factors20, factors15):
    """Returns dict of results for one Monte Carlo trial (common random numbers)."""
    draws = {(a, t, s): model.draw_run(rng, a, t) for t in TASKS for a in ARMS for s in range(SEEDS)}
    res = {"serial_rows": {}, "inter_alone_rows": {}, "serial_batch_spans": {}, "inter_pool_row_finish": {},
           "serial_row_batch_spans": {}}
    # --- SERIAL_PER_RUN ---
    total_serial = 0.0
    for t in TASKS:
        for a in ARMS:
            row = 0.0
            spans_all = []
            for s in range(SEEDS):
                rt, spans = run_serial(draws[(a, t, s)], model.overhead(a), factors15)
                row += rt; spans_all.append(spans)
            res["serial_rows"][(a, t)] = row
            res["serial_row_batch_spans"][(a, t)] = float(np.mean(np.concatenate(spans_all)))
            total_serial += row
    # --- INTERLEAVED: each row alone (5 seeds pooled on 20 workers) ---
    for t in TASKS:
        for a in ARMS:
            ms, _ = run_interleaved([draws[(a, t, s)] for s in range(SEEDS)], [model.overhead(a)] * SEEDS, factors20)
            res["inter_alone_rows"][(a, t)] = ms
    # --- INTERLEAVED: full D7 phases (all arms x seeds of a task pooled) ---
    total_inter = 0.0
    for t in TASKS:
        keys = [(a, t, s) for a in ARMS for s in range(SEEDS)]
        ms, fin = run_interleaved([draws[k] for k in keys], [model.overhead(k[0]) for k in keys], factors20)
        for a in ARMS:
            res["inter_pool_row_finish"][(a, t)] = max(f for k, f in zip(keys, fin) if k[0] == a)
        res["inter_phase_" + t] = ms
        total_inter += ms
    res["serial_total"] = total_serial
    res["interleaved_total"] = total_inter
    return res


def summarize(x):
    x = np.asarray(x, float)
    n = len(x)
    return {"mean": x.mean(), "median": float(np.median(x)), "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)), "sd": x.std(ddof=1),
            "ci95_of_mean": [x.mean() - 1.96 * x.std(ddof=1) / math.sqrt(n), x.mean() + 1.96 * x.std(ddof=1) / math.sqrt(n)],
            "n_trials": n}


def hours(d):
    return {k: (v / 3600 if isinstance(v, (int, float, np.floating)) else [y / 3600 for y in v]) for k, v in d.items() if k != "n_trials"} | {"n_trials": d["n_trials"]}


def campaign(model, n_trials, seed, factors20, factors15):
    rng = np.random.default_rng(seed)
    tr = [one_trial(model, rng, factors20, factors15) for _ in range(n_trials)]
    out = {"serial_total_h": hours(summarize([r["serial_total"] for r in tr])),
           "interleaved_total_h": hours(summarize([r["interleaved_total"] for r in tr])),
           "difference_serial_minus_interleaved_h": hours(summarize([r["serial_total"] - r["interleaved_total"] for r in tr])),
           "ratio_serial_over_interleaved": summarize([r["serial_total"] / r["interleaved_total"] for r in tr]),
           "phase_h": {t: hours(summarize([r["inter_phase_" + t] for r in tr])) for t in TASKS},
           "rows": {}}
    for a in ARMS:
        for t in TASKS:
            k = (a, t)
            out["rows"][f"{a}|{t}"] = {
                "serial_h": hours(summarize([r["serial_rows"][k] for r in tr])),
                "interleaved_row_alone_h": hours(summarize([r["inter_alone_rows"][k] for r in tr])),
                "interleaved_pooled_phase_finish_h": hours(summarize([r["inter_pool_row_finish"][k] for r in tr])),
                "difference_alone_h": hours(summarize([r["serial_rows"][k] - r["inter_alone_rows"][k] for r in tr])),
                "mean_serial_batch_eval_makespan_s": float(np.mean([r["serial_row_batch_spans"][k] for r in tr])),
            }
    return out, tr


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    inp = load_inputs()
    ratio = inp["ratio"]
    f20 = speed_factors(worker_speeds(20), ratio)
    f15 = speed_factors(worker_speeds(15), ratio)
    result = {
        "distribution_case": "SUMMARY STATS ONLY (n=3 raw t_PPO observations per (arm,task), n=100 t_sentinel): "
                             "too few to resample, so a lognormal is fitted to the measured mean/sd.",
        "inputs": {"pe_ratio": ratio, "pe_ratio_source": f"{RUN}/pe_core_ratio_measurement_production_scale.json",
                   "valid_rate": inp["valid"], "t_ppo_mean_sd": {f"{a}|{t}": v for (a, t), v in inp["ppo"].items()},
                   "t_sentinel_mean_sd": inp["sent"], "t_develop_mean": inp["tdev"],
                   "orchestration_overhead_s_per_gen": inp["orch"],
                   "worker_layout_20": {"n_P": sum(1 for x in f20 if x == 1.0), "n_E": sum(1 for x in f20 if x != 1.0)},
                   "worker_layout_15": {"n_P": sum(1 for x in f15 if x == 1.0), "n_E": sum(1 for x in f15 if x != 1.0)}},
        "variants": {}}
    vg = json.load(open(f"{OUT}/validity_by_generation.json"))["arms"]
    curves = {}
    for a, r in vg.items():
        if a == "direct":   # monotone rise: logistic fit, logit(p) = b0 + b1*g over gens 0-19
            f = r["logit_fit_gens0_19"]
            curves[a] = [1 / (1 + math.exp(-(f["intercept"] + f["slope_per_gen"] * g))) for g in range(GENS)]
        else:               # step-then-plateau, not logistic: use the empirical pooled per-generation rate
            curves[a] = r["rate_by_generation_pooled_5_seeds"]
    trajs = {a: r["per_seed_rate_by_generation"] for a, r in vg.items()}
    cg = json.load(open(f"{OUT}/validity_round3_analysis.json"))["arms"]      # current-calibration measurement (round 3)
    cur_curves = {}
    for a, r in cg.items():
        if a == "direct":
            b = r["logit_slope"]; cur_curves[a] = [1 / (1 + math.exp(-(math.log(b["fitted_gen0"] / (1 - b["fitted_gen0"])) + b["slope"] * g))) for g in range(GENS)]
        else:
            cur_curves[a] = r["rate_by_generation"]
    cur_trajs = {a: r["per_seed_rate"] for a, r in cg.items()}
    variants = {
        "BASE": dict(),                                            # measured CV, B6 validity, decode counted in overhead
        "BASE_orch_only_no_decode": dict(decode_in_launcher=False),
        "SENS_cv10pct": dict(cv_floor=0.10),
        "SENS_cv25pct": dict(cv_floor=0.25),
        "SENS_bound_all_valid_after_gen0": dict(later_gen_valid=1.0),
        "HIST_curve_fitted_validity": dict(validity_curves=curves),
        "HIST_resampled_run_validity": dict(validity_trajs=trajs),
        "CUR_curve_validity": dict(validity_curves=cur_curves),
        "CUR_resampled_run_validity": dict(validity_trajs=cur_trajs),
    }
    t0 = time.time()
    for name, kw in variants.items():
        m = Model(inp, **kw)
        out, _ = campaign(m, n_trials, 20260924, f20, f15)
        result["variants"][name] = out
        b = out
        print(f"{name}: serial {b['serial_total_h']['mean']:.2f}h  interleaved {b['interleaved_total_h']['mean']:.2f}h "
              f"[p5 {b['interleaved_total_h']['p5']:.2f}, p95 {b['interleaved_total_h']['p95']:.2f}]", flush=True)
    result["historical_validity_curves_used"] = curves
    result["current_validity_curves_used"] = cur_curves
    result["n_trials"] = n_trials
    result["sim_wall_s"] = time.time() - t0
    json.dump(result, open(f"{OUT}/simulation_results.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
