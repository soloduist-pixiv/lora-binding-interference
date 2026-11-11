"""Equal-batch/shape comparison of four objectives from each existing Cedar adapter."""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT/"outputs/model_compare_open_20261108"
OUT = MAIN/"binding_constraints"
sys.path.insert(0, str(ROOT/"work/round5"))
import pilot as p
import codebook as c

ARMS = ("plain", "replay", "kl", "boundary")
GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
LAMBDA, ETA, CAP, EPSILON, DELTA = .5, .25, 3., .1, 1.
TOLERANCE = 2
MIN_GAIN = 9


def forward(engine, rows):
    sequences, positions, targets, sizes = [], [], [], []
    for i, row in enumerate(rows):
        prompt = engine.encode(row["text"])
        labels = engine.tok.encode(str(row["answer"]), add_special_tokens=False) + [engine.tok.eos_token_id]
        sequences.append(prompt + labels[:-1])
        sizes.append(len(labels))
        positions.extend((i, len(prompt)-1+j) for j in range(len(labels)))
        targets.extend(labels)
    z = p.logits(engine, sequences, positions).log_softmax(-1)
    return z.split(sizes), p.torch.tensor(targets).split(sizes), sum(map(len, sequences))


def sequence_logp(z, target):
    return z.gather(1, target[:, None]).sum()


def swapped(row, answer):
    return {**row, "answer": answer}


def key(row):
    return int(row["id"].split("/")[0]) + (row["variant"] == "counterfactual")


def unit(engine, pairs, rows, teacher=None, opt=None, arm=None):
    z, labels, tokens = forward(engine, rows)
    sums = [sequence_logp(value, y) for value, y in zip(z, labels)]
    if teacher is None:
        return z[4].detach(), float(sums[4]-sums[5]), len(labels[4])
    target, reference, reliable = teacher
    new = -sum(sums[:4])/sum(map(len, labels[:4]))
    old_margin = sums[4]-sums[5]
    new_margin = sums[0]-sums[6]
    if arm == "replay":
        penalty = -LAMBDA*sums[4]/len(labels[4])
    elif arm == "kl":
        penalty = LAMBDA*p.F.kl_div(z[4], target, log_target=True, reduction="batchmean")
    elif arm == "boundary":
        old = (min(reference, CAP)-EPSILON-old_margin).relu()
        penalty = LAMBDA*old*reliable + ETA*(DELTA-new_margin).relu()
    else:
        penalty = new.new_zeros(())
    opt.zero_grad(set_to_none=True)
    loss = new + penalty
    assert p.torch.isfinite(loss)
    loss.backward()
    opt.step()
    if p.torch.cuda.max_memory_reserved() > 6*1024**3:
        raise RuntimeError("6 GiB reserved limit reached")
    return {"new_ce": new.item(), "penalty": penalty.item(), "old_margin": old_margin.item(),
            "new_margin": new_margin.item(), "tokens": tokens, "sequences": len(rows),
            "scored_tokens": sum(map(len, labels)), "key": key(pairs)}


def matching(curves, minimum=MIN_GAIN):
    def choose(names):
        grid = [[x for x in curves[n] if x["gain_count"] >= minimum] for n in names]
        candidates = []
        for points in itertools.product(*grid):
            gains = [x["gain_count"] for x in points]
            spread = max(gains)-min(gains)
            if spread <= TOLERANCE:
                steps = tuple(x["step"] for x in points)
                candidates.append(((max(gains), spread, sum(steps), steps), steps))
        return dict(zip(names, min(candidates)[1])) if candidates else None
    return {"all": choose(ARMS), **{f"boundary_vs_{n}": choose(("boundary", n)) for n in ARMS[:-1]}}


def exploratory(e, initial, references, trial_rows, folder, old_val, new_val, old_test, new_test):
    """Replay only missing points for a secondary, lower-gain common support check."""
    raw = {n: json.loads((folder/f"{n}.json").read_text()) for n in ARMS}
    primary = json.loads((folder/"matching.json").read_text())
    low = matching({n:r["curve"] for n,r in raw.items()}, minimum=5)
    selected = {k:pair for k,pair in low.items() if pair and primary[k] is None}
    scores = {}
    for arm in ARMS:
        steps = {m[arm] for m in selected.values() if arm in m} - {int(s) for s in raw[arm]["test"]}
        if not steps:
            continue
        e.restore(initial)
        opt = p.optimizer(e.ps, .001)
        scores[arm] = {}
        for i in range(max(steps)):
            lp, margin, valid = references[i]
            rows = trial_rows[i]
            got = unit(e, rows[0], rows, (lp.float().log_softmax(-1), margin, valid), opt, arm)
            expect = raw[arm]["trace"][i]
            assert all(abs(got[k]-expect[k]) < 1e-5 for k in ("new_ce","penalty","old_margin","new_margin"))
            step = i+1
            if step in steps:
                point = next(q for q in raw[arm]["curve"] if q["step"] == step)
                for task, examples in (("old",old_val), ("new",new_val)):
                    assert [x["response"] for x in p.evaluate(e, examples)] == [x["response"] for x in point[task]]
                scores[arm][str(step)] = {"old":p.evaluate(e,old_test), "new":p.evaluate(e,new_test)}
                e.save(folder/f"{arm}_{step}.pt")
    p.write(folder/"exploratory.json", {"minimum":5, "selected":selected, "extra_test":scores})
    print("LOW GAIN", selected, flush=True)


def run(seed, low_gain=False, light=False):
    global LAMBDA, ETA
    if light:
        LAMBDA, ETA = .10, .05
    folder = OUT/(f"light_seed_{seed}" if light else f"seed_{seed}")
    folder.mkdir(parents=True, exist_ok=True)
    if low_gain:
        curves = {n:json.loads((folder/f"{n}_training.json").read_text())["curve"] for n in ARMS}
        primary = json.loads((folder/"matching.json").read_text())
        options = matching(curves, minimum=5)
        if not any(pair and primary[k] is None for k,pair in options.items()):
            p.write(folder/"exploratory.json", {"minimum":5, "selected":{}, "extra_test":{}})
            print("LOW GAIN no eligible supplemental comparisons", flush=True)
            return
    src = MAIN/"binding_followup"/f"expanded_s{seed}"
    proto = json.loads((src/"protocol.json").read_text())
    c.KEYS, c.OLD, c.NEW = proto["keys"], proto["cedar_values"], proto["maple_conflict_values"]
    p.problem = c.problem
    splits = proto["splits"]
    free, _ = p.torch.cuda.mem_get_info()
    assert free >= 6*1024**3, "Requires at least 6 GiB free"
    e = p.Engine("Qwen3-0.6B")
    ck = p.torch.load(src/"old.pt", map_location="cpu", weights_only=True)
    assert ck["names"] == [n for n,_ in e.adapters] and set(ck["modes"]) == {"add"}
    e.restore(ck["parameters"])
    initial = e.snapshot()
    ref = json.loads((src/"old_reference.json").read_text())
    old_val = list(p.rows("old", splits["validation"], "plain", p.VARIANTS))
    old_test = list(p.rows("old", splits["test"], "plain", p.VARIANTS))
    new_val = list(p.rows("related", splits["validation"], "plain", ("original",)))
    new_test = list(p.rows("related", splits["test"], "plain", ("original",)))
    pool = list(p.rows("related", splits["train"], "plain", ("original", "counterfactual")))
    old_pool = {(row["id"], row["variant"]): row for row in
                p.rows("old", splits["train"], "plain", p.VARIANTS)}
    batches = p.sample(pool, seed+1, 64)
    trial_rows, references = [], []
    reliable = {}
    started = time.perf_counter()
    with p.torch.no_grad():
        for i, batch in enumerate(batches):
            sample = batch[0]
            k = key(sample)
            variant = "equivalent" if sample["variant"] == "original" and i % 2 else sample["variant"]
            old = old_pool[sample["id"], variant]
            assert key(old) == k and old["answer"] == c.OLD[k] and sample["answer"] == c.NEW[k]
            rows = [*batch, old, swapped(old, c.NEW[k]), swapped(sample, c.OLD[k])]
            # Check the reference on all three old prompts for the sampled family.
            if sample["id"] not in reliable:
                triplet = [old_pool[sample["id"], v] for v in p.VARIANTS]
                reliable[sample["id"]] = all(r["correct"] for r in p.evaluate(e, triplet))
            lp, margin, _ = unit(e, sample, rows)
            references.append((lp.to(p.torch.float16), margin, reliable[sample["id"]] and margin > 0))
            trial_rows.append(rows)
    teacher_seconds = time.perf_counter()-started
    assert any(valid for _,_,valid in references)
    original = p.evaluate(e, old_val)
    assert [r["response"] for r in original] == [r["response"] for r in ref["validation"]]
    baseline = p.evaluate(e, new_val)
    assert len(baseline) == 72
    if low_gain:
        exploratory(e, initial, references, trial_rows, folder, old_val, new_val, old_test, new_test)
        return
    p.write(folder/"protocol.json", {"seed": seed, "arms": ARMS, "grid": GRID, "steps": 64, "lr": .001,
        "new_batch": 4, "score_sequences_per_step": 7, "lambda": LAMBDA, "eta": ETA, "cap": CAP,
        "epsilon": EPSILON, "delta": DELTA, "min_gain_count": MIN_GAIN, "max_match_gap_count": TOLERANCE,
        "light_sensitivity": light,
        "teacher_seconds": teacher_seconds, "reliable_memory_steps": sum(x[2] for x in references),
        "gpu_free_before_bytes": free, "old_checkpoint": str(src/"old.pt"),
        "batches": [[(r["id"],r["variant"],r["answer"]) for r in rows] for rows in trial_rows]})
    curves, states, runs = {}, {}, {}
    for arm in ARMS:
        t = time.perf_counter()
        e.restore(initial)
        opt = p.optimizer(e.ps, .001)
        trace, curve = [], []
        states[arm] = {}
        p.torch.cuda.reset_peak_memory_stats()
        for step in range(65):
            if step:
                lp, margin, valid = references[step-1]
                target = lp.float().log_softmax(-1)
                rows = trial_rows[step-1]
                trace.append(unit(e, rows[0], rows, (target, margin, valid), opt, arm))
            if step in GRID:
                new = p.evaluate(e, new_val) if step else baseline
                old = (p.evaluate(e, old_val) if not light else None) if step else original
                gain = sum(r["correct"] for r in new)-sum(r["correct"] for r in baseline)
                curve.append({"step": step, "gain_count": gain, "old": old, "new": new})
                states[arm][step] = [x.cpu() for x in e.snapshot()]
                print(seed, arm, step, gain, flush=True)
        curves[arm] = curve
        runs[arm] = {"curve": curve, "trace": trace, "seconds": time.perf_counter()-t,
                     "peak_allocated_mib": p.torch.cuda.max_memory_allocated()/2**20,
                     "peak_reserved_mib": p.torch.cuda.max_memory_reserved()/2**20}
        p.write(folder/f"{arm}_training.json", runs[arm])
    selected = matching(curves)
    p.write(folder/"matching.json", selected)
    for arm in ARMS:
        steps = {64} | {pair[arm] for pair in selected.values() if pair and arm in pair}
        tests = {}
        for step in sorted(steps):
            e.restore(states[arm][step])
            tests[str(step)] = {"old": p.evaluate(e, old_test), "new": p.evaluate(e, new_test)}
            e.save(folder/f"{arm}_{step}.pt")
        p.write(folder/f"{arm}.json", {"test": tests, **runs[arm]})
    budgets = [[(r["tokens"],r["scored_tokens"],r["sequences"]) for r in runs[a]["trace"]] for a in ARMS]
    assert all(b == budgets[0] for b in budgets)
    p.write(folder/"completed.json", {"matched": selected, "equal_budget": True, "seconds": time.perf_counter()-started})
    print("DONE", seed, selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, choices=(29, 43))
    parser.add_argument("--low-gain", action="store_true")
    parser.add_argument("--light", action="store_true")
    args = parser.parse_args()
    if args.low_gain and args.light:
        parser.error("low-gain replay applies to the main configuration")
    run(args.seed, args.low_gain, args.light)
