"""Three task relationships from the exact existing Qwen3-0.6B Cedar adapter."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT/"outputs/model_compare_open_20261108"
OUT = MAIN/"binding_followup"
sys.path.insert(0, str(ROOT/"work/round5"))
import pilot as p
import codebook as c

MODEL = "Qwen3-0.6B"
ARMS = ("conflict", "consistent", "disjoint")
GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def problem(arm, a, b, variant):
    text, value = c.problem("related", a, b, variant)
    index = a + (variant == "counterfactual")
    if arm == "consistent":
        value = c.OLD[index]
    if arm in ("disjoint", "independent"):
        key = c.KEYS[index]
        replacement = key.lower() if arm == "disjoint" else independent_key(key)
        text = text.replace(f"key {key} in", f"key {replacement} in")
    return text, value


def independent_key(key):
    return {"Q": "J", "X": "V"}[key[0]] + chr(65+(ord(key[1])-65+13)%26)


def run(arms, seed=None):
    global OUT
    free, _ = p.torch.cuda.mem_get_info()
    if free < 6*1024**3:
        raise RuntimeError("Requires 6 GiB free GPU memory")
    engine = p.Engine(MODEL)
    root = MAIN/MODEL
    if seed is None:
        old_checkpoint = root/"checkpoints/old.pt"
        ck = p.torch.load(old_checkpoint, map_location="cpu", weights_only=True)
        assert ck["names"] == [name for name, _ in engine.adapters]
        assert set(ck["modes"]) == {"add"}
        engine.restore(ck["parameters"])
        splits = json.loads((root/"protocol.json").read_text())["splits"]
        old_ref = json.loads((root/"old_reference.json").read_text())
        grid = GRID
        training_seed = 18
    else:
        OUT = OUT/f"expanded_s{seed}"
        old_checkpoint = OUT/"old.pt"
        if arms == ("independent",):
            protocol = json.loads((OUT/"protocol.json").read_text())
            splits = protocol["splits"]
            c.KEYS = protocol["keys"]
            c.OLD = protocol["cedar_values"]
            c.NEW = protocol["maple_conflict_values"]
            ck = p.torch.load(old_checkpoint, map_location="cpu", weights_only=True)
            assert ck["names"] == [name for name, _ in engine.adapters]
            engine.restore(ck["parameters"])
            old_ref = json.loads((OUT/"old_reference.json").read_text())
            assert set(c.KEYS).isdisjoint(independent_key(key) for key in c.KEYS[1:])
        else:
            splits, old_ref = expanded(engine, seed)
        if old_ref is None:
            return
        grid = (0, 1, 2, 4, 8, 16, 32, 64)
        training_seed = seed+1
    state = engine.snapshot()
    p.problem = c.problem
    val_old = list(p.rows("old", splits["validation"], "plain", p.VARIANTS))
    test_old = list(p.rows("old", splits["test"], "plain", p.VARIANTS))
    check = p.evaluate(engine, val_old)
    assert [(r["id"], r["response"]) for r in check] == [
        (r["id"], r["response"]) for r in old_ref["validation"]]
    protocol_name = "independent_protocol.json" if arms == ("independent",) else "protocol.json"
    p.write(OUT/protocol_name, {"model": MODEL, "arms": arms, "grid": grid, "splits": splits,
            "training_seed": training_seed, "selection_delta": .125, "batch": 4, "lr": .001,
            "gpu_free_before_bytes": free, "old_checkpoint": str(old_checkpoint),
            "keys": list(c.KEYS), "cedar_values": c.OLD, "maple_conflict_values": c.NEW})
    for arm in arms:
        start = time.perf_counter()
        engine.restore(state)
        opt = p.optimizer(engine.ps, .001)

        def select_problem(task, a, b, variant):
            return problem(arm, a, b, variant) if task == "related" else c.problem(task, a, b, variant)

        p.problem = select_problem
        train = list(p.rows("related", splits["train"], "plain", ("original", "counterfactual")))
        val_new = list(p.rows("related", splits["validation"], "plain", ("original",)))
        test_new = list(p.rows("related", splits["test"], "plain", ("original",)))
        batches = p.sample(train, training_seed)
        trace, curve = [], []
        chosen = None
        for step in range(65):
            if step:
                trace.append(p.train_step(engine, opt, batches[step-1]))
                if p.torch.cuda.max_memory_reserved() > 6*1024**3:
                    raise RuntimeError("6 GiB peak reserved memory limit exceeded")
            if step not in grid:
                continue
            new = p.evaluate(engine, val_new)
            old = check if not step else p.evaluate(engine, val_old)
            gain = p.acc(new) - (p.acc(curve[0]["new"]) if curve else p.acc(new))
            curve.append({"step": step, "gain": gain, "new": new, "old": old})
            if chosen is None and gain >= .125:
                chosen = step
                selected_state = engine.snapshot()
            print(arm, step, f"gain={gain:.3f}", flush=True)
        fixed = {"old": p.evaluate(engine, test_old), "new": p.evaluate(engine, test_new)}
        engine.save(OUT/f"{arm}_64.pt")
        matched = None
        if chosen is not None:
            engine.restore(selected_state)
            matched = fixed if chosen == 64 else {"old": p.evaluate(engine, test_old),
                                                  "new": p.evaluate(engine, test_new)}
            if chosen != 64:
                engine.save(OUT/f"{arm}_{chosen}.pt")
        p.write(OUT/f"{arm}.json", {"curve": curve, "trace": trace, "chosen": chosen, "fixed": fixed,
                                    "matched": matched, "seconds": time.perf_counter()-start,
                                    "peak_allocated_bytes": p.torch.cuda.max_memory_allocated()})
    print("Finished", ", ".join(arms), flush=True)


def expanded(engine, seed, cedar_values=None):
    from itertools import product
    from string import ascii_uppercase

    rng = p.random.Random(seed)
    c.KEYS = ("_",) + tuple(a+b for a, b in list(product("QXZ", ascii_uppercase))[:48])
    sampled = rng.sample(range(10, 100), 48)
    c.OLD = [0] + (sampled if cedar_values is None else list(cedar_values))
    c.NEW = [0] + c.OLD[25:] + c.OLD[1:25]
    contexts = list(range(1, 24))
    rng.shuffle(contexts)
    splits = {k: list(product(range(1, 49, 2), v)) for k, v in
              (("train", contexts[:14]), ("validation", contexts[14:17]), ("test", contexts[17:]))}
    engine.reset(seed, "add")
    p.problem = c.problem
    val = list(p.rows("old", splits["validation"], "plain", p.VARIANTS))
    test = list(p.rows("old", splits["test"], "plain", p.VARIANTS))
    pool = list(p.rows("old", splits["train"], "plain", p.VARIANTS))
    baseline = p.evaluate(engine, val)
    opt = p.optimizer(engine.ps, .001)
    trace, checkpoints = [], []
    for step, batch in enumerate(p.sample(pool, seed, 1440), 1):
        trace.append(p.train_step(engine, opt, batch))
        if p.torch.cuda.max_memory_reserved() > 6*1024**3:
            raise RuntimeError("6 GiB peak reserved memory limit exceeded")
        if step not in (480, 960, 1440):
            continue
        predictions = p.evaluate(engine, val)
        gain = p.acc(predictions[::3])-p.acc(baseline[::3])
        tri = sum(all(r["correct"] for r in predictions[i:i+3])
                  for i in range(0, len(predictions), 3))/(len(predictions)//3)
        checkpoints.append({"step": step, "original_gain": gain, "triplet": tri})
        p.write(OUT/"old_gate.json", {"seed": seed, "splits": splits, "keys": list(c.KEYS),
                "cedar_values": c.OLD, "maple_conflict_values": c.NEW, "trace": trace,
                "validation_checkpoints": checkpoints, "gate": tri >= .75 and gain >= .125,
                "peak_mib": p.torch.cuda.max_memory_allocated()/2**20})
        print("EXPANDED", seed, step, "old triple", round(tri, 3), flush=True)
        if tri >= .75 and gain >= .125:
            reference = {"validation": predictions, "test": p.evaluate(engine, test)}
            p.write(OUT/"old_reference.json", reference)
            engine.save(OUT/"old.pt")
            return splits, reference
    return splits, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("arms", nargs="*", choices=(*ARMS, "independent"))
    args = parser.parse_args()
    arms = tuple(args.arms) or ARMS
    if "independent" in arms and (arms != ("independent",) or args.seed is None):
        parser.error("independent must be run alone with an existing expanded --seed")
    run(arms, args.seed)

