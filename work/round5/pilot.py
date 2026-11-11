"""Round 5 open-answer pilot. Run from project root: python -u work/round5/pilot.py."""
import json
import random
import re
import sys
import time
from itertools import product
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round4"))
from compare_models import Engine
from core import ROOT, optimizer, torch, write
from torch.nn import functional as F

OUT = ROOT / "outputs/round5_20261108"
SEED = 17
OLD_STEPS = 64
STUDY = "arithmetic"
GRID = (0, 1, 2, 4, 8, 16, 32, 64)
TASKS = ("old", "related", "unrelated")
VARIANTS = ("original", "equivalent", "counterfactual")
OLD_TRAIN_VARIANTS = VARIANTS[:2]
NEW_TRAIN_VARIANTS = ("original",)
ARMS = tuple(product(TASKS[1:], ("plain", "tagged")))
FORMATS = {
    "plain": "Reply with only the integer.",
    "tagged": "Reply with exactly the word Result, a colon, a space, and the integer.",
}


def problem(task, a, b, variant):
    if variant == "counterfactual":
        a += 1
    if task == "old":
        value = 3 * a + 2 * b
        text = (f"How many points altogether? There are {b} blue chips worth 2 points each "
                f"and {a} red chips worth 3 each." if variant == "equivalent" else
                f"You have {a} red chips at 3 points each and {b} blue chips at 2 points each. "
                "What is the total score?")
    elif task == "related":
        value = 2 * a + 3 * b
        text = (f"Find the combined score of {b} blue chips worth 3 points apiece and "
                f"{a} red chips worth 2 apiece." if variant == "equivalent" else
                f"You have {a} red chips at 2 points each and {b} blue chips at 3 points each. "
                "What is the total score?")
    else:
        value = 2 * a + 3 * b
        letters = "ABCDEFGHIJKLMNOP"
        rows = [f"{letters[i]}: {2*i+3*b}" for i in range(16)]
        random.Random(b).shuffle(rows)
        text = f"Registry entries: {', '.join(rows)}. What number is listed for key {letters[a]}?"
    return text, value


def data():
    # Hold out the entire (a,b) family, including the adjacent a counterfactual.
    families = list(product(range(1, 15, 2), range(1, 47)))
    random.Random(SEED).shuffle(families)
    splits = {"train": families[:144], "validation": families[144:168], "test": families[168:296]}
    assert len(set(sum(splits.values(), []))) == 296
    return splits


def rows(task, pairs, protocol, variants):
    for a, b in pairs:
        for variant in variants:
            text, answer = problem(task, a, b, variant)
            yield {"id": f"{a}/{b}", "variant": variant, "answer": answer,
                   "text": f"{text}\n{FORMATS[protocol]}", "protocol": protocol}


def logits(engine, sequences, positions):
    model = engine.model.model
    length = max(map(len, sequences))
    tokens = torch.tensor([s + [0] * (length-len(s)) for s in sequences])
    embeddings = model.embed_tokens(tokens).to("cuda", dtype=torch.bfloat16)
    hidden = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
    batch = torch.tensor([i for i, _ in positions], device="cuda")
    index = torch.tensor([p for _, p in positions], device="cuda")
    return engine.model.lm_head(hidden[batch, index].float().cpu())


def train_step(engine, opt, batch):
    sequences, positions, labels = [], [], []
    for i, row in enumerate(batch):
        prefix = engine.encode(row["text"])
        target = f"Result: {row['answer']}" if row["protocol"] == "tagged" else str(row["answer"])
        tail = engine.tok.encode(target, add_special_tokens=False) + [engine.tok.eos_token_id]
        sequences.append(prefix + tail[:-1])
        positions.extend((i, len(prefix)-1+k) for k in range(len(tail)))
        labels.extend(tail)
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(logits(engine, sequences, positions), torch.tensor(labels))
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    return loss.item(), len(labels), sum(map(len, sequences))


def parse(text, protocol):
    text = text.strip()
    expected = r"-?\d+" if protocol == "plain" else r"Result: -?\d+"
    format_ok = re.fullmatch(expected, text) is not None
    numbers = re.findall(r"(?<![\w.])-?\d+(?![\w.])", text)
    return (int(numbers[0]) if len(numbers) == 1 else None), format_ok


@torch.no_grad()
def evaluate(engine, examples):
    results = []
    eos = engine.loaded.generation_config.eos_token_id
    stop = set(eos if isinstance(eos, (tuple, list)) else [eos]) | {engine.tok.eos_token_id}
    for start in range(0, len(examples), 32):
        group = examples[start:start+32]
        prefixes = [engine.encode(r["text"]) for r in group]
        generated = [[] for _ in group]
        active = list(range(len(group)))
        for _ in range(8):
            if not active:
                break
            seqs = [prefixes[i] + generated[i] for i in active]
            ids = logits(engine, seqs, [(i, len(s)-1) for i, s in enumerate(seqs)]).argmax(-1).tolist()
            remaining = []
            for i, token in zip(active, ids):
                if token not in stop:
                    generated[i].append(token)
                    remaining.append(i)
            active = remaining
        for row, tokens in zip(group, generated):
            text = engine.tok.decode(tokens, skip_special_tokens=True)
            pred, fmt = parse(text, row["protocol"])
            results.append({**row, "response": text, "prediction": pred,
                            "correct": pred == row["answer"], "format_ok": fmt})
    return results


def sample(pool, seed, n=64):
    rng = random.Random(seed)
    return [rng.sample(pool, 4) for _ in range(n)]


def acc(examples):
    return mean(r["correct"] for r in examples)


def run(model="MiniCPM5-1B"):
    started = time.perf_counter()
    splits = data()
    protocol = {"seed": SEED, "model": model, "grid": GRID, "lr": .001, "batch": 4,
                "arms": ARMS, "splits": splits, "old_steps": OLD_STEPS, "study": STUDY,
                "old_train_variants": OLD_TRAIN_VARIANTS, "new_train_variants": NEW_TRAIN_VARIANTS,
                "matching": "first checkpoint at +12.5pp validation numeric ACC; no fallback match",
                "old_gate": "+12.5pp validation original numeric ACC",
                "max_generated_tokens": 8}
    write(OUT/"protocol.json", protocol)
    engine = Engine(model)
    engine.reset(SEED, "add")
    val_old = list(rows("old", splits["validation"], "plain", VARIANTS))
    test_old = list(rows("old", splits["test"], "plain", VARIANTS))
    original = lambda examples: [r for r in examples if r["variant"] == "original"]
    baseline = {"validation": evaluate(engine, val_old), "test": evaluate(engine, test_old)}
    write(OUT/"old_base.json", baseline)
    opt = optimizer(engine.ps, .001)
    old_pool = list(rows("old", splits["train"], "plain", OLD_TRAIN_VARIANTS))
    old_trace = [train_step(engine, opt, batch) for batch in sample(old_pool, SEED, OLD_STEPS)]
    engine.save(OUT/"checkpoints/old.pt")
    reference = {"validation": evaluate(engine, val_old), "test": evaluate(engine, test_old),
                 "trace": old_trace}
    write(OUT/"old_reference.json", reference)
    gain = acc(original(reference["validation"]))-acc(original(baseline["validation"]))
    print("OLD VALIDATION GAIN", gain, flush=True)
    reference_state = engine.snapshot()
    for task, protocol_name in ARMS:
        start = time.perf_counter()
        engine.restore(reference_state)
        opt = optimizer(engine.ps, .001)
        name = f"{task}_{protocol_name}"
        pool = list(rows(task, splits["train"], protocol_name, NEW_TRAIN_VARIANTS))
        val_new = list(rows(task, splits["validation"], protocol_name, ("original",)))
        test_new = list(rows(task, splits["test"], protocol_name, ("original",)))
        trace, curve = [], []
        batches = sample(pool, SEED+1)
        initial = acc(evaluate(engine, val_new))
        selected = None
        for step in range(65):
            if step:
                trace.append(train_step(engine, opt, batches[step-1]))
            if step not in GRID:
                continue
            new = evaluate(engine, val_new)
            old = evaluate(engine, val_old)
            curve.append({"step": step, "new": new, "old": old})
            print(name, step, "new", round(acc(new), 3), "old", round(acc(original(old)), 3), flush=True)
            if step and selected is None and acc(new)-initial >= .125:
                selected = step
                selected_state = engine.snapshot()
        final = {"old": evaluate(engine, test_old), "new": evaluate(engine, test_new)}
        engine.save(OUT/f"checkpoints/{name}_64.pt")
        chosen = None
        if selected is not None:
            engine.restore(selected_state)
            chosen = final if selected == 64 else {"old": evaluate(engine, test_old), "new": evaluate(engine, test_new)}
            if selected != 64:
                engine.save(OUT/f"checkpoints/{name}_{selected}.pt")
        write(OUT/f"{name}.json", {"curve": curve, "trace": trace, "selected": selected,
                                 "fixed_test": final, "matched_test": chosen,
                                 "seconds": time.perf_counter()-start})
    write(OUT/"completed.json", {"old_gate_met": gain >= .125, "old_validation_gain": gain,
                               "seconds": time.perf_counter()-started,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "trainable": sum(p.numel() for p in engine.ps)})


if __name__ == "__main__":
    run()


