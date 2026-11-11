"""Serial, resumable review experiments with fixed learning-progress targets."""
import argparse
import importlib.util
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/prospective_updates"))
from experiment import c, d, p, x
sys.path.insert(0, str(ROOT / "work/binding_review"))
from cached_score import CandidateScorer
from compare_models import Engine as StandardEngine

OUT = ROOT / "outputs/binding_revision2_20261111"
HISTORICAL = d.MAIN / "prospective_updates/disjoint_mass"


def values_for(seed, pairing, tokenizer):
    rng = random.Random(seed + 8000)
    pairs = [(v, v + 1) for v in range(100, 196, 2)]
    rng.shuffle(pairs)
    pairs = [pair if rng.random() < .5 else pair[::-1] for pair in pairs]
    old, new = map(list, zip(*pairs))
    if pairing == "permuted":
        ordered = sorted(new)
        mapping = dict(zip(ordered, ordered[24:] + ordered[:24], strict=True))
        new = [mapping[v] for v in new]
        assert all(a // 10 != b // 10 for a, b in zip(old, new))
    active_rng = random.Random()
    active_rng.setstate(rng.getstate())
    if pairing == "support_permuted":
        sample_rng = random.Random()
        sample_rng.setstate(active_rng.getstate())
        active = set(sample_rng.sample(list(range(1,49,2)),16))
        groups = [[key-1 for key in sorted(active)],
                  [key for key in sorted(active)],
                  [i for i in range(48) if i+1 not in active and i not in active]]
        permutation_rng = random.Random(seed+48295)
        before = list(new)
        for indices in groups:
            values = [before[i] for i in indices]
            for _ in range(10000):
                permutation_rng.shuffle(values)
                if all(old[i]//10 != value//10 for i,value in zip(indices,values)):
                    break
            else:
                raise RuntimeError("Could not construct support-preserving prefix derangement")
            for i,value in zip(indices,values):
                new[i] = value
            assert {before[i] for i in indices} == {new[i] for i in indices}
    if pairing == "alphabetic":
        pool = []
        while len(pool) < 96:
            word = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
            if word not in pool and len(tokenizer.encode(word, add_special_tokens=False)) == 3:
                pool.append(word)
        old, new = pool[:48], pool[48:]
    return old, new, active_rng


def restore(engine, checkpoint, rank, seed):
    p.torch.manual_seed(seed)
    for (_, adapter), old_a, old_b in zip(engine.adapters, checkpoint["parameters"][::2],
                                         checkpoint["parameters"][1::2], strict=True):
        a = p.torch.empty((rank, old_a.shape[1]), device="cuda", dtype=old_a.dtype)
        b = p.torch.zeros((old_b.shape[0], rank), device="cuda", dtype=old_b.dtype)
        p.torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        a[:old_a.shape[0]] = old_a
        b[:, :old_b.shape[1]] = old_b
        adapter.A = p.torch.nn.Parameter(a)
        adapter.B = p.torch.nn.Parameter(b)
    engine.ps = [value for _, adapter in engine.adapters for value in (adapter.A, adapter.B)]


def rows_for(splits, task, variants, code_format):
    rows = list(p.rows(task, splits, "plain", variants))
    if code_format:
        rows = [dict(row, text=row["text"].replace("Which integer corresponds", "Which code corresponds")
                     .replace("Reply with only the integer.", "Reply with only the code."))
                for row in rows]
    return rows


@p.torch.no_grad()
def generate(scorer, rows):
    engine = scorer.engine
    results = []
    eos = engine.tok.eos_token_id
    for start in range(0, len(rows), 24):
        group = rows[start:start + 24]
        tokens = [[] for _ in group]
        active = list(range(len(group)))
        for _ in range(12):
            if not active:
                break
            seqs = [engine.encode(group[i]["text"]) + tokens[i] for i in active]
            length = max(map(len, seqs))
            ids = p.torch.tensor([s + [0] * (length - len(s)) for s in seqs])
            model = engine.model.model
            embeddings = model.embed_tokens(ids).to("cuda", dtype=p.torch.bfloat16)
            hidden = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
            index = p.torch.tensor([len(s) - 1 for s in seqs], device="cuda")
            output = scorer.logits(hidden[p.torch.arange(len(seqs), device="cuda"), index]).argmax(-1).tolist()
            remaining = []
            for i, token in zip(active, output):
                if token != eos:
                    tokens[i].append(token)
                    remaining.append(i)
            active = remaining
        for row, generated in zip(group, tokens):
            answer = engine.tok.decode(generated, skip_special_tokens=True).strip()
            results.append(dict(id=row["id"], variant=row["variant"], answer=str(row["answer"]),
                                response=answer, correct=answer == str(row["answer"])))
    return results


def fresh_anchor(engine, scorer, seed, folder, code_format):
    gate_file = folder / "anchor.json"
    checkpoint_file = folder / "old.pt"
    if checkpoint_file.exists() and gate_file.exists():
        meta = d.read(gate_file)
        if not meta["passed"]:
            raise RuntimeError("Recorded Cedar anchor did not pass")
        checkpoint = p.torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        restore(engine, checkpoint, 4, seed)
        return meta["splits"], checkpoint
    contexts = list(range(1, 24))
    random.Random(seed).shuffle(contexts)
    splits = {name: list(itertools.product(range(1, 49, 2), context))
              for name, context in (("train", contexts[:14]), ("validation", contexts[14:17]),
                                    ("test", contexts[17:]))}
    engine.reset(seed, "add")
    validation = rows_for(splits["validation"], "old", p.VARIANTS, code_format)
    baseline = generate(scorer, validation)
    base_accuracy = np.mean([row["correct"] for row in baseline[::3]])
    pool = rows_for(splits["train"], "old", p.VARIANTS, code_format)
    opt = p.optimizer(engine.ps, .001)
    trace = []
    for step, batch in enumerate(p.sample(pool, seed, 1920), 1):
        p.train_step(engine, opt, batch)
        x.check_memory()
        if step % 480:
            continue
        evaluated = generate(scorer, validation)
        original = np.mean([row["correct"] for row in evaluated[::3]])
        triples = np.mean([all(row["correct"] for row in evaluated[i:i+3])
                           for i in range(0, len(evaluated), 3)])
        passed = original - base_accuracy >= .125 and triples >= .75
        trace.append(dict(step=step, original=float(original), triplet=float(triples)))
        p.write(gate_file, dict(splits=splits, passed=bool(passed), trace=trace,
                                old=c.OLD[1:], keys=c.KEYS))
        print("ANCHOR", seed, folder.name, trace[-1], flush=True)
        if passed:
            engine.save(checkpoint_file)
            checkpoint = p.torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            return splits, checkpoint
    raise RuntimeError("Cedar validation gate not reached in 1920 steps")


def run(model, seed, pairing, lr, rank, replay, historical=False, minimum_steps=640, maximum_steps=1280, exposure="plain"):
    if p.torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        raise RuntimeError("Initial free GPU memory below 6 GiB")
    name = f"{model}_s{seed}_{pairing}_lr{lr:g}_r{rank}" + ("_replay" if replay else "")
    if minimum_steps != 64:
        name += f"_min{minimum_steps}"
    if exposure == "duplicate_maple":
        name += "_duplicate_maple"
    folder = OUT / "runs" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        print("ALREADY COMPLETE", name, flush=True)
        return
    engine = p.Engine(model) if model == "Qwen3-0.6B" else StandardEngine(model)
    old, new, rng = values_for(seed, pairing, engine.tok)
    c.KEYS = ("_",) + tuple(a+b for a, b in list(itertools.product("QXZ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))[:48])
    c.OLD, c.NEW = [0] + old, [0] + new
    p.problem = c.problem
    scorer = CandidateScorer(engine, old + new, batch_size=2 if pairing == "alphabetic" else 4)
    code_format = not historical
    if historical:
        anchor_folder = HISTORICAL / f"s{seed}"
        meta = d.read(anchor_folder / "old_gate.json")
        assert meta["cedar_values"] == c.OLD
        c.KEYS = meta["keys"]
        splits = meta["splits"]
        checkpoint = p.torch.load(anchor_folder / "old.pt", map_location="cpu", weights_only=True)
    else:
        anchor_name = f"{model}_s{seed}_" + ("alphabetic" if pairing == "alphabetic" else "numeric")
        anchor_folder = OUT / "anchors" / anchor_name
        anchor_folder.mkdir(parents=True, exist_ok=True)
        splits, checkpoint = fresh_anchor(engine, scorer, seed, anchor_folder, code_format)
    restore(engine, checkpoint, rank, seed)
    active = set(rng.sample(list(range(1, 49, 2)), 16))
    train = [r for r in rows_for(splits["train"], "related", ("original", "counterfactual"), code_format)
             if int(r["id"].split("/")[0]) in active]
    replay_pool = rows_for(splits["train"], "old", p.VARIANTS, code_format)
    old_rows = rows_for(splits["test"], "old", ("original",), code_format)
    maple_rows = [dict(row, text=row["text"].replace("Cedar register", "Maple register"))
                  for row in old_rows]
    validation = [r for r in rows_for(splits["validation"], "related", ("original",), code_format)
                  if int(r["id"].split("/")[0]) in active]
    protocol = dict(model=model, seed=seed, pairing=pairing, lr=lr, rank=rank, replay=replay,
                    historical_anchor=historical, old=old, new=new, keys=list(c.KEYS),
                    minimum_steps=minimum_steps, maximum_steps=maximum_steps, exposure=exposure,
                    active_original=sorted(active), splits=splits, old_eval_ids=[r["id"] for r in old_rows],
                    validation_ids=[r["id"] for r in validation],
                    adapter_parameters=sum(v.numel() for v in engine.ps),
                    token_lengths=[len(engine.tok.encode(str(v), add_special_tokens=False)) for v in old+new],
                    value_tokens=[engine.tok.encode(str(v), add_special_tokens=False) for v in old+new],
                    torch=p.torch.__version__, gpu=p.torch.cuda.get_device_name(),
                    scoring_batch_size=scorer.batch_size,
                    scoring_method=scorer.method,
                    scoring_precision=scorer.precision,
                    dense_contexts=2, full_context_steps=[0,16,64])
    p.write(folder / "protocol.json", protocol)
    optimizer = p.optimizer(engine.ps, lr)
    batches = p.sample(train, seed+901, maximum_steps)
    old_batches = p.sample(replay_pool, seed+3901, maximum_steps)
    dense_indices = [i for i in range(len(old_rows)) if i % 6 < 2]
    saved = folder / "state.pt"
    start_step = 0
    trace = []
    if saved.exists():
        state = p.torch.load(saved, map_location="cpu", weights_only=True)
        restore(engine, state, rank, seed)
        optimizer = p.optimizer(engine.ps, lr)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"] + 1
        trace = d.read(folder / "trace.json")
    path_length = state.get("effective_path", 0.) if saved.exists() else 0.
    path_origin = state.get("path_origin", start_step-1) if saved.exists() else 0
    started = time.perf_counter()
    for step in range(start_step, maximum_steps+1):
        if step:
            before = [v.detach().clone() for v in engine.ps]
            batch = batches[step-1] + (old_batches[step-1] if replay else batches[step-1] if exposure=="duplicate_maple" else [])
            p.train_step(engine, optimizer, batch)
            with p.torch.no_grad():
                norm2 = 0.
                for i in range(0,len(engine.ps),2):
                    a,b=engine.ps[i:i+2]; oa,ob=before[i:i+2]
                    left=p.torch.cat([b-ob,ob],dim=1); right=p.torch.cat([a,a-oa],dim=0)
                    norm2 += float(((left.T@left)*(right@right.T).T).sum())*4
                path_length += math.sqrt(max(0.,norm2))
            x.check_memory()
        if (step <= 64 and step % 2) or (step > 64 and step % 16):
            continue
        indices = list(range(len(old_rows))) if step in (0,16,64) else dense_indices
        old_lp = scorer([old_rows[i] for i in indices])
        maple_lp = scorer([maple_rows[i] for i in indices])
        val_lp = np.array(scorer.gold(validation))
        if step == 0:
            full = scorer(validation[:3])
            columns = [48+new.index(r["answer"]) for r in validation[:3]]
            error = float(np.max(np.abs(np.exp(val_lp[:3]) -
                                        np.exp(full[np.arange(3),columns]))))
            if error > 1e-4:
                raise RuntimeError(f"Validation scorer disagreement {error}")
            p.write(folder/"validation_score_agreement.json",dict(maximum_probability_error=error))
        maple_progress = float(val_lp.mean())
        old_mass = np.exp(old_lp)
        maple_mass = np.exp(maple_lp)
        original_keys = [int(old_rows[i]["id"].split("/")[0])-1 for i in indices]
        correct = float(old_mass[np.arange(len(indices)), original_keys].mean())
        np.savez_compressed(folder / f"s{step:04d}.npz", old=old_mass, maple=maple_mass,
                            validation_gold_logp=val_lp, indices=np.array(indices))
        row = dict(step=step, maple=maple_progress, old_correct=correct,
                   old_other=float((1-old_mass.sum(1)).mean()),
                   new_examples=step*(8 if exposure=="duplicate_maple" else 4), old_examples=step*4 if replay else 0,
                   unique_new_draws=step*4, effective_path=path_length, path_origin=path_origin,
                   seconds=time.perf_counter()-started,
                   peak_reserved_mib=p.torch.cuda.max_memory_reserved()/2**20)
        trace.append(row)
        p.write(folder / "trace.json", trace)
        state = dict(parameters=[v.detach().cpu() for v in engine.ps],
                     optimizer=optimizer.state_dict(), step=step, rank=rank, scale=2.,
                     effective_path=path_length,path_origin=path_origin)
        temporary = saved.with_suffix(".tmp")
        p.torch.save(state, temporary)
        temporary.replace(saved)
        p.torch.save(dict(parameters=state["parameters"], rank=rank, scale=2.),
                     folder / f"adapter_{step:04d}.pt")
        print(name, step, round(maple_progress, 3), round(correct, 4), flush=True)
        if step >= minimum_steps and maple_progress >= -.1:
            break
    if step >= minimum_steps:
        old_final = rows_for(splits["test"],"old",("original","counterfactual"),code_format)
        maple_final = [r for r in rows_for(splits["test"],"related",("original","counterfactual"),code_format)
                       if int(r["id"].split("/")[0]) in active]
        generations = {"cedar":generate(scorer,old_final),"maple":generate(scorer,maple_final)}
        p.write(folder/"final_generations.json",generations)
        p.write(folder/"final_generation_summary.json",
                {name:dict(context_queries=len(rows),accuracy=float(np.mean([r["correct"] for r in rows])),
                           unique_facts=48 if name=="cedar" else 32)
                 for name,rows in generations.items()})
    target_row = next((r for r in trace if r["maple"] >= -1.),None)
    if target_row is not None:
        target_step=target_row["step"]
        target_state=p.torch.load(folder/f"adapter_{target_step:04d}.pt",map_location="cpu",weights_only=True)
        restore(engine,target_state,rank,seed)
        active_old=[r for r in old_rows if int(r["id"].split("/")[0]) in active]
        active_new=[dict(r,text=r["text"].replace("Cedar register","Maple register"),answer=new[int(r["id"].split("/")[0])-1]) for r in active_old]
        decoded={"cedar":generate(scorer,active_old),"maple":generate(scorer,active_new)}
        p.write(folder/"first_minus1_generation.json",dict(step=target_step,validation_logp=target_row["maple"],queries=decoded))
    p.write(folder / "complete.json", dict(last_step=step, target_reached=maple_progress >= -.1,
                                          seconds=time.perf_counter()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairing", choices=("adjacent", "permuted", "alphabetic","support_permuted"),
                        default="adjacent")
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--minimum-steps", type=int, default=640)
    parser.add_argument("--maximum-steps", type=int, default=1280)
    parser.add_argument("--exposure", choices=("plain","duplicate_maple"), default="plain")
    args = parser.parse_args()
    if args.output_root:
        OUT = ROOT / args.output_root
    run(args.model, args.seed, args.pairing, args.lr, args.rank, args.replay,
        args.historical, args.minimum_steps,args.maximum_steps,args.exposure)
