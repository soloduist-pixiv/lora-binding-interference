"""Disjoint-value binding trajectories and probability confusion matrices."""
import argparse
import importlib.util
import math
import random
from pathlib import Path

from experiment import OUT, c, d, p, x

GRID=(0,1,2,4,8,16,32,64)


def cuda_scorer(engine):
    torch = p.torch
    weight = engine.model.lm_head.weight.detach().to("cuda")

    def score(engine, sequences, positions):
        model = engine.model.model
        length = max(map(len, sequences))
        tokens = torch.tensor([s + [0] * (length - len(s)) for s in sequences])
        embeddings = model.embed_tokens(tokens).to("cuda", dtype=torch.bfloat16)
        hidden = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        batch, index = torch.tensor(positions, device=hidden.device).unbind(1)
        return torch.nn.functional.linear(hidden[batch, index].float(), weight)

    return score


def verify_scores(grouped, reference):
    assert grouped.keys() == reference.keys()
    error = max(abs(a - b) for key, row in grouped.items()
                for column in ("cedar", "maple")
                for a, b in zip(row[column], reference[key][column], strict=True))
    if error > 3e-4:
        raise RuntimeError(f"CPU/CUDA score mismatch: {error:.3g}")
    return error


def run(seed, cuda_score=False, break_prefix=False):
    rng=random.Random(seed+8000)
    pairs=[(v,v+1) for v in range(100,196,2)]
    rng.shuffle(pairs)
    pairs=[pair if rng.random()<.5 else pair[::-1] for pair in pairs]
    old_values,new_values=map(list,zip(*pairs))
    folder=OUT/"disjoint_mass"/f"s{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("disjoint_anchor",Path(__file__).resolve().parents[1]/"binding_followup/run.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT=folder
    assert p.torch.cuda.mem_get_info()[0]>=6*1024**3
    e=p.Engine("Qwen3-0.6B")
    if (folder/"old.pt").exists():
        protocol=d.read(folder/"old_gate.json")
        c.KEYS,c.OLD=protocol["keys"],protocol["cedar_values"]
        splits=protocol["splits"]
        ck=p.torch.load(folder/"old.pt",map_location="cpu",weights_only=True)
        d.configure(e,ck,4,seed)
    else:
        if break_prefix:
            raise RuntimeError("The prefix control requires the original Cedar checkpoint")
        splits,reference=mod.expanded(e,seed,old_values)
        if reference is None:
            raise RuntimeError("Disjoint Cedar anchor did not meet validation gate")
    assert c.OLD[1:]==old_values
    if break_prefix:
        ordered = sorted(new_values)
        half = len(ordered) // 2
        mapping = dict(zip(ordered, ordered[half:] + ordered[:half], strict=True))
        new_values = [mapping[v] for v in new_values]
        assert all(old // 10 != new // 10 for old, new in zip(old_values, new_values, strict=True))
    grid = (0, 8, 16, 64) if break_prefix else GRID
    c.NEW=[0]+new_values
    output = folder / ("mass_prefix_control.json" if break_prefix else "mass.json")
    cpu_reference = folder / "mass_cpu_reference.json"
    if cuda_score and not break_prefix and cpu_reference.exists():
        previous = d.read(cpu_reference)
        assert previous["cedar"] == old_values and previous["maple"] == new_values
        reference = {int(step): {int(key): row for key, row in block["keys"].items()}
                     for step, block in previous["result"].items()}
    else:
        reference = {}
    score = cuda_scorer(e) if cuda_score else None
    p.problem=c.problem
    old=list(p.rows("old",splits["test"],"plain",("original",)))
    assert len(old)==144
    idx=lambda row:int(row["id"].split("/")[0])
    active=set(rng.sample(sorted({idx(r) for r in old}),16))
    train=[r for r in p.rows("related",splits["train"],"plain",("original","counterfactual")) if idx(r) in active]
    opt=p.optimizer(e.ps,.001)
    result={}
    seen=set()
    for step,batch in enumerate([None]+p.sample(train,seed+901,64)):
        if batch is not None:
            p.train_step(e,opt,batch)
            seen.update(idx(r)+int(r["variant"]=="counterfactual") for r in batch)
            x.check_memory()
        if step not in grid:
            continue
        scores=d.masses(e,old,old_values+new_values,logits=score)
        x.check_memory()
        grouped={}
        for row,lp in zip(old,scores):
            k=idx(row)
            dest=grouped.setdefault(k,dict(cedar=[0.]*48,maple=[0.]*48))
            for j in range(48):
                dest["cedar"][j]+=math.exp(lp[old_values[j]])/6
                dest["maple"][j]+=math.exp(lp[new_values[j]])/6
        for k,r in grouped.items():
            r["categories"]=[r["cedar"][k-1],sum(r["cedar"])-r["cedar"][k-1],r["maple"][k-1],
                sum(v for j,v in enumerate(r["maple"],1) if j!=k and j in seen),
                sum(v for j,v in enumerate(r["maple"],1) if j!=k and j not in seen)]
            r["outside"]=max(0.,1-sum(r["categories"]))
        result[step]=dict(seen=sorted(seen),keys=grouped)
        if step in reference:
            error = verify_scores(grouped, reference[step])
            print(f"CPU/CUDA maximum probability error at step {step}: {error:.3g}", flush=True)
        if step >= max(reference, default=0):
            p.write(output,dict(cedar=old_values,maple=new_values,result=result,
                                scoring="cuda-fp32" if cuda_score else "cpu-fp32"))
        print(seed,step,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--cuda-score", action="store_true")
    parser.add_argument("--break-prefix", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.cuda_score, args.break_prefix)
