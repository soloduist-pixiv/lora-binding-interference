"""Serial optimizer/capacity scan and exact two-digit candidate probabilities."""
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"work/binding_constraints"))
import run as base
p,c=base.p,base.c
MAIN=base.MAIN
OUT=MAIN/"update_diagnostics"
GRID=(0,1,2,4,8,16,32,48,64)
ARMS={"adam_1e-4":(.0001,4,"adam"),"adam_3e-4":(.0003,4,"adam"),
      "adam_1e-3":(.001,4,"adam"),"warmup":(.001,4,"warmup"),
      "sgd":(.001,4,"sgd"),"rank16":(.001,16,"adam"),"rank64":(.001,64,"adam")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure(e, ck, rank, seed):
    p.torch.manual_seed(seed)
    for (_,adapter),a,b in zip(e.adapters,ck["parameters"][::2],ck["parameters"][1::2]):
        aa=p.torch.empty((rank,a.shape[1]),device="cuda",dtype=a.dtype)
        p.torch.nn.init.kaiming_uniform_(aa,a=math.sqrt(5))
        bb=p.torch.zeros((b.shape[0],rank),device="cuda",dtype=b.dtype)
        aa[:4]=a
        bb[:,:4]=b
        adapter.A=p.torch.nn.Parameter(aa)
        adapter.B=p.torch.nn.Parameter(bb)
    e.ps=[v for _,a in e.adapters for v in (a.A,a.B)]


@p.torch.no_grad()
def norm(e,ck):
    total=0.
    for (_,adapter),a,b in zip(e.adapters,ck["parameters"][::2],ck["parameters"][1::2]):
        change=adapter.scale*(adapter.B@adapter.A-b.cuda()@a.cuda())
        total+=float(change.square().sum())
    return math.sqrt(total)


def triplets(rows):
    return sum(all(r["correct"] for r in rows[i:i+3]) for i in range(0,len(rows),3))/ (len(rows)//3)


def load(seed):
    free,_=p.torch.cuda.mem_get_info()
    assert free>=6*1024**3
    src=MAIN/"binding_followup"/f"expanded_s{seed}"
    proto=read(src/"protocol.json")
    c.KEYS,c.OLD,c.NEW=proto["keys"],proto["cedar_values"],proto["maple_conflict_values"]
    p.problem=c.problem
    e=p.Engine("Qwen3-0.6B")
    ck=p.torch.load(src/"old.pt",map_location="cpu",weights_only=True)
    assert ck["names"]==[n for n,_ in e.adapters]
    return e,ck,proto["splits"]


def scan(seed):
    folder=OUT/f"s{seed}"
    e,ck,splits=load(seed)
    old=list(p.rows("old",splits["validation"],"plain",p.VARIANTS))
    new=list(p.rows("related",splits["validation"],"plain",("original",)))
    test_old=list(p.rows("old",splits["test"],"plain",p.VARIANTS))
    test_new=list(p.rows("related",splits["test"],"plain",("original",)))
    pool=list(p.rows("related",splits["train"],"plain",("original","counterfactual")))
    batches=p.sample(pool,seed+1)
    configure(e,ck,4,seed)
    baseline=p.evaluate(e,old)
    baseline_new=p.acc(p.evaluate(e,new))
    test_ref=triplets(p.evaluate(e,test_old))
    # Calibrate plain SGD against the first actual dense Adam update on the same batch.
    calibrations={}
    for name in ("adam","sgd"):
        configure(e,ck,4,seed)
        opt=p.optimizer(e.ps,.001) if name=="adam" else p.torch.optim.SGD(e.ps,lr=.001)
        p.train_step(e,opt,batches[0])
        calibrations[name]=norm(e,ck)
    assert calibrations["sgd"]>0
    sgd_lr=.001*calibrations["adam"]/calibrations["sgd"]
    p.write(folder/"protocol.json",{"seed":seed,"grid":GRID,"arms":ARMS,"baseline_old":triplets(baseline),
            "baseline_new":baseline_new,"baseline_test_old":test_ref,"sgd_lr":sgd_lr,"calibration_norms":calibrations,
            "rank_scale":2,"warmup_steps":8})
    for arm,(lr,rank,kind) in ARMS.items():
        if (folder/f"{arm}.json").exists():
            continue
        start=time.perf_counter()
        configure(e,ck,rank,seed)
        assert norm(e,ck)<1e-5
        if rank!=4:
            assert [r["response"] for r in p.evaluate(e,old)]==[r["response"] for r in baseline]
        p.torch.cuda.reset_peak_memory_stats()
        opt=p.torch.optim.SGD(e.ps,lr=sgd_lr) if kind=="sgd" else p.optimizer(e.ps,lr)
        curve=[]
        for i in range(65):
            if i:
                if kind=="warmup":
                    opt.param_groups[0]["lr"]=lr*min(i/8,1)
                p.train_step(e,opt,batches[i-1])
                assert p.torch.cuda.max_memory_reserved()<=6*1024**3
            if i not in GRID:
                continue
            old_score=p.evaluate(e,old) if i else baseline
            gain=p.acc(p.evaluate(e,new))-baseline_new if i else 0.
            curve.append({"step":i,"gain":gain,"old_triplet":triplets(old_score),"old_original":p.acc(old_score[::3]),
                          "dense_norm":norm(e,ck),"lr":opt.param_groups[0]["lr"]})
            print(seed,arm,i,round(gain,3),round(curve[-1]["old_triplet"],3),flush=True)
        p.write(folder/f"{arm}.json",{"curve":curve,"fixed":{"old_triplet":triplets(p.evaluate(e,test_old)),
                "new":p.acc(p.evaluate(e,test_new))},"parameters":sum(t.numel() for t in e.ps),
                "seconds":time.perf_counter()-start,"peak_reserved_mib":p.torch.cuda.max_memory_reserved()/2**20})
    print("SCAN DONE",seed,flush=True)


@p.torch.no_grad()
def masses(e,rows,candidates=range(10,100),logits=None):
    # Trie of every canonical two-digit completion followed by this tokenizer's EOS.
    answers={v:tuple(e.tok.encode(str(v),add_special_tokens=False)+[e.tok.eos_token_id]) for v in candidates}
    assert len(set(answers.values()))==len(answers)
    prefixes={}
    for tokens in answers.values():
        for j,t in enumerate(tokens):
            prefixes.setdefault(tokens[:j],set()).add(t)
    work=[(i,prefix,sorted(children)) for i in range(len(rows)) for prefix,children in prefixes.items()]
    logps=[{} for _ in rows]
    score = p.logits if logits is None else logits
    for start in range(0,len(work),48):
        group=work[start:start+48]
        seqs=[e.encode(rows[i]["text"])+list(prefix) for i,prefix,_ in group]
        z=score(e,seqs,[(j,len(s)-1) for j,s in enumerate(seqs)]).log_softmax(-1).cpu()
        for (i,prefix,children),prob in zip(group,z):
            logps[i][prefix]={t:float(prob[t]) for t in children}
    return [{v:sum(tree[tokens[:j]][t] for j,t in enumerate(tokens)) for v,tokens in answers.items()}
            for tree in logps]


def offline(seed):
    e,ck,splits=load(seed)
    configure(e,ck,4,seed)
    # One independently held-out context per original key.
    seen=set()
    pairs=[]
    for a,b in splits["test"]:
        if a not in seen:
            pairs.append((a,b))
            seen.add(a)
    rows=list(p.rows("old",pairs,"plain",("original",)))
    src=MAIN/"binding_factorial"/f"s{seed}"
    meta=read(src/"protocol.json")
    arms={"reference":ck["parameters"]}
    for name in ("shared_overlap","shared_disjoint","independent_overlap","independent_disjoint"):
        arms[name]=p.torch.load(src/f"{name}_64.pt",map_location="cpu",weights_only=True)["parameters"]
    output={}
    for name,state in arms.items():
        e.restore(state)
        result=masses(e,rows)
        output[name]=[]
        values={int(k):v for k,v in meta["maps"]["disjoint" if name.endswith("_disjoint") else "overlap"].items()}
        for row,lp in zip(rows,result):
            k=int(row["id"].split("/")[0])
            gold=row["answer"]
            groups={g:0. for g in ("old","paired","other_maple","other_cedar","other_legal")}
            for val,logp in lp.items():
                group=("old" if val==gold else "paired" if val==values.get(k) else
                       "other_maple" if val in values.values() else "other_cedar" if val in c.OLD else "other_legal")
                groups[group]+=math.exp(logp)
            assert sum(groups.values())<=1.00001
            groups["outside_exact_two_digit_eos"]=max(0.,1-sum(groups.values()))
            output[name].append({"id":row["id"],"gold":gold,"logp":lp,"mass":groups})
        print(seed,"offline",name,flush=True)
    p.write(OUT/f"s{seed}"/"offline.json",output)


if __name__=="__main__":
    mode,seed=sys.argv[1:]
    {"scan":scan,"offline":offline}[mode](int(seed))


