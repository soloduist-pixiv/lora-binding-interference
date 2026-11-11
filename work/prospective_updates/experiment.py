"""Prospective update tests, long LR/projection scan, and candidate mass trajectories."""
import copy
import math
import os
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD","1")
sys.path.insert(0,str(ROOT/"work/update_diagnostics"))
import diagnostics as d
import extra as x
from stream import BufferedEngine

d.p.Engine=BufferedEngine
p,c=d.p,d.c
OUT=d.MAIN/"prospective_updates"
RELATIONS=("alias","unrelated","conflict")


def verify_reference(engine,ck,seed):
    d.configure(engine,ck,4,seed)
    path=d.MAIN/"binding_followup"/f"expanded_s{seed}"/"old_reference.json"
    recorded=d.read(path)["validation"]
    evaluated=p.evaluate(engine,recorded)
    fields=("prediction","correct","format_ok")
    assert len(recorded)==len(evaluated)
    assert all(all(a[k]==b[k] for k in fields) for a,b in zip(recorded,evaluated))
    print(seed,"recorded Cedar validation matches",len(recorded),flush=True)


def prospective(engine,opt,batch):
    objective=-x.logp(engine,batch).sum()/sum(
        len(engine.tok.encode(str(r["answer"]),add_special_tokens=False))+1 for r in batch)
    gradients=p.torch.autograd.grad(objective,engine.ps)
    # The real adapter and optimizer are untouched until the proposal is evaluated.
    shadow=[p.torch.nn.Parameter(t.detach().clone()) for t in engine.ps]
    proposal=p.optimizer(shadow,opt.param_groups[0]["lr"])
    proposal.load_state_dict(copy.deepcopy(opt.state_dict()))
    for v,g in zip(shadow,gradients):
        v.grad=g.detach()
    proposal.step()
    return [a.detach()-b.detach() for a,b in zip(shadow,engine.ps)],proposal.state_dict()


def flat(vectors):
    return p.torch.cat([v.flatten() for v in vectors])


def grads(engine,rows):
    jac,values=[],[]
    for row in rows:
        v=x.logp(engine,[row])[0]
        jac.append(flat(p.torch.autograd.grad(v,engine.ps)))
        values.append(v.item())
    return p.torch.stack(jac),values


@p.torch.no_grad()
def values(engine,rows,size=24):
    return [v for i in range(0,len(rows),size) for v in x.logp(engine,rows[i:i+size]).tolist()]


def relation_rows(splits,relation):
    def problem(task,a,b,variant):
        text,y=c.problem(task,a,b,variant)
        if relation=="conflict" or task=="old":
            return text,y
        k=a+(variant=="counterfactual")
        old=c.KEYS[k]
        alias={"Q":"J","X":"V"}[old[0]] + chr(65+(ord(old[1])-65+13)%26)
        text=text.replace(f"key {old} in",f"key {alias} in")
        if relation=="alias":
            text+=f" In Maple, key {alias} is an alias of Cedar key {old}."
            y=c.OLD[k]
        return text,y
    p.problem=problem
    fit=list(p.rows("related",splits["train"],"plain",("original","counterfactual")))
    held=list(p.rows("related",splits["validation"],"plain",("original",)))
    p.problem=c.problem
    return fit,held


def signal(seed,mid_only=False,confirm=False):
    e,ck,splits=d.load(seed)
    folder=OUT/f"s{seed}"
    verify_reference(e,ck,seed)
    anchor=e.snapshot()
    p.problem=c.problem
    old_test=list(p.rows("old",splits["test"],"plain",("original",)))
    # Distinct contexts within each key for controller signal versus verification.
    by_key={}
    for r in old_test:
        by_key.setdefault(r["id"].split("/")[0],[]).append(r)
    old_probe=[v[0] for v in by_key.values()]
    old_verify=[v[1] for v in by_key.values()]
    relations={rel:relation_rows(splits,rel) for rel in RELATIONS}
    output=folder/("confirm.json" if confirm else "signals.json")
    result=d.read(output) if output.exists() else {}
    recorded=d.read(d.MAIN/"binding_followup"/f"expanded_s{seed}"/"old_gate.json")["trace"]
    stages=(("mid320",320),) if mid_only else (("mature",0),("early120",120))
    for stage,n_old in stages:
        if stage in result:
            continue
        replay_error=None
        if n_old:
            e.reset(seed,"add")
            old_pool=list(p.rows("old",splits["train"],"plain",p.VARIANTS))
            opt=p.optimizer(e.ps,.001)
            replay_error=0.
            for b,reference in zip(p.sample(old_pool,seed,n_old),recorded[:n_old],strict=True):
                loss,*counts=p.train_step(e,opt,b)
                assert counts==reference[1:]
                replay_error=max(replay_error,abs(loss-reference[0]))
                x.check_memory()
        else:
            e.restore(anchor)
        initial=e.snapshot()
        old_acc=p.acc(p.evaluate(e,old_probe))
        old_j,old_log=grads(e,old_probe)
        old_verify_log=values(e,old_verify)
        old_norm=old_j.norm(dim=1).tolist()
        trials=[]
        for rel,(pool,held) in relations.items():
            grouped={}
            for row in held:
                grouped.setdefault(row["id"].split("/")[0],[]).append(row)
            new_probe=[v[0] for v in grouped.values()]
            new_verify=[v[1] for v in grouped.values()]
            new_j,_=grads(e,new_probe)
            anchor_new=values(e,new_verify)
            for block,b in enumerate(p.sample(pool,seed+(1901 if confirm else 901),8)):
                opt=p.optimizer(e.ps,.0001)
                delta,_=prospective(e,opt,b)
                v=flat(delta)
                old_pred=(old_j@v).tolist()
                new_pred=(new_j@v).tolist()
                cosine=((old_j@v)/(old_j.norm(dim=1)*v.norm()).clamp_min(1e-12)).tolist()
                e.restore(initial,delta,1.)
                old_actual=[a-b for a,b in zip(values(e,old_verify),old_verify_log)]
                new_actual=[a-b for a,b in zip(values(e,new_verify),anchor_new)]
                e.restore(initial)
                x.check_memory()
                trials.append({"relation":rel,"block":block,"keys":list(by_key),
                               "dot":old_pred,"cosine":cosine,"norm":old_norm,
                               "headroom":[1-math.exp(z) for z in old_log],
                               "old_anchor_logp":old_verify_log,"old_actual":old_actual,
                               "new_pred":new_pred,"new_actual":new_actual,"delta_norm":v.norm().item()})
            print(seed,stage,rel,flush=True)
        result[stage]={"old_probe_accuracy":old_acc,"trials":trials,"cedar_replay_max_loss_error":replay_error}
        p.write(output,result)
    if confirm or mid_only or (folder/"sequential.json").exists():
        return
    # Repeated before-step forecasts with carried Adam moments, re-anchored at each proposal.
    e.restore(anchor)
    fit,_=relations["conflict"]
    opt=p.optimizer(e.ps,.0001)
    steps=[]
    for i,b in enumerate(p.sample(fit,seed+902,8),1):
        state=e.snapshot()
        jac,lp=grads(e,old_probe)
        baseline=values(e,old_verify)
        delta,shadow=prospective(e,opt,b)
        v=flat(delta)
        dot=jac@v
        predicted=dot.tolist()
        cosine=(dot/(jac.norm(dim=1)*v.norm()).clamp_min(1e-12)).tolist()
        e.restore(state,delta,1.)
        opt.load_state_dict(shadow)
        actual=[a-b for a,b in zip(values(e,old_verify),baseline)]
        x.check_memory()
        steps.append({"step":i,"keys":list(by_key),"dot":predicted,"cosine":cosine,
                      "norm":jac.norm(dim=1).tolist(),"headroom":[1-math.exp(z) for z in lp],"old_actual":actual})
    p.write(folder/"sequential.json",steps)


def long(seed):
    e,ck,splits=d.load(seed)
    verify_reference(e,ck,seed)
    folder=OUT/f"s{seed}"
    p.problem=c.problem
    held_context=min(b for _,b in splits["train"])
    pool=list(p.rows("related",[(a,b) for a,b in splits["train"] if b!=held_context],"plain",("original","counterfactual")))
    new=list(p.rows("related",splits["validation"],"plain",("original",)))
    old=list(p.rows("old",splits["validation"],"plain",p.VARIANTS))
    new_test=list(p.rows("related",splits["test"],"plain",("original",)))
    old_test=list(p.rows("old",splits["test"],"plain",p.VARIANTS))
    keys=sorted({a for a,_ in splits["train"]})
    probes=list(p.rows("old",[(a,held_context) for a in keys[::3]],"plain",("original",)))
    batches=p.sample(pool,seed+1,640)
    arms=[("lr1e-3",.001,64,None),("lr3e-4",.0003,214,None),("lr1e-4",.0001,640,None)]
    if seed==29:
        arms += [("proj0",.0003,214,0.),("proj01",.0003,214,.01)]
    output=folder/"long.json"
    result=d.read(output) if output.exists() else {}
    for name,lr,steps,budget in arms:
        if name in result:
            continue
        d.configure(e,ck,4,seed)
        opt=p.optimizer(e.ps,lr)
        grid=sorted({0,1,4,16,steps}|{round(steps*j/8) for j in range(1,8)})
        curve=[]
        projections=0
        for i in range(steps+1):
            if i:
                batch=batches[i-1]
                if budget is None:
                    p.train_step(e,opt,batch)
                else:
                    state=e.snapshot()
                    proposal,shadow=prospective(e,opt,batch)
                    # Projection of the actual Adam candidate onto a local average logP half-space.
                    score=x.logp(e,probes).mean()
                    g=p.torch.autograd.grad(score,e.ps)
                    dot=sum((a*b).sum() for a,b in zip(g,proposal))
                    norm2=sum(a.square().sum() for a in g)
                    if dot.item() < -budget and norm2.item()>0:
                        scale=(-budget-dot)/norm2
                        proposal=[a+scale*b for a,b in zip(proposal,g)]
                        assert sum((a*b).sum() for a,b in zip(g,proposal)).item()>=-budget-1e-5
                        projections+=1
                    e.restore(state,proposal,1.)
                    opt.load_state_dict(shadow)
                x.check_memory()
            if i in grid:
                new_logp=values(e,new)
                curve.append({"step":i,"new_logp":sum(new_logp)/len(new_logp),
                              "old_triplet":d.triplets(p.evaluate(e,old)),
                              "new_acc":p.acc(p.evaluate(e,new)),"dense_norm":d.norm(e,ck)})
                if name=="lr1e-3" and i in ((0,1,4,16,64) if seed==29 else (0,64)):
                    e.save(folder/f"mass_{i}.pt")
                print(seed,name,i,round(curve[-1]["new_logp"],3),flush=True)
        result[name]={"lr":lr,"steps":steps,"budget_nats":budget,"curve":curve,"projections":projections,
                      "peak_reserved_mib":p.torch.cuda.max_memory_reserved()/2**20,
                      "test":{"new_logp":sum(values(e,new_test))/len(new_test),
                              "new_acc":p.acc(p.evaluate(e,new_test)),"old_triplet":d.triplets(p.evaluate(e,old_test))}}
        p.write(output,result)


def mass(seed):
    folder=OUT/f"s{seed}"
    files={str(i):folder/f"mass_{i}.pt" for i in ((0,1,4,16,64) if seed==29 else (0,64))}
    missing=[str(path.name) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run long {seed} first; missing checkpoints: {', '.join(missing)}")
    e,ck,splits=d.load(seed)
    verify_reference(e,ck,seed)
    rows=list(p.rows("old",splits["test"],"plain",("original",)))
    assert len(rows)==144
    path_out=folder/"mass.json"
    output=d.read(path_out) if path_out.exists() else {}
    for step,path in files.items():
        if step in output:
            continue
        e.restore(p.torch.load(path,map_location="cpu",weights_only=True)["parameters"])
        vals=d.masses(e,rows)
        x.check_memory()
        output[step]=[{"id":r["id"],"gold":r["answer"],"logp":lp} for r,lp in zip(rows,vals)]
        p.write(path_out,output)
        print(seed,"mass",step,flush=True)


if __name__=="__main__":
    mode,seed=sys.argv[1:]
    {"signal":signal,"mid":lambda s:signal(s,mid_only=True),
     "confirm":lambda s:signal(s,confirm=True),"confirm_mid":lambda s:signal(s,mid_only=True,confirm=True),"long":long,"mass":mass,"migration":x.migration,"trajectory":lambda s:x.migration(s,trajectory=True)}[mode](int(seed))


