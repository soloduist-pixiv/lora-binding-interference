"""Update prediction, controlled-transfer feasibility and a probe-based line-search pilot."""
import copy
import sys
import time

import diagnostics as d
p,c=d.p,d.c


def logp(engine,rows):
    z,y,_=d.base.forward(engine,rows)
    return p.torch.stack([d.base.sequence_logp(a,b) for a,b in zip(z,y)])


def check_memory():
    assert p.torch.cuda.max_memory_reserved()<=6*1024**3


def carry(seed):
    e,ck,splits=d.load(seed)
    folder=d.OUT/f"s{seed}"
    source=d.MAIN/"binding_followup"/f"expanded_s{seed}"
    old_protocol=d.read(source/"old_gate.json")
    old_steps=len(old_protocol["trace"])
    old_pool=list(p.rows("old",splits["train"],"plain",p.VARIANTS))
    new_pool=list(p.rows("related",splits["train"],"plain",("original","counterfactual")))
    old_val=list(p.rows("old",splits["validation"],"plain",p.VARIANTS))
    old_test=list(p.rows("old",splits["test"],"plain",p.VARIANTS))
    new_val=list(p.rows("related",splits["validation"],"plain",("original",)))
    new_test=list(p.rows("related",splits["test"],"plain",("original",)))
    e.reset(seed,"add")
    opt=p.optimizer(e.ps,.001)
    max_loss_error=0.
    for i,batch in enumerate(p.sample(old_pool,seed,old_steps),1):
        actual=p.train_step(e,opt,batch)
        recorded=old_protocol["trace"][i-1]
        assert list(actual[1:])==recorded[1:]
        max_loss_error=max(max_loss_error,abs(actual[0]-recorded[0]))
        check_memory()
        if i%120==0:
            print("CARRY old replay",i,flush=True)
    max_weight_error=max(float((x.detach().cpu()-y).abs().max()) for x,y in zip(e.ps,ck["parameters"]))
    assert max_loss_error<1e-5 and max_weight_error<1e-6
    anchor=p.acc(p.evaluate(e,new_val))
    curve=[]
    batches=p.sample(new_pool,seed+1)
    for i in range(65):
        if i:
            p.train_step(e,opt,batches[i-1])
            check_memory()
        if i in d.GRID:
            curve.append({"step":i,"gain":p.acc(p.evaluate(e,new_val))-anchor,
                          "old_triplet":d.triplets(p.evaluate(e,old_val)),"dense_norm":d.norm(e,ck)})
            print("CARRY new",i,curve[-1]["gain"],flush=True)
    p.write(folder/"carry.json",{"curve":curve,"old_steps":old_steps,"max_loss_error":max_loss_error,
             "max_weight_error":max_weight_error,"fixed":{"new":p.acc(p.evaluate(e,new_test)),
             "old_triplet":d.triplets(p.evaluate(e,old_test))}})


def first_order(seed):
    e,ck,splits=d.load(seed)
    pool=list(p.rows("related",splits["train"],"plain",("original","counterfactual")))
    batches=p.sample(pool,seed+1)
    pairs={}
    for a,b in splits["test"]:
        pairs.setdefault(a,[]).append(b)
    rows=list(p.rows("old",[(a,b) for a,bs in pairs.items() for b in bs[:2]],"plain",("original",)))
    d.configure(e,ck,4,seed)
    initial=e.snapshot()
    grads,anchor=[],[]
    for row in rows:
        value=logp(e,[row])[0]
        anchor.append(value.item())
        grads.append(p.torch.cat([g.flatten() for g in p.torch.autograd.grad(value,e.ps)]))
    gradients=p.torch.stack(grads)
    results={}
    for name,lr,kind in (("adam_low",.0001,"adam"),("adam_high",.001,"adam"),
                         ("sgd",d.read(d.OUT/f"s{seed}"/"protocol.json")["sgd_lr"],"sgd")):
        e.restore(initial)
        opt=p.optimizer(e.ps,lr) if kind=="adam" else p.torch.optim.SGD(e.ps,lr=lr)
        results[name]={}
        for i,batch in enumerate(batches[:4],1):
            p.train_step(e,opt,batch)
            check_memory()
            if i not in (1,4):
                continue
            delta=p.torch.cat([(x-y).detach().flatten() for x,y in zip(e.ps,initial)])
            pred=(gradients@delta).tolist()
            with p.torch.no_grad():
                measured=[logp(e,[row])[0].item() for row in rows]
            results[name][str(i)]=[{"id":r["id"],"predict":f,"actual":a-b} for r,f,a,b in zip(rows,pred,measured,anchor)]
            print(seed,name,i,flush=True)
    p.write(d.OUT/f"s{seed}"/"first_order.json",results)


def migration(seed,trajectory=False):
    e,ck,splits=d.load(seed)
    folder=d.OUT/f"s{seed}"
    old_test=list(p.rows("old",splits["test"],"plain",p.VARIANTS))
    probes=[r for r in old_test if r["variant"]=="original"]
    rng=p.random.Random(seed+930)
    selected=sorted(rng.sample(list(range(1,49,2)),20))
    def task_rows(split,variants):
        return list(p.rows("related",[pair for pair in splits[split] if pair[0] in selected],"plain",variants))
    result={}
    maple_state=None
    initial_old=None
    for relation in ("conflict","unrelated","alias","compose"):
        def problem(task,a,b,variant):
            text,y=c.problem(task,a,b,variant)
            if task=="old" or relation=="conflict":
                return text,y
            k=a+(variant=="counterfactual")
            old=c.KEYS[k]
            alias={"Q":"J","X":"V"}[old[0]] + chr(65+(ord(old[1])-65+13)%26)
            text=text.replace(f"key {old} in",f"key {alias} in")
            if relation=="alias":
                text+=f" In Maple, key {alias} is an alias of Cedar key {old}."
                y=c.OLD[k]
            elif relation=="compose":
                text+=f" The value for key {alias} is the Cedar value for key {old} plus 2."
                y=c.OLD[k]+2
            return text,y
        p.problem=problem
        train=task_rows("train",("original","counterfactual"))
        test=task_rows("test",("original",))
        batches=p.sample(train,seed+1,16)
        result[relation]={}
        for warm in (False,True):
            d.configure(e,ck,4,seed)
            if not warm:
                e.reset(seed,"add")
            opt=p.optimizer(e.ps,.001)
            record={}
            for i in (range(17) if trajectory else (0,16)):
                if i:
                    for batch in (batches[i-1:i] if trajectory else batches):
                        p.train_step(e,opt,batch)
                        check_memory()
                with p.torch.no_grad():
                    scores=[]
                    for j in range(0,len(test),16):
                        scores.extend(logp(e,test[j:j+16]).tolist())
                if trajectory:
                    with p.torch.no_grad():
                        old_scores=[z for j in range(0,len(probes),16) for z in logp(e,probes[j:j+16]).tolist()] if warm else []
                    record[str(i)]={"logp":scores,"old_logp":old_scores}
                    continue
                record[str(i)]={"new":p.evaluate(e,test),"logp":scores}
                if warm:
                    if i==0:
                        if initial_old is None:
                            initial_old=p.evaluate(e,old_test)
                        record[str(i)]["old"]=initial_old
                    else:
                        record[str(i)]["old"]=p.evaluate(e,old_test)
                if relation=="conflict" and not warm and i==16:
                    maple_state=e.snapshot()
                    e.save(folder/"routed_maple.pt")
            result[relation]["warm" if warm else "cold"]=record
            print("TRANSFER",relation,warm,flush=True)
        p.write(folder/("migration_trajectory.json" if trajectory else "migration.json"),result)
        if trajectory:
            continue
        if relation=="conflict":
            assert maple_state is not None
            mixed=old_test+test
            cedar=[r for r in mixed if "Cedar register" in r["text"]]
            maple=[r for r in mixed if "Maple register" in r["text"]]
            assert len(cedar)+len(maple)==len(mixed)
            d.configure(e,ck,4,seed)
            routed_old=p.evaluate(e,cedar)
            e.restore(maple_state)
            routed_new=p.evaluate(e,maple)
            assert routed_old==result[relation]["warm"]["0"]["old"]
            assert routed_new==result[relation]["cold"]["16"]["new"]
            p.write(folder/"route.json",{"old":routed_old,"new":routed_new,"rank_per_task":4,"tasks":2})


def oracle(seed):
    e,ck,splits=d.load(seed)
    folder=d.OUT/f"s{seed}"
    d.configure(e,ck,4,seed)
    initial=e.snapshot()
    p.problem=c.problem
    held_context=sorted({b for _,b in splits["train"]})[0]
    held=[(a,b) for a,b in splits["train"] if b==held_context]
    held_keys={a for a,b in held[::6]}
    fit=[(a,b) for a,b in splits["train"] if b!=held_context and a not in held_keys]
    old_probe=list(p.rows("old",held[::2],"plain",("original",)))
    new_probe=list(p.rows("related",[r for r in held if r[0] not in held_keys],"plain",("original",)))
    new_pool=list(p.rows("related",fit,"plain",("original","counterfactual")))
    batches=p.sample(new_pool,seed+1,64)
    old_val=list(p.rows("old",splits["validation"],"plain",p.VARIANTS))
    new_val=list(p.rows("related",[r for r in splits["validation"] if r[0] not in held_keys],"plain",("original",)))
    old_test=list(p.rows("old",splits["test"],"plain",p.VARIANTS))
    new_test=list(p.rows("related",[r for r in splits["test"] if r[0] not in held_keys],"plain",("original",)))
    with p.torch.no_grad():
        old_anchor=logp(e,old_probe).mean().item()
    result={}
    for arm in ("plain","oracle"):
        started=time.perf_counter()
        e.restore(initial)
        opt=p.optimizer(e.ps,.001)
        baseline=p.acc(p.evaluate(e,new_val))
        curves,trace,states=[],[],{}
        for i in range(65):
            if i:
                start=e.snapshot()
                old_opt=copy.deepcopy(opt.state_dict()) if arm=="oracle" else None
                p.train_step(e,opt,batches[i-1])
                check_memory()
                if arm=="oracle":
                    direction=[a-b for a,b in zip(e.snapshot(),start)]
                    new_probe_sample=new_probe[((i-1)*4)%len(new_probe):((i-1)*4)%len(new_probe)+4]
                    options=[]
                    with p.torch.no_grad():
                        for scale in (0.,.25,.5,1.):
                            e.restore(start,direction,scale)
                            scores=logp(e,old_probe+new_probe_sample)
                            harm=old_anchor-scores[:len(old_probe)].mean().item()
                            benefit=scores[len(old_probe):].mean().item()
                            options.append((scale,harm,benefit))
                    feasible=[q for q in options if q[1]<=.25]
                    selected=max(feasible,key=lambda q:(q[2],-q[0]))[0] if feasible else 0.
                    e.restore(start,direction,selected)
                    if selected==0.:
                        opt.load_state_dict(old_opt)
                    trace.append({"chosen":selected,"candidates":options})
            if i in d.GRID:
                gain=p.acc(p.evaluate(e,new_val))-baseline
                curves.append({"step":i,"gain":gain,"old_triplet":d.triplets(p.evaluate(e,old_val))})
                states[i]=[x.cpu() for x in e.snapshot()]
                print("ORACLE",arm,i,gain,flush=True)
        e.restore(states[64])
        result[arm]={"curve":curves,"trace":trace,"states":states,
                     "seconds_training_validation":time.perf_counter()-started,
                     "test":{"old":p.evaluate(e,old_test),"new":p.evaluate(e,new_test)}}
    import itertools
    points=[(x,y) for x,y in itertools.product(*(result[a]["curve"] for a in ("plain","oracle")))
            if min(x["gain"],y["gain"])>=.125 and abs(x["gain"]-y["gain"])<=2/len(new_val)+1e-8]
    matched=min(points,key=lambda pair:(max(x["gain"] for x in pair),abs(pair[0]["gain"]-pair[1]["gain"]),
                                         sum(x["step"] for x in pair))) if points else None
    for i,arm in enumerate(("plain","oracle")):
        states=result[arm].pop("states")
        if matched:
            e.restore(states[matched[i]["step"]])
            result[arm]["matched"]={"point":matched[i],"old":p.evaluate(e,old_test),"new":p.evaluate(e,new_test)}
    p.write(folder/"oracle.json",{"arms":result,"held_context":held_context,"held_keys":sorted(held_keys),
                                 "risk_budget_nats":.25,
                                 "peak_reserved_mib":p.torch.cuda.max_memory_reserved()/2**20})


if __name__=="__main__":
    mode,seed=sys.argv[1:]
    {"first_order":first_order,"migration":migration,"oracle":oracle,"carry":carry}[mode](int(seed))


