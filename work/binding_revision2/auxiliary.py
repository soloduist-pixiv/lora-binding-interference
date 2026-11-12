"""Serial three-register probes; separate from the eight-anchor primary study."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from resource_guard import require_gpu_mode

if __name__=="__main__":require_gpu_mode()

import numpy as np
from aux_data import OUT,SEEDS

# Use a unique module name: historical diagnostics itself imports a module named run.
spec=importlib.util.spec_from_file_location("revision2_runner",Path(__file__).with_name("run.py"))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
p=r.p

def save(path,parameters,**fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    p.torch.save(dict(parameters=[v.detach().cpu() for v in parameters],rank=4,scale=2.,**fields),temporary)
    temporary.replace(path)

def restore(engine,path):
    state=p.torch.load(path,map_location="cpu",weights_only=True)
    r.restore(engine,state,4,0)
    return state

def anchor(engine,scorer,data,folder):
    folder.mkdir(exist_ok=True,parents=True)
    if (folder/"old.pt").exists():
        restore(engine,folder/"old.pt");return
    old_val=data["rows"]["validation"][0]
    engine.reset(data["seed"],"add")
    optimizer=p.optimizer(engine.ps,.001)
    start=0;trace=[]
    if (folder/"state.pt").exists():
        state=restore(engine,folder/"state.pt");start=state["step"]+1
        optimizer=p.optimizer(engine.ps,.001);optimizer.load_state_dict(state["optimizer"])
        trace=json.loads((folder/"trace.json").read_text())
    else:
        baseline=r.generate(scorer,old_val)
        p.write(folder/"baseline.json",dict(validation_logp=float(np.mean(scorer.gold(old_val))),
                    queries=baseline,accuracy=float(np.mean([v["correct"] for v in baseline]))))
    batches=p.sample(data["rows"]["train"][0],data["seed"]+32000,1280)
    for step in range(start,1281):
        if step:p.train_step(engine,optimizer,batches[step-1]);r.x.check_memory()
        if step%32:continue
        lp=float(np.mean(scorer.gold(old_val)))
        decoded=r.generate(scorer,old_val) if lp>=-.1 else None
        acc=float(np.mean([v["correct"] for v in decoded])) if decoded is not None else None
        passed=lp>=-.1 and acc>=.875
        trace.append(dict(step=step,validation_logp=lp,accuracy=acc,passed=passed))
        p.write(folder/"trace.json",trace)
        save(folder/"state.pt",engine.ps,step=step,optimizer=optimizer.state_dict())
        print("ANCHOR",data["kind"],data["seed"],trace[-1],flush=True)
        if passed:
            save(folder/"old.pt",engine.ps,step=step)
            p.write(folder/"gate.json",dict(passed=True,step=step,validation_logp=lp,accuracy=acc))
            return
    p.write(folder/"gate.json",dict(passed=False,step=1280))
    raise RuntimeError("Auxiliary old-fact gate failed; retain this seed, do not replace")

def main(args):
    if p.torch.cuda.mem_get_info()[0]<6*1024**3:raise RuntimeError("Requires 6 GiB free GPU memory")
    input_file=OUT/"auxiliary_data"/f"{args.kind}_s{args.seed}.json"
    data=json.loads(input_file.read_text(encoding="utf8"))
    root=OUT/"auxiliary"/f"{args.model}_{args.kind}_s{args.seed}"
    folder=root/("replay" if args.replay else "plain");folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():return
    engine=p.Engine(args.model) if args.model=="Qwen3-0.6B" else r.StandardEngine(args.model)
    scorer=r.CandidateScorer(engine,data["candidates"],batch_size=2,diagnostic_directory=folder)
    if args.kind=="closed":
        tokens=[engine.tok.encode(word,add_special_tokens=False) for word in data["candidates"]]
        assert all(len(t)==1 for t in tokens) and len({t[0] for t in tokens})==48
    protocol=dict(model=args.model,kind=args.kind,seed=args.seed,replay=args.replay,
        new_lr=.0003,rank=4,scale=2.,scope="last4_qv",batch_new=4,batch_old=4 if args.replay else 0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
        maximum_new_steps=640,scoring="full canonical sequence plus EOS; unnormalized vocabulary probabilities",
        scoring_precision="float32",training_precision="BF16 base, FP32 adapters",
        targets=[-1.,-.5,-.3,-.1],gpu=p.torch.cuda.get_device_name(),
        adapter_parameters=sum(v.numel() for v in engine.ps),
        candidates=data["candidates"],answers=data["answers"],keys=data["keys"],labels=data["labels"])
    p.write(folder/"protocol.json",protocol)
    anchor(engine,scorer,data,root/"anchor")
    for stage in (1,2):
        current=folder/f"stage{stage}";current.mkdir(exist_ok=True)
        if (current/"complete.json").exists():
            restore(engine,current/"finish.pt");continue
        optimizer=p.optimizer(engine.ps,.0003)
        start=0;trace=[];path=0.
        if (current/"state.pt").exists():
            state=restore(engine,current/"state.pt")
            start=state["step"]+1;path=state["effective_path"]
            optimizer=p.optimizer(engine.ps,.0003);optimizer.load_state_dict(state["optimizer"])
            trace=json.loads((current/"trace.json").read_text())
        batches=p.sample(data["rows"]["train"][stage],args.seed+34000+stage,640)
        replay_pool=[v for rows in data["rows"]["train"][:stage] for v in rows]
        previous=p.sample(replay_pool,args.seed+35000+stage,640)
        begun=time.monotonic()
        for step in range(start,641):
            if step:
                before=[v.detach().clone() for v in engine.ps]
                p.train_step(engine,optimizer,batches[step-1]+(previous[step-1] if args.replay else []))
                with p.torch.no_grad():
                    norm=0.
                    for i in range(0,len(engine.ps),2):
                        a,b=engine.ps[i:i+2];oa,ob=before[i:i+2]
                        left=p.torch.cat([b-ob,ob],1);right=p.torch.cat([a,a-oa],0)
                        norm+=4*float(((left.T@left)*(right@right.T).T).sum())
                    path+=max(0.,norm)**.5
                r.x.check_memory()
            if (step<=64 and step%2) or (step>64 and step%16):continue
            arrays={}
            validations=[]
            scorer.phase=f"stage{stage}_checkpoint{step}"
            with scorer.evaluation_precision():
                for label in range(3):
                    arrays[f"register{label}"]=np.exp(scorer(data["rows"]["test"][label]))
                    validations.append(float(np.mean(scorer.gold(data["rows"]["validation"][label]))))
            np.savez_compressed(current/f"s{step:04d}.npz",**arrays)
            trace.append(dict(step=step,validation_logp=validations[stage],
                validation_by_registry=validations,effective_path=path,path_origin=0,
                new_examples=step*4,old_examples=step*4 if args.replay else 0,
                seconds=time.monotonic()-begun,peak_reserved_mib=p.torch.cuda.max_memory_reserved()/2**20))
            p.write(current/"trace.json",trace)
            save(current/f"adapter_{step:04d}.pt",engine.ps,step=step)
            save(current/"state.pt",engine.ps,step=step,optimizer=optimizer.state_dict(),effective_path=path)
            print(args.kind,args.seed,args.replay,"stage",stage,step,validations,flush=True)
        save(current/"finish.pt",engine.ps,step=640)
        native={label:r.generate(scorer,rows) for label,rows in enumerate(data["rows"]["test"])}
        p.write(current/"final_generations.json",native)
        first=next((v for v in trace if v["validation_logp"]>=-1),None)
        if first:
            restore(engine,current/f"adapter_{first['step']:04d}.pt")
            p.write(current/"first_minus1_generation.json",dict(step=first["step"],
                validation_logp=first["validation_logp"],queries={
                label:r.generate(scorer,rows) for label,rows in enumerate(data["rows"]["test"])}))
            restore(engine,current/"finish.pt")
        p.write(current/"complete.json",dict(last_step=640,stop_reason="fixed_budget",convergence_claim=False))
    p.write(folder/"complete.json",dict(stages_completed=[1,2],status="complete"))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--kind",choices=("closed","temporal"),required=True)
    parser.add_argument("--model",choices=("Qwen3-0.6B","SmolLM2-360M-Instruct"),default="Qwen3-0.6B")
    parser.add_argument("--seed",choices=SEEDS,type=int,required=True)
    parser.add_argument("--replay",action="store_true")
    main(parser.parse_args())
