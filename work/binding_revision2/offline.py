"""Matched-row probability budgets and window audits; no model or CUDA imports."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"outputs/binding_review_20261111"
OUT=ROOT/"outputs/binding_revision2_20261111"
NAMES=("correct_cedar","other_cedar","paired_maple","same_family_maple",
       "other_active_original_maple","other_active_counterfactual_maple","inactive_maple","other")

def pava(y):
    """Equal checkpoint weights; retrospective isotonic least squares."""
    blocks=[]
    for i,v in enumerate(y):
        blocks.append([float(v),1,i,i+1])
        while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
            b=blocks.pop(); a=blocks.pop(); n=a[1]+b[1]
            blocks.append([(a[0]*a[1]+b[0]*b[1])/n,n,a[2],b[3]])
    result=np.empty(len(y))
    for v,n,start,end in blocks: result[start:end]=v
    return result

def measures(p,q,keys,active):
    a=np.array([k in active for k in keys])
    result={}
    for label,mask in (("active",a),("all",np.ones(len(keys),dtype=bool))):
        x,y=p[mask],q[mask]; ks=keys[mask]; ids=np.arange(len(ks))
        po=np.maximum(0,1-x.sum(1)); qo=np.maximum(0,1-y.sum(1))
        pp=np.c_[x,po]; qq=np.c_[y,qo]; mid=(pp+qq)/2
        ent=lambda v: -(v*np.log(np.maximum(v,np.finfo(float).tiny))).sum(1)
        js=ent(mid)-(ent(pp)+ent(qq))/2
        tv=.5*np.abs(pp-qq).sum(1)
        d=x[ids,ks-1]; m=y[ids,48+ks-1]
        budgets=np.zeros((len(ks),8))
        budgets[:,0]=d; budgets[:,1]=x[:,:48].sum(1)-d
        budgets[:,2]=x[ids,48+ks-1]; budgets[:,-1]=po
        for row,k in enumerate(ks):
            for j in range(1,49):
                if j==k: continue
                group=3 if j==k+1 else 4 if j in active else 5 if j-1 in active else 6
                budgets[row,group]+=x[row,48+j-1]
        assert np.allclose(budgets.sum(1),1,atol=1e-5)
        D,M=float(d.mean()),float(m.mean()); J=float(js.mean()); T=float(tv.mean())
        sigma=D+M-1
        assert sigma<=T+1e-5 and T<=np.sqrt(2*max(J,0))+1e-5
        result[label]=dict(D=D,M=M,sigma=sigma,blind_bound=1-M,
            blind_ratio=D/(1-M) if M<1-1e-12 else None,
            js=J,tv=T,tv_upper=float(min(1,np.sqrt(2*max(J,0)))),
            signed_cedar=float((d-y[ids,ks-1]).mean()),
            signed_maple=float((m-x[ids,48+ks-1]).mean()),
            budget=dict(zip(NAMES,map(float,budgets.mean(0)))))
    grouped=np.array([p[keys==k].mean(0) for k in sorted(set(keys))])
    grouped=np.c_[grouped,np.maximum(0,1-grouped.sum(1))]
    entropy=lambda v: -(v*np.log(np.maximum(v,np.finfo(float).tiny))).sum(-1)
    result["information"]=float(entropy(grouped.mean(0))-entropy(grouped).mean())
    return result

def load(folder,protocol,step):
    with np.load(folder/f"s{step:04d}.npz") as z:
        keep=z["indices"]%6<2
        keys=np.array([int(protocol["old_eval_ids"][i].split("/")[0]) for i in z["indices"][keep]])
        return z["old"][keep],z["maple"][keep],keys

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    assert np.allclose(pava([1,3,2,4]),[1,2.5,2.5,4])
    records={}; pairs=[]
    for folder in sorted((SOURCE/"runs").iterdir()):
        if not (folder/"complete.json").exists(): continue
        protocol=json.loads((folder/"protocol.json").read_text())
        trace=json.loads((folder/"trace.json").read_text())
        raw=np.array([r["maple"] for r in trace]); smooth=pava(raw)
        checks=[]
        for row in trace:
            p,q,keys=load(folder,protocol,row["step"])
            checks.append(dict(step=row["step"],validation_logp=row["maple"],
                               measurements=measures(p,q,keys,protocol["active_original"])))
        targets=[]
        for method,curve in (("raw_first_crossing",raw),("isotonic_retrospective",smooth)):
            for goal in (-1.,-.5,-.3,-.1):
                hits=np.flatnonzero(curve>=goal)
                if not len(hits):
                    targets.append(dict(method=method,target=goal,covered=False)); continue
                j=int(hits[0]); lo=max(0,j-1)
                w=0 if j==0 else float((goal-curve[lo])/(curve[j]-curve[lo]))
                p,q,keys=load(folder,protocol,trace[lo]["step"])
                p1,q1,_=load(folder,protocol,trace[j]["step"])
                targets.append(dict(method=method,target=goal,covered=True,
                    step=(1-w)*trace[lo]["step"]+w*trace[j]["step"],
                    observed_step=trace[j]["step"],
                    observed=checks[j]["measurements"],
                    measurements=measures((1-w)*p+w*p1,(1-w)*q+w*q1,keys,protocol["active_original"])))
        records[folder.name]=dict(protocol=protocol,checkpoints=checks,targets=targets,
                                  smoothed_validation=smooth.tolist())
    for seed in (107,108,109):
        names=[f"Qwen3-0.6B_s{seed}_adjacent_lr{x}_r4" for x in ("0.0001","0.001")]
        base=f"Qwen3-0.6B_s{seed}_adjacent_lr0.0003_r4"
        for goal in (-1.,-.5,-.3,-.1):
            def point(name):
                return next(t for t in records[name]["targets"] if t["method"]=="isotonic_retrospective" and t["target"]==goal)
            for label,a,b in (("low_minus_high",names[0],names[1]),("replay_minus_plain",base+"_replay",base)):
                x,y=point(a),point(b)
                pairs.append(dict(seed=seed,contrast=label,target=goal,covered=x["covered"] and y["covered"],
                    difference=x["measurements"]["active"]["D"]-y["measurements"]["active"]["D"]
                    if x["covered"] and y["covered"] else None))
    (OUT/"matched_audit.json").write_text(json.dumps(dict(runs=records,paired_differences=pairs),indent=2,allow_nan=False))
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,3,figsize=(9,6),layout="constrained")
    for i,model in enumerate(("Qwen3-0.6B","SmolLM2-360M-Instruct")):
        for j,seed in enumerate((211,223,227)):
            r=records[f"{model}_s{seed}_adjacent_lr0.0003_r4"]; rows=r["checkpoints"]
            v=np.array([x["measurements"]["information"] for x in rows])
            delta=v[None,:]-v[:,None]
            delta[np.tril_indices(len(v),0)]=np.nan
            im=axes[i,j].imshow(delta,vmin=-3.2,vmax=3.2,cmap="RdBu",aspect="equal")
            ticks=np.linspace(0,len(v)-1,5,dtype=int); labels=[rows[k]["step"] for k in ticks]
            axes[i,j].set(xticks=ticks,xticklabels=labels,yticks=ticks,yticklabels=labels,
                          xlabel="End step",ylabel="Start step",title=f"{model}, seed {seed}")
    fig.colorbar(im,ax=axes,label="MI(end) - MI(start), nats")
    for ext in ("png","pdf"): fig.savefig(OUT/f"all_windows.{ext}",dpi=300)
    plt.close(fig)
    lines=["# 第二轮离线审计","",
      "所有概率与界限均在同一权重、同一测试上下文、同一键集合上计算。主表限 16 个活跃原题键（每键两个上下文），不把 24 键的 D 与 16 键的 M 相加。",
      "使用候选+其余的粗化 TV、JS，Σ≤TV≤√(2 JS) 在该共同粗化空间即成立。比值为聚合 D/(1−M)，不是逐题比值的均值；它衡量相对于完全标签盲混合界的大小，不是准确率。",
      "验证曲线采用等检查点权重 PAVA；这是使用整条已观测轨迹的回顾性规格，不能作为在线停机算法。原始首次越过结果同时保留。高目标缺失禁止外推。","",
      "|分支|目标|步数|D %|M %|Σ|D/(1−M)|有符号 Cedar 路由|TV|√(2 JS)|",
      "|---|---|---|---|---|---|---|---|---|---|"]
    for name,r in records.items():
        for t in r["targets"]:
            if t["method"]!="isotonic_retrospective": continue
            if not t["covered"]:
                lines.append(f"|{name}|{t['target']}|未覆盖|—|—|—|—|—|—|—|");continue
            m=t["measurements"]["active"]
            lines.append(f"|{name}|{t['target']}|{t['step']:.1f}|{100*m['D']:.2f}|{100*m['M']:.2f}|{m['sigma']:.4f}|{m['blind_ratio']:.3f}|{m['signed_cedar']:.4f}|{m['tv']:.4f}|{m['tv_upper']:.4f}|")
    lines+=["","## 完整质量预算（共同目标 −1.0）","",
        "|分支|"+"|".join(NAMES)+"|","|---|"+"---|"*8]
    for name,r in records.items():
        t=next(t for t in r["targets"] if t["method"]=="isotonic_retrospective" and t["target"]==-1)
        if t["covered"]:
            b=t["measurements"]["active"]["budget"]
            lines.append("|"+name+"|"+"|".join(f"{100*b[k]:.2f}" for k in NAMES)+"|")
    lines+=["","## 旧种子的配对差（描述性，不能替代新确认种子）","",
            "|对照|目标|覆盖|正方向数|逐种子差 pp|","|---|---|---|---|---|"]
    for label in ("low_minus_high","replay_minus_plain"):
        for goal in (-1.,-.5,-.3,-.1):
            group=[p for p in pairs if p["contrast"]==label and p["target"]==goal and p["covered"]]
            lines.append(f"|{label}|{goal}|{len(group)}/3|{sum(p['difference']>0 for p in group)}|"+
                         ", ".join(f"{p['seed']}: {p['difference']*100:+.2f}" for p in group)+"|")
    (OUT/"离线审计.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Audited",len(records),"runs; matched-row bounds and complete budgets verified.")

if __name__=="__main__": main()
