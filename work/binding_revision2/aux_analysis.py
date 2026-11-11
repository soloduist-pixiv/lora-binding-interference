"""Unnormalized probability and declared-alias audits for auxiliary probes; CPU only."""
import json
import os
from pathlib import Path
import unicodedata
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from aux_data import OUT
from offline import pava

def normalize(text):
    text=unicodedata.normalize("NFKD",text)
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in text if not unicodedata.combining(c)).split())

def measure(arrays,data):
    n=len(data["keys"]); keys=np.repeat(np.arange(n),3);idx=np.arange(len(keys))
    columns=np.array([[data["candidates"].index(v) for v in group] for group in data["answers"]])
    complete=[];qualities=[];coverage=[]
    for label,x in enumerate(arrays):
        assert x.shape==(n*3,len(data["candidates"]))
        assert x.min()>=0 and x.sum(1).max()<=1+1e-5
        other=np.maximum(0,1-x.sum(1));complete.append(np.c_[x,other])
        qualities.append(float(x[idx,columns[label,keys]].mean()))
        coverage.append(dict(mean_other=float(other.mean()),p95_other=float(np.quantile(other,.95)),max_other=float(other.max())))
    pairs={}
    ent=lambda x:-(x*np.log(np.maximum(x,np.finfo(float).tiny))).sum(1)
    for a,b in ((0,1),(0,2),(1,2)):
        x,y=complete[a],complete[b];mid=(x+y)/2
        js=float((ent(mid)-.5*(ent(x)+ent(y))).mean())
        tv=float((.5*np.abs(x-y).sum(1)).mean())
        sigma=qualities[a]+qualities[b]-1
        assert sigma<=tv+1e-5 and tv<=np.sqrt(max(0.,2*js))+1e-5
        pairs[f"{a}_{b}"]=dict(sigma=sigma,js=js,tv=tv,tv_upper=float(min(1,np.sqrt(max(0.,2*js)))),
            signed_old=float((arrays[a][idx,columns[a,keys]]-arrays[b][idx,columns[a,keys]]).mean()),
            blind_ratio=qualities[a]/(1-qualities[b]) if qualities[b]<1-1e-12 else None)
    budget={};x=arrays[0]
    for label in range(3):
        correct=x[idx,columns[label,keys]]
        budget[f"matched_register{label}"]=float(correct.mean())
        budget[f"other_register{label}"]=float((x[:,columns[label]].sum(1)-correct).mean())
    budget["other"]=coverage[0]["mean_other"]
    assert abs(sum(budget.values())-1)<1e-5
    grouped=complete[0].reshape(n,3,-1).mean(1)
    mi=float(ent(grouped.mean(0,keepdims=True))[0]-ent(grouped).mean())
    return dict(correct_probability=qualities,coverage=coverage,pairs=pairs,cedar_budget=budget,cedar_mi=mi)

def main():
    root=OUT/"auxiliary";results={}
    aliases=json.loads((OUT/"temporal_fact_sources.json").read_text(encoding="utf8"))["canonicalization"]
    for protocol_file in root.glob("*/*/protocol.json"):
        folder=protocol_file.parent
        protocol=json.loads(protocol_file.read_text())
        data=json.loads((OUT/"auxiliary_data"/f"{protocol['kind']}_s{protocol['seed']}.json").read_text(encoding="utf8"))
        for stage in (1,2):
            current=folder/f"stage{stage}"
            if not (current/"complete.json").exists():continue
            trace=json.loads((current/"trace.json").read_text());curve=pava([v["validation_logp"] for v in trace])
            def load(step):
                with np.load(current/f"s{step:04d}.npz") as z:return [z[f"register{s}"].copy() for s in range(3)]
            targets=[]
            for goal in (-1.,-.5,-.3,-.1):
                hits=np.flatnonzero(curve>=goal)
                if not len(hits):targets.append(dict(target=goal,covered=False));continue
                j=int(hits[0]);i=max(0,j-1);w=0 if j==0 else float((goal-curve[i])/(curve[j]-curve[i]))
                a=load(trace[i]["step"]);b=load(trace[j]["step"])
                targets.append(dict(target=goal,covered=True,step=(1-w)*trace[i]["step"]+w*trace[j]["step"],
                    effective_path=(1-w)*trace[i]["effective_path"]+w*trace[j]["effective_path"],
                    measurements=measure([(1-w)*x+w*y for x,y in zip(a,b)],data)))
            final=measure(load(640),data)
            decoded=json.loads((current/"final_generations.json").read_text())
            native={}
            for label,rows in decoded.items():
                correct=[]
                for row in rows:
                    allowed=[row["answer"],*aliases.get(row["answer"],[])] if protocol["kind"]=="temporal" else [row["answer"]]
                    correct.append(normalize(row["response"]) in {normalize(v) for v in allowed})
                native[label]=dict(canonical_accuracy=float(np.mean([v["correct"] for v in rows])),
                    declared_alias_accuracy=float(np.mean(correct)),queries=len(rows))
            name=str(current.relative_to(root)).replace("\\","/")
            results[name]=dict(protocol=protocol,stage=stage,targets=targets,
                initial_validation_logp=trace[0]["validation_logp"],endpoint=final,native=native,
                terminal_coverage_criterion=all(v["mean_other"]<=.02 for v in final["coverage"][:stage+1]),
                coverage_rule="Every trained register's terminal mean other mass <= .02; no test-based retuning or seed replacement")
    temporary=OUT/"auxiliary_analysis.tmp";temporary.write_text(json.dumps(results,indent=2,allow_nan=False))
    temporary.replace(OUT/"auxiliary_analysis.json")
    lines=["# 封闭词表与真实时间事实审计","",
           "本表只列完成的阶段。合成臂 16 键、48 个单 token 词、三个注册表；真实事实臂 8 个职位、24 个规范人名、三个历史日期。都在独立措辞模板上测试，不宣称新国家或事实泛化。候选集合只写入提示，不限制解码或归一化模型概率。",
           "每阶段固定 640 步，不称收敛。平均候选外质量任一已训练注册表超过 2% 时，封闭词表臂不得进入主文作为高覆盖证据，原始结果仍保留。","",
           "|分支/阶段|D0 %|D1 %|D2 %|Σ01|Σ02|Σ12|最大已训练标签平均其余 %|覆盖规则通过|","|---|---|---|---|---|---|---|---|---|"]
    for name,row in results.items():
        m=row["endpoint"];q=m["correct_probability"]
        other=max(v["mean_other"] for v in m["coverage"][:row["stage"]+1])
        lines.append(f"|{name}|{q[0]*100:.2f}|{q[1]*100:.2f}|{q[2]*100:.2f}|"
                     +"|".join(f"{m['pairs'][k]['sigma']:.4f}" for k in ("0_1","0_2","1_2"))
                     +f"|{100*other:.3f}|{row['terminal_coverage_criterion']}|")
    if not results:lines.append("\n尚无完成的辅助训练阶段；输入数据已冻结，不能把输入校验当作实验结果。")
    (OUT/"辅助实验审计.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Completed auxiliary stages audited:",len(results))

if __name__=="__main__":main()
