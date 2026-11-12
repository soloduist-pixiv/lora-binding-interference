"""Post-hoc candidate-source readouts. CPU only, no model loading or forced-mass claims."""
import hashlib
import json
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from offline import ROOT,OUT,SOURCE,load,pava

PERMUTATIONS=2000

def tail(observed,null):
    return dict(observed=float(observed),null_mean=float(null.mean()),
                null_quantiles=np.quantile(null,[.025,.5,.975]).tolist(),
                exploratory_upper_p=float((1+np.sum(null>=observed-1e-12))/(len(null)+1)))

def diagnose(p,keys,protocol,seed):
    active=np.array(sorted(protocol["active_original"]));mask=np.isin(keys,active)
    full=p[mask];p=full[:,:48];keys=keys[mask];true=keys-1
    assert set(keys)==set(active)
    old=np.array(protocol["old"]);new=np.array(protocol["new"])
    rng=np.random.default_rng(seed)
    all_max=full.max(1)
    all_credit=(full[np.arange(len(keys)),true]==all_max)/np.sum(full==all_max[:,None],axis=1)
    result=dict(raw_correct=float(p[np.arange(len(keys)),true].mean()),
                all96_candidate_old_top1=float(all_credit.mean()),
                old_set_mass_mean=float(p.sum(1).mean()),old_set_mass_min=float(p.sum(1).min()),
                original_query_count=len(keys),original_key_count=len(active),readouts={})
    for label,columns in (("all48",np.arange(48)),("active16",active-1)):
        x=p[:,columns];mass=x.sum(1);valid=mass>0
        lookup={int(c):i for i,c in enumerate(columns)}
        targets=np.array([lookup[int(c)] for c in true])
        maximum=x.max(1,keepdims=True);credit=(x==maximum)/np.sum(x==maximum,axis=1,keepdims=True)
        conditional=np.divide(x,mass[:,None],out=np.zeros_like(x),where=mass[:,None]>0)
        good_keys=[int(k) for k in active if np.all(valid[keys==k])]
        if not good_keys:
            result["readouts"][label]=dict(defined=False,valid_key_count=0);continue
        c=np.array([credit[keys==k].mean(0) for k in good_keys])
        q=np.array([conditional[keys==k].mean(0) for k in good_keys])
        target=np.array([lookup[k-1] for k in good_keys]);ii=np.arange(len(good_keys))
        permutations=np.array([rng.permutation(len(columns)) for _ in range(PERMUTATIONS)])
        assigned=permutations[:,target]
        obs_c=c[ii,target].mean();obs_q=q[ii,target].mean()
        readout=dict(defined=True,candidate_count=len(columns),valid_key_count=len(good_keys),
            oracle_top1=tail(obs_c,c[ii[None,:],assigned].mean(1)),
            conditional_correct=tail(obs_q,q[ii[None,:],assigned].mean(1)),
            exact_null_expectation=1/len(columns),mass_floor_sensitivity={})
        for floor in (0.,1e-8,1e-4):
            keep=[int(k) for k in active if np.all(mass[keys==k]>floor)]
            idx=np.isin(keys,keep)
            readout["mass_floor_sensitivity"][str(floor)]=dict(keys=len(keep),planned_keys=16,
                top1=float(credit[idx,targets[idx]].mean()) if idx.any() else None,
                conditional_correct=float(conditional[idx,targets[idx]].mean()) if idx.any() else None)
        result["readouts"][label]=readout
        if label=="all48":
            # Rows = fixed old-word predictions; columns = same trained Maple assignments.
            prefix=np.array([[float((row*(old//10==new[k-1]//10)).sum()) for k in good_keys] for row in c])
            permutation=np.array([rng.permutation(len(good_keys)) for _ in range(PERMUTATIONS)])
            result["maple_prefix_match"]=tail(np.diag(prefix).mean(),prefix[ii[None,:],permutation].mean(1))
            result["maple_prefix_match"]["exact_null_mean"]=float(prefix.mean())
            result["key_details"]=[dict(key=k,cedar=int(old[k-1]),maple=int(new[k-1]),
                raw_correct=float(p[keys==k,k-1].mean()),old_mass=float(p[keys==k].sum(1).mean()),
                old_oracle_top_values=[int(v) for v in old[c[j]==c[j].max()]],
                old_oracle_correct=float(c[j,k-1]),
                maple_prefix_match=float(prefix[j,j])) for j,k in enumerate(good_keys)]
    return result

def main():
    records={};excluded=[]
    for source,label in ((SOURCE,"first_revision"),(OUT,"second_revision")):
        for folder in sorted((source/"runs").iterdir()):
            if not (folder/"complete.json").exists():continue
            protocol=json.loads((folder/"protocol.json").read_text())
            if protocol.get("minimum_steps",64)<64:continue
            if protocol["pairing"]=="alphabetic":excluded.append(label+"/"+folder.name);continue
            trace=json.loads((folder/"trace.json").read_text());curve=pava([v["maple"] for v in trace])
            points={};stages={};steps={v["step"] for v in trace}
            actual={"initial":0,"terminal":trace[-1]["step"]}
            for step in (64,640):
                if step in steps:actual[f"step{step}"]=step
            raw=next((v["step"] for v in trace if v["maple"]>=-1),None)
            if raw is not None:actual["raw_first_minus1"]=raw
            def calculate(point,p,keys):
                nonce=f"source_oracle_v1|{label}|{folder.name}|{point}"
                seed=int.from_bytes(hashlib.sha256(nonce.encode()).digest()[:8],"little")
                return dict(seed=seed,measurements=diagnose(p,keys,protocol,seed))
            for stage,step in actual.items():
                point=f"step_{step}"
                if point not in points:
                    p,_,keys=load(folder,protocol,step);points[point]=calculate(point,p,keys)
                    points[point]["step"]=step
                stages[stage]=point
            for goal in (-1.,-.5,-.3,-.1):
                hits=np.flatnonzero(curve>=goal);stage=f"goal{goal}"
                if not len(hits):stages[stage]=None;continue
                j=int(hits[0]);i=max(0,j-1)
                w=0 if j==0 else float((goal-curve[i])/(curve[j]-curve[i]))
                p,_,keys=load(folder,protocol,trace[i]["step"]);p1,_,_=load(folder,protocol,trace[j]["step"])
                point=stage;points[point]=calculate(point,(1-w)*p+w*p1,keys)
                points[point]["step"]=(1-w)*trace[i]["step"]+w*trace[j]["step"]
                stages[stage]=point
            native=None;generation=folder/"first_minus1_generation.json"
            if generation.exists() and raw is not None:
                g=json.loads(generation.read_text());assert g["step"]==raw
                with np.load(folder/f"s{raw:04d}.npz") as z:
                    chosen=[int(i) for i in z["indices"] if i%6<2]
                ids={protocol["old_eval_ids"][i] for i in chosen
                     if int(protocol["old_eval_ids"][i].split("/")[0]) in protocol["active_original"]}
                queries=[v for v in g["queries"]["cedar"] if v["id"] in ids and v["variant"]=="original"]
                assert len(queries)==32 and len(ids)==32
                native=dict(step=raw,actual_validation_logp=g["validation_logp"],queries=len(queries),
                    accuracy=float(np.mean([v["correct"] for v in queries])),point=stages["raw_first_minus1"])
            records[label+"/"+folder.name]=dict(protocol=protocol,stages=stages,points=points,
                initial_validation_logp=trace[0]["maple"],native_raw_minus1=native,statistical_status="post_hoc_exploratory")
    output=dict(status="Exploratory CPU-only diagnostic; source constraint is an external oracle, not model conditioning",
                permutations=PERMUTATIONS,excluded_alphabetic=excluded,runs=records)
    (OUT/"source_oracle_audit.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=["# 候选来源约束的离线诊断（探索性）","",
       "只使用已完成的数字答案分支，主集合为同一 16 个活跃原题键、每键两个固定上下文。外部限制只允许 48 个 Cedar 规范答案，单独计算受限排序；这不是原生生成准确率，也不是模型自行条件化。条件正确份额经过集合内归一化，必须与原始正确质量和集合质量同时阅读。",
       f"共 {len(records)} 个分支记录（含续训副本和共享锚点），不等于独立种子数；排除 {len(excluded)} 个覆盖失败的字母臂。每点 {PERMUTATIONS} 次按键的标签置换，所有旧值均经过 Cedar 训练。尾概率为未校正的探索性诊断。","",
       "## 配对结构：同一低目标 −1.0 的三个已分析种子","",
       "该目标不代表可用的新任务质量。表中 oracle 只回答限制候选后最高分是否为旧正确值。前缀匹配指该受限最高分是否与同键 Maple 值共享前两位数字；零参照在训练过的 16 个 Maple 赋值内置换。","",
       "|种子|配对|步数|原始 D %|Cedar 集合质量 %|48值 oracle 命中 %|oracle 条件正确份额 %|Maple 前缀匹配 %|前缀零参照 %|","|---|---|---|---|---|---|---|---|---|"]
    for seed in (211,223,227):
        for pairing in ("adjacent","support_permuted","permuted"):
            name=f"first_revision/Qwen3-0.6B_s{seed}_{pairing}_lr0.0003_r4";r=records[name]
            point=r["stages"]["goal-1.0"]
            if point is None:lines.append(f"|{seed}|{pairing}|未达目标|—|—|—|—|—|—|");continue
            t=r["points"][point];m=t["measurements"];a=m["readouts"]["all48"];f=m["maple_prefix_match"]
            lines.append(f"|{seed}|{pairing}|{t['step']:.1f}|{100*m['raw_correct']:.2f}|{100*m['old_set_mass_mean']:.2f}|{100*a['oracle_top1']['observed']:.2f}|{100*a['conditional_correct']['observed']:.2f}|{100*f['observed']:.2f}|{100*f['exact_null_mean']:.2f}|")
    lines += ["","## 所有完成分支的终点","",
              "各终点步数不同，仅作描述，不能当作等进度比较。完整 JSON 包含所有目标、未覆盖项、质量阈值敏感性、16值版本、逐键去向和零分布分位数。48值随机标签期望为 2.083%，16值为 6.25%。","",
              "|分支|终点步数|原始 D %|Cedar 集合质量 %|48值 oracle 命中 %|16值 oracle 命中 %|48值 oracle 命中（集合质量>1e-4）%|阈值后覆盖键数/16|","|---|---|---|---|---|---|---|---|"]
    for name,r in records.items():
        t=r["points"][r["stages"]["terminal"]];m=t["measurements"];a=m["readouts"]["all48"];b=m["readouts"]["active16"]
        s=a["mass_floor_sensitivity"]["0.0001"]
        value=f"{100*s['top1']:.2f}" if s["top1"] is not None else "未定义"
        lines.append(f"|{name}|{t['step']}|{100*m['raw_correct']:.3f}|{100*m['old_set_mass_mean']:.3f}|{100*a['oracle_top1']['observed']:.2f}|{100*b['oracle_top1']['observed']:.2f}|{value}|{s['keys']}/16|")
    lines += ["","## 同一实测检查点、同一上下文的原生生成对照","",
              "仅在保存过首次实测 −1.0 生成记录的分支中报告。抽取和矩阵相同的 16 键×2 上下文，实际 logP 各自列出；不使用插值权重。oracle 是候选序列的受限最大概率读出，原生列是未限制词表的逐 token 贪心生成，两者不是相同解码器。","",
              "|分支|实测步数|实际验证 logP|原生贪心旧正确 %|96值候选旧正确 %|48值 oracle 旧正确 %|原始 D %|","|---|---|---|---|---|---|---|"]
    for name,r in records.items():
        g=r["native_raw_minus1"]
        if g is None:continue
        m=r["points"][g["point"]]["measurements"];a=m["readouts"]["all48"]["oracle_top1"]["observed"]
        lines.append(f"|{name}|{g['step']}|{g['actual_validation_logp']:.4f}|{100*g['accuracy']:.2f}|{100*m['all96_candidate_old_top1']:.2f}|{100*a:.2f}|{100*m['raw_correct']:.3f}|")
    lines += ["","限制：候选来源是额外 oracle；正的受限排序不证明知识未被擦除，接近随机也不证明所有潜在读出均失败。共享数值前缀可支持受限排序，本分析不定位内部绑定机制。未完成的新锚点仍然缺失。"]
    (OUT/"候选来源约束诊断.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Exploratory numeric branches:",len(records),"; excluded alphabetic:",len(excluded))

if __name__=="__main__":main()
