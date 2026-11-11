"""Prespecified fresh-anchor contrasts, including all missing seed pairs."""
import json
from offline import OUT

SEEDS=(401,409,419,431,443,449,457,463)
TARGETS=(-1.,-.5,-.3,-.1)

def report(audits):
    fresh={}
    for name,run in audits.items():
        p=run["protocol"]
        if not p["historical_anchor"] and p["pairing"]=="support_permuted" and p.get("scope","last4_qv")=="last4_qv":
            fresh[(p["seed"],p["lr"],p["rank"],p["replay"],p["exposure"])]=(name,run)
    records=[]
    for rank in (4,16):
        contrasts=[]
        for replay in (False,True):
            contrasts.append(("lr_1e-4_minus_1e-3",dict(rank=rank,replay=replay),
                              (.0001,rank,replay,"plain"),(.001,rank,replay,"plain")))
        for lr in (.0001,.0003,.001):
            contrasts.append(("replay_minus_plain",dict(rank=rank,lr=lr),
                              (lr,rank,True,"plain"),(lr,rank,False,"plain")))
        if rank==4:
            contrasts.append(("duplicate_minus_plain",dict(rank=4,lr=.0003),
                              (.0003,4,False,"duplicate_maple"),(.0003,4,False,"plain")))
        for label,condition,left,right in contrasts:
            for stage in (*TARGETS,"step640","terminal"):
                rows=[]
                for seed in SEEDS:
                    a=fresh.get((seed,*left)); b=fresh.get((seed,*right))
                    row=dict(seed=seed,covered=False,left=a[0] if a else None,right=b[0] if b else None)
                    if a and b:
                        def point(run):
                            if stage in TARGETS:
                                t=next(t for t in run["targets"] if t["target"]==stage)
                                return (t["measurements"]["active"],t["step"]) if t["covered"] else (None,None)
                            key="640" if stage=="step640" else str(max(map(int,run["terminals"])))
                            t=run["terminals"].get(key)
                            return (t["measurements"]["active"],int(key)) if t else (None,None)
                        x,sx=point(a[1]); y,sy=point(b[1])
                        if x is not None and y is not None:
                            row.update(covered=True,left_step=sx,right_step=sy,
                                       difference={k:x[k]-y[k] for k in ("D","M","sigma","signed_cedar","js")})
                        else:row["missing_reason"]="at least one completed branch did not cover this target"
                    else:row["missing_reason"]="at least one branch not completed"
                    rows.append(row)
                covered=[r for r in rows if r["covered"]]
                records.append(dict(contrast=label,condition=condition,stage=stage,
                    covered=len(covered),planned=len(SEEDS),
                    positive_D=sum(r["difference"]["D"]>0 for r in covered),
                    negative_D=sum(r["difference"]["D"]<0 for r in covered),rows=rows))
    status=dict(completed_fresh_branches=len(fresh),planned_fresh_branches=104,
                seeds=list(SEEDS),contrasts=records)
    temporary=OUT/"paired_contrasts.tmp"
    temporary.write_text(json.dumps(status,indent=2,allow_nan=False))
    temporary.replace(OUT/"paired_contrasts.json")
    lines=["# 新锚点配对对照（持续更新）","",
           f"已完成 {len(fresh)}/104 个新分支；预定八个锚点，未完成或未达目标均保留缺失。仅在配对两臂都覆盖目标时计算差，分母始终列为 8。",
           "所有概率限同一 16 个活跃原题键、每键两个上下文。高目标来自回顾性 PAVA 概率矩阵插值。终点可能步数不同，不能当作等预算结果；固定第 640 步另列。未完成八种子前不作确认性总体结论。","",
           "|对照|条件|阶段|覆盖/8|D 正/负方向数|逐种子 D 差 pp|","|---|---|---|---|---|---|"]
    for r in records:
        differences=", ".join(f"{x['seed']}: {100*x['difference']['D']:+.2f}" if x["covered"]
                              else f"{x['seed']}: 缺失" for x in r["rows"])
        lines.append(f"|{r['contrast']}|{r['condition']}|{r['stage']}|{r['covered']}/8|"
                     f"{r['positive_D']}/{r['negative_D']}|{differences}|")
    lines += ["","## 各分支初始难度、达标步数和路径","",
              "有效路径为每次更新的 LoRA 矩阵 Frobenius 位移累加；表中在拟合进度处作线性插值。以 path_origin 为起点；旧轨迹恢复点不能算作第零步。",
              "|分支|初始验证 logP|目标|拟合步数|首次实测步数|有效路径|路径起点|终点原因|","|---|---|---|---|---|---|---|---|"]
    for name,run in sorted(fresh.values()):
        for t in run["targets"]:
            raw=next(x["observed_step"] for x in run["raw_first_crossings"] if x["target"]==t["target"])
            path=t.get("effective_path_since_origin")
            lines.append(f"|{name}|{run['initial_validation_logp']:.4f}|{t['target']}|"
                         f"{t['step'] if t['covered'] else '未达'}|{raw}|{path}|{t.get('path_origin')}|"
                         f"{run['completion'].get('stop_reason','fixed_budget')}|")
    (OUT/"新锚点配对差.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Fresh completed branches:",len(fresh),"/ 104; missing pairs preserved.")

if __name__=="__main__":
    report(json.loads((OUT/"training_analysis.json").read_text()))
