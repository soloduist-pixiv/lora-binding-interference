"""Apply the frozen stage 0 gate to complete stage 1 trajectories."""
import json
import numpy as np
from stage0_power import GRID, OUT, ROOT, TARGET, decisions, fold_auc

SEEDS=range(101,107)
arms=("false","true")


def harm(rows):
    lp=np.array([r["test"] for r in rows]).reshape(-1,8,5)
    log=np.maximum(0,(lp[0]-lp).mean(2)).mean(1)
    prob=np.maximum(0,(np.exp(lp[0])-np.exp(lp)).mean(2)).mean(1)
    return np.stack((log,prob),axis=1)


def remaining(rows,k):
    b=np.array([r["maple"] for r in rows])
    h=harm(rows)
    hits=np.flatnonzero(b>=TARGET)
    if not len(hits) or hits[0]<=k:
        return None
    j=hits[0]
    f=(TARGET-b[j-1])/(b[j]-b[j-1])
    reference=h[j-1]+f*(h[j]-h[j-1])
    if np.any(reference<=0):
        return None
    return (np.maximum(0,reference-h[k])/reference).tolist()


def main():
    records=[json.loads((OUT/f"s{s}/stage1_trajectories.json").read_text(encoding="utf8")) for s in SEEDS]
    assert all(len(r)==5 for r in records)
    features=np.array([[[[[np.mean(block[a][i]["probe"])-np.mean(block[a][0]["probe"]),block[a][i]["dot"]] for i in GRID] for a in arms] for block in r] for r in records])
    auc,lower,good=decisions(fold_auc(features[None]))
    k=next((int(GRID[i]) for i in range(2,len(GRID)) if all(good[0,i-2:i+1])),None)
    fractions={a:{s:[remaining(block[a],k) if k is not None else None for block in r] for s,r in zip(SEEDS,records)} for a in arms}
    medians=[np.median([v for v in rows if v is not None],axis=0) for rows in fractions["false"].values() if any(v is not None for v in rows)]
    pooled=np.median(medians,axis=0).tolist() if len(medians)==len(SEEDS) else None
    gate="不通过" if k is None or k>8 else "不可比较" if pooled is None else "通过" if min(pooled)>=.4 else "不通过"
    probability=np.array([[[[np.exp(block[a][i]["test"]).mean() for i in GRID] for a in arms] for block in r] for r in records]).mean((0,1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,sharex=True,layout="constrained")
    axes[0].plot(GRID,auc[0],"o-",label="Joint AUC")
    axes[0].fill_between(GRID,np.maximum(0,lower[0]),np.minimum(1,2*auc[0]-lower[0]),alpha=.2)
    axes[0].axhline(.8,ls=":",color="gray")
    axes[0].set(ylabel="AUC",ylim=(0,1.02))
    for a,curve in zip(arms,probability):
        axes[1].plot(GRID,curve,"o-",label=a)
    axes[1].set(xlabel="Maple updates",ylabel="Old answer probability")
    for ax in axes:
        ax.legend()
    fig.savefig(OUT/"stage1_auc.png",dpi=180)
    result=dict(gate=gate,k=k,auc=auc[0].tolist(),lower=lower[0].tolist(),fractions=fractions,median=pooled,probability=probability.tolist())
    (OUT/"stage1_gate.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    heading="## 附录｜阶段 1 留出轨迹门槛"
    lines=[heading,"",f"格式匹配子门槛为 **{gate}**。决策步为 {k}，假别名可避免伤害的种子中位数（logP / 概率）为 {pooled}。","",
           "| 步 | AUC | 同时带下界 | 旧值概率（假别名 / 真别名） |","|---:|---:|---:|---:|"]
    for j,i in enumerate(GRID):
        lines.append(f"| {i} | {auc[0,j]:.3f} | {max(0,lower[0,j]):.3f} | {probability[0,j]:.2%} / {probability[1,j]:.2%} |")
    lines+=["","![阶段 1 AUC 和旧值概率](prospective_updates/stage1_auc.png)","",
            "该表只覆盖阶段 1 的格式匹配子实验。互斥值集和一致更新对照另行验收。"]
    report=ROOT/"六模型开放生成与持续学习对比报告.md"
    report.write_text(report.read_text(encoding="utf-8-sig").split(heading)[0].rstrip()+"\n\n"+"\n".join(lines)+"\n",encoding="utf8")
    print(gate,k,pooled)


if __name__=="__main__":
    main()
