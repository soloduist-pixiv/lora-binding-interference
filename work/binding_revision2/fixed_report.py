"""Matched-budget report with the blind-mixture feasibility region."""
import json
import os
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from offline import ROOT,OUT,load,measures

def main():
    audits=json.loads((OUT/"training_analysis.json").read_text())
    lines=["# 固定 640 步的对照（旧锚点，探索性）","",
       "概率指标统一使用 16 个活跃原题键、每键两个相同测试上下文。D、M 为算术平均概率，Σ=D+M−1；不是贪心准确率。旧锚点已多次分析，不计入八个新锚点的确认样本。","",
       "|种子|rank|回放|重复 Maple|D %|M %|Σ|有符号 Cedar 路由|JS nats|",
       "|---|---|---|---|---|---|---|---|---|"]
    first=["","## 首次实测达到 −1.0 时的贪心准确率","",
           "以下均为同一 16 个活跃原题键、六个测试上下文，Cedar 与 Maple 各 96 个查询。列出实际验证 logP，不冒充插值权重的解码结果。","",
           "|分支|步数|实际验证 logP|Cedar %|Maple %|","|---|---|---|---|---|"]
    for name,r in audits.items():
        p=r["protocol"]
        if not p["historical_anchor"]:continue
        m=r["terminals"]["640"]["measurements"]["active"]
        lines.append(f"|{p['seed']}|{p['rank']}|{p['replay']}|{p['exposure']=='duplicate_maple'}|{100*m['D']:.3f}|{100*m['M']:.3f}|{m['sigma']:.4f}|{m['signed_cedar']:.4f}|{m['js']:.4f}|")
        g=r["greedy_at_first_minus1"]
        if g:
            first.append(f"|{name}|{g['step']}|{g['validation_logp']:.3f}|{100*g['accuracy']['cedar']:.2f}|{100*g['accuracy']['maple']:.2f}|")
    lines+=first
    (OUT/"固定预算对照.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    fig,axes=plt.subplots(1,3,figsize=(10,3.9),layout="constrained",sharex=True,sharey=True)
    styles=[(4,False,"plain","rank 4","#155993"),(16,False,"plain","rank 16","#9A5151"),
            (4,True,"plain","rank 4 replay","#2B8D83"),(4,False,"duplicate_maple","rank 4 duplicate","#85679B")]
    for ax,seed in zip(axes,(107,108,109)):
        xx=np.linspace(0,1,100); ax.fill_between(xx,0,1-xx,color="#eeeeee")
        ax.plot(xx,1-xx,"--",color="gray",lw=1)
        for rank,replay,exposure,label,color in styles:
            group=[(n,r) for n,r in audits.items() if r["protocol"]["historical_anchor"] and
                   r["protocol"]["seed"]==seed and r["protocol"]["rank"]==rank and
                   r["protocol"]["replay"]==replay and r["protocol"]["exposure"]==exposure]
            if not group:continue
            name,r=group[0]; folder=OUT/"runs"/name; trace=json.loads((folder/"trace.json").read_text())
            values=[]
            for row in trace:
                p,q,keys=load(folder,r["protocol"],row["step"])
                values.append(measures(p,q,keys,r["protocol"]["active_original"])["active"])
            ax.plot([v["M"] for v in values],[v["D"] for v in values],color=color,lw=1.5,label=label)
            ax.scatter(values[-1]["M"],values[-1]["D"],color=color,s=18)
        ax.set(title=f"Seed {seed}",xlabel="Mean correct Maple probability",xlim=(0,1.02),ylim=(0,1.02),aspect="equal")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Mean correct Cedar probability")
    axes[0].legend(frameon=False,fontsize=7,loc="lower left")
    fig.suptitle("Matched test rows; shaded region permits a label-blind mixture",fontsize=11)
    for ext in ("png","pdf"):fig.savefig(OUT/f"blind_region_trajectories.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)
    print("Fixed-budget branches reported:",sum(r["protocol"]["historical_anchor"] for r in audits.values()))

if __name__=="__main__":main()
