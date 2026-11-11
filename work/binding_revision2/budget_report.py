"""Unnormalized disjoint candidate budgets at every requested target and endpoint."""
import json
from offline import OUT,NAMES

def main():
    audits=json.loads((OUT/"training_analysis.json").read_text())
    lines=["# 第二轮完整质量预算","",
           "每行为 Cedar 标签下同一批 16 个活跃原题键、每键两个测试上下文的算术平均概率，单位 %。互斥分桶覆盖全部输出质量，未对候选集重新归一化。",
           "同题族 Maple 是原题的另一个换键变体；其余活跃原题及换键值分别列出，非活跃值另列。高目标使用 PAVA 对齐后的概率矩阵，终点为实际保存权重。未达目标不填零。","",
           "|分支|阶段|"+"|".join(NAMES)+"|合计|",
           "|---|---|"+"---|"*9]
    rows=0
    for name,r in audits.items():
        points=[(str(t["target"]),t.get("measurements")) for t in r["targets"]]
        points += [("step"+step,t["measurements"]) for step,t in r["terminals"].items()]
        for stage,measure in points:
            if measure is None:
                lines.append("|"+name+"|"+stage+"|未达目标|"+"—|"*8)
                continue
            b=measure["active"]["budget"]
            total=sum(b.values()); assert abs(total-1)<1e-5
            lines.append("|"+name+"|"+stage+"|"+"|".join(f"{100*b[k]:.3f}" for k in NAMES)+f"|{total*100:.3f}|")
            rows+=1
    (OUT/"完整质量预算.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Complete probability budgets verified:",rows)

if __name__=="__main__":main()
