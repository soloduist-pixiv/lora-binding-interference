"""CPU lower bounds from stored adapter chords; never call this exact step path."""
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"outputs/binding_review_20261111/runs"
OUT=ROOT/"outputs/binding_revision2_20261111"
torch.set_num_threads(2)

def distance(old,new):
    squared=0.
    for i in range(0,len(old),2):
        a,b=(x.numpy().astype(np.float64) for x in new[i:i+2])
        oa,ob=(x.numpy().astype(np.float64) for x in old[i:i+2])
        left=np.concatenate([b-ob,ob],1); right=np.concatenate([a,a-oa],0)
        squared+=4*float(np.sum((left.T@left)*(right@right.T).T))
    return np.sqrt(max(0.,squared))

def main():
    records={}; lines=["# 初始学习距离、达标步数与检查点路径下界","",
       "以下路径是保存检查点之间有效 LoRA 矩阵的弦长和，单位为固定 scale=2 的 Frobenius 范数。它是全程逐更新路径的下界；不同检查点覆盖与间距会影响数值，不能当作已匹配的真实更新距离。新实验另记录精确逐更新路径。","",
       "|分支|初始验证 logP|首次实测 ≥−1 步|首次实测 ≥−0.5 步|末步|记录路径下界|","|---|---|---|---|---|---|"]
    for folder in sorted(SOURCE.iterdir()):
        if not (folder/"complete.json").exists():continue
        trace=json.loads((folder/"trace.json").read_text()); previous=None; length=0.; path=[]
        for row in trace:
            file=folder/f"adapter_{row['step']:04d}.pt"
            if not file.exists():raise RuntimeError(f"Missing local checkpoint {file}")
            current=torch.load(file,map_location="cpu",weights_only=True)["parameters"]
            if previous is not None:length+=float(distance(previous,current))
            path.append(dict(step=row["step"],polygon_length=length))
            previous=current
        hits={str(g):next((r["step"] for r in trace if r["maple"]>=g),None) for g in (-1.,-.5,-.3,-.1)}
        records[folder.name]=dict(initial_validation_logp=trace[0]["maple"],first_observed_crossings=hits,checkpoint_chord_path=path)
        lines.append(f"|{folder.name}|{trace[0]['maple']:.3f}|{hits['-1.0']}|{hits['-0.5']}|{trace[-1]['step']}|{length:.3f}|")
    (OUT/"checkpoint_path_lower_bounds.json").write_text(json.dumps(records,indent=2))
    (OUT/"初始难度与路径下界.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Audited saved-checkpoint path lower bounds:",len(records))

if __name__=="__main__":main()
