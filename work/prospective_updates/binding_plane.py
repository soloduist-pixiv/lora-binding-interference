"""Offline binding plane on exact canonical completion masses."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]/"outputs/model_compare_open_20261108"
OUT=ROOT/"prospective_updates"

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def entropy(p,axis=-1):
    return -(p*np.log(np.maximum(p,np.finfo(float).tiny))).sum(axis=axis)

def rows_disjoint(raw):
    for step,block in raw["result"].items():
        keys=sorted(block["keys"],key=int)
        yield int(step),np.array([block["keys"][k]["cedar"]+block["keys"][k]["maple"] for k in keys]),[int(k)-1 for k in keys]

def rows_overlap(raw):
    for step,block in raw.items():
        groups={}
        for row in block:
            groups.setdefault(row["id"].split("/")[0],[]).append(row)
        keys=sorted(groups,key=int)
        assert len(keys)==24 and all(len(groups[k])==6 for k in keys)
        yield int(step),np.array([np.exp([[r["logp"][str(v)] for v in range(10,100)] for r in groups[k]]).mean(0) for k in keys]),[groups[k][0]["gold"]-10 for k in keys]

def main():
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    result={}
    for j,seed in enumerate((29,43)):
        result[seed]={}
        for label,records in (("disjoint",rows_disjoint(read(OUT/"disjoint_mass"/f"s{seed}"/"mass.json"))),
                              ("overlap",rows_overlap(read(OUT/f"s{seed}"/"mass.json")))):
            rows=[]
            for step,m,gold in sorted(records,key=lambda x:x[0]):
                assert m.shape[0]==24 and np.all(m.sum(1)<=1+1e-6)
                p=np.c_[m,np.maximum(0,1-m.sum(1))]
                p/=p.sum(1,keepdims=True)
                conditional=entropy(p).mean()
                marginal=entropy(p.mean(0))
                info=marginal-conditional
                assert -1e-9<=info<=np.log(len(p))+1e-9
                rows.append(dict(step=step,D=float(p[np.arange(len(p)),gold].mean()),I=float(info),
                    conditional=float(conditional),marginal=float(marginal),outside=float(p[:,-1].mean())))
            for r in rows:
                r["BIR"]=r["I"]/rows[0]["I"] if rows[0]["I"]>0 else None
            result[seed][label]=rows
            axes[0,j].plot([r["D"] for r in rows],[r["BIR"] for r in rows],"o-",label=label)
            for r in rows:
                axes[0,j].annotate(str(r["step"]),(r["D"],r["BIR"]),fontsize=7)
            if label=="disjoint":
                axes[1,j].stackplot([r["step"] for r in rows],[r["conditional"] for r in rows],[r["I"] for r in rows],
                                    labels=("H(V|K)","I(K;V)"),alpha=.8)
        axes[0,j].set(title=f"Seed {seed}",xlabel="Correct mass D",ylabel="Binding information retention")
        axes[1,j].set(xlabel="Maple updates",ylabel="Marginal entropy (nats)")
        for ax in axes[:,j]:
            ax.legend()
    fig.savefig(OUT/"binding_plane.png",dpi=180)
    (OUT/"binding_plane.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    for seed,data in result.items():
        for label,rows in data.items():
            print(seed,label,[(r["step"],round(r["D"],3),round(r["BIR"],3),round(r["conditional"],3)) for r in rows])


if __name__=="__main__":
    main()
