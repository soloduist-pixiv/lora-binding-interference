"""Stage 0 planning simulation. Pilot-calibrated, conditional on heterogeneity scenarios."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]/"outputs/model_compare_open_20261108"
OUT=ROOT/"prospective_updates"
GRID=np.array([2,4,6,8,12,16])
TARGET=-3.0
RNG=np.random.default_rng(61027)


def fold_auc(x):
    means=x.mean(axis=2)
    sums=x.sum(axis=(1,2),keepdims=True)
    n=x.shape[1]
    train=(sums-x.sum(axis=2,keepdims=True))/((n-1)*x.shape[2])
    pooled=(x*x).sum(axis=(1,2),keepdims=True)-(x*x).sum(axis=2,keepdims=True)
    var=(pooled/((n-1)*x.shape[2])-train**2).mean(axis=3)
    w=(train[:,:,:,1]-train[:,:,:,0]).squeeze(2)/(var.squeeze(2)+1e-8)
    score=np.einsum("rsbatf,rstf->rsbat",x,w)
    pair=score[:,:, :,1,None,:]-score[:,:,None,:,0,:]
    return ((pair>0)+.5*(pair==0)).mean(axis=(2,3))


def decisions(folds):
    n=folds.shape[1]
    avg=folds.mean(axis=1)
    lower=avg-t.ppf(1-.05/(2*len(GRID)),n-1)*folds.std(axis=1,ddof=1)/np.sqrt(n)
    good=(avg>=.8)&(lower>.5)
    return avg,lower,good


def main():
    raw=[]
    for seed in (29,43):
        rows=json.loads((OUT/f"s{seed}"/"matched_trajectories.json").read_text(encoding="utf8"))
        raw.append([[[[np.mean(r[a][i]["probe"])-np.mean(r[a][0]["probe"]),r[a][i]["dot"]] for i in GRID] for a in ("false","true")] for r in rows])
    pilot=np.asarray(raw)
    center=pilot.mean((0,1))
    seed=pilot.mean(1)-center
    block=(pilot-pilot.mean(1,keepdims=True)).reshape(-1,2,len(GRID),2)
    scale=block.std(0)
    estimates=[]
    for n in (6,8,10,12,16,20,24,32):
        for b in (5,8,12):
            for multiplier in (.5,1.,2.):
                shared=RNG.normal(size=(1500,n,1,1,1,1))*seed[0]
                jitter=RNG.normal(size=(1500,n,1,2,1,2))*scale
                residual=block[RNG.integers(len(block),size=(1500,n,b))]
                x=center+multiplier*(shared+.25*jitter)+residual
                _,_,good=decisions(fold_auc(x))
                power=np.mean((good[:,0]&good[:,1]&good[:,2])|(good[:,1]&good[:,2]&good[:,3]))
                estimates.append(dict(seeds=n,blocks=b,heterogeneity=multiplier,power_by_8=float(power)))
    (OUT/"stage0_power.json").write_text(json.dumps(estimates,indent=2),encoding="utf8")
    for m in (.5,1.,2.):
        eligible=[r for r in estimates if r["heterogeneity"]==m and r["power_by_8"]>=.8]
        print(m,min(eligible,key=lambda r:r["seeds"]*r["blocks"]) if eligible else "none")


if __name__=="__main__":
    main()
