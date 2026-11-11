"""Exploratory coordinate-permutation null, keeping the probability matrix fixed."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/binding_revision2_20261111"

def main():
    rows=json.loads((ROOT/"outputs/binding_review_20261111/matched_routes.json").read_text())
    audit=json.loads((OUT/"matched_audit.json").read_text())["runs"]
    results={}
    for name,row in rows.items():
        matrix=np.array(row["matrix"]); p=audit[name]["protocol"]; active=set(p["active_original"])
        left=[]; right=[]; mass=[]
        for i,k in enumerate(range(1,49,2)):
            for j in range(1,49):
                if j in active or j-1 in active:
                    if j not in (k,k+1):
                        left.append(k-1); right.append(j-1); mass.append(matrix[i,j-1])
        left,right,mass=map(np.array,(left,right,mass))
        def statistic(position):
            distance=np.abs(position[left]-position[right])
            return float(np.dot(distance,mass)/mass.sum()-distance.mean())
        observed=statistic(np.arange(48)); rng=np.random.default_rng(20261112+p["seed"])
        null=np.array([statistic(rng.permutation(48)) for _ in range(2000)])
        results[name]=dict(statistic="mass-weighted minus opportunity-weighted ordinal distance",
            observed=observed,null_mean=float(null.mean()),null_025=float(np.quantile(null,.025)),
            null_975=float(np.quantile(null,.975)),one_sided_closer_p=float((1+(null<=observed).sum())/2001),
            permutations=2000,null_values=null.tolist())
    (OUT/"key_coordinate_null.json").write_text(json.dumps(results,indent=2))
    print("Coordinate permutation nulls:",len(results))

if __name__=="__main__": main()
