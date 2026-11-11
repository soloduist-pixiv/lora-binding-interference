"""Rebuild completed new-run endpoint and high-target probability audits."""
import json
import os
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from offline import OUT,load,measures,pava

def main():
    results={}
    for folder in sorted((OUT/"runs").glob("*")):
        if not (folder/"complete.json").exists(): continue
        protocol=json.loads((folder/"protocol.json").read_text())
        trace=json.loads((folder/"trace.json").read_text())
        raw=np.array([r["maple"] for r in trace]); curve=pava(raw); rows=[]; raw_crossings=[]
        for goal in (-1.,-.5,-.3,-.1):
            raw_hits=np.flatnonzero(raw>=goal)
            raw_crossings.append(dict(target=goal,observed_step=trace[int(raw_hits[0])]["step"]
                                      if len(raw_hits) else None))
            hits=np.flatnonzero(curve>=goal)
            if not len(hits):
                rows.append(dict(target=goal,covered=False));continue
            j=int(hits[0]); lo=max(0,j-1)
            w=0 if j==0 else float((goal-curve[lo])/(curve[j]-curve[lo]))
            p,q,keys=load(folder,protocol,trace[lo]["step"])
            p1,q1,_=load(folder,protocol,trace[j]["step"])
            rows.append(dict(target=goal,covered=True,observed_step=trace[j]["step"],
                step=(1-w)*trace[lo]["step"]+w*trace[j]["step"],
                effective_path_since_origin=(1-w)*trace[lo]["effective_path"]+w*trace[j]["effective_path"]
                    if all("effective_path" in trace[k] for k in (lo,j)) else None,
                path_origin=trace[j].get("path_origin"),
                measurements=measures((1-w)*p+w*p1,(1-w)*q+w*q1,keys,protocol["active_original"])))
        terminals={}
        for step in sorted({640,trace[-1]["step"]}):
            entry=next((r for r in trace if r["step"]==step),None)
            if entry is None:continue
            p,q,keys=load(folder,protocol,step)
            terminals[str(step)]=dict(validation_logp=entry["maple"],
                effective_path_since_origin=entry.get("effective_path"),
                path_origin=entry.get("path_origin"),
                measurements=measures(p,q,keys,protocol["active_original"]))
        decode=folder/"first_minus1_generation.json"
        greedy=None
        if decode.exists():
            g=json.loads(decode.read_text())
            greedy=dict(step=g["step"],validation_logp=g["validation_logp"],
                        accuracy={k:float(np.mean([r["correct"] for r in v])) for k,v in g["queries"].items()})
        results[folder.name]=dict(protocol=protocol,initial_validation_logp=trace[0]["maple"],
            completion=json.loads((folder/"complete.json").read_text()),
            targets=rows,raw_first_crossings=raw_crossings,terminals=terminals,greedy_at_first_minus1=greedy)
    temporary=OUT/"training_analysis.tmp"
    temporary.write_text(json.dumps(results,indent=2,allow_nan=False))
    temporary.replace(OUT/"training_analysis.json")
    from paired_report import report
    report(results)
    from budget_report import main as budgets
    budgets()
    print("New completed branches audited:",len(results))

if __name__=="__main__":main()
