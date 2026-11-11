"""Rebuild completed new-run endpoint and high-target probability audits."""
import json
from pathlib import Path
import numpy as np
from offline import OUT,load,measures,pava

def main():
    results={}
    for folder in sorted((OUT/"runs").glob("*")):
        if not (folder/"complete.json").exists(): continue
        protocol=json.loads((folder/"protocol.json").read_text())
        trace=json.loads((folder/"trace.json").read_text())
        curve=pava([r["maple"] for r in trace]); rows=[]
        for goal in (-1.,-.5,-.3,-.1):
            hits=np.flatnonzero(curve>=goal)
            if not len(hits):
                rows.append(dict(target=goal,covered=False));continue
            j=int(hits[0]); lo=max(0,j-1)
            w=0 if j==0 else float((goal-curve[lo])/(curve[j]-curve[lo]))
            p,q,keys=load(folder,protocol,trace[lo]["step"])
            p1,q1,_=load(folder,protocol,trace[j]["step"])
            rows.append(dict(target=goal,covered=True,observed_step=trace[j]["step"],
                step=(1-w)*trace[lo]["step"]+w*trace[j]["step"],
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
            targets=rows,terminals=terminals,greedy_at_first_minus1=greedy)
    (OUT/"training_analysis.json").write_text(json.dumps(results,indent=2,allow_nan=False))
    print("New completed branches audited:",len(results))

if __name__=="__main__":main()
