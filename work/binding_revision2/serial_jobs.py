"""Serialized fixed-budget continuations, then fresh-anchor confirmation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/binding_revision2_20261111"
OLD=ROOT/"outputs/binding_review_20261111"
SEEDS=(401,409,419,431,443,449,457,463)

def cases():
    result=[]
    for seed in (107,108,109):
        for rank,replay,exposure in ((4,False,"plain"),(16,False,"plain"),
                                    (4,True,"plain"),(4,False,"duplicate_maple")):
            args=["--seed",str(seed),"--historical","--rank",str(rank),
                  "--minimum-steps","640","--maximum-steps","640","--exposure",exposure]
            if replay: args+=["--replay"]
            name=f"Qwen3-0.6B_s{seed}_adjacent_lr0.0003_r{rank}"+("_replay" if replay else "")+"_min640"
            if exposure=="duplicate_maple": name+="_duplicate_maple"
            source=None if exposure=="duplicate_maple" else name.replace("_min640","")
            result.append(dict(phase="fixed640",name=name,args=args,source=source))
    for seed in SEEDS:
        for lr in (.0001,.0003,.001):
            for rank in (4,16):
                for replay in (False,True):
                    args=["--seed",str(seed),"--pairing","support_permuted","--lr",str(lr),"--rank",str(rank)]
                    if replay: args+=["--replay"]
                    name=f"Qwen3-0.6B_s{seed}_support_permuted_lr{lr:g}_r{rank}"+("_replay" if replay else "")+"_min640"
                    result.append(dict(phase="fresh",name=name,args=args,source=None))
        args=["--seed",str(seed),"--pairing","support_permuted","--exposure","duplicate_maple"]
        result.append(dict(phase="fresh_exposure",name=f"Qwen3-0.6B_s{seed}_support_permuted_lr0.0003_r4_min640_duplicate_maple",args=args,source=None))
    return result

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    def write(path,value):
        temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
    plan=cases(); write(OUT/"queue_plan.json",plan)
    status=OUT/"queue_status.json"; history_file=OUT/"queue_attempts.json"
    records=json.loads(status.read_text()) if status.exists() else []
    history=json.loads(history_file.read_text()) if history_file.exists() else list(records)
    for i,case in enumerate(plan):
        folder=OUT/"runs"/case["name"]
        known=next((row for row in records if row["index"]==i+1),None)
        if known and known["name"]!=case["name"]:raise RuntimeError("Existing queue plan mismatch")
        if known and known["exit_code"]==0 and (folder/"complete.json").exists():
            continue
        records=[row for row in records if row["index"]<i+1]
        write(status,records)
        if case["source"] and not folder.exists():
            # Copy only owned experiment artifacts; preserve all originals.
            shutil.copytree(OLD/"runs"/case["source"],folder)
            (folder/"complete.json").unlink()
        started=time.monotonic()
        attempt=1+sum(row["index"]==i+1 for row in history)
        log=OUT/f"run_{i+1:03d}.log" if attempt==1 else OUT/f"run_{i+1:03d}_attempt{attempt:02d}.log"
        print("START",i+1,len(plan),case["name"],flush=True)
        try:
            with log.open("w",encoding="utf8") as f:
                p=subprocess.run([sys.executable,"-u",str(Path(__file__).with_name("run.py")),*case["args"]],
                                 cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
            code=p.returncode
        except subprocess.TimeoutExpired: code=None
        records.append(dict(index=i+1,name=case["name"],exit_code=code,seconds=time.monotonic()-started,attempt=attempt))
        history.append(records[-1]);write(history_file,history)
        write(status,records)
        print("END",records[-1],flush=True)
        if code!=0:
            raise RuntimeError("Queue stopped on failure; retain the seed and inspect its log")
        subprocess.run([sys.executable,str(Path(__file__).with_name("analyze_training.py"))],cwd=ROOT,check=True)

if __name__=="__main__": main()
