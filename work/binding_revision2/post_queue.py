"""Wait for the existing GPU queue, then run the registered extensions serially."""
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from aux_data import OUT,ROOT,SEEDS

def plan():
    cases=[]
    for scope in ("all_qv","all_linear"):
        cases.append(dict(name="technical_"+scope,script="run.py",args=["--seed","401",
            "--pairing","support_permuted","--scope",scope,"--minimum-steps","2","--maximum-steps","2"],technical=True))
    for seed in (401,409,419):
        for scope in ("all_qv","all_linear"):
            for replay in (False,True):
                args=["--seed",str(seed),"--pairing","support_permuted","--scope",scope]
                if replay:args += ["--replay"]
                cases.append(dict(name=f"scope_s{seed}_{scope}_replay{replay}",script="run.py",args=args,technical=False))
    for kind,models in (("closed",("Qwen3-0.6B","SmolLM2-360M-Instruct")),("temporal",("Qwen3-0.6B",))):
        for model in models:
            for seed in SEEDS:
                for replay in (False,True):
                    args=["--kind",kind,"--model",model,"--seed",str(seed)]
                    if replay:args += ["--replay"]
                    cases.append(dict(name=f"{kind}_{model}_s{seed}_replay{replay}",script="auxiliary.py",args=args,technical=False))
    return cases

def write(path,value):
    temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def main():
    jobs=plan();write(OUT/"extension_plan.json",jobs)
    print("Waiting for all 116 primary/fixed-budget jobs; no GPU model loaded.",flush=True)
    while True:
        try:
            records=json.loads((OUT/"queue_status.json").read_text())
            expected=len(json.loads((OUT/"queue_plan.json").read_text()))
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
        if records and records[-1]["exit_code"]!=0:raise RuntimeError("Primary queue failed; extensions not launched")
        if len(records)==expected:break
        time.sleep(20)
    records=[]
    for i,job in enumerate(jobs,1):
        print("EXTENSION START",i,len(jobs),job["name"],flush=True)
        begun=time.monotonic()
        with (OUT/f"extension_{i:03d}.log").open("w",encoding="utf8") as log:
            try:
                result=subprocess.run([sys.executable,"-u",str(Path(__file__).with_name(job["script"])),*job["args"]],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
                code=result.returncode
            except subprocess.TimeoutExpired:code=None
        records.append(dict(index=i,name=job["name"],exit_code=code,seconds=time.monotonic()-begun))
        write(OUT/"extension_status.json",records)
        if code!=0:raise RuntimeError("Extension failed; retain seed and inspect log, no fallback substitution")
        if job["technical"]:
            scope=job["args"][job["args"].index("--scope")+1]
            folder=OUT/"runs"/f"Qwen3-0.6B_s401_support_permuted_lr0.0003_r4_min2_{scope}"
            protocol=json.loads((folder/"protocol.json").read_text())
            config=json.loads((ROOT/"Models/Qwen3-0.6B/config.json").read_text())
            expected=config["num_hidden_layers"]*(2 if scope=="all_qv" else 7)
            assert len(set(protocol["target_modules"]))==expected
            assert json.loads((folder/"complete.json").read_text())["last_step"]==2
            with zipfile.ZipFile(folder/"state.pt") as saved:assert saved.testzip() is None
            audit_file=OUT/"scope_gpu_smoke.json"
            audit=json.loads(audit_file.read_text()) if audit_file.exists() else {}
            audit[scope]=dict(passed=True,steps=2,target_modules=expected,checkpoint_zip_crc=True)
            write(audit_file,audit)
        subprocess.run([sys.executable,str(Path(__file__).with_name("analyze_training.py"))],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name("aux_analysis.py"))],cwd=ROOT,check=True)
        print("EXTENSION END",records[-1],flush=True)

if __name__=="__main__":main()
