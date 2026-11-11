"""One supervisor: numerical validation, unchanged serial jobs, then extensions."""
import json
import msvcrt
from pathlib import Path
import subprocess
import sys
from aux_data import OUT,ROOT
from resource_guard import wait_for_model,record
from serial_jobs import cases

def pending():
    for case in cases():
        folder=OUT/"runs"/case["name"]
        if (folder/"complete.json").exists():continue
        trace=folder/"trace.json"
        checkpoint=json.loads(trace.read_text())[-1]["step"] if trace.exists() else "Cedar anchor"
        return dict(next_case=case["name"],checkpoint=checkpoint)
    return dict(next_case="extensions",checkpoint="see extension status")

def main():
    task=pending();print("Pending:",task,flush=True)
    value=wait_for_model(OUT/"resource_recovery.json")
    record(OUT/"resource_recovery.json","resources_ready",**task,**value)
    subprocess.run([sys.executable,"-u",str(Path(__file__).with_name("verify_scoring.py"))],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-u",str(Path(__file__).with_name("serial_jobs.py"))],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-u",str(Path(__file__).with_name("post_queue.py"))],cwd=ROOT,check=True)

if __name__=="__main__":
    with (OUT/"resource_recovery.lock").open("a+b") as lock:
        if lock.tell()==0:lock.write(b"0");lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError("A recovery supervisor is already active")
        main()
