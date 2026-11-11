"""Resume the unchanged saved job only after resources are actually released."""
import ctypes
from ctypes import wintypes
import datetime
import json
import msvcrt
from pathlib import Path
import subprocess
import sys
import time
from aux_data import OUT,ROOT

class MemoryStatus(ctypes.Structure):
    _fields_=[("length",wintypes.DWORD),("load",wintypes.DWORD),
        *[(name,ctypes.c_ulonglong) for name in ("total_physical","available_physical",
          "total_pagefile","available_pagefile","total_virtual","available_virtual","extended_virtual")]]

def resources():
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise OSError("Cannot read memory status")
    result=subprocess.run(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],
                           capture_output=True,text=True,check=True)
    free_gpu=min(float(line.strip()) for line in result.stdout.splitlines() if line.strip())
    return dict(free_gpu_mib=free_gpu,available_commit_mib=state.available_pagefile/2**20,
                available_physical_mib=state.available_physical/2**20)

def write(value):
    temporary=OUT/"resource_recovery.tmp";temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(OUT/"resource_recovery.json")

def main():
    begun=time.monotonic()
    while True:
        status=resources();ready=status["free_gpu_mib"]>=6144 and status["available_commit_mib"]>=8192
        write(dict(checked_at=datetime.datetime.now().astimezone().isoformat(),
                   status="ready" if ready else "waiting_for_resource_release",**status,
                   checkpoint=752,seed=401,lr=.0003,rank=4,replay=False,
                   note="No model loaded while waiting; no automatic reboot or driver reset"))
        if ready:break
        if time.monotonic()-begun>3600:
            write(dict(status="resource_release_required",**status,checkpoint=752,
                       note="One-hour resource wait ended; retain all data. Resume after resources or driver state change."))
            print("Resources not released; checkpoint preserved. No GPU retry launched.",flush=True)
            return
        time.sleep(20)
    print("Resources released; resume original seed and saved optimizer.",flush=True)
    subprocess.run([sys.executable,"-u",str(Path(__file__).with_name("serial_jobs.py"))],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-u",str(Path(__file__).with_name("post_queue.py"))],cwd=ROOT,check=True)

if __name__=="__main__":
    with (OUT/"resource_recovery.lock").open("a+b") as lock:
        if lock.tell()==0:lock.write(b"0");lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError("A recovery supervisor is already active")
        main()
