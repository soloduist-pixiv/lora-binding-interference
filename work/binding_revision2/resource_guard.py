"""Read-only Windows memory checks; wait before allocating another model."""
import ctypes
from ctypes import wintypes
import datetime
import json
from pathlib import Path
import subprocess
import time

class MemoryStatus(ctypes.Structure):
    _fields_=[("length",wintypes.DWORD),("load",wintypes.DWORD),
        *[(name,ctypes.c_ulonglong) for name in ("total_physical","available_physical",
          "total_pagefile","available_pagefile","total_virtual","available_virtual","extended_virtual")]]

def host_memory():
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise OSError("Cannot read host memory")
    return dict(available_commit_mib=state.available_pagefile/2**20,
                available_physical_mib=state.available_physical/2**20)

def snapshot():
    result=subprocess.run(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],
                          capture_output=True,text=True,check=True)
    return dict(host_memory(),free_gpu_mib=min(float(v.strip()) for v in result.stdout.splitlines() if v.strip()))

def ready(value):
    return value["free_gpu_mib"]>=6144 and value["available_commit_mib"]>=10240 and value["available_physical_mib"]>=3072

def record(path,status,**fields):
    if path is None:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(checked_at=datetime.datetime.now().astimezone().isoformat(),status=status,**fields),indent=2))
    tmp.replace(path)

def wait_for_model(path=None,timeout=3600):
    begun=time.monotonic();announced=False
    while True:
        value=snapshot();ok=ready(value)
        record(path,"ready" if ok else "waiting_before_model_load",**value)
        if ok:return value
        if not announced:print("Waiting before model load:",value,flush=True);announced=True
        if time.monotonic()-begun>=timeout:raise RuntimeError("Model not loaded: resource headroom unavailable")
        time.sleep(20)

def wait_for_scoring(path=None,timeout=3600):
    begun=time.monotonic();announced=False
    while True:
        value=host_memory()
        if value["available_commit_mib"]>=1536 and value["available_physical_mib"]>=256:return
        record(path,"waiting_before_FP32_scoring",**value)
        if not announced:print("Waiting before FP32 scoring:",value,flush=True);announced=True
        if time.monotonic()-begun>=timeout:raise RuntimeError("Scoring not started: host memory headroom unavailable")
        time.sleep(5)
