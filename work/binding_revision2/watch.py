"""Update CPU audits only when a new serial job completes; no CUDA imports."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/binding_revision2_20261111"
last=-1
while True:
    p=OUT/"queue_status.json"
    records=json.loads(p.read_text()) if p.exists() else []
    if len(records)!=last:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name("analyze_training.py"))],cwd=ROOT)
        if result.returncode:raise RuntimeError("CPU audit failed")
        last=len(records)
    if records and records[-1]["exit_code"]!=0:break
    plan=OUT/"queue_plan.json"
    if plan.exists() and len(records)==len(json.loads(plan.read_text())):break
    time.sleep(10)
