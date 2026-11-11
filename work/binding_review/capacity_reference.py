"""Registered extension: rank-16 replay with a fixed 640-step budget."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/binding_review_20261111"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-main",action="store_true")
    args = parser.parse_args()
    if args.wait_for_main:
        deadline = time.monotonic()+6*3600
        while time.monotonic()<deadline:
            path = OUT/"pipeline.json"
            if path.exists():
                status = json.loads(path.read_text(encoding="utf8"))
                if len(status["completed"]) == len(status["cases"]):
                    break
            time.sleep(10)
        else:
            raise RuntimeError("Main experiment queue has not finished")
    records = []
    for seed in (107,108,109):
        path = OUT / f"capacity_{seed}.log"
        started = time.perf_counter()
        try:
            with path.open("w",encoding="utf8") as stream:
                result = subprocess.run(
                    [sys.executable,"-u",str(Path(__file__).with_name("run.py")),
                     "--seed",str(seed),"--historical","--rank","16","--replay",
                     "--minimum-steps","640"],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=5400)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = None
        records.append(dict(seed=seed,exit_code=code,
                            seconds=time.perf_counter()-started))
        (OUT/"capacity.json").write_text(json.dumps(records,indent=2),encoding="utf8")
        print("CAPACITY",records[-1],flush=True)


if __name__ == "__main__":
    main()
