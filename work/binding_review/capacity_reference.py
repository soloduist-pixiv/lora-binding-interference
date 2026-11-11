"""Registered extension: rank-16 replay with a fixed 640-step budget."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/binding_review_20261111"


def main():
    records = []
    for seed in (107,108,109):
        path = OUT / f"capacity_{seed}.log"
        started = time.perf_counter()
        with path.open("w",encoding="utf8") as stream:
            result = subprocess.run(
                [sys.executable,"-u",str(Path(__file__).with_name("run.py")),
                 "--seed",str(seed),"--historical","--rank","16","--replay",
                 "--minimum-steps","640"],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=5400)
        records.append(dict(seed=seed,exit_code=result.returncode,
                            seconds=time.perf_counter()-started))
        (OUT/"capacity.json").write_text(json.dumps(records,indent=2),encoding="utf8")
        print("CAPACITY",records[-1],flush=True)


if __name__ == "__main__":
    main()
