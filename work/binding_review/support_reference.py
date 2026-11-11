"""Support-preserving shuffle control, serialized after the capacity extension."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/binding_review_20261111"


def main():
    deadline = time.monotonic()+8*3600
    while time.monotonic()<deadline:
        path = OUT/"capacity.json"
        if path.exists() and len(json.loads(path.read_text(encoding="utf8"))) == 3:
            break
        time.sleep(10)
    else:
        raise RuntimeError("Capacity reference has not finished")
    records = []
    for seed in (211,223,227):
        started = time.perf_counter()
        with (OUT/f"support_{seed}.log").open("w",encoding="utf8") as stream:
            result = subprocess.run(
                [sys.executable,"-u",str(Path(__file__).with_name("run.py")),
                 "--seed",str(seed),"--pairing","support_permuted"],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=5400)
        records.append(dict(seed=seed,exit_code=result.returncode,
                            seconds=time.perf_counter()-started))
        (OUT/"support.json").write_text(json.dumps(records,indent=2),encoding="utf8")
        print("SUPPORT",records[-1],flush=True)


if __name__ == "__main__":
    main()
