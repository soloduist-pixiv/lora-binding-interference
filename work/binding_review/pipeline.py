"""Run the registered experiments one subprocess at a time."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/round4"))
from core import write

OUT = ROOT / "outputs/binding_review_20261111"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    output = ROOT / args.output_root if args.output_root else OUT
    output.mkdir(exist_ok=True, parents=True)
    if args.wait_for:
        deadline = time.monotonic() + 5400
        while not (ROOT / args.wait_for).exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Initial pilot did not complete within 90 minutes")
            time.sleep(10)
    cases = []
    for seed in (107, 108, 109):
        for lr in (.0001, .0003, .001):
            cases.append(["--seed", str(seed), "--lr", str(lr), "--historical"])
        cases.append(["--seed", str(seed), "--lr", ".0003", "--rank", "16", "--historical"])
        cases.append(["--seed", str(seed), "--lr", ".0003", "--replay", "--historical"])
    for seed in (211, 223, 227):
        for pairing in ("adjacent", "permuted", "alphabetic"):
            cases.append(["--seed", str(seed), "--pairing", pairing])
    for seed in (211, 223, 227):
        cases.append(["--seed", str(seed), "--model", "SmolLM2-360M-Instruct"])
    records = []
    for index, case in enumerate(cases):
        label = " ".join(case)
        print(f"START {index+1}/{len(cases)} {label}", flush=True)
        log_path = output / f"job_{index+1:02d}.log"
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf8") as stream:
                result = subprocess.run(
                    [sys.executable, "-u", str(Path(__file__).with_name("run.py")), *case,
                     *(["--output-root", args.output_root] if args.output_root else [])],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=5400,
                )
            record = dict(case=case, exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            record = dict(case=case, exit_code=None, error="90-minute execution timeout")
        record.update(seconds=time.perf_counter()-started, log=log_path.name)
        records.append(record)
        write(output / "pipeline.json", dict(cases=cases, completed=records))
        print("END", index+1, record["exit_code"], flush=True)
    print("REGISTERED PIPELINE FINISHED", flush=True)


if __name__ == "__main__":
    main()
