"""Check longer value sequences and the second model against full-prefix scoring."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"outputs/binding_review_20261111"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-support",action="store_true")
    args = parser.parse_args()
    if args.wait_for_support:
        deadline = time.monotonic()+6*3600
        while time.monotonic()<deadline:
            path = OUT/"support.json"
            if path.exists() and len(json.loads(path.read_text(encoding="utf8"))) == 3:
                break
            time.sleep(10)
        else:
            raise RuntimeError("Support-preserving controls have not finished")
    spec = importlib.util.spec_from_file_location("extended_review_runner",Path(__file__).with_name("run.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    from verify_scorer import reference_fp32
    import numpy as np
    records = []
    for model,pairing in (("Qwen3-0.6B","alphabetic"),("SmolLM2-360M-Instruct","adjacent")):
        name = f"{model}_s211_{pairing}_lr0.0003_r4"
        folder = OUT/"runs"/name
        protocol = json.loads((folder/"protocol.json").read_text(encoding="utf8"))
        end = json.loads((folder/"complete.json").read_text(encoding="utf8"))["last_step"]
        engine = runner.p.Engine(model) if model=="Qwen3-0.6B" else runner.StandardEngine(model)
        runner.c.KEYS = protocol["keys"]
        runner.c.OLD,runner.c.NEW = [0]+protocol["old"],[0]+protocol["new"]
        runner.p.problem = runner.c.problem
        old = runner.rows_for(protocol["splits"]["test"],"old",("original",),True)
        rows = [old[0],old[6]]
        rows += [dict(r,text=r["text"].replace("Cedar register","Maple register")) for r in rows]
        candidates = protocol["old"]+protocol["new"]
        scorer = runner.CandidateScorer(engine,candidates,batch_size=2)
        anchor = OUT/"anchors"/f"{model}_s211_{'alphabetic' if pairing=='alphabetic' else 'numeric'}"/"old.pt"
        for step,checkpoint in ((0,anchor),(end,folder/f"adapter_{end:04d}.pt")):
            state = runner.p.torch.load(checkpoint,map_location="cpu",weights_only=True)
            runner.restore(engine,state,4,211)
            fast = scorer(rows)
            reference = reference_fp32(engine,rows,candidates,scorer)
            expected = np.array([[r[v] for v in candidates] for r in reference])
            error = float(np.max(np.abs(np.exp(fast)-np.exp(expected))))
            record = dict(model=model,pairing=pairing,seed=211,step=step,
                          prompts=4,keys=2,register_labels=2,
                          context_ids=sorted({r["id"].split("/")[1] for r in rows}),
                          candidates=96,maximum_probability_error=error,passed=error<1e-4)
            records.append(record)
            runner.p.write(OUT/"extended_scorer_verification.json",records)
            print(record,flush=True)
            if not record["passed"]:
                raise RuntimeError("Extended FP32 scorer verification failed")
        del scorer,engine
        gc.collect()
        runner.p.torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
