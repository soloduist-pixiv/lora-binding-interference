"""Verify the allocation-only scorer against archived FP32 probabilities; no updates."""
import hashlib
import importlib.util
import json
from pathlib import Path
from aux_data import OUT
from resource_guard import wait_for_model

def main():
    target=OUT/"scoring_memory_validation.json"
    policy=hashlib.sha256(Path(__file__).with_name("stable_score.py").read_bytes()).hexdigest()
    if target.exists():
        previous=json.loads(target.read_text())
        if previous.get("passed") and previous.get("policy_sha256")==policy:
            print("Allocation policy already validated against archived scores.",flush=True)
            return
    wait_for_model(OUT/"scoring_validation_resource_wait.json")
    spec=importlib.util.spec_from_file_location("revision2_scoring_validation",Path(__file__).with_name("run.py"))
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    import numpy as np
    source=OUT/"runs/Qwen3-0.6B_s401_support_permuted_lr0.0003_r4_min640"
    protocol=json.loads((source/"protocol.json").read_text())
    r.c.KEYS=tuple(protocol["keys"]);r.c.OLD=[0]+protocol["old"];r.c.NEW=[0]+protocol["new"]
    r.p.problem=r.c.problem
    engine=r.p.Engine("Qwen3-0.6B")
    anchor=OUT/"anchors/Qwen3-0.6B_s401_numeric/old.pt"
    state=r.p.torch.load(anchor,map_location="cpu",weights_only=True)
    r.restore(engine,state,4,401)
    scorer=r.CandidateScorer(engine,protocol["old"]+protocol["new"],batch_size=4,
                             diagnostic_directory=OUT/"scoring_validation")
    old=r.rows_for(protocol["splits"]["test"],"old",("original",),True)
    assert [v["id"] for v in old]==protocol["old_eval_ids"]
    maple=[dict(v,text=v["text"].replace("Cedar register","Maple register")) for v in old]
    active=set(protocol["active_original"])
    validation=[v for v in r.rows_for(protocol["splits"]["validation"],"related",("original",),True)
                if int(v["id"].split("/")[0]) in active]
    cedar=r.rows_for(protocol["splits"]["validation"],"old",("original","counterfactual"),True)
    with np.load(source/"s0000.npz") as z:
        indices=z["indices"].copy();old_reference=z["old"].copy();maple_reference=z["maple"].copy()
        gold_reference=z["validation_gold_logp"].copy()
    scorer.phase="archived_checkpoint_zero_equivalence"
    with scorer.evaluation_precision():
        old_new=np.exp(scorer([old[i] for i in indices]))
        maple_new=np.exp(scorer([maple[i] for i in indices]))
        gold_new=np.array(scorer.gold(validation))
        cedar_mean=float(np.mean(scorer.gold(cedar)))
    trace=json.loads((source/"trace.json").read_text())
    errors=dict(old_probability=float(np.abs(old_new-old_reference).max()),
                maple_probability=float(np.abs(maple_new-maple_reference).max()),
                validation_logp=float(np.abs(gold_new-gold_reference).max()),
                cedar_validation_mean_logp=abs(cedar_mean-trace[0]["cedar_validation_logp"]))
    passed=max(errors["old_probability"],errors["maple_probability"])<=1e-6 and max(errors["validation_logp"],errors["cedar_validation_mean_logp"])<=1e-4
    result=dict(passed=passed,policy_sha256=policy,training_updates=0,
                reference=source.name,reference_checkpoint=0,reference_precision="float32",
                compared_candidate_rows_each_label=len(indices),compared_gold_validation_rows=len(validation),
                absolute_error=errors,probability_tolerance=1e-6,logp_tolerance=1e-4,
                peak_reserved_mib=r.p.torch.cuda.max_memory_reserved()/2**20,
                anchor_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest())
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)
    if not passed:raise RuntimeError("Scoring equivalence did not meet fixed tolerances; training remains stopped")

if __name__=="__main__":main()
