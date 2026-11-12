"""Cross-check the post-hoc report, table, figure and original training support on CPU."""
import hashlib
import argparse
import json
from pathlib import Path
import random
import subprocess
from offline import ROOT,OUT,SOURCE

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main(require_local_artifacts=False):
    audit=json.loads((OUT/"source_oracle_audit.json").read_text());runs=audit["runs"]
    assert len(runs)==47 and len(audit["excluded_alphabetic"])==3
    points=0;native=0
    for r in runs.values():
        assert len(set(r["protocol"]["old"]+r["protocol"]["new"]))==96
        for p in r["points"].values():
            m=p["measurements"];a=m["readouts"]["all48"]
            assert 0<=m["raw_correct"]<=m["old_set_mass_mean"]+1e-8<=1+1e-5
            assert m["all96_candidate_old_top1"]<=a["oracle_top1"]["observed"]+1e-8
            assert m["original_key_count"]==16 and m["original_query_count"]==32
            assert a["valid_key_count"]==16
            for readout in m["readouts"].values():
                assert 0<=readout["oracle_top1"]["observed"]<=1
                assert m["raw_correct"]<=readout["conditional_correct"]["observed"]+1e-5
            points+=1
        if r["native_raw_minus1"] is not None:
            n=r["native_raw_minus1"];assert n["queries"]==32
            assert r["points"][n["point"]]["step"]==n["step"];native+=1
    figure=json.loads((OUT/"source_oracle_figure_data.json").read_text())
    text=(ROOT/"paper/共享LoRA中的条件绑定干扰_第二轮工作稿.md").read_text(encoding="utf8")
    support=[]
    for seed in (211,223,227):
        anchor=json.loads((SOURCE/"anchors"/f"Qwen3-0.6B_s{seed}_numeric/anchor.json").read_text())
        steps=anchor["trace"][-1]["step"];counts={key:0 for key in range(1,49)}
        # pilot.rows enumerates each training pair, then original/equivalent/counterfactual.
        pool=len(anchor["splits"]["train"])*3;rng=random.Random(seed)
        for _ in range(steps):
            for index in rng.sample(range(pool),4):
                key=anchor["splits"]["train"][index//3][0]+(index%3==2)
                counts[key]+=1
        assert min(counts.values())>0
        support.append(dict(seed=seed,anchor_steps=steps,distinct_old_values_seen=len(counts),
                            minimum_draws_per_old_value=min(counts.values())))
        for pairing,label in (("adjacent","相邻"),("support_permuted","支持集保持打散")):
            r=runs[f"first_revision/Qwen3-0.6B_s{seed}_{pairing}_lr0.0003_r4"]
            m=r["points"][r["stages"]["goal-1.0"]]["measurements"]
            expected=f"|{seed}|{label}|{100*m['old_set_mass_mean']:.2f}|{100*m['readouts']['all48']['oracle_top1']['observed']:.2f}|{100*m['maple_prefix_match']['observed']:.2f}|"
            assert expected in text,expected
            row=next(v for v in figure if v["seed"]==seed and v["pairing"]==pairing)
            assert row["old_set_mass"]==m["old_set_mass_mean"]
            assert row["oracle_top1"]==m["readouts"]["all48"]["oracle_top1"]["observed"]
    repo=ROOT if (ROOT/".git").exists() else ROOT/"publish/lora-binding-interference"
    baseline=subprocess.run(["git","show","6729184c5d936e0d7f0ac758652ac4a26d21c049:manifest.json"],
                            cwd=repo if repo.exists() else ROOT,capture_output=True,text=True)
    historical_verified=False
    if baseline.returncode==0:
        first=json.loads(baseline.stdout)
        old=next(v for v in first if v["path"]=="paper/共享LoRA中的旧事实干扰.pdf")
        assert digest(ROOT/old["path"])==old["sha256"]
        historical_verified=True
    pdf=ROOT/"paper/共享LoRA中的条件绑定干扰_第二轮工作稿.pdf"
    build=ROOT/"paper/build_revision2/manuscript_revision2.pdf"
    build_verified=False
    if build.exists():
        assert digest(pdf)==digest(build)
        build_verified=True
    if require_local_artifacts:
        assert historical_verified and build_verified
    result=dict(numeric_branch_records=47,excluded_alphabetic=3,observation_points=points,
        same_checkpoint_native_comparisons=native,primary_figure_observations=len(figure),
        figure_and_manuscript_values_match=True,primary_old_training_support=support,
        original_release_pdf_unchanged=True if historical_verified else None,
        working_pdf_matches_local_build=True if build_verified else None,
        working_pdf_sha256=digest(pdf),
        source_sha256=digest(Path(__file__).with_name("source_oracle.py")))
    (OUT/"source_oracle_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--require-local-artifacts",action="store_true")
    main(parser.parse_args().require_local_artifacts)
