"""Analyze registered runs at fixed steps and validation-selected progress."""
import json
from pathlib import Path

import numpy as np

from offline import OUT, distribution_metrics, entropy


def metrics(old, maple, keys, active):
    assert np.isfinite(old).all() and np.isfinite(maple).all()
    assert min(old.min(),maple.min()) >= 0
    assert max(old.sum(1).max(),maple.sum(1).max()) <= 1+1e-5
    active_rows = np.array([key in active for key in keys])
    active_gold = np.array(keys)[active_rows]-1
    active_probabilities = maple[active_rows,48+active_gold]
    active_test_logp = float(np.log(np.maximum(active_probabilities,np.finfo(float).tiny)).mean())
    row_p = np.c_[old, np.maximum(0,1-old.sum(1))]
    row_q = np.c_[maple, np.maximum(0,1-maple.sum(1))]
    row_js = entropy((row_p+row_q)/2)-(entropy(row_p)+entropy(row_q))/2
    row_upper = row_js+(row_p[:,-1]+row_q[:,-1])/2*np.log(2)
    groups = sorted(set(keys))
    old = np.array([old[np.array(keys) == key].mean(0) for key in groups])
    maple = np.array([maple[np.array(keys) == key].mean(0) for key in groups])
    gold = np.array(groups) - 1
    p = np.c_[old, np.maximum(0, 1-old.sum(1))]
    q = np.c_[maple, np.maximum(0, 1-maple.sum(1))]
    mixture = (p+q)/2
    js = entropy(mixture) - (entropy(p)+entropy(q))/2
    assert np.all(js >= -1e-6) and np.all(js <= np.log(2)+1e-6)
    per_key_js = [float(row_js[np.array(keys)==key].mean()) for key in groups]
    buckets = np.c_[p[:, :48].sum(1), p[:, 48:96].sum(1), p[:, -1]]
    info = entropy(p.mean(0)) - entropy(p).mean()
    assert -1e-6 <= info <= np.log(len(groups))+1e-6
    parts = []
    for group, total in zip((p[:, :48], p[:, 48:96]), buckets.T[:2]):
        part = entropy(group.mean(0)) - entropy(group).mean()
        part += entropy(total[:, None]).mean() - entropy(np.array([total.mean()]))
        parts.append(float(part))
    between = entropy(buckets.mean(0)) - entropy(buckets).mean()
    assert np.isclose(info, between+sum(parts), atol=1e-6)
    trained = np.array([key in active for key in groups])
    target_columns = np.array(sorted(active))-1
    return dict(information=float(info), cedar_information=parts[0], maple_information=parts[1],
                between=float(between), old_correct=float(old[np.arange(len(gold)), gold].mean()),
                old_prompt_paired_new=float(old[np.arange(len(gold)), 48+gold].mean()),
                maple_prompt_correct=float(maple[np.arange(len(gold)), 48+gold].mean()),
                maple_prompt_cedar=float(maple[np.arange(len(gold)), gold].mean()),
                active_maple_test_logp=active_test_logp,
                active_maple_test_probability=float(active_probabilities.mean()),
                old_prompt_maple_total=float(buckets[:, 1].mean()),
                trained_paired=float(old[trained][:, 48:][np.arange(trained.sum()), gold[trained]].mean()),
                trained_null=float(old[trained][:, 48+target_columns].mean()),
                js=float(row_js.mean()), js_per_key=per_key_js, pooled_js=float(js.mean()),
                js_upper=float(min(np.log(2), row_upper.mean())),
                other=float(p[:, -1].mean()), other_max=float(p[:, -1].max()),
                maple_other=float(q[:, -1].mean()))


def analyze_run(folder):
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf8"))
    trace = json.loads((folder / "trace.json").read_text(encoding="utf8"))
    result = []
    for row in trace:
        with np.load(folder / f"s{row['step']:04d}.npz") as data:
            indices = data["indices"]
            keys = [int(protocol["old_eval_ids"][i].split("/")[0]) for i in indices]
            dense = indices % 6 < 2
            key_subset = [key for key, keep in zip(keys, dense) if keep]
            measured = metrics(data["old"][dense], data["maple"][dense], key_subset,
                               protocol["active_original"])
            full = metrics(data["old"], data["maple"], keys, protocol["active_original"]) if len(indices) == 144 else None
        result.append(dict(step=row["step"], maple=row["maple"], metrics=measured, full=full))
    progress = []
    for target in (-1.0, -.5):
        initial = result[0]["maple"]
        for fraction in (.25, .5, .75, 1.):
            threshold = initial + fraction*(target-initial)
            hits = [i for i, row in enumerate(result) if row["maple"] >= threshold]
            if initial >= target or not hits:
                progress.append(dict(target=target, fraction=fraction, comparable=False))
                continue
            j = hits[0]
            if j == 0:
                raise RuntimeError("Nonpositive progress target")
            a, b = result[j-1], result[j]
            f = (threshold-a["maple"])/(b["maple"]-a["maple"])
            values = {name: a["metrics"][name]+f*(b["metrics"][name]-a["metrics"][name])
                      for name in a["metrics"] if name != "js_per_key"}
            progress.append(dict(target=target, fraction=fraction, comparable=True,
                                 step=a["step"]+f*(b["step"]-a["step"]),
                                 interpolated=values, observed=b["metrics"],observed_step=b["step"]))
    fixed = {row["step"]: row["full"] for row in result if row["full"] is not None}
    fixed_change = {key: fixed[64][key]-fixed[16][key] for key in fixed[16]
                    if key != "js_per_key"} if 16 in fixed and 64 in fixed else None
    return dict(protocol=protocol, trace=result, progress=progress, fixed_change=fixed_change,
                complete=(folder / "complete.json").exists())


def main():
    runs = {}
    for folder in sorted((OUT / "runs").glob("*")):
        if (folder / "trace.json").exists():
            runs[folder.name] = analyze_run(folder)
    target = OUT / "analysis.json"
    target.write_text(json.dumps(runs, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    for name, run in runs.items():
        last = run["trace"][-1]
        print(name, "complete", run["complete"], "step", last["step"],
              "Maple", round(last["maple"], 3), "JS", round(last["metrics"]["js"], 4))


if __name__ == "__main__":
    main()
