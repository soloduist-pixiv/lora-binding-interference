"""Re-evaluate saved pilot weights with the verified tree scorer."""
import shutil
import time
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("review_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
OUT, CandidateScorer = runner.OUT, runner.CandidateScorer
c, d, p = runner.c, runner.d, runner.p
rows_for, restore = runner.rows_for, runner.restore


def main(folder, engine):
    protocol = d.read(folder / "protocol.json")
    trace = d.read(folder / "trace.json")
    reference = folder / "cache_reference"
    reference.mkdir(exist_ok=True)
    for path in [folder / "trace.json", *folder.glob("s*.npz")]:
        target = reference / path.name
        if not target.exists():
            shutil.copy2(path, target)
    c.KEYS, c.OLD, c.NEW = protocol["keys"], [0]+protocol["old"], [0]+protocol["new"]
    p.problem = c.problem
    old_rows = rows_for(protocol["splits"]["test"], "old", ("original",), False)
    maple_rows = [dict(row, text=row["text"].replace("Cedar register", "Maple register"))
                  for row in old_rows]
    validation = [r for r in rows_for(protocol["splits"]["validation"], "related", ("original",), False)
                  if int(r["id"].split("/")[0]) in protocol["active_original"]]
    scorer = CandidateScorer(engine, protocol["old"]+protocol["new"])
    differences = []
    started = time.perf_counter()
    for row in trace:
        step = row["step"]
        checkpoint = p.torch.load(folder / f"adapter_{step:04d}.pt", map_location="cpu", weights_only=True)
        restore(engine, checkpoint, protocol["rank"], protocol["seed"])
        with np.load(reference / f"s{step:04d}.npz") as prior:
            indices = prior["indices"]
            previous = prior["old"].copy()
        old = np.exp(scorer([old_rows[i] for i in indices]))
        maple = np.exp(scorer([maple_rows[i] for i in indices]))
        val = scorer(validation)
        gold = [48+protocol["new"].index(r["answer"]) for r in validation]
        old_gold = [int(old_rows[i]["id"].split("/")[0])-1 for i in indices]
        differences.append(dict(step=step, maximum=float(np.abs(old-previous).max()),
                                mean_absolute=float(np.abs(old-previous).mean()),
                                correct_mass_difference=float(abs(
                                    (old[np.arange(len(indices)), old_gold] -
                                     previous[np.arange(len(indices)), old_gold]).mean()))))
        np.savez_compressed(folder / f"s{step:04d}.npz", old=old, maple=maple,
                            validation_logp=val, indices=indices)
        row["maple"] = float(val[np.arange(len(validation)), gold].mean())
        row["old_correct"] = float(old[np.arange(len(indices)), old_gold].mean())
        row["old_other"] = float((1-old.sum(1)).mean())
        print("RESCORE", step, differences[-1], flush=True)
    protocol.update(scoring_method=scorer.method, scoring_batch_size=scorer.batch_size,
                    scoring_precision=scorer.precision)
    p.write(folder / "protocol.json", protocol)
    p.write(folder / "trace.json", trace)
    p.write(folder / "tree_comparison.json", dict(comparisons=differences,
                                                 seconds=time.perf_counter()-started))


if __name__ == "__main__":
    engine = p.Engine("Qwen3-0.6B")
    for folder in sorted((OUT / "runs").glob("Qwen3-0.6B_s107_adjacent_lr*_r4")):
        if (folder / "trace.json").exists():
            main(folder, engine)
