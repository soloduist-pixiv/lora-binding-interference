"""Compare the cached scorer to the established full-prefix scorer."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/prospective_updates"))
from experiment import c, d, p
from disjoint_mass import cuda_scorer
from cached_score import CandidateScorer
import numpy as np


@p.torch.no_grad()
def reference_fp32(engine, rows, candidates, scorer):
    frozen = [(v, v.dtype) for v in engine.model.model.parameters()
              if v.device.type == "cuda" and not v.requires_grad]
    try:
        for v, _ in frozen:
            v.data = v.data.float()
        def logits(engine, sequences, positions):
            model = engine.model.model
            length = max(map(len, sequences))
            ids = p.torch.tensor([s+[0]*(length-len(s)) for s in sequences])
            embeddings = model.embed_tokens(ids).to("cuda", dtype=p.torch.float32)
            hidden = model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
            batch, index = p.torch.tensor(positions, device="cuda").unbind(1)
            return p.torch.nn.functional.linear(hidden[batch,index], scorer.weight)
        return d.masses(engine, rows, candidates, logits=logits)
    finally:
        for v, dtype in frozen:
            v.data = v.data.to(dtype=dtype)


def main():
    folder = d.MAIN / "prospective_updates/disjoint_mass/s107"
    protocol = d.read(folder / "old_gate.json")
    raw = d.read(folder / "mass.json")
    engine = p.Engine("Qwen3-0.6B")
    c.KEYS, c.OLD, c.NEW = protocol["keys"], protocol["cedar_values"], [0] + raw["maple"]
    checkpoint = p.torch.load(folder / "old.pt", map_location="cpu", weights_only=True)
    d.configure(engine, checkpoint, 4, 107)
    p.problem = c.problem
    all_rows = list(p.rows("old", protocol["splits"]["test"], "plain", ("original",)))
    rows = all_rows[:4]
    candidates = raw["cedar"] + raw["maple"]
    scorer = CandidateScorer(engine, candidates)
    opt = p.optimizer(engine.ps, .0003)
    result = []
    for step in (0, 1, 16, 64):
        if step == 1:
            p.train_step(engine, opt, list(p.rows("related", protocol["splits"]["train"],
                                               "plain", ("original",)))[:4])
        if step in (16,64):
            path = ROOT / "outputs/binding_review_20261111/runs/Qwen3-0.6B_s107_adjacent_lr0.0003_r4"
            mature = p.torch.load(path/f"adapter_{step:04d}.pt", map_location="cpu", weights_only=True)
            d.configure(engine, mature, 4, 107)
        start = time.perf_counter()
        fast = scorer(rows)
        seconds = time.perf_counter() - start
        reference = reference_fp32(engine, rows, candidates, scorer)
        expected = np.array([[row[v] for v in candidates] for row in reference])
        error = np.max(np.abs(np.exp(expected) - np.exp(fast)))
        if error > 1e-4:
            raise RuntimeError(f"Cache probability error {error:.4g}")
        result.append(dict(step=step, method=scorer.method, precision=scorer.precision, seconds=seconds,
                           comparison="FP32 tree versus FP32 full prefix",
                           maximum_probability_error=float(error)))
        print(result[-1], flush=True)
    d.configure(engine, checkpoint, 4, 107)
    start = time.perf_counter()
    scored = np.exp(scorer(all_rows)).reshape(24, 6, 96).mean(1)
    expected = np.array([raw["result"]["0"]["keys"][str(k)]["cedar"] +
                         raw["result"]["0"]["keys"][str(k)]["maple"] for k in range(1,49,2)])
    error = np.max(np.abs(scored-expected))
    mean_error = np.mean(np.abs(scored-expected))
    gold = np.arange(0,48,2)
    correct_difference = abs(np.mean(scored[np.arange(24), gold] - expected[np.arange(24), gold]))
    result.append(dict(method=scorer.method, all_144_prompts=True,
                       comparison="FP32 tree versus historical BF16 scoring",
                       seconds=time.perf_counter()-start, maximum_probability_error=float(error),
                       mean_absolute_error=float(mean_error), correct_mass_difference=float(correct_difference)))
    print(result[-1], flush=True)
    assert error <= .02 and mean_error <= .001 and correct_difference <= .002
    target = ROOT / "outputs/binding_review_20261111/scorer_verification.json"
    target.parent.mkdir(exist_ok=True, parents=True)
    p.write(target, result)


if __name__ == "__main__":
    main()
