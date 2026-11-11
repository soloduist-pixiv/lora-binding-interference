"""Historical-data corrections requested in the manuscript review."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/model_compare_open_20261108/prospective_updates"
OUT = ROOT / "outputs/binding_review_20261111"
SEEDS = (29, 43, 107, 108, 109)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def entropy(p):
    return -(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(-1)


def distribution_metrics(mass):
    assert np.all(mass >= 0) and np.all(mass.sum(1) <= 1 + 1e-5)
    p = np.c_[mass, np.maximum(0, 1 - mass.sum(1))]
    return dict(information=float(entropy(p.mean(0)) - entropy(p).mean()),
                outside=float(p[:, -1].mean()), outside_max=float(p[:, -1].max()))


def audit_mass(raw):
    records = []
    for step, block in sorted(raw["result"].items(), key=lambda row: int(row[0])):
        keys = sorted(block["keys"], key=int)
        original = np.array(keys, dtype=int)
        mass = np.array([block["keys"][k]["cedar"] + block["keys"][k]["maple"] for k in keys])
        metrics = distribution_metrics(mass)
        new = mass[:, 48:]
        paired = new[np.arange(24), original - 1]
        trained = np.array([key in block["seen"] for key in original])
        trained_values = original[trained] - 1
        seen_values = np.array(block["seen"], dtype=int) - 1
        metrics.update(
            step=int(step), old_correct=float(mass[np.arange(24), original - 1].mean()),
            paired=float(paired.mean()),
            trained_paired=float(paired[trained].mean()) if trained.any() else None,
            trained_original_null=float(new[trained][:, trained_values].mean()) if trained.any() else None,
            trained_all_seen_null=float(new[trained][:, seen_values].mean()) if trained.any() else None,
        )
        records.append(metrics)
    keys = sorted(raw["result"]["64"]["keys"], key=int)
    new = np.array([raw["result"]["64"]["keys"][k]["maple"] for k in keys])
    originals = np.array(keys, dtype=int)
    seen = set(raw["result"]["64"]["seen"])
    routes = []
    for i, key in enumerate(originals):
        for j in range(48):
            if j + 1 == key:
                group = "paired"
            elif j + 1 not in seen:
                group = "untrained_target"
            elif abs(j + 1 - key) <= 2:
                group = "near_key"
            elif (j // 26) == ((key - 1) // 26):
                group = "same_initial_other"
            else:
                group = "different_initial"
            routes.append(dict(key=int(key), target_key=j+1, group=group,
                               probability=float(new[i, j]), distance=int(abs(j+1-key))))
    group_mass = {g: sum(r["probability"] for r in routes if r["group"] == g) / 24
                  for g in sorted({r["group"] for r in routes})}
    assert np.isclose(sum(group_mass.values()), new.sum(1).mean())
    eligible = [r for r in routes if r["group"] not in ("paired", "untrained_target")]
    route_null = {}
    for group in ("near_key", "same_initial_other", "different_initial"):
        expected = []
        for key in originals:
            selected = [r for r in eligible if r["key"] == key]
            mass = sum(r["probability"] for r in selected)
            expected.append(mass * sum(r["group"] == group for r in selected) / len(selected))
        route_null[group] = float(np.mean(expected))
    distance_profile = []
    for lo, hi in ((1, 2), (3, 8), (9, 16), (17, 32), (33, 47)):
        selected = [r["probability"] for r in eligible if lo <= r["distance"] <= hi]
        distance_profile.append(dict(lo=lo, hi=hi, n=len(selected),
                                     mean=float(np.mean(selected)) if selected else None))
    return dict(trace=records, routes=routes, route_mass=group_mass, route_null=route_null,
                distance_profile=distance_profile,
                matrix=new.tolist(), keys=list(map(int, keys)))


def crossing(rows, target):
    progress = np.array([r["maple"] for r in rows])
    if progress[0] >= target:
        return None
    hits = np.flatnonzero(progress >= target)
    if not len(hits):
        return None
    j = hits[0]
    f = (target - progress[j-1]) / (progress[j] - progress[j-1])
    probability = np.exp([r["test"] for r in rows]).mean(1)
    value = probability[j-1] + f * (probability[j] - probability[j-1])
    return dict(step=float(j-1+f), harm=float(probability[0] - value))


def progress_audit():
    seeds = range(101, 107)
    records = {s: read(SOURCE / f"s{s}/stage1_trajectories.json") for s in seeds}
    output = {"absolute_train_required": {}, "gap_closure": {}}
    for target in (-6, -4.5, -3):
        pairs = []
        for seed, blocks in records.items():
            for j, block in enumerate(blocks):
                if any(block[a][0]["maple"] >= target for a in ("true", "false")):
                    continue
                arms = {a: crossing(block[a], target) for a in ("true", "false")}
                if all(arms.values()):
                    pairs.append(dict(seed=seed, block=j, **arms))
        output["absolute_train_required"][str(target)] = summarize_pairs(pairs)
    for target in (-1.0, -.5):
        for fraction in (.25, .5, .75):
            pairs = []
            excluded = 0
            for seed, blocks in records.items():
                for j, block in enumerate(blocks):
                    if any(block[a][0]["maple"] >= target for a in ("true", "false")):
                        excluded += 1
                        continue
                    arms = {a: crossing(block[a], block[a][0]["maple"] +
                                        fraction * (target - block[a][0]["maple"]))
                            for a in ("true", "false")}
                    if all(arms.values()):
                        pairs.append(dict(seed=seed, block=j, **arms))
            summary = summarize_pairs(pairs)
            summary["baseline_excluded"] = excluded
            output["gap_closure"][f"{target}/{fraction}"] = summary
    return output


def summarize_pairs(pairs):
    per_seed = {}
    for seed in range(101, 107):
        selected = [p for p in pairs if p["seed"] == seed]
        per_seed[seed] = dict(
            n=len(selected),
            difference=float(np.mean([p["false"]["harm"] - p["true"]["harm"] for p in selected]))
            if selected else None,
        )
    comparable = [s["difference"] for s in per_seed.values() if s["n"]]
    return dict(n=len(pairs), per_seed=per_seed, pairs=pairs,
                seed_mean=float(np.mean(comparable)) if comparable else None)


def figures(audits, control):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    for name,seeds in (("historical_all_seeds",tuple(audits)),
                       ("historical_development",("29","43")),
                       ("historical_holdout",("107","108","109"))):
        fig, axes = plt.subplots(3,len(seeds),figsize=(13 if len(seeds)==5 else 7,5.8),
                                 sharex=True,sharey="row",layout="constrained")
        for j,label in enumerate(seeds):
            rows = audits[label]["trace"]
            x = [r["step"] for r in rows]
            axes[0,j].set_title(f"Seed {label}")
            for field,line,color in (("old_correct","Cedar correct","#0F4D92"),
                                     ("paired","Maple paired","#B64342")):
                axes[0,j].plot(x,[r[field] for r in rows],"o-",ms=3,label=line,color=color)
            axes[1,j].plot(x,[r["information"] for r in rows],"o-",ms=3,color="#272727")
            axes[2,j].plot(x,[r["outside"] for r in rows],"o-",ms=3,color="#767676")
            axes[2,j].set_xlabel("Maple steps")
            for ax in axes[:,j]:
                ax.set_xticks([0,16,32,64])
                ax.grid(axis="y",alpha=.2)
        axes[0,0].set_ylabel("Canonical mass")
        axes[1,0].set_ylabel("I(key; answer), nats")
        axes[2,0].set_ylabel("Other mass")
        axes[0,0].legend(frameon=False,fontsize=8)
        for extension in ("png","pdf"):
            fig.savefig(OUT/f"{name}.{extension}",dpi=300)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), layout="constrained", sharey=True)
    for ax, name, audit in zip(axes, ("107", "107_prefix_control"), (audits["107"], control)):
        matrix = np.array(audit["matrix"])
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Maple target key index (1–48)")
        ax.set_ylabel("Cedar query key (odd indices)")
    fig.colorbar(im, ax=axes, label="Canonical mass")
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"prefix_routes.{extension}", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audits = {str(seed): audit_mass(read(SOURCE / f"disjoint_mass/s{seed}/mass.json"))
              for seed in SEEDS}
    control = audit_mass(read(SOURCE / "disjoint_mass/s107/mass_prefix_control.json"))
    result = dict(historical=audits, prefix_control=control, alignment=progress_audit())
    (OUT / "offline.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    figures(audits, control)
    for key, audit in audits.items():
        print(key, audit["trace"][-1])
    print("control routes", control["route_mass"])
    for key, value in result["alignment"]["gap_closure"].items():
        print("gap", key, value["n"], value["seed_mean"])


if __name__ == "__main__":
    main()
