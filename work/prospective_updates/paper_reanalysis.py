"""CPU-only reanalysis of the saved Cedar–Maple trajectories."""
import json
from pathlib import Path

import numpy as np

from binding_plane import entropy, read, rows_disjoint
from stage0_power import GRID, OUT, TARGET, decisions, fold_auc
from stage1_report import remaining


def mass_audit(seed, prefix_control=False):
    filename = "mass_prefix_control.json" if prefix_control else "mass.json"
    raw = read(OUT / "disjoint_mass" / f"s{seed}" / filename)
    assert len(set(raw["cedar"] + raw["maple"])) == 96
    maple_prefix = np.array(raw["maple"]) // 10
    common_prefix = np.array(raw["cedar"]) // 10 == maple_prefix
    assert np.all(common_prefix == (not prefix_control))
    result = []
    for step, mass, gold in sorted(rows_disjoint(raw), key=lambda row: row[0]):
        assert mass.shape == (24, 96)
        assert np.all(mass >= 0) and np.all(mass.sum(1) <= 1 + 1e-6)
        distribution = np.c_[mass, np.maximum(0, 1 - mass.sum(1))]
        distribution /= distribution.sum(1, keepdims=True)
        old, new = distribution[:, :48], distribution[:, 48:96]
        buckets = np.c_[old.sum(1), new.sum(1), distribution[:, -1]]
        information = entropy(distribution.mean(0)) - entropy(distribution).mean()
        between = entropy(buckets.mean(0)) - entropy(buckets).mean()
        within = [
            float(entropy(values.mean(0)) - entropy(values).mean()
                  + np.mean(entropy(bucket[:, None])) - entropy(np.array([bucket.mean()])))
            for values, bucket in zip((old, new), buckets.T[:2])
        ]
        assert np.isclose(information, between + sum(within))
        assert min(between, *within) >= -1e-10
        paired = new[np.arange(len(gold)), gold]
        permuted = new[:, gold].mean()
        assert np.all(buckets[:, 1] > 0)
        same_prefix = maple_prefix[gold, None] == maple_prefix[None, :]
        prefix_mass = (new * same_prefix).sum(1)
        assert np.all(prefix_mass > 0)
        result.append(dict(
            step=step, old_correct=old[np.arange(len(gold)), gold].mean(),
            new_total=new.sum(1).mean(), new_paired=paired.mean(),
            new_paired_share_uniform_keys=np.mean(paired / buckets[:, 1]),
            new_paired_random_permutation_mean=permuted,
            new_paired_same_prefix_share=np.mean(paired / prefix_mass),
            new_same_prefix_uniform_share=np.mean(1 / same_prefix.sum(1)),
            information=information, between_sets=between,
            within_cedar=within[0], within_maple=within[1],
        ))
    expected = [0, 8, 16, 64] if prefix_control else [0, 1, 2, 4, 8, 16, 32, 64]
    assert [row["step"] for row in result] == expected
    return result


def exposure_audit(seed):
    raw = read(OUT / "disjoint_mass" / f"s{seed}" / "mass.json")["result"]
    start, end = raw["0"], raw["64"]
    seen = set(end["seen"])
    by_group = {}
    for group, trained in (("exposed", True), ("unexposed", False)):
        keys = [key for key in end["keys"] if (int(key) in seen) == trained]
        by_group[group] = dict(
            keys=len(keys),
            start_old_correct=np.mean([start["keys"][key]["categories"][0] for key in keys]),
            end_categories=np.mean([end["keys"][key]["categories"] for key in keys],
                                   axis=0).tolist(),
        )
    return by_group


def first_crossing(rows, target, relative, interpolate):
    progress = np.array([row["maple"] for row in rows])
    if relative:
        progress -= progress[0]
    hits = np.flatnonzero(progress >= target)
    if not len(hits):
        return None
    j = hits[0]
    probability = np.exp([row["test"] for row in rows]).mean(1)
    if j == 0:
        return [0.0, 0.0]
    if not interpolate:
        return [float(j), float(probability[0] - probability[j])]
    fraction = (target - progress[j - 1]) / (progress[j] - progress[j - 1])
    interpolated = (1 - fraction) * probability[j - 1] + fraction * probability[j]
    return [float(j - 1 + fraction), float(probability[0] - interpolated)]


def paired_audit(records, target, relative=False, interpolate=True):
    by_seed = {}
    for seed, blocks in records.items():
        paired = []
        for i, block in enumerate(blocks):
            arms = {arm: first_crossing(block[arm], target, relative, interpolate)
                    for arm in ("true", "false")}
            paired.append(dict(block=i, **arms))
        valid = [item for item in paired if item["true"] is not None and item["false"] is not None]
        effects = np.array([np.subtract(row["false"], row["true"]) for row in valid])
        by_seed[seed] = dict(n=len(valid), blocks=paired,
                             mean_delta=effects.mean(0).tolist() if len(effects) else None)
    return by_seed


def replication_audit(rows):
    reference = min((row for row in rows if row["step"] in (8, 16)),
                    key=lambda row: row["information"])
    final = rows[-1]
    rebound = final["information"] - reference["information"]
    old_change = final["old_correct"] - reference["old_correct"]
    maple_change = final["within_maple"] - reference["within_maple"]
    paired_share = final["new_paired"] / final["new_total"]
    return dict(reference=reference["step"], rebound=rebound, old_change=old_change,
                maple_change=maple_change, paired_share=paired_share,
                primary_pass=bool(rebound >= .25 and old_change <= .02),
                support_pass=bool(rebound > 0 and maple_change >= .5 * rebound and paired_share > .5))


def validate_prefix_control(original, control):
    assert original["cedar"] == control["cedar"]
    assert set(original["maple"]) == set(control["maple"])
    columns = [control["maple"].index(value) for value in original["maple"]]
    original_anchor = original["result"]["0"]["keys"]
    control_anchor = control["result"]["0"]["keys"]
    assert original_anchor.keys() == control_anchor.keys()
    differences = [
        np.abs(np.r_[original_anchor[key]["cedar"], original_anchor[key]["maple"]]
               - np.r_[control_anchor[key]["cedar"], np.array(control_anchor[key]["maple"])[columns]])
        for key in original_anchor
    ]
    for step, block in control["result"].items():
        assert original["result"][step]["seen"] == block["seen"]
    error = np.max(differences)
    assert error <= 3e-4
    return float(error)


def reproduce_gate(records):
    arms = ("false", "true")
    features = np.array([
        [[[[np.mean(block[a][i]["probe"]) - np.mean(block[a][0]["probe"]),
            block[a][i]["dot"]] for i in GRID] for a in arms] for block in blocks]
        for blocks in records.values()
    ])
    auc, lower, good = decisions(fold_auc(features[None]))
    k = next(int(GRID[i]) for i in range(2, len(GRID)) if all(good[0, i-2:i+1]))
    fractions = [[remaining(block["false"], k) for block in blocks]
                 for blocks in records.values()]
    median = np.median([np.median([v for v in blocks if v is not None], axis=0)
                        for blocks in fractions], axis=0)
    probability = np.array([
        [[[np.exp(block[arm][i]["test"]).mean() for i in GRID] for arm in arms]
         for block in blocks] for blocks in records.values()
    ]).mean((0, 1))
    saved = read(OUT / "stage1_gate.json")
    assert k == saved["k"]
    for actual, field in ((auc[0], "auc"), (lower[0], "lower"), (median, "median"),
                          (probability, "probability")):
        np.testing.assert_allclose(actual, saved[field])
    return dict(step=k, auc=auc[0].tolist(), lower=lower[0].tolist(),
                median=median.tolist(), old_probability=dict(zip(arms, probability.tolist())),
                comparable_per_seed=[
                    sum(v is not None for v in blocks) for blocks in fractions])


def plot_development(data, folder):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.7), sharex=True, sharey="row",
                             layout="constrained")
    panels = (
        (("old_correct", "Correct Cedar", "#0F4D92", "-"),
         ("new_paired", "Paired Maple", "#B64342", "-"),
         ("new_paired_random_permutation_mean", "Permuted pairing mean", "#767676", "--")),
        (("information", "Total", "#272727", "-"),
         ("within_cedar", "Within Cedar", "#0F4D92", "-"),
         ("within_maple", "Within Maple", "#B64342", "-"),
         ("between_sets", "Between sets", "#767676", "--")),
    )
    for j, (seed, rows) in enumerate(data.items()):
        steps = [r["step"] for r in rows]
        axes[0, j].set_title(f"Development seed {seed}")
        axes[0, j].set_ylim(-.02, 1.02)
        axes[0, j].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        for i, fields in enumerate(panels):
            ax = axes[i, j]
            for field, label, color, style in fields:
                ax.plot(range(len(steps)), [r[field] for r in rows], style,
                        marker="o" if style == "-" else None, ms=3.5,
                        color=color, label=label, lw=1.5)
            ax.set_xticks(range(len(steps)), steps)
            ax.grid(axis="y", alpha=.22)
        axes[1, j].set_xlabel("Maple update step (sampled checkpoints)")
    axes[0, 0].set_ylabel("Mass on Cedar prompts")
    axes[1, 0].set_ylabel("Mutual information (nats)")
    axes[1, 0].set_ylim(-.12, 3.57)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.savefig(folder / "binding_trajectories.png", dpi=300)
    fig.savefig(folder / "binding_trajectories.pdf")
    plt.close(fig)


def main():
    records = {s: read(OUT / f"s{s}" / "stage1_trajectories.json") for s in range(101, 107)}
    for blocks in records.values():
        assert len(blocks) == 5
        for block in blocks:
            for trajectory in block.values():
                assert 17 <= len(trajectory) <= 65
                assert all(len(row["test"]) == 40 for row in trajectory)
            np.testing.assert_allclose(block["true"][0]["test"], block["false"][0]["test"])
    replications = {}
    for seed in (107, 108, 109):
        raw_path = OUT / "disjoint_mass" / f"s{seed}" / "mass.json"
        if raw_path.is_file() and "64" in read(raw_path)["result"]:
            rows = mass_audit(seed)
            replications[seed] = dict(mass=rows, outcome=replication_audit(rows))
    prefix_path = OUT / "disjoint_mass" / "s107" / "mass_prefix_control.json"
    prefix_control = None
    if prefix_path.is_file() and "64" in read(prefix_path)["result"]:
        rows = mass_audit(107, prefix_control=True)
        error = validate_prefix_control(read(prefix_path.with_name("mass.json")), read(prefix_path))
        prefix_control = dict(mass=rows, outcome=replication_audit(rows),
                              anchor_max_probability_delta=error)
    result = dict(
        mass={s: mass_audit(s) for s in (29, 43)},
        by_exposure={s: exposure_audit(s) for s in (29, 43, *replications)},
        replication=replications,
        prefix_control=prefix_control,
        maple_initial={a: [block[a][0]["maple"] for blocks in records.values() for block in blocks]
                       for a in ("true", "false")},
        absolute={t: paired_audit(records, t) for t in (-6.0, -4.5, TARGET)},
        gain={t: paired_audit(records, t, relative=True) for t in (0.5, 1.0, 2.0, 3.0, 4.0)},
        observed_gain={t: paired_audit(records, t, relative=True, interpolate=False)
                       for t in (0.5, 1.0, 2.0)},
        gate=reproduce_gate(records),
    )
    target = Path(__file__).resolve().parents[2] / "paper" / "cedar_maple_reanalysis.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    plot_development(result["mass"], target.parent)
    for seed, rows in result["mass"].items():
        for row in rows:
            print(seed, " ".join(f"{key}={value:.4f}" for key, value in row.items()))
    for mode in ("absolute", "gain"):
        for t, seeds in result[mode].items():
            print(mode, t, [(s, r["n"], np.round(r["mean_delta"], 4).tolist()
                            if r["n"] else None) for s, r in seeds.items()])
    print(result["gate"])
    for seed, data in replications.items():
        print("replication", seed, data["outcome"])
    if prefix_control is not None:
        print("prefix control", prefix_control["outcome"])


if __name__ == "__main__":
    main()
