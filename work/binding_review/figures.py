"""Publication figures using complete runs and explicit progress coverage."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from offline import OUT

COLORS = ("#0F4D92", "#42949E", "#B64342")


def finish(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=300)
    plt.close(fig)


def selected(runs, *, model="Qwen3-0.6B", historical=True, rank=4, replay=False,
             lr=None, pairing="adjacent"):
    return [r for r in runs.values() if r["complete"] and r["protocol"]["model"] == model
            and r["protocol"]["historical_anchor"] == historical
            and r["protocol"]["rank"] == rank and r["protocol"]["replay"] == replay
            and r["protocol"]["pairing"] == pairing
            and (lr is None or r["protocol"]["lr"] == lr)]


def progress_values(run, field, target=-1.):
    entries = [r for r in run["progress"] if r["target"] == target and r["comparable"]]
    return {r["fraction"]: r["interpolated"][field] for r in entries}


def lr_trajectories(runs):
    fig, axes = plt.subplots(3, 3, figsize=(7, 5.8), sharex=True, sharey="row", layout="constrained")
    for j, seed in enumerate((107, 108, 109)):
        axes[0, j].set_title(f"Seed {seed}")
        for lr, color in zip((.0001, .0003, .001), COLORS):
            matches = [r for r in selected(runs, lr=lr) if r["protocol"]["seed"] == seed]
            if not matches:
                continue
            rows = [r for r in matches[0]["trace"] if r["step"] <= 64]
            for i, field in enumerate(("information", "old_correct", "js")):
                axes[i, j].plot([r["step"] for r in rows],
                                [r["metrics"][field] for r in rows], color=color,
                                lw=1.6, label=f"lr {lr:g}")
        for ax in axes[:, j]:
            ax.grid(axis="y", alpha=.2)
            ax.set_xticks([0,16,32,48,64])
        axes[2, j].set_xlabel("Maple update steps")
    for ax, label in zip(axes[:, 0], ("I(key; answer), nats", "Correct Cedar mass", "Register JS, nats")):
        ax.set_ylabel(label)
    axes[1, 0].set_ylim(-.02, 1.02)
    axes[0, 0].legend(frameon=False, fontsize=8)
    finish(fig, "lr_trajectories")


def progress_panel(runs, choices, name, fields=("old_correct","information","js"),
                   labels=("Correct Cedar mass","I(key; answer), nats","Register JS, nats")):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), layout="constrained")
    for label, group, color in choices:
        for ax, field in zip(axes, fields):
            values = [progress_values(r, field) for r in group]
            x, y, lo, hi = [], [], [], []
            for fraction in (.25, .5, .75, 1.):
                available = [v[fraction] for v in values if fraction in v]
                if not available:
                    continue
                x.append(fraction)
                y.append(np.median(available))
                lo.append(min(available))
                hi.append(max(available))
                ax.scatter([fraction] * len(available), available, s=12, color=color, alpha=.5)
            ax.plot(x, y, "o-", color=color, ms=4, label=label, lw=1.6)
            ax.fill_between(x, lo, hi, alpha=.08, color=color)
    for ax, field in zip(axes, labels):
        ax.set(xlabel="Maple gap closure", ylabel=field)
        ax.set_xticks([.25, .5, .75, 1.], ["25%", "50%", "75%", "100%"])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylim(-.02, 1.02)
    axes[0].legend(frameon=False, fontsize=7)
    finish(fig, name)


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    runs = json.loads((OUT / "analysis.json").read_text(encoding="utf8"))
    lr_trajectories(runs)
    progress_panel(runs, [(f"lr {lr:g}", selected(runs, lr=lr), color)
                         for lr, color in zip((.0001, .0003, .001), COLORS)], "lr_progress")
    progress_panel(runs, [
        ("rank 4", selected(runs, lr=.0003), COLORS[0]),
        ("rank 16", selected(runs, lr=.0003, rank=16), COLORS[2]),
        ("rank 4 + replay", selected(runs, lr=.0003, replay=True), COLORS[1]),
        ("rank 16 + replay*", selected(runs, lr=.0003, rank=16, replay=True), "#8E6C8A"),
    ], "reference_progress")
    progress_panel(runs, [
        (pairing, selected(runs, lr=.0003, historical=False, pairing=pairing), color)
        for pairing, color in zip(("adjacent", "permuted", "alphabetic"), COLORS)
    ], "factorial_progress")
    progress_panel(runs, [
        (pairing, selected(runs, lr=.0003, historical=False, pairing=pairing), color)
        for pairing, color in zip(("adjacent","permuted","alphabetic"),COLORS)
    ], "factorial_coverage", fields=("other","other_max","js_upper"),
                   labels=("Other canonical mass","Largest key other mass","Register JS upper bound, nats"))
    progress_panel(runs, [
        (model, selected(runs, model=model, lr=.0003, historical=False), color)
        for model, color in zip(("Qwen3-0.6B", "SmolLM2-360M-Instruct"), COLORS)
    ], "model_progress")
    print("Saved complete-run figures")


if __name__ == "__main__":
    main()
