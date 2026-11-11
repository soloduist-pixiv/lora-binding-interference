"""Describe destinations of Maple mass at validation-selected matched progress."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from offline import OUT


def matrix_at(folder, run, fraction=1.):
    match = next((p for p in run["progress"] if p["target"] == -1 and
                  p["fraction"] == fraction and p["comparable"]), None)
    if match is None:
        return None
    step = match["step"]
    above = next(i for i, r in enumerate(run["trace"]) if r["step"] >= step)
    lower, upper = run["trace"][above-1], run["trace"][above]
    weight = (step-lower["step"])/(upper["step"]-lower["step"])
    matrices = []
    for row in (lower, upper):
        with np.load(folder/f"s{row['step']:04d}.npz") as data:
            mask = data["indices"] % 6 < 2
            matrices.append(data["old"][mask].reshape(24,2,96).mean(1)[:,48:])
    return (1-weight)*matrices[0]+weight*matrices[1]


def summarize(matrix, protocol):
    active = set(protocol["active_original"])
    active_counter = {k+1 for k in active}
    items = []
    for i, key in enumerate(range(1,49,2)):
        for j in range(48):
            target = j+1
            if target == key:
                group = "paired"
            elif target == key+1:
                group = "same_family_counterfactual"
            elif target in active:
                group = "other_active_original"
            elif target in active_counter:
                group = "other_active_counterfactual"
            else:
                group = "inactive"
            a, b = protocol["keys"][key], protocol["keys"][target]
            items.append(dict(key=key,target=target,group=group,
                              distance=abs(key-target),
                              character_distance=sum(c != d for c,d in zip(a,b)),
                              mass=float(matrix[i,j])))
    groups = {}
    for group in sorted({r["group"] for r in items}):
        selected = [r for r in items if r["group"] == group]
        groups[group] = dict(total_mass=sum(r["mass"] for r in selected)/24,
                             per_candidate_mean=np.mean([r["mass"] for r in selected]),
                             pairs=len(selected))
    near = []
    for low, high in ((1,2),(3,8),(9,16),(17,32),(33,47)):
        selected = [r["mass"] for r in items if r["group"].startswith("other_active")
                    and low <= r["distance"] <= high]
        near.append(dict(low=low,high=high,n=len(selected),
                         mean=float(np.mean(selected)) if selected else None))
    return dict(groups=groups,ordinal_distance=near,matrix=matrix.tolist(),
                pairing=protocol["pairing"])


def main():
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
    runs = json.loads((OUT/"analysis.json").read_text(encoding="utf8"))
    data = {}
    for name, run in runs.items():
        p = run["protocol"]
        if not run["complete"] or p["historical_anchor"] or p["model"] != "Qwen3-0.6B":
            continue
        matrix = matrix_at(OUT/"runs"/name,run)
        if matrix is not None:
            data[name] = summarize(matrix,p)
            early = matrix_at(OUT/"runs"/name,run,fraction=.75)
            if early is not None:
                data[name]["gap75"] = summarize(early,p)
    (OUT/"matched_routes.json").write_text(json.dumps(data,indent=2),encoding="utf8")
    fig, axes = plt.subplots(2,2,figsize=(7,5.8),layout="constrained",sharey=True)
    axes = axes.ravel()
    for ax, pairing in zip(axes,("adjacent","permuted","support_permuted","alphabetic")):
        matrices = [np.array(row["matrix"]) for row in data.values() if row["pairing"] == pairing]
        ax.set(title=f"{pairing}, n={len(matrices)}",xlabel="Maple target key index")
        if matrices:
            image = ax.imshow(np.mean(matrices,0),aspect="auto",vmin=0,vmax=1,cmap="Blues")
    for ax in axes[::2]:
        ax.set_ylabel("Cedar query key (odd indices)")
    if data:
        fig.colorbar(image,ax=axes,label="Canonical mass on Cedar prompts")
    for extension in ("png","pdf"):
        fig.savefig(OUT/f"matched_routes.{extension}",dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(7,3.1),layout="constrained")
    groups = ("paired","same_family_counterfactual","other_active_original",
              "other_active_counterfactual","inactive")
    labels = ("Paired","Same-family CF","Other active\noriginal",
              "Other active CF","Inactive")
    for offset,(pairing,color) in enumerate(zip(
            ("adjacent","permuted","support_permuted","alphabetic"),
            ("#0F4D92","#42949E","#8E6C8A","#B64342"))):
        rows = [r for r in data.values() if r["pairing"] == pairing]
        if not rows:
            continue
        x = np.arange(len(groups))+(offset-1.5)*.18
        y = np.array([[r["groups"][g]["total_mass"] for g in groups] for r in rows])
        axes[0].barh(x,y.mean(0),.17,color=color,label=pairing,alpha=.8)
        for values in y:
            axes[0].scatter(values,x,s=10,color="black",alpha=.45,zorder=3)
        distance = np.array([[r["ordinal_distance"][i]["mean"] for i in range(5)] for r in rows])
        axes[1].plot(range(5),distance.mean(0),"o-",color=color,label=pairing)
        axes[1].fill_between(range(5),distance.min(0),distance.max(0),color=color,alpha=.08)
    axes[0].set(yticks=range(5),yticklabels=labels,xlabel="Maple mass on Cedar prompts")
    axes[0].tick_params(axis="y",labelsize=8)
    axes[1].set(xticks=range(5),xticklabels=("1–2","3–8","9–16","17–32","33–47"),
                xlabel="Ordinal key distance",
                ylabel="Mass per eligible candidate")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[0] else "y",alpha=.2)
    axes[0].legend(frameon=False,fontsize=7)
    for extension in ("png","pdf"):
        fig.savefig(OUT/f"routing_profiles.{extension}",dpi=300)
    plt.close(fig)
    print("Matched routing matrices",len(data))


if __name__ == "__main__":
    main()
