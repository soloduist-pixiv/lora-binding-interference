"""Three paired-seed panels for the exploratory source-constrained readout."""
import json
from offline import OUT
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    runs=json.loads((OUT/"source_oracle_audit.json").read_text())["runs"]
    plt.rcParams.update({"font.size":8,"axes.titlesize":8.5,"pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout="constrained")
    data=[];colors=("#155993","#B96B32","#23867F");markers=("o","s","^")
    for index,(seed,color,marker) in enumerate(zip((211,223,227),colors,markers)):
        group=[]
        for pairing in ("adjacent","support_permuted"):
            r=runs[f"first_revision/Qwen3-0.6B_s{seed}_{pairing}_lr0.0003_r4"]
            point=r["points"][r["stages"]["goal-1.0"]];m=point["measurements"]
            assert m["readouts"]["all48"]["mass_floor_sensitivity"]["0.0001"]["keys"]==16
            row=dict(seed=seed,pairing=pairing,step=point["step"],raw_D=m["raw_correct"],
                old_set_mass=m["old_set_mass_mean"],oracle_top1=m["readouts"]["all48"]["oracle_top1"]["observed"],
                maple_prefix_match=m["maple_prefix_match"]["observed"],
                prefix_null_mean=m["maple_prefix_match"]["exact_null_mean"])
            group.append(row);data.append(row)
        x=[-.045+index*.045,.955+index*.045]
        for ax,key in zip(axes,("old_set_mass","oracle_top1","maple_prefix_match")):
            ax.plot(x,[100*v[key] for v in group],color=color,marker=marker,ms=4,lw=1.1,alpha=.95)
        axes[2].scatter(x,[100*v["prefix_null_mean"] for v in group],edgecolors=color,
                        facecolors="none",marker="D",s=19,zorder=4)
    for ax,title in zip(axes,("(a) Cedar-set mass","(b) Cedar-only top-1","(c) Maple-prefix match")):
        ax.set(title=title,xticks=(0,1),xticklabels=("Adjacent","Support-preserving\nshuffled"),
               xlim=(-.22,1.22),ylabel="Percent",ylim=(0,105))
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].set(ylim=(0,50),yticks=(0,10,20,30,40,50))
    axes[1].axhline(100/48,color="#777777",lw=.8,ls="--",zorder=0)
    legend=[Line2D([],[],color=c,marker=m,ms=4,label=f"Seed {s}") for s,c,m in zip((211,223,227),colors,markers)]
    legend.append(Line2D([],[],color="#777777",marker="D",mfc="none",ls="none",label="Prefix null mean"))
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,-.09),ncol=4,frameon=False,fontsize=7)
    fig.suptitle("Exploratory old anchors; aligned at validation logP = -1.0",fontsize=9)
    for ext in ("png","pdf"):fig.savefig(OUT/f"source_oracle_pairing.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)
    (OUT/"source_oracle_figure_data.json").write_text(json.dumps(data,indent=2))
    print("Rendered six paired observations; probability mass and constrained accuracy are separate panels.")

if __name__=="__main__":main()
