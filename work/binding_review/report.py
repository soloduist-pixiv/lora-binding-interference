"""Build the revised paper from seed-level measurements and public source links."""
import json
from pathlib import Path
import re
import shutil

import numpy as np

from offline import OUT, ROOT
from figures import selected

REPO = "https://github.com/soloduist-pixiv/lora-binding-interference"


def point(run, target=-1., fraction=1.):
    return next((p["interpolated"] for p in run["progress"]
                 if p["target"] == target and p["fraction"] == fraction and p["comparable"]), None)


def estimate(values, scale=1., digits=2):
    values = [v*scale for v in values if v is not None]
    if not values:
        return "未覆盖"
    return f"{np.mean(values):.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def points(runs, **kwargs):
    return [p for r in runs if (p := point(r, **kwargs)) is not None]


def table(header, rows):
    return "\n".join(["| "+" | ".join(header)+" |",
                      "| "+" | ".join(["---"]*len(header))+" |",
                      *["| "+" | ".join(map(str,row))+" |" for row in rows]])


def image(name, alt):
    path = f"outputs/binding_review_20261111/{name}.png"
    return f"![{alt}](https://raw.githubusercontent.com/soloduist-pixiv/lora-binding-interference/main/{path})"


def main():
    runs = json.loads((OUT/"analysis.json").read_text(encoding="utf8"))
    pipeline = json.loads((OUT/"pipeline.json").read_text(encoding="utf8"))
    count = sum(r["complete"] for r in runs.values())
    lr_rows = []
    for lr in (.0001,.0003,.001):
        group = selected(runs,lr=lr)
        early, late = points(group,fraction=.75),points(group)
        delta = [r["fixed_change"]["information"] for r in group if r["fixed_change"]]
        lr_rows.append([f"{lr:g}",len(group),
                        estimate([p["old_correct"] for p in early],100),
                        estimate([p["old_correct"] for p in late],100),
                        estimate([p["active_maple_test_logp"] for p in late]),
                        estimate([p["js"] for p in late],digits=4),
                        estimate(delta,digits=3)])
    control_rows = []
    for label,rank,replay in (("普通，rank 4",4,False),("普通，rank 16",16,False),
                              ("回放，rank 4",4,True),("回放，rank 16，探索性",16,True)):
        group = selected(runs,lr=.0003,rank=rank,replay=replay)
        late = points(group,target=-1.)
        control_rows.append([label,len(late),
                             estimate([p["old_correct"] for p in late],100),
                             estimate([p["active_maple_test_logp"] for p in late]),
                             estimate([p["js"] for p in late],digits=4),
                             estimate([p["js_upper"] for p in late],digits=4)])
    factor_rows = []
    factor_notes = []
    for pairing,label in (("adjacent","相邻整数"),("permuted","全局打散整数"),
                           ("alphabetic","非数值代码"),("support_permuted","支持集保持打散，探索性")):
        group = selected(runs,historical=False,lr=.0003,pairing=pairing)
        late = points(group)
        factor_rows.append([label,len(late),
                            estimate([p["old_correct"] for p in late],100),
                            estimate([p["old_prompt_paired_new"] for p in late],100),
                            estimate([p["old_prompt_paired_new"]/p["old_prompt_maple_total"]
                                      for p in late],100),
                            estimate([p["active_maple_test_logp"] for p in late]),
                            estimate([p["other"] for p in late],100)])
        if late:
            factor_notes.append(f"{label}在共同目标处的其余质量为 "
                                +estimate([p["other"] for p in late],100)+"%。")
            if pairing in ("adjacent","alphabetic"):
                factor_notes.append("该水平初始 Maple 验证 logP 为 "
                                    +estimate([r["trace"][0]["maple"] for r in group])+"。")
    model_rows = []
    model_notes = []
    for model in ("Qwen3-0.6B","SmolLM2-360M-Instruct"):
        group = selected(runs,historical=False,model=model,lr=.0003)
        late = points(group)
        delta = [r["fixed_change"]["information"] for r in group if r["fixed_change"]]
        model_rows.append([model,len(late),
                           estimate(delta,digits=3),
                           str(sum(v>0 for v in delta))+"/"+str(len(delta)),
                           estimate([p["old_correct"] for p in late],100),
                           estimate([p["js"] for p in late],digits=4),
                           estimate([p["js_upper"] for p in late],digits=4)])
        if model == "SmolLM2-360M-Instruct" and len(group)==3 and len(late)==3:
            pairs = [(point(r,fraction=.75),point(r)) for r in group]
            increase = [b["information"]-a["information"] for a,b in pairs]
            damage = [(b["old_correct"]-a["old_correct"])*100 for a,b in pairs]
            model_notes.append(
                "SmolLM2 在三个种子上均通过 Cedar 门槛，并达到 Maple 验证 logP −0.5。"
                "其固定 16→64 步互信息均下降；在 75% 缺口闭合位置至共同绝对目标 −1.0 的辅助比较中，"
                "互信息变化依种子为 "+ "、".join(f"{v:+.3f}" for v in increase)+" nats，"
                "旧正确质量同时变化 "+ "、".join(f"{v:+.2f}" for v in damage)+" pp。"
                "后一比较使用了 100% 这个辅助端点，区别于预定的 25%、50%、75% 主比较。")
    capacity_rows = []
    for run in runs.values():
        p = run["protocol"]
        if p.get("minimum_steps",64) != 640 or not run["complete"]:
            continue
        name = next(k for k,v in runs.items() if v is run)
        summary = json.loads((OUT/"runs"/name/"final_generation_summary.json").read_text(encoding="utf8"))
        last = run["trace"][-1]
        capacity_rows.append([p["seed"],last["step"],f"{last['maple']:.3f}",
                              f"{summary['cedar']['accuracy']*100:.2f}",
                              f"{summary['maple']['accuracy']*100:.2f}",
                              f"{last['metrics']['js']:.4f}"])
    history = json.loads((OUT/"offline.json").read_text(encoding="utf8"))
    gap_rows = []
    for key,value in history["alignment"]["gap_closure"].items():
        target,fraction = key.split("/")
        nseeds = sum(r["n"]>0 for r in value["per_seed"].values())
        gap_rows.append([target,f"{float(fraction):.0%}",value["n"],nseeds,
                         "—" if value["seed_mean"] is None else f"{value['seed_mean']*100:.2f}"])
    subset_rows = []
    for target,value in history["alignment"]["absolute_train_required"].items():
        nseeds = sum(r["n"]>0 for r in value["per_seed"].values())
        subset_rows.append([target,value["n"],nseeds,
                            "—" if value["seed_mean"] is None else f"{value['seed_mean']*100:.2f}"])
    source = r"""# 共享 LoRA 顺序学习中的事实绑定干扰

*Factual Binding Interference during Sequential Low-Rank Adaptation*

2026 年 11 月 11 日修订工作稿。

## 摘要

共享 LoRA 顺序学习后，输出对查询键的区分与对旧事实的正确保持可能分离。我们在两套同时有效的合成注册表中分解键与答案的互信息，固定权重替换注册表标签，并设置学习率、回放、秩与值配对方式的对照。在三个旧任务锚点上，Maple 验证 logP 同为 −1.0 时，普通 rank-4 分支在 lr=1e-4 与 lr=1e-3 下的旧正确序列概率均值分别为 32.30% 和 20.83%，lr=3e-4 的 rank-4 回放参照为 61.00%，固定步数的互信息回升则随学习率改变。证据来自两款小模型上的合成代码事实与输入干预，非数值条件的候选覆盖不足，尚不能推广到真实事实。

**关键词**　持续学习，低秩适配，事实绑定，提示条件化

## 1　引言

一个模型先学习 Cedar 注册表，再学习 Maple 注册表。两张表中的同一个键可以有不同的值，且各自在所属注册表中正确。例如，Cedar 的 QA 对应 127，Maple 的 QA 对应 126。模型若把每个 Cedar 问题都答成该键的 Maple 值，仍能区分键，却无法按注册表正确回答。这与对所有键输出同一个高频值产生相同低正确率时的行为不同。

本文考察这种绑定干扰随顺序更新如何出现，以及哪些训练和数据条件会改变它。早期试点在高学习率下观察到键与输出的互信息先降后升，但正确旧值质量维持低位。新增学习率对照表明，这条时间轨迹需要限定其更新尺度与训练阶段。本文因此将正确值质量、答案来源、键依赖和注册表输入效应共同呈现，并在较高的新任务目标处比较旧值保持。

研究的具体贡献在于对这一受控现象进行可复核的范围检验：区分错误映射中的键结构与正确旧事实，检验学习率和旧样本回放如何改变两者的关系，再用配对方式与注册表输入干预排查表面结构和条件区分的影响。互信息分解、JS 与回放均为已有工具；本文不将其组合命名为新算法。

## 2　相关工作

LoRA 通过低秩矩阵更新适配冻结基座。O-LoRA、InfLoRA 等研究任务间子空间干扰，SLoRA 则从后续 LoRA 更新中的干扰成分出发进行去噪。本文采用普通加法 LoRA 和常规回放作为受控条件，关注输出如何绑定到查询键与注册表。 [Hu 等，2022](https://arxiv.org/abs/2106.09685) · [Wang 等，2023](https://aclanthology.org/2023.findings-emnlp.715/) · [Liang 和 Li，2024](https://arxiv.org/abs/2404.00228) · [Yang 等，2026](https://aclanthology.org/2026.acl-long.247/)

Palit 等发现，旧事实召回可在只训练新事实时先崩溃、再恢复、最后衰退，并通过关联记忆、归一化及表示干预解释共同位移的可逆影响。他们衡量的是正确旧事实的恢复。本文的互信息只衡量键与输出事件的依赖，错误但逐键不同的 Maple 答案也可以使它很高。两种读数对应不同的问题，本文的标签替换实验也未定位共同表示位移。 [Palit 等，2026，§3–4](https://arxiv.org/abs/2610.08718)

O’Neill 等在 Qwen3 上连续写入虚构事实，比较训练数据、目标与参数更新。在其重复陈述训练、严格问题均失败的事实子集中，225/327 个错误回答含有后续写入事实的内容。这已经确立了新内容可进入旧问题的行为现象。本文补充显式注册表条件下的同键与其他键概率分解、学习进度比较和值配对对照。 [O’Neill 等，2026，Appendix F](https://www.alphaxiv.org/abs/2607.11020.md)

Kaplan 等将微调后的事实错误与语义表示重叠及关联记忆干扰联系起来。本文使用共享字符串键，但不同注册表提示下的隐藏表示重叠未经测量，因此不能将其直接设为 1。后文的键距离分析描述字符串与索引上的邻近关系。Guo 等在上下文更新中区分当前值、同变量旧值和其他变量值，并进行探针和注意力干预。这里两套表同时有效，Cedar 查询的旧值仍然正确，答案分型按其所指注册表定义。 [Kaplan 等，COLM 2026](https://arxiv.org/pdf/2604.15574) · [Guo 等，2026](https://arxiv.org/pdf/2609.38866)

知识编辑中的特异性、副作用与时间事实共存也构成本研究的背景。逐步而非仅在任务终点评估持续学习，已有 stability gap 研究提供了依据。分布监测和 LoRA 更新门控同样有先例。本文的主要证据由具体任务对照给出。 [Hoelscher-Obermaier 等，2023](https://aclanthology.org/2023.findings-acl.733/) · [Huang 等，2026](https://aclanthology.org/2026.emnlp-main.1719/) · [De Lange、van de Ven 和 Tuytelaars，ICLR 2023](https://arxiv.org/abs/2205.13452) · [Paula 等，2026](https://arxiv.org/abs/2606.27634) · [Hoy 和 Celik，2025](https://arxiv.org/abs/2510.16089)

## 3　任务、测量与实验设计

### 3.1　两套同时有效的事实

Cedar 含 48 个键，按相邻键组成 24 个原题与换键题族。训练、验证和测试各使用 14、3、6 个彼此隔离的上下文，同一事实跨上下文复用。旧阶段检查原题、等价改写和换键三联，要求原题准确率相对基座提升至少 12.5 pp，验证三联通过率至少为 75%。新阶段在固定选出的 16 个原题键族及其换键样本上训练 Maple，共 32 个活跃事实。

上下文由三种固定的工单或请求前缀及编号组成，集合按编号切分，前缀模板跨集合复用。因此，测试衡量已见事实在未见编号上下文中的可访问性，不是开放语言改写或新事实泛化。

旧任务的概率评估覆盖 24 个原题键，Maple 验证覆盖 16 个活跃原题键。独立 Maple 测试指标也只在这 16 个键上汇总。密集轨迹每两步记录至第 64 步，之后每 16 步记录至最多 640 步。密集评估每键使用两个测试上下文，第 0、16、64 步另用全部六个。跨上下文重复用于检查同一事实的可访问性，种子是独立重复单位。

### 3.2　正确绑定与键依赖

令 $K$ 在 24 个查询键上均匀分布，$Y$ 为模型的完整规范 token 完成序列及 EOS。每个键先在所用上下文内平均概率。令 $c_k$、$m_k$ 分别为同键的 Cedar 与 Maple 值，报告

$$
D_C=\mathbb{E}_K P(Y=c_K\mid K,Cedar),\qquad
M_C=\mathbb{E}_K P(Y=m_K\mid K,Cedar).
$$

所有旧值、新值及“其余”组成完整的粗化输出分布。令 $S$ 表示值来自 Cedar、Maple 或其余，链式法则给出

$$
I(K;Y)=I(K;S)+\sum_s P(S=s)I(K;Y\mid S=s).
$$

其余在观测空间只有一个类别，故其条件项为零。集合内项按联合分布诱导的 $P(K\mid S)$ 计算，并乘以集合质量。该分解沿用已知信息论恒等式。一个始终答对 Cedar 的模型与一个始终输出同键 Maple 值的模型，均可有 $I(K;Y)=\log 24$，而 $D_C$ 分别为 1 和 0。合并其余序列会丢失其中的键信息，因此同时报告候选覆盖。

同键新值的置换参照在预定的 16 个活跃原题键内部计算。其均值为这 16 个查询行与对应 16 个 Maple 列的子矩阵平均，保留列的边际质量。历史第 64 步另报告在所有实际训练过的 32 个值中的参照。早期检查点的“活跃”表示任务设计中的活跃集合。

### 3.3　固定权重的注册表输入干预

在同一权重、键与测试上下文上，仅把 Cedar register 替换为 Maple register。对两个标签的规范序列分布逐题计算 JS，随后按上下文和键平均。候选之外的质量保留为其余类别。以 $p,q$ 表示这两个粗化分布，有

$$
J=\tfrac12 KL(p\Vert (p+q)/2)+\tfrac12 KL(q\Vert (p+q)/2).
$$

该量以 nats 计，最大为 $\log 2$。候选覆盖不足时，小 JS 仍可能遗漏其余桶内部的标签效应。由互信息链式法则，完整序列 JS 满足

$$
J\ \leq\ J_{\rm full}\ \leq\
\min\{\log 2,\ J+\tfrac12(p_O+q_O)\log 2\}.
$$

表中同时给出这个上界。此输入干预测量对两个注册表字符串的行为敏感性，其解释范围限于给定提示与输出事件。

### 3.4　新任务进度、预算与统计单位

令 $L_t$ 是 Maple 验证正确完成序列平均 logP，定义到固定目标 $L^\star$ 的缺口闭合比例

$$
G_t=\frac{L_t-L_0}{L^\star-L_0}.
$$

主要目标为 −1.0，−0.5 为敏感性目标，报告 25%、50%、75% 处的旧任务读数。不同配对水平的 $L_0$ 可能相差很大，相同闭合比例未必代表相同的绝对新任务质量，因此另报首次达到共同绝对目标的辅助比较。$\exp(L_t)$ 是规范完成事件概率的几何平均，解码正确率另行计数。

首次越过验证目标的相邻检查点用于插值独立测试读数，覆盖不足单列；首次实测达标点保存在逐种子数据中。固定步数的新增描述使用第 16→64 步，密集曲线完整公开。表中给出种子均值及种子范围，进度和固定步数表采用各自明示的两个或六个上下文。

### 3.5　条件与登记顺序

学习率、rank 与回放对照复用历史 Cedar 种子 107、108、109。普通共享 LoRA 的学习率取 1e-4、3e-4、1e-3，rank 为 4，另加 lr=3e-4 的 rank 16 和 rank-4 回放。普通分支每步四条 Maple，回放每步另加四条 Cedar。任务切换重置 AdamW，权重衰减为零，LoRA 作用于末四层的 Q、V 投影，缩放因子为 2。

AdamW 使用 $\beta=(0.9,0.999)$、$\epsilon=10^{-8}$，无学习率调度和梯度裁剪，损失按批内答案及 EOS 的 token 平均。提示使用各模型的原生聊天模板，单条 user 消息，不启用思考模式。Qwen3-0.6B 与 SmolLM2-360M-Instruct 的去重基座参数数分别为 596,049,920 和 361,821,120；rank 4 可训练参数分别为 81,920 和 51,200。新训练的 Cedar 锚点使用 lr=1e-3，在 480、960、1440、1920 步检查门槛；通过后才进入 Maple 阶段，实际步数保存在 anchor.json。

配对方式实验使用新种子 211、223、227，每种子运行相邻整数、全局打散整数和非数值代码三个水平，均为 lr=3e-4、rank 4。相邻与打散共享 Cedar 权重及完整值集合，打散按升序 Maple 值循环移动 24 位。非数值水平为两套互斥的六字母代码，每个规范答案恰好三个 tokenizer token，重新训练 Cedar，三个水平统一要求仅输出 code。SmolLM2-360M-Instruct 在相同三个新种子上复核相邻整数条件。

两模型的三位整数答案也各为三个 token。相邻配对共享前两个数字 token；全局打散与支持集保持打散使两位前缀不同，首个词元仍都是数字“1”。配对干预并未消除所有词元重合。

主三水平因子实验与第二模型在产生结果前已公开登记；历史锚点的稳健性分支在公开登记时已有部分运行。其后又登记两个探索性扩展。一个固定运行 640 步的 rank-16 回放参照检验容量与保持能力，并对 48 个 Cedar 事实和 32 个 Maple 事实做六上下文自由生成。另一个把置换限制在活跃原题、活跃换键和其余键内部，保留训练与验证值支持集；各值的抽样频数仍会改变。 [初始公开登记](https://github.com/soloduist-pixiv/lora-binding-interference/commit/804bde4) · [完整协议](https://github.com/soloduist-pixiv/lora-binding-interference/blob/main/paper/review_protocol.md)

## 4　结果

### 4.1　固定步数的回升依赖更新尺度

历史三个留出种子在原 lr=1e-3 条件下，以固定第 16 步为参照，至第 64 步的互信息增加分别为 0.318、0.638、1.034 nats，旧正确质量变化分别为 −19.55、−9.44、+0.50 pp。这是本轮对历史轨迹的固定点重算，原 8/16 步取低值规则保留在历史协议中。新增 FP32 复跑的三个高学习率分支仍显示正变化，低学习率分支的固定时间窗变化则可以为负。

新增高学习率分支在第 64 步的总互信息为 2.810、2.765、2.867 nats，旧正确质量仅为 3.51%、13.03%、20.57%，同键 Maple 质量为 65.42%、60.19%、55.28%。第 16→64 步的 Maple 集合加权互信息分别增加 1.035、1.129、1.153 nats，同时 Cedar 集合的贡献下降。因而，输出重新呈现键结构并不表示旧映射随之恢复。

__LR_TABLE__

表 1．旧正确质量以百分比计；种子均值后的方括号为种子范围。75% 为到 −1.0 目标的缺口闭合比例，目标处为平均验证 logP 首次达到 −1.0。独立 Maple 测试 logP 仅覆盖活跃原题键。固定变化使用六个上下文，其余为两个。

__LR_IMAGE__

图 1．三个历史锚点的新学习率轨迹。每两步记录至第 64 步。上行为键与粗化答案的互信息，中行为旧正确质量，下行为注册表输入 JS。不同学习率在同一步数处可能处于不同新任务学习阶段。

__LR_PROGRESS_IMAGE__

图 2．按新任务缺口闭合比例比较。线为种子中位数，点为各个种子，淡色带为种子最小至最大值。较小学习率在相同验证学习目标处保留更多旧值，但同一固定窗口的回升形状不能跨学习率直接推广。

### 4.2　回放与扩大 rank 的参照

__CONTROL_TABLE__

表 2．lr=3e-4，在 Maple 验证平均 logP 首次达到主要目标 −1.0 处比较。旧正确质量以百分比计，JS 和上界以 nats 计。回放增加旧样本与计算开销，所有分支每步仍见四条 Maple；其样本暴露量随原始轨迹公开。最后一行来自后续登记的容量扩展，取首次达标而非最终第 640 步。−0.5 敏感性目标的结果随逐种子全表公开。

__REFERENCE_IMAGE__

图 3．rank 与回放参照在目标 −1.0 的缺口闭合曲线。星号表示后续登记的探索性容量扩展。增大 rank 与加入旧样本是不同的操作，分别报告。

固定预算容量参照的自由生成验收如下。每个种子中，Cedar 对 48 个不同事实、共 288 个测试查询评分，Maple 对 32 个活跃事实、共 192 个测试查询评分。该扩展为观察到部分基础参照后登记的探索性实验。

__CAPACITY_TABLE__

表 3．rank 16 回放在固定第 640 步的结果。生成准确率采用 BF16 原生前向、贪心解码及逐字严格匹配；最后一列规范分布 JS 使用 FP32。事实数和上下文数分别计数。

这一终点参照检验两套映射在同一基座与适配器中能否共存。回放再次提供旧事实，末期表现可以包括旧映射的重新学习；它不等同于更新全程没有暂时损伤，也不能单独证明 rank 4 的容量已经充足。

### 4.3　注册表输入效应与值配对方式

历史 Qwen 锚点的普通分支在共同目标处的注册表 JS 较小，回放参照可以改变这一输入效应。这个结果描述两个标签下的输出条件差异，并不定位模型内部的注意力或表示机制。候选覆盖和其余桶上界随同读数报告；第二模型的标签效应另见 §4.4。

__FACTOR_TABLE__

表 4．配对方式在共同 Maple 验证 logP −1.0 处的辅助比较。旧正确、同键 Maple、其余质量及 Maple 集合内同键占比均以百分比计。前三行为运行前公开登记的三水平实验；最后一行为后续登记的支持集保持打散控制。非数值水平同时改变值类型和 Cedar 学习问题。

__FACTOR_NOTES__

相邻整数的 Maple 集合内同键占比均值为 71.10%，全局打散与支持集保持打散分别为 44.73% 和 47.45%。支持集保持打散的旧正确质量为 6.61%，相邻条件为 34.12%。因此，改变活跃目标支持集并不足以单独解释数字配对差异；词元结构、值的抽样频数及学习路径仍需共同考虑。

非数值条件的候选外质量明显较大。因此，该水平的粗化互信息不能与高覆盖的数字水平等同解释，小粗化 JS 也不足以断言完整生成分布对标签不敏感。Maple 集合内的同键占比必须与其绝对质量和其余桶一起读取。

__FACTOR_IMAGE__

图 4．三个登记水平的进度轨迹。不同水平的初始 Maple logP 不同，图中相同闭合比例与表 4 的同一绝对目标分别回答不同问题。

__ROUTE_IMAGE__

图 5．在首次达到 Maple 验证 logP −1.0 的区间插值 Cedar 提示下的 Maple 候选质量，并按种子平均。列按该值所对应的 Maple 键排序，包含原题键和换键键，行是 24 个 Cedar 原题键。数值大小不决定列顺序。逐种子的同键、同题族换键、其他活跃键及非活跃键分型、索引距离曲线在公开数据中给出。这些邻近分析为描述性结果。

__ROUTE_PROFILE_IMAGE__

图 6．错误去向的分组质量及索引距离。左图柱为三种子均值、点为种子；右图先对每个距离组按可选候选数归一化，再画种子均值与范围。距离只针对其他活跃键，字符串与索引距离未经隐藏表示测量。

__COVERAGE_IMAGE__

图 7．候选以外质量的均值、最大键值及注册表 JS 的上界。其余桶较大时，粗化分布对完整输出差异的约束较弱，互信息中的桶内键信息也未被观测。

### 4.4　第二个小模型的范围复核

__MODEL_TABLE__

表 5．两个模型在新种子上的相邻整数条件，均为 lr=3e-4、rank 4。固定变化为第 16→64 步六上下文读数，正变化数仅作描述。目标处的旧质量和 JS 为首次达到 Maple 验证平均 logP −1.0 的插值。模板、tokenizer 和预训练不同，因此这些结果用于复核现象范围。

__MODEL_NOTES__

SmolLM2 的注册表 JS 在共同目标处高于 Qwen 新种子的相邻整数条件。因此，本文不把接近零的标签效应作为跨模型的必要条件。JS 衡量两个标签下的输出区分，并不保证任一标签下的答案正确。

__MODEL_IMAGE__

图 8．第二模型的进度比较。每款模型各三个登记种子。不可比较的目标不连线，覆盖数见逐种子全表。

### 4.5　主张与证据边界

| 主张 | 证据 | 主要对照 | 分析类型 |
|---|---|---|---|
| 键依赖与正确旧值保持可以分离 | 互信息分解、正确质量、答案来源矩阵 | 同键值与其他键值、完整候选覆盖 | 历史观察及本轮稳健性复核 |
| 固定时间窗的回升形状依赖更新尺度 | 三种子、三学习率、每两步轨迹 | 共同验证目标、独立 Maple 测试 | 运行前约定的稳健性规格，公开登记时部分已启动 |
| 注册表输入效应可以测量并受训练条件影响 | 固定权重、键与上下文的标签替换 | 回放、rank、候选外质量上界 | 已登记的输入干预 |
| 值配对影响同键和其他键质量分配 | 三水平、三个新种子及匹配进度矩阵 | 数字共享锚点、非数值压力条件 | 前瞻公开登记；支持集控制和邻近分析为探索性 |
| 对旧值保持存在可检验的容量参照 | 固定 640 步回放及逐题自由生成 | rank 4 与 rank 16、样本和计算预算 | 部分结果产生后登记的探索性扩展 |

表 6．主张、证据及主要对照，按数据与分析规则的登记顺序区分证据类型。

## 5　讨论与局限

互信息衡量输出事件中的键依赖，正确旧值质量衡量该依赖是否符合 Cedar 映射。两者的联合读数避免将一个有结构的错误映射误读为旧事实恢复。学习率对照进一步限定了时间轨迹的解释：相同步数不代表相同新任务学习状态；较高学习率在目标之后继续更新，也可能改变回升幅度。

注册表替换给出了直接的输入层干预。普通顺序更新下低 JS 与同键新值进入旧提示相容，回放参照则检验标签条件区分和旧值保持能否同时改善。关于隐藏知识是否擦除、共同表示位移是否回撤、哪个注意力模块控制答案选择，仍需要相应的表示或模块干预。本文结果与表观遗忘、写入副作用和上下文更新失败相联系，但各自测量的恢复对象与干预位置不同。

Cedar 起点只训练过一个注册表，其标签 JS 本来就可能很低。因此，末期低 JS 不能证明新更新破坏了一个已建立的注册表路由器；它说明在所测提示中，顺序训练尚未形成充分的标签条件区分。回放的改善还伴随旧样本暴露与额外计算，不能单独归结为某个低秩方向。

数字的相邻与打散条件共享 Cedar 权重，非数值水平则同时改变字符串、词元频率和旧任务学习过程。即使控制 token 数，初始新任务概率也可能不同。原全局重配还会改变活跃值支持集，支持集保持扩展用于检验这一因素，但抽样频数与目标绑定关系仍随置换改变。字符串距离也不能替代隐藏表示重叠。

实验集中于两款小模型、两阶段顺序更新和合成事实。候选规范序列及 EOS 的概率与自由生成正确率分别报告；其余桶可能包含其他表面形式。种子只有三个，本轮给出逐种子配对读数及范围。真实事实、更多任务阶段和更大模型仍是外部有效性的主要缺口。

普通新任务分支在至少第 64 步且验证平均 logP 达到 −0.5 后停止，或在第 640 步达到预算上限。因此，本文刻画所观测学习区间中的分离，不回答只训练新事实更长时间后旧正确值是否再恢复；固定预算的回放参照也不能代替这一测试。

## 6　复现材料

代码、完整协议、参数、环境、模型文件校验、原始矩阵与逐题生成输出均在公开仓库。模型基座从官方仓库获取，本次公开快照包含合成 Cedar 适配器锚点及数值核验所需的代表性检查点。新增训练使用 BF16 基座与 FP32 LoRA，概率评估统一使用 FP32。Qwen 数字试点中，FP32 前缀树与完整前缀评分在旧起点、一次更新及第 16、64 步抽查的最大概率差约为 1.1e-6。对非数值 Qwen 与第二模型的起点和终点另取两个键、两个标签，共四个提示，逐一核对 96 个候选，最大差为 2.1e-5，均通过事先使用的 1e-4 核验阈值。早期 BF16 矩阵保留供精度审计。

硬件为 RTX 4060 Laptop、8188 MiB 显存及约 16 GiB RAM，训练串行。PyTorch 2.10.0+cu130、Transformers 5.10.2、Python 3.14.3，完整环境与依赖版本随数据给出。预留显存上限为 6 GiB。 [公开仓库](https://github.com/soloduist-pixiv/lora-binding-interference) · [逐种子全表](https://github.com/soloduist-pixiv/lora-binding-interference/blob/main/paper/review_tables.md) · [引用核对](https://github.com/soloduist-pixiv/lora-binding-interference/blob/main/paper/references_audit.md)

~~~text
python work/binding_review/offline.py
python work/binding_review/analyze.py
python work/binding_review/figures.py
python work/binding_review/routes.py
~~~

上述重算入口在 CPU 上运行。重新训练可指定独立的输出目录，具体命令、模型下载位置和矩阵格式见仓库 README。

## 附录 A　历史轨迹、置换参照与覆盖

__HISTORY_DEV_IMAGE__

图 A1．历史开发种子 29、43，横轴采用实际步数。

__HISTORY_IMAGE__

图 A2．历史留出种子 107–109。全部五种子的其余桶均值轨迹最高值为 4.15%–9.48%，第 64 步种子 109 个别键的其余质量达到 26.37%。

在历史第 64 步的 16 个实际训练原题键内部置换后，种子 29、43 的同键 Maple 质量与置换参照分别为 66.26% 对 4.86%、86.53% 对 5.49%。这一参照排除了八个未参与 Maple 更新的原题键。原单种子打散试点中，流向其他已训练值的质量在近邻索引组为 4.39%，按组机会数校正的均匀参照为 3.10%；同首字母的其余组为 17.67% 对 15.83%。这些分型只作描述，支持集保持控制与新三种子数据另行公开。

## 附录 B　真假别名中的进度口径审计

历史真假别名实验使用种子 101–106、每种子五个配对区块，均为 lr=1e-3。原绝对目标较低，且真别名的部分轨迹在第 0 步已达标。先排除两臂任一在起点已达绝对目标的区块，结果如下。

__SUBSET_TABLE__

表 B1．仅保留真假两臂均需要训练的区块。损伤差为假减真，以 pp 计；先在种子内求均值，再对可比较种子等权平均。子集的种子覆盖随目标改变。

改用到 −1.0 或 −0.5 的缺口闭合比例后，历史数据的损伤差如下。较高进度的区块覆盖下降，原“相对 +1 nat 时方向反转”不作为主文结论。

__GAP_TABLE__

表 B2．历史别名轨迹的闭合比例比较，损伤差为假减真，单位 pp。独立单位为 Cedar 种子。该方法学审计与主文绑定干扰实验的数据构造不同。

闭合比例比较各自从起点走向固定目标的相对进度，未保证两臂具有相同的绝对新答案质量。排除零步达标也未消除初始学习距离的差异。这里不再用原 +77 pp 差值单独论证别名真伪的机制效应。

## 附录 C　历史低功效探针

历史冻结规则使用旧探针累计 logP 变化及事前梯度点积，在第 2、4、6、8、12、16 步按种子留一拟合并判断。第 8、12、16 步 AUC 分别为 0.853、0.967、0.993，连续三点规则到第 16 步才满足。假别名相对共同目标的剩余损伤比例中位数为 14.78%（logP）和 2.69%（概率），29/30 条可比较。既有功效模拟为 14.6%–52.1%。这些读数仅属于原 lr=1e-3 的历史规格，跨学习率的判别器检测门槛未复跑。

## 附录 D　早期自由生成的范围

早期七键试点中，Qwen3-0.6B、Qwen3.5-0.8B 和 SmolLM2-1.7B 的同收益观察点都只学会一个新键。该键的 19 个 Cedar 上下文有 19、19、16 次生成同键 Maple 值，独立事实仍每模型一个。此段只作早期探索记录。主文容量参照另外覆盖 48 个 Cedar 事实与 32 个活跃 Maple 事实，逐题输出和自动评分均公开。
"""
    replacements = {
        "__LR_TABLE__":table(["lr","种子数","75% 处旧质量 %","共同目标处旧质量 %","独立 Maple 测试 logP","目标处 JS nats","固定变化 nats"],lr_rows),
        "__CONTROL_TABLE__":table(["参照","种子数","旧质量 %","独立 Maple 测试 logP","JS nats","JS 上界 nats"],control_rows),
        "__FACTOR_TABLE__":table(["配对方式","覆盖种子","旧质量 %","同键 Maple 质量 %","Maple 内同键 %","独立 Maple 测试 logP","其余 %"],factor_rows),
        "__FACTOR_NOTES__":"".join(factor_notes),
        "__MODEL_TABLE__":table(["模型","覆盖种子","固定互信息变化","正变化数","目标处旧质量 %","目标处 JS","JS 上界"],model_rows),
        "__MODEL_NOTES__":"".join(model_notes),
        "__CAPACITY_TABLE__":table(["种子","固定步数","Maple 验证 logP","Cedar 生成准确率 %","Maple 生成准确率 %","JS nats"],capacity_rows),
        "__GAP_TABLE__":table(["目标","闭合比例","配对区块","覆盖种子","损伤差 pp"],gap_rows),
        "__SUBSET_TABLE__":table(["绝对目标","配对区块","覆盖种子","损伤差 pp"],subset_rows),
        "__LR_IMAGE__":image("lr_trajectories","图 1 学习率轨迹"),
        "__LR_PROGRESS_IMAGE__":image("lr_progress","图 2 学习率进度对齐"),
        "__REFERENCE_IMAGE__":image("reference_progress","图 3 rank 与回放参照"),
        "__FACTOR_IMAGE__":image("factorial_progress","图 4 配对方式进度对齐"),
        "__ROUTE_IMAGE__":image("matched_routes","图 5 匹配质量处的 Maple 目标质量"),
        "__ROUTE_PROFILE_IMAGE__":image("routing_profiles","图 6 错误答案去向"),
        "__COVERAGE_IMAGE__":image("factorial_coverage","图 7 候选覆盖"),
        "__MODEL_IMAGE__":image("model_progress","图 8 第二模型复核"),
        "__HISTORY_DEV_IMAGE__":image("historical_development","图 A1 历史开发种子"),
        "__HISTORY_IMAGE__":image("historical_holdout","图 A2 历史留出种子"),
    }
    for key,value in replacements.items():
        source = source.replace(key,value)
    references = []
    audit = (ROOT/"paper/references_audit.md").read_text(encoding="utf8")
    for line in audit.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        match = re.search(r"\[([^\]]+)\]\(([^)]+)\)",cells[0]) if cells else None
        if match and match.group(2) in source and len(cells)>1:
            references.append(f"{len(references)+1}. {cells[0]}．{cells[1].replace('**','')}。")
    source = source.replace("## 附录 A", "## 参考文献\n\n"+"\n\n".join(references)+"\n\n## 附录 A",1)
    if count < 33:
        source = source.replace("## 摘要",f"当前为进行中的修订工作稿，已完成 {count}/33 条分支。\n\n## 摘要",1)
    target = ROOT/"paper/共享LoRA中的旧事实干扰.md"
    backup = ROOT/"paper/history/initial_20261111.md"
    if not backup.exists():
        backup.parent.mkdir(exist_ok=True)
        shutil.copy2(target,backup)
    target.write_text(source,encoding="utf8")
    response = f"""# 对修订意见的处理记录

本轮主实验 27 条，容量参照 3 条，训练值支持集保持对照 3 条。当前完成 {count}/33 条。科学上未支持的推广会收窄，未执行的工作按下表注明。

| 修订意见 | 本轮处理 | 证据入口 |
|---|---|---|
| 学习率与学习进度 | 同一旧锚点的三个种子，lr=1e-4、3e-4、1e-3；报告到 −1.0/−0.5 的 25%/50%/75% 缺口闭合和共同绝对目标辅助读数 | 主文表 1、图 1–2；逐种子全表 |
| 联合或回放与 rank | rank 4、16 的普通分支和回放；后续容量扩展固定 640 步，逐题检验两套映射是否可共存 | 主文表 2–3、图 3；生成原始输出 |
| 注册表干预 | 每个新检查点固定权重、键和上下文，替换 Cedar/Maple；计算逐上下文 JS，保留其余桶，报告完整分布 JS 上界 | 主文 §3.3、表 2、图 7 |
| 前缀因子实验 | 三水平 × 三新种子；相邻与全局打散共享旧锚点；另外完成三种子的训练与验证值支持集保持打散 | 主文 §4.3、表 4、图 4–6；协议注明后续扩展 |
| 密集检查点与参考点 | 至第 64 步每两步；新增固定 16→64 比较，历史 8/16 取低点规则保留并标明 | 图 1；历史协议和附录 A |
| 置换参照 | 在 16 个活跃原题键内部取均匀置换的参照期望，另给全部 32 个已见目标值的参照 | 附录 A；offline.json |
| 候选覆盖与错误去向 | 报告其余均值、最大键值、JS 上界；匹配学习目标后的同键、同题族、其他活跃键和非活跃键质量及索引距离 | 图 5–7；matched_routes.json |
| 留出种子图 | 主图含 107–109 的密集学习率曲线，附录分别给出 29/43 和 107–109 的原历史轨迹 | 图 1、图 A1–A2 |
| 原 §4.2 目标与零步达标 | 排除任一臂起点已达目标的区块，另用目标 −1.0/−0.5 的 25%/50%/75% 缺口闭合，公开覆盖数 | 附录 B；逐种子全表 |
| 原 §4.3 功效不足 | 移入附录 C，明确只测过 lr=1e-3；本轮未跨学习率重训探针判别器 | 附录 C |
| 开放生成事实太少 | 单键旧试点降为附录 D；容量扩展每种子评估 48 个 Cedar 和 32 个 Maple 事实，各六个独立测试上下文，严格自动评分 | 表 3；final_generations.json |
| 第二模型 | SmolLM2-360M-Instruct 的三个新种子均通过旧学习门槛并达到新任务目标；固定窗口与学习进度结果分别解释 | §4.4、表 5、图 8 |
| 主张、确认性与探索性 | 主线聚焦绑定干扰；按登记时序标明历史观察、稳健性复核、前瞻因子与后续探索扩展 | 主文表 6（主张—证据—对照表）和协议 |
| 文献与复现 | 核对近邻论文及既有引文，纠正 stability gap 作者；记录种子、超参、硬件、模型文件指纹、公开矩阵与文件校验 | 引用核对表、README、manifest.json |

## 根据新证据收窄的表述

固定 16→64 步的互信息回升未跨学习率保持一致，不能作为共享 LoRA 的普遍时间规律。相同 Maple 验证 logP −0.5 时，普通 rank-4 分支的旧正确质量均值在 lr=1e-4 与 lr=1e-3 下分别为 25.12% 与 12.67%；lr=3e-4 的 rank-4 回放为 66.07%。这些读数是三个种子的描述性均值。

非数值代码条件的其余桶很大，因此不能声称该条件的规范候选覆盖充分，或把小粗化 JS 等同于完整生成分布中的标签不敏感。Cedar 起点本来也可能缺少标签区分，输入干预没有证明内部路由器被破坏或旧知识被擦除。

第二模型固定早期窗口无互信息回升，但从后段闭合位置至共同绝对目标的辅助比较可出现互信息增加与旧正确质量继续下降。该结果不改变预先指定的主比较时点。

## 尚未覆盖

本轮没有真实事实基准、更多任务阶段、隐藏表示重叠的剂量实验或内部模块因果干预。历史探针的跨学习率检验也未执行。当前稿件适合作为可复核的受控研究工作稿；这些范围限制在主文保留。

[修订稿]({REPO}/blob/main/paper/共享LoRA中的旧事实干扰.md) · [逐种子全表]({REPO}/blob/main/paper/review_tables.md) · [登记协议]({REPO}/blob/main/paper/review_protocol.md) · [引用核对]({REPO}/blob/main/paper/references_audit.md)
"""
    (ROOT/"paper/revision_response.md").write_text(response,encoding="utf8")
    print("Revised manuscript",count,"complete branches",target)


if __name__ == "__main__":
    main()
