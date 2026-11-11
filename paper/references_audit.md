# 引用核对记录

核对日期为 2026-11-11。下表记录题名、作者与可核实的发表状态，以及本文可以引用的具体内容。会议状态只采用会议官网、论文首页或正式论文集信息。预印本的年份采用 arXiv 记录。表中“等”省略其余作者。

| 文献与原始来源 | 作者、年份与状态 | 原文支持的内容及修订处理 |
|---|---|---|
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Edward J. Hu 等，2021 年预印本，后发表于 ICLR 2022 | 冻结基座并训练低秩更新。引用时区分预印本年份和会议年份。 |
| [Orthogonal Subspace Learning for Language Model Continual Learning](https://aclanthology.org/2023.findings-emnlp.715/) | Xiao Wang 等，Findings of EMNLP 2023 | O-LoRA 以近似正交低秩子空间减少顺序任务干扰。 |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](https://arxiv.org/abs/2404.00228) | Yan-Shuo Liang、Wu-Jun Li，CVPR 2024 | 低秩适配器与干扰约束。其主要图像持续学习设置与本文文本共享适配器设置不同。 |
| [SLoRA: Balancing Plasticity and Forgetting in Large Language Models for Continual Learning](https://aclanthology.org/2026.acl-long.247/) | Lina Yang、Yusheng Liao、Yanfeng Wang、Yu Wang，ACL 2026 | 删除第二任务更新的部分成分可改善第一任务表现，据此提出子空间去噪。旧稿引用的作者与年份相符。 |
| [Pareto Continual Learning](https://arxiv.org/abs/2503.23390) | Song Lai 等，2025 | 将稳定性与可塑性作为多目标问题。AAAI 网页本轮受访问限制，保留可访问的作者预印本作为内容依据，修订稿主线不依赖该文。 |
| [Detecting Edit Failures In Large Language Models: An Improved Specificity Benchmark](https://aclanthology.org/2023.findings-acl.733/) | Jason Hoelscher-Obermaier 等，Findings of ACL 2023 | CounterFact+ 检测旧特异性评估遗漏的编辑副作用，并使用 KL 散度。 |
| [Evaluating the Ripple Effects of Knowledge Editing in Language Models](https://aclanthology.org/2024.tacl-1.16/) | Roi Cohen、Eden Biran、Ori Yoran、Amir Globerson、Mor Geva，TACL 2024 | 编辑应传播到逻辑相关事实，同时保留不应改变的事实。 |
| [Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://aclanthology.org/2024.findings-acl.902/) | Akshat Gupta、Anurag Rao、Gopala Anumanchipalli，Findings of ACL 2024 | ROME、MEMIT 的多次写入可出现渐进遗忘与突发失效，不能从单次编辑有效性推出长期稳定性。 |
| [PRISM Edit: One Vector for All Temporal Answers](https://aclanthology.org/2026.emnlp-main.1719/) | Chen Huang、Qi Zheng、Zhengruiqin、Long Zeng、Yuantong Xu，EMNLP 2026 | 不同时间上下文中的旧、新答案可以同时有效。作者列表按正式页面保留。 |
| [CHRONOEXPERT](https://aclanthology.org/2026.findings-emnlp.1823/) | Xudong Li 等，Findings of EMNLP 2026 | 时间事实链中，最新写入可能覆盖历史值。其独立专家与时间路由是已有方法。 |
| [Continual evaluation for lifelong learning: Identifying the stability gap](https://arxiv.org/abs/2205.13452) | **Matthias De Lange、Gido M. van de Ven、Tinne Tuytelaars，ICLR 2023** | 逐步评估揭示任务切换后的暂时性大幅失分。**旧稿误写为 Hess 等，现纠正。** |
| [Continual Learning for Sequential Personalization of Small Language Models: A Stability Monitoring Analysis](https://arxiv.org/abs/2606.27634) | Thomas S. Paula、Lucas S. Kupssinskü、Rodrigo C. Barros，2026 年预印本，v2 为 2026-09-12 | 阶段检查点上的任务分数和参考集 KL、熵、置信边际。本文不把常规分布监测视为原创。 |
| [STABLE: Gated Continual Learning for Large Language Models](https://arxiv.org/abs/2510.16089) | William Hoy、Nurcin Celik，2025 年预印本 | 以 EM、bits 或 KL 控制 LoRA 候选合并，超过预算时缩放或拒绝。 |
| [Spurious Forgetting in Continual Learning of Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a774503daed55eb53c634847ae071ec7-Abstract-Conference.html) | Junhao Zheng、Xidi Cai、Shengjie Qiu、Qianli Ma，ICLR 2025 | 合成人物事实召回和恢复训练用于区分任务对齐与知识保持。 |
| [Task Confusion and Catastrophic Forgetting in Class-Incremental Learning](https://papers.nips.cc/paper/2024/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html) | Milad Khademi Nori、II-Min Kim，NeurIPS 2024 | 类增量学习中的任务混淆与遗忘需要分别分析。本文显式给出注册表名称，任务设置有所不同。 |
| [In-Context Binding Capacity in Language Models](https://arxiv.org/abs/2609.30634) | Manas Venkata Sai Ravulapalli、Samrath Singh Chadha，2026 年预印本 | 静态上下文键值绑定的容量及不同归一化的解释。本文研究参数更新后的绑定行为。 |
| [When Forgetting is not Catastrophic: On the Mechanics of Spurious Forgetting](https://arxiv.org/abs/2610.08718) | Vedant Palit、Florent Draye、Nicolas Zucchet、Zhijing Jin、Bernhard Schölkopf，2026 年预印本 | 旧召回可以先崩溃、再恢复、再衰退。共享表示位移和归一化可解释可逆失访，并有表示与更新方向干预。本文的输出互信息回升本身不能替代这些机制证据。 |
| [Can a Language Model Continually Learn Facts in its Weights?](https://www.alphaxiv.org/abs/2607.11020.md) | Charles O’Neill、Max Kirkby、Jonathon Liu、Michael Psenka，2026 年预印本，v3 首页注明 NeurIPS 2026 workshop | 连续写入二十个虚构事实。Appendix F 在重复陈述训练、每个严格问题均失败的事实中记录 225/327（68.8%）错误回答含有**后续写入**事实内容。“后续”不限于最后一次写入。 |
| [Why Fine-Tuning Encourages Hallucinations and How to Fix It](https://arxiv.org/pdf/2604.15574) | Guy Kaplan、Zorik Gekhman、Zhen Zhu、Lotem Rozner、Yuval Reif、Swabha Swayamdipta、Derek Hoiem、Roy Schwartz，COLM 2026，见论文首页 | 新旧事实的语义表示重叠与局部干扰相关，关联记忆模型提供解释，自蒸馏可缓解。本文共享**字符串**键，并未实测隐藏表示重叠，不能把这一重叠直接当作数值 1。 |
| [When Context Changes: Understanding Update Failures in LLMs](https://arxiv.org/pdf/2609.38866) | Junyu Guo、Yuchen Fang、Shangding Gu、Costas Spanos、James Demmel、Javad Lavaei，2026 年预印本，v2 为 2026-10-29 | 上下文更新中的当前值、同变量旧值、其他变量值等错误分型，结合探针和注意力干预。本文采用相容的输出分类，但两套注册表同时有效，实验的“旧值”仍是旧注册表的正确答案。 |

修订稿中，注册表 JS 仅描述固定权重下的输入标签敏感性。它不能独自证明隐藏表示的平移、内部擦除、注意力漂移或因果路由机制。
