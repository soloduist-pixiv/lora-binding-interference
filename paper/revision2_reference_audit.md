# 新增相关工作核对

依据原论文 arXiv 元数据、论文全文首页或官方会议页面核对。以下六篇已核实标识符；原稿其余引文的 DOI/ID 补齐仍待逐条完成，不把现有作者年份核对等同于完成全部 DOI 核对。

| 文献 | 作者及状态 | 标识符与原始来源 | 本文可引用的范围 |
|---|---|---|---|
| LoRA Learns Less and Forgets Less | Dan Biderman 等；TMLR 2024，arXiv v2 注明 2024-08 接收 | [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)，DOI:10.48550/arXiv.2405.09673 | LoRA 与全微调在数学、代码上的学习与域外保持存在权衡；不能把 lr/rank 对保持的影响表述为本文首次发现。具体学习率/rank 推荐需结合全文具体实验，不从摘要推出最优值。 |
| Understanding the Role of Training Regimes in Continual Learning | Seyed Iman Mirzadeh、Mehrdad Farajtabar、Razvan Pascanu、Hassan Ghasemzadeh；NeurIPS 2020 | [arXiv:2006.06958](https://arxiv.org/abs/2006.06958)，DOI:10.48550/arXiv.2006.06958；[官方论文集](https://papers.neurips.cc/paper_files/paper/2020/hash/518a38cc9a0173d0b2dc088166981cf8-Abstract.html) | 训练设置、学习率衰减、dropout、batch 大小影响遗忘，需避免把不匹配的训练设置误认为新机制。 |
| Experience Replay for Continual Learning | David Rolnick、Arun Ahuja、Jonathan Schwarz、Timothy P. Lillicrap、Greg Wayne；NeurIPS 2019 | [arXiv:1811.11682](https://arxiv.org/abs/1811.11682)，DOI:10.48550/arXiv.1811.11682 | 强化学习中经验回放、行为克隆与新旧样本混合的已有证据；其 Atari/DMLab 设置不直接证明本文的语言事实结果。 |
| LoRA+: Efficient Low Rank Adaptation of Large Models | Soufiane Hayou、Nikhil Ghosh、Bin Yu；2024，v2 为 2024-07-04 | [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)，DOI:10.48550/arXiv.2402.12354 | A/B 使用相同学习率会限制特征学习，提出不同学习率比例；本文固定同率的 rank 对照不等于优化后最优 LoRA。 |
| A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA | Damjan Kalajdzievski；2023 年预印本 | [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)，DOI:10.48550/arXiv.2312.03732 | rsLoRA 研究按 rank 的平方根缩放；本文固定有效 scale=2，不是 rsLoRA 或固定 alpha 的纯容量扫描，需注明优化尺度混淆。 |
| How do Language Models Bind Entities in Context? | Jiahai Feng、Jacob Steinhardt；ICLR 2024 | [arXiv:2310.17191](https://arxiv.org/abs/2310.17191)，DOI:10.48550/arXiv.2310.17191；[官方论文](https://openreview.net/pdf?id=zb3b6oKO77) | 通过激活因果干预识别实体—属性 binding ID；其上下文激活机制不能由本文输出 JS 或互信息直接推断。 |

这里列出的 10.48550 DOI 是 arXiv 版本标识，不冒充会议出版 DOI。摘要与正文应将“训练条件影响遗忘”和“实体绑定机制”作为已有工作背景。
