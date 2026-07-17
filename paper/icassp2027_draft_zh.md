# ICASSP 2027 中文草稿

## 暂定标题

面向场景图生成中新颖谓词识别的语义结构自适应方法

备选标题：

1. 面向新颖谓词场景图预测的低秩语义结构自适应方法
2. 结合语义结构对齐与特征生成的新颖谓词识别方法
3. 面向 PredCls 场景图生成的结构保持语义自适应方法

## 当前论文定位

本文研究的是场景图生成中 PredCls 设置下的新颖谓词识别问题。方法基于 SDSGG 进行改进，但不使用 APT。APT 作为 2026 ICLR 的外部 SOTA 对比方法出现在实验表中。当前论文的贡献需要谨慎表述：我们的方法在 Visual Genome 的 PredCls 设置下提升了 SDSGG，并超过了报告的 APT 结果；现阶段最强的证据集中在 novel predicate，而不是完整的 SGCls / SGDet 泛化能力。

## 核心主张

本文证明，在 SDSGG 框架中引入低秩语义自适应和结构保持的视觉-文本对齐，可以提升新颖谓词识别性能；进一步加入基于 SHIP 的新颖谓词特征生成，可以继续提升 novel predicate recall，但会带来可观察的 base-novel 性能权衡。

## 贡献点

1. 提出一种基于 SDSGG 的语义结构自适应框架，用于 PredCls 设置下的新颖谓词识别。
2. 引入低秩语义自适应模块和结构保持对齐目标，以增强 base predicate 到 novel predicate 的迁移能力。
3. 加入基于 SHIP 的新颖谓词特征生成机制，并通过消融实验分析其对 base 类和 novel 类的不同影响。
4. 在 Visual Genome PredCls 设置下，实验结果显示本文方法优于 SDSGG，并超过报告的 APT 结果。

## 摘要初稿

场景图生成需要模型识别目标对之间的视觉关系，但谓词类别通常呈现明显的长尾分布，新颖谓词由于监督样本不足而难以准确预测。本文关注 PredCls 设置下的新颖谓词识别问题，并基于 SDSGG 提出一种语义结构自适应框架。该方法结合低秩语义自适应、结构保持的视觉-文本对齐以及基于 SHIP 的新颖谓词特征生成。低秩语义自适应用于降低谓词语义表示中的冗余信息，结构对齐约束视觉关系特征保持文本语义空间中的邻域结构，而新颖谓词特征生成则为低频或未充分观测的谓词提供额外训练信号。在 Visual Genome 上的实验表明，本文方法相较 SDSGG baseline 取得明显提升，并在当前评估设置下超过报告的 APT 结果。消融实验进一步说明，SVD 语义自适应能够带来稳定的初始增益，结构对齐增强了表示一致性，而 SHIP 特征生成主要提升 novel predicate recall，同时引入一定的 base-novel 性能权衡。这些结果表明，显式建模谓词语义结构是提升场景图生成中新颖谓词识别能力的一条有效路径。

## 1. 引言初稿

场景图生成旨在将图像表示为由目标及其两两关系组成的结构化图。通过生成 subject-predicate-object 三元组，场景图为图像理解、视觉推理、图像检索以及视觉语言任务提供了重要的中间表示。尽管近年来场景图生成取得了持续进展，谓词识别仍然是其中的核心挑战。与目标类别相比，视觉谓词通常更加抽象、依赖上下文，并且类别分布高度不均衡。诸如 “on” 或 “near” 等高频关系在训练数据中占据主导，而许多具有实际语义价值的谓词只出现少量样本，甚至在训练划分中不可见。因此，模型往往容易学习到偏向高频谓词的判别边界，在 base predicates 上表现较好，但难以泛化到 novel predicates。

已有研究尝试通过语言先验、语义描述或开放词汇谓词表示来缓解这一问题。SDSGG 通过引入 scene-specific descriptions，将文本知识注入关系预测过程，从而提升谓词识别能力。然而，仅依赖文本引导的谓词表示并不能完全解决 base-to-novel 的迁移问题。谓词语义不仅由单个词的含义决定，还受到谓词之间结构关系以及 subject-object 视觉上下文兼容性的影响。如果学习到的视觉-文本映射不能保持这种语义结构，那么在 base predicates 上获得的提升未必能够可靠迁移到 novel predicates。

本文聚焦于 PredCls 设置下的新颖谓词识别。在该设置中，模型已知目标框和目标类别，需要预测目标对之间的谓词类别。我们基于 SDSGG 提出一种语义结构自适应框架。该框架包含三个主要模块。首先，我们利用基于奇异值分解（SVD）的低秩语义自适应，构建更加紧凑的谓词语义空间。其次，我们引入结构保持对齐目标，使视觉关系特征和文本谓词特征在邻域结构上保持一致。最后，我们使用基于 SHIP 的新颖谓词特征生成，为低频或未充分观测的 novel predicates 提供额外训练信号。

在 Visual Genome 上的实验结果表明，本文方法在 PredCls 设置下优于 SDSGG，并超过报告的 APT 结果。更重要的是，消融实验揭示了不同模块的作用差异。仅加入 SVD 的设置已经能够提升 baseline，说明低秩语义建模对谓词迁移是有效的；进一步加入 structure loss 后，表示结构的一致性得到增强；最终加入基于 SHIP 的新颖谓词生成后，novel predicate 的召回性能进一步提升，但 base predicate 性能会出现一定下降。这表明该模块带来的不是所有类别上的均匀提升，而是一种有意义的 base-novel 权衡。

本文贡献总结如下：

- 提出一种面向 SDSGG 场景图生成框架的新颖谓词语义结构自适应方法。
- 引入低秩语义自适应和结构保持的视觉-文本对齐，以增强 base-to-novel 谓词迁移。
- 加入基于 SHIP 的新颖谓词特征生成，并分析其对 base 类和 novel 类的不同影响。
- 在 Visual Genome PredCls 设置下，相较 SDSGG 和报告的 APT 结果取得提升，并通过消融实验证明各模块的有效性。

## 2. 相关工作提纲

### 2.1 场景图生成

需要简要介绍：

- 场景图生成预测 subject-predicate-object 三元组。
- PredCls 通过给定真实目标框和目标类别，将问题聚焦到谓词分类。
- 谓词长尾分布是影响场景图生成性能的重要因素。

### 2.2 长尾与新颖谓词识别

需要简要介绍：

- 谓词类别不平衡导致模型偏向高频 base predicates。
- novel predicate recognition 需要利用语义、上下文或结构信息进行迁移。
- 本文关注的是 base-to-novel predicate transfer，而不是完整 SGCls / SGDet 设置下的泛化。

### 2.3 文本引导与开放词汇场景图生成

需要简要介绍：

- 语言先验和文本描述可以为谓词识别提供语义知识。
- SDSGG 是本文的直接 baseline，因为它使用 scene-specific descriptions。
- APT 是近期 SOTA 对比方法，但本文方法不使用 APT。

## 3. 方法初稿提纲

### 3.1 问题定义

在 PredCls 设置下，给定图像中的真实目标框和目标类别，模型需要为每个候选 subject-object 对预测谓词类别。记目标对 \((i, j)\) 的视觉关系特征为 \(v_{ij}\)，谓词类别 \(p\) 的文本语义特征为 \(t_p\)。本文目标是在保持 base predicate 性能的同时提升 novel predicate 的召回率，尤其关注 base-to-novel 的语义迁移能力。

### 3.2 低秩语义自适应

第一个模块是基于 SVD 的低秩语义自适应。其动机是，原始谓词文本特征中可能包含冗余或噪声维度，而谓词之间可迁移的语义结构可以在一个更低维的子空间中得到表达。通过将谓词表示投影到低秩语义空间，模型能够获得更加紧凑的语义基，从而提升从 base predicates 向 novel predicates 的迁移能力。

后续需要从代码中补充：

- 输入语义矩阵的定义；
- SVD 分解形式；
- 保留的秩或主成分数量；
- 低秩语义特征如何参与关系预测。

### 3.3 结构保持对齐损失

第二个模块用于约束视觉关系特征和文本谓词特征保持一致的结构关系。与只对齐单个视觉特征和对应文本目标不同，structure loss 进一步鼓励视觉特征空间中的相似性结构与文本语义空间中的相似性结构一致。这样可以减少视觉-文本映射中的结构扭曲，使从 base predicates 学到的关系结构更容易迁移到 novel predicates。

后续需要从代码中补充：

- visual structure term；
- text structure term；
- 相似度或距离函数；
- loss 权重和 warmup 策略。

### 3.4 基于 SHIP 的新颖谓词特征生成

第三个模块是基于 SHIP 的新颖谓词特征生成。在训练过程中，模型根据 subject-object 兼容性和谓词文本信息采样 novel predicate，并生成相应的视觉关系特征。这些生成特征为低频或未充分观测的新颖谓词提供额外监督信号。预期效果是提升 novel predicate recall，但同时可能对 base predicate 的稳定性造成一定影响。

后续需要从代码中补充：

- novel predicate candidate 的采样方式；
- generator 的输入和输出；
- reconstruction / KL / alignment loss；
- pseudo ratio 和 ramp schedule。

### 3.5 训练目标

完整训练目标由 SDSGG 原始关系预测损失和本文提出的自适应/生成损失组成：

\[
\mathcal{L} = \mathcal{L}_{rel} + \lambda_s \mathcal{L}_{structure} + \lambda_g \mathcal{L}_{ship}.
\]

其中，SVD only 设置去掉 structure loss 和 SHIP loss，仅保留低秩语义自适应；SVD + Structure 设置保留 SVD 和 structure loss，但关闭 SHIP 生成；完整模型同时启用 SVD、structure loss 和 SHIP-based novel generation。

## 4. 实验初稿提纲

### 4.1 数据集与评估设置

本文在 Visual Genome 数据集上的 PredCls 设置进行实验。根据当前实验设置，谓词类别被划分为 base 类和 novel 类。我们分别报告 base 和 novel predicates 上的 Recall@20、Recall@50 和 Recall@100。

需要补充：

- Visual Genome 的具体划分；
- base / novel predicate 数量；
- 当前表格中左右两组指标的准确含义；
- 训练细节、学习率、epoch / iteration、batch size、checkpoint 选择方式。

### 4.2 对比方法

实验对比包括：

- SDSGG：原始 baseline，也是本文实现的基础框架。
- APT：2026 ICLR 的外部 SOTA 对比方法。
- SVD only：仅加入 SVD 低秩语义自适应，去掉 structure loss 和 SHIP loss。
- SVD + Structure：加入 SVD 和 structure loss，但关闭 SHIP-based novel generation。
- Full Model：同时启用 SVD、structure loss 和 SHIP-based novel generation。

### 4.3 主实验结果

当前结果可以这样解释：

- 完整模型在多数 novel predicate 指标上优于 SDSGG 和报告的 APT。
- SVD only 已经超过 SDSGG，说明低秩语义自适应本身有效。
- SVD + Structure 在 base 类上表现更强，说明 structure loss 有助于稳定表示学习。
- 完整模型在 novel 类上提升最明显，说明 SHIP-based novel generation 主要贡献于 novel predicate 泛化。

### 4.4 消融实验

建议论文表格命名如下：

| Method | SVD | Structure Loss | SHIP / Novel Generation |
|---|---|---|---|
| SDSGG | No | No | No |
| SVD only | Yes | No | No |
| SVD + Structure | Yes | Yes | No |
| Full Model | Yes | Yes | Yes |

需要把当前表中的 base / novel R@20、R@50、R@100 数字整理进来。

### 4.5 讨论与局限性

需要主动说明：

- 当前实验主要集中在 VG PredCls 设置。
- 尚未充分验证 SGCls 和 SGDet。
- 如果没有完成 GQA overlap 实验，则跨数据集泛化仍需后续验证。
- SHIP-based generation 提升 novel recall，但可能降低 base predicate 性能，体现出 base-novel trade-off。

## 5. 结论初稿

本文提出了一种面向 PredCls 场景图生成中新颖谓词识别的语义结构自适应框架。该方法基于 SDSGG，结合低秩语义自适应、结构保持的视觉-文本对齐以及基于 SHIP 的新颖谓词特征生成。在 Visual Genome 上的实验表明，本文方法优于 SDSGG，并在当前评估设置下超过报告的 APT 结果。消融实验说明，SVD 提供了有效的语义基础，structure loss 增强了表示一致性，而 SHIP-based generation 主要提升 novel predicate recall。未来工作将进一步扩展到 SGCls、SGDet 以及更广泛的跨数据集泛化评估。

## 写作 TODO

- [ ] 从代码中补充 SVD、structure loss、SHIP loss 的准确公式。
- [ ] 整理当前实验表格，并统一命名为 SVD only / SVD + Structure / Full Model。
- [ ] 明确当前表中 base 类和 novel 类左右两组指标的具体含义。
- [ ] 决定是否补 GQA overlap 或新的 VG split。
- [ ] 核对 SDSGG、APT、Visual Genome、SGG evaluation metrics 的引用信息。
- [ ] 在实验结果稳定后压缩成 ICASSP 4 页英文正式稿。
