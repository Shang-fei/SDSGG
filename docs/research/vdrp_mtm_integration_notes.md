# VDRP 与当前 MTM 框架融合笔记

## 1. VDRP 这部分方法在做什么

VDRP 采用两阶段 HOI 框架：

1. 第一阶段用冻结 DETR 做 object / human 检测，得到每个 instance 的检测分数、类别 embedding 和 box。
2. 第二阶段用这些 instance 构造 human-object pair，并用冻结 CLIP 视觉编码器提取区域特征，再和 verb prompts 做匹配。

关键不是 DETR 本身，而是它把检测结果变成一种 **instance prior**：

```text
p = ProjDown([score; label_embedding; box])
```

这个 prior 被送入 CLIP visual transformer 中的 lightweight adapter。adapter 先把 patch features 降维，再让 patch features cross-attend 到 instance prior，最后升维并 residual 回原 CLIP feature：

```text
X'_i   = CrossAttn(ProjDown(X_i), p, p)
X_i+1 = X_i + ProjUp(X'_i)
```

因此，VDRP 不是直接微调整个 CLIP，而是用检测先验轻量调控 CLIP patch features。

之后 VDRP 从 CLIP feature map 中用 RoIAlign 提取三类区域特征：

```text
x_h: human region
x_o: object region
x_u: union/context region
```

union 特征还会结合 human/object features 和 box spatial encoding，得到带空间先验的 context / union relation feature。最后 human、object、context 三个区域分别和 verb prompt 表示做匹配。

## 2. 代码中可确认的具体机制

VDRP 代码里有几个值得关注的实现点：

- `Adapter`：插入到 CLIP visual transformer block 中，使用 bottleneck + cross-attention 到 instance prior。
- `get_prior`：把 detection score、box、object text embedding 等组合成 prior，再降维。
- `compute_roi_embeddings`：从 CLIP local feature map 中对 human、object、union boxes 做 RoIAlign。
- `spatial_head` / `spatial_fusion`：把 union visual feature 与 box spatial encoding 融合。
- `CueSelectorModule`：对 human/object/context concept bank 做 cosine matching，并用 sparsemax/softmax/top-k 选择相关 concept。
- `PromptUpdater`：把选中的 concept vector 加到 verb prompt embedding 上：

```text
prompt' = prompt + cue_scale * selected_concept
```

- `GaussianSampler`：用预先统计的 visual covariance 对 text embedding 加扰动，模拟 visual diversity。

这些点和我们当前讨论的 region-aware prompt augmentation、text distribution、MTM structure loss 是可以接上的。

## 3. 对当前 SDSGG / MTM 框架的启发

当前 SDSGG 里的 MTM 实际版本是：

```text
triplet text
-> frozen CLIP text encoder
-> SVD/common-component removal
-> refined triplet text teacher

relation union crop feature
-> relation MTM
-> predicted text-like embedding
```

MTM loss 包括：

```text
L_align: q_rel 对齐 refined triplet text teacher
L_visual_structure: 保持 MTM 前后视觉结构
L_text_structure: 让 q_rel 的相似度结构接近 text teacher
```

VDRP 给出的启发是：不要只依赖一个固定 triplet text teacher，可以让 teacher 具有 **区域感知能力**。但是这个区域感知不一定要通过大改 CLIP visual encoder 实现。

## 4. 最推荐的融合方式

### 4.1 不建议直接迁移 VDRP 的 DETR + CLIP adapter

当前 SDSGG 已经有自己的检测、pair construction、MVA 和 relation scoring 逻辑。直接把 VDRP 的 DETR 两阶段 HOI 框架搬进来，会破坏 baseline，并且工程量很大。

不建议第一版做：

```text
frozen DETR
CLIP visual transformer adapters
prior-guided patch extraction
完整 human-object HOI pipeline
```

### 4.2 建议迁移 region-aware prompt augmentation

最适合当前 MTM 的是 VDRP 的 concept bank + cue selector 思路。

对场景图关系，可以把 HOI 的 human/object/context 改成：

```text
subject branch
object branch
union / relation branch
```

每个 predicate 准备若干 visual concepts：

```text
C_sub[p]   = subject-side concepts of predicate p
C_obj[p]   = object-side concepts of predicate p
C_union[p] = relation/context concepts of predicate p
```

训练或推理时，对当前 pair 的视觉特征做检索：

```text
c_sub   = Select(sub_feature,   C_sub[p])
c_obj   = Select(obj_feature,   C_obj[p])
c_union = Select(rel_feature,   C_union[p])
```

然后增强文本 teacher：

```text
t_teacher = normalize(t_svd_triplet + beta_u * c_union)
```

第一版建议只用 `union / relation concept`，因为它最直接对应 predicate 判别；subject/object concept 后续再加。

### 4.3 保持 MTM 的结构保持定位

VDRP 的 region-aware prompt 可以增强 teacher，但 MTM 的核心仍然是 structure-preserving transfer。

融合后：

```text
region concept retrieval -> 增强 text teacher
MTM align loss           -> 对齐增强 teacher
MTM structure loss       -> 保持视觉关系空间结构
```

不能让 region-aware prompt 替代 MTM structure loss。否则 teacher 变成动态视觉调控目标，容易让 MTM 只追随当前样本，破坏原视觉空间的关系结构。

## 5. 和 SVD refined text teacher 的关系

当前 SVD 是：

```text
t_raw = CLIP(triplet text)
t_svd = normalize(t_raw - U_top U_top^T t_raw)
```

VDRP-style concept augmentation 可以接在 SVD 后面：

```text
t_teacher = normalize(t_svd + beta * c_union)
```

这样逻辑清楚：

- SVD 去掉 triplet text 中的公共主方向。
- concept augmentation 补充 predicate-relevant region concept。
- MTM structure loss 保持视觉几何结构。

## 6. 和 Bayesian Prompt Learning 的关系

如果后续加入 Bayesian Prompt Learning，VDRP 的 cue selector 可以作为 BPL 的条件信息来源，但第一版不要同时大改。

可能路径：

```text
relation feature -> cue selector -> selected concept
selected concept / relation feature -> Bayesian prompt posterior
sample residual prompt tokens
-> CLIP text encoder
-> SVD refine
-> MTM teacher
```

但最小可跑版本应先做：

```text
fixed concept bank + sparsemax/top-k retrieval + additive teacher enhancement
```

不要第一版就做 BPL token sampling、visual-conditioned posterior 和 candidate-wise dynamic text encoding。

## 7. 当前最小实现建议

第一版建议实现以下内容：

1. 离线准备 predicate-level relation concept bank。
   - 每个 predicate 5 到 10 条短视觉概念即可。
   - 用 frozen CLIP text encoder 编码并保存。

2. 构建 SGG pair prior，而不是 HOI human-object prior。
   - VDRP 的 prior 是 `[score; label embedding; box]`。
   - 在 SGG 中应改成：

```text
subject score / label / box
+ object score / label / box
+ union box
+ pair spatial encoding
+ optional relationness score
```

   - 该 prior 不建议送入 CLIP ViT blocks，第一版更适合作为 MTM adapter 或 teacher updater 的条件。

3. 在 MTM 训练正样本上检索 concept。
   - 输入：当前 positive pair 的 relation visual feature。
   - 对应 label 的 concept bank：`C_union[p]`。
   - 用 cosine + softmax/top-k/sparsemax 得到 `c_union`。

4. 构造增强 teacher：

```text
t_teacher = normalize(t_svd_triplet + beta * c_union)
```

5. MTM align target 从 `t_svd_triplet` 改为 `t_teacher.detach()`。

6. MTM visual structure loss 保持不变。

7. 推理时先做保守版本：
   - 每个候选 predicate 根据当前 relation feature 检索该 predicate 的 concept。
   - 构造 `t_teacher_candidate`。
   - 用 `cos(q_rel, t_teacher_candidate)` 得到 MTM score。

如果推理太慢，可以先只对 top-k baseline candidate 做 concept augmentation。

## 7.1 Subagent 补充：更贴近当前代码的迁移路线

subagent 的独立调研结论强调：VDRP 的 DETR-HOI 框架不适合直接迁移到 SDSGG，但它的 prior-guided adaptation 可以低侵入地迁移到 MTM。

更稳的路线是：

1. **不改 CLIP backbone**。
   - VDRP 在 CLIP ViT 多层插入 adapter，但当前 SDSGG 里已经有大量 crop encode、device、half precision 和 relation head 逻辑。
   - 第一版直接改 CLIP backbone 风险高。

2. **优先在 MTM 前后加 prior-conditioned adapter**。
   - 当前已有 `RelationModalityTransfer` 做视觉到文本空间映射。
   - 可以在 relation MTM 输入前，加入一个轻量 pair-prior gate：

```text
h_rel' = h_rel + Gate(pair_prior) * Adapter(h_rel)
```

   - 或者在 MTM 内部用 pair prior 做 cross-attention / FiLM-style modulation。

3. **把 VDRP 的 human/object/context 改成 subject/object/union**。
   - SGG 没有固定 human subject。
   - 所有 region-aware cue 都应以 arbitrary subject-object pair 为单位。

4. **保留 triplet text teacher**。
   - 不要把 VDRP 的 verb prompt 直接替换成 predicate prompt。
   - SDSGG 中 subject/object 对 predicate 判断影响很强，因此 teacher 仍应是：

```text
a photo of a {subject} {predicate} a {object}
```

   - 然后再做 SVD refine 和 region-aware augmentation。

5. **不要直接迁移 HICO covariance / Gaussian sampler**。
   - VDRP 的 covariance 是 HOI/HICO verb 分布。
   - 如果要用，必须重新在 VG/SDSGG 上统计 predicate/triplet visual distribution。

建议 ablation 顺序：

```text
baseline MTM
+ pair prior adapter
+ subject/object/union feature fusion
+ region-aware teacher
+ Bayesian prompt / distribution sampling
```

## 8. 风险

1. 推理成本会上升。
   - 每个 pair × 每个候选 predicate 都做 concept retrieval 会比较贵。
   - 可用 top-k candidate 或缓存 concept bank 降低成本。

2. concept quality 决定上限。
   - LLM 生成概念如果太抽象，可能无法提升 predicate 判别。
   - 概念应偏视觉短语，而不是解释性长句。

3. teacher 过度依赖视觉 feature。
   - 如果 concept retrieval 太强，text teacher 会变成 visual-conditioned target。
   - 因此 `beta` 要小，并且 MTM structure loss 必须保留。

4. subject/object 分支可能重新引入实体主导。
   - 第一版只做 union/relation concept 更稳。

## 9. 方法摘要版本

受 VDRP 的区域感知提示机制启发，我们在当前结构保持 MTM 框架中引入 relation-aware concept augmentation。具体而言，首先通过 frozen CLIP text encoder 和 SVD common-component removal 构建 refined triplet text teacher，以削弱通用文本成分对三元组表示的主导影响。随后，为每个 predicate 构建一组视觉关系概念，并根据当前 subject-object pair 的 union relation feature 检索最相关的 predicate concepts。检索得到的 region-aware concept 被用于增强 refined triplet teacher，从而得到更具样本适配性和关系判别性的文本目标。

MTM 仍然负责学习结构保持的跨模态迁移：一方面将视觉关系特征对齐到增强后的文本 teacher，另一方面通过 pairwise similarity structure loss 保持迁移前后视觉关系结构的一致性。该设计避免直接改动主干检测和 baseline relation classifier，同时利用外部视觉概念增强文本 teacher 的关系判别能力，适合作为当前 SDSGG/MTM 框架的低侵入式扩展。

## 10. 不建议直接照搬的部分

- 不建议直接替换为 VDRP 的 DETR-HOI pipeline。
- 不建议直接在 CLIP visual transformer 多层插入 adapter，第一版工程风险太高。
- 不建议同时引入 visual prior adapter、BPL sampling、concept bank 和 generator。
- 不建议先做 subject/object/context 三分支全量增强，容易让实体重新主导关系判断。

最稳妥的路线是：

```text
SVD refined triplet teacher
+ predicate union concept augmentation
+ MTM structure-preserving loss
```
