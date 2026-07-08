# Bayesian Prompt Learning 文本侧融合笔记

参考仓库：

- GitHub: https://github.com/saic-fi/Bayesian-Prompt-Learning
- 论文: Bayesian Prompt Learning for Image-Language Model Generalization, ICCV 2023
- 关键文件:
  - `trainers/vpt.py`: Bayesian / variational prompt learning 主实现
  - `trainers/coop.py`: 普通 CoOp soft prompt baseline

用户提供的图对应 `trainers/vpt.py`，不是普通 CoOp。

## 1. BPL / VPT 原实现核心

BPL 不是在最终 text embedding 上加噪声，而是在 **prompt token space** 中采样 residual / bias，然后再送入 frozen CLIP text encoder。

原始流程：

```text
image x
-> frozen CLIP image encoder
-> image feature f_x
-> posterior network pi_phi(f_x)
-> mu(x), logvar(x)
-> sample bias r ~ N(mu(x), sigma(x))
-> context prompt tokens ctx + r
-> concatenate [SOS, ctx+r, class name, EOS]
-> frozen CLIP text encoder
-> text embedding
```

代码对应：

```python
bias_mu, bias_logvar = self.meta_net(im_features)
bias = self.sample(bias_mu, bias_logvar, self.L)
ctx_shifted = ctx + bias
prompts = construct_prompts(ctx_shifted, prefix, suffix)
text_features = self.text_encoder(prompts, tokenized_prompts)
```

其中 `L` 是采样次数。训练时对 `L` 个 sampled prompts 的分类结果做 log-mean-exp 聚合。

## 2. Posterior network

代码里的 posterior 是 `Amortized`：

```python
class Amortized(nn.Module):
    self.weight_mean = InferenceBlock(...)
    self.weight_log_variance = InferenceBlock(...)
```

输入是 image feature，输出：

```text
mu:     [B, ctx_dim]
logvar: [B, ctx_dim]
```

注意原实现输出的是每个样本一个 `ctx_dim` bias，而不是每个 token 一个独立分布。然后这个 bias 会 broadcast 到所有 context tokens：

```text
ctx:        [n_ctx, ctx_dim]
bias:       [L, B, ctx_dim]
ctx_shifted [L, B, n_ctx, ctx_dim]
```

所以它是 **shared token bias**，不是 `[n_ctx, ctx_dim]` 的 full token-wise posterior。第一版迁移时保持这个简单形式即可。

## 3. Sampling

原实现采样：

```python
shape = (L,) + mu.size()
eps = torch.randn(shape).type_as(mu)
bias = mu.unsqueeze(0) + eps * logvar.exp().sqrt().unsqueeze(0)
```

即：

```text
r = mu + eps * sigma
sigma = exp(0.5 * logvar)
```

注意这里 `logvar` 是 log variance，不是 std。

## 4. KL regularization

原实现使用标准正态先验：

```text
p(r) = N(0, I)
q(r|x) = N(mu(x), sigma(x))
```

KL：

```python
post = Normal(mu, exp(0.5 * logvar))
prior = Normal(0, 1)
KL(post || prior).mean(dim=-1)
```

训练 loss：

```python
task_loss + 0.001 * KL
```

迁移到 MTM 时，应保留 KL，但权重需要小，例如：

```text
MTM.BPL_KL_WEIGHT = 1e-3
```

## 5. 与 CoOp 的区别

`trainers/coop.py` 是普通 soft prompt：

```text
learnable ctx tokens + class name -> CLIP text encoder
```

没有 posterior、没有采样、没有 KL。

BPL / VPT 是：

```text
learnable ctx tokens + sampled residual/bias + class name -> CLIP text encoder
```

并且 residual/bias 可以由图像条件生成。

## 6. 迁移到当前 SDSGG / MTM 的建议

当前 MTM 文本侧是：

```text
triplet text
-> frozen CLIP text encoder
-> optional SVD refine
-> target text embedding
```

BPL 加入后，建议变成：

```text
relation visual condition c
-> BPL posterior
-> mu(c), logvar(c)
-> sample bias r

[SOS] + soft ctx + r + triplet label tokens + [EOS]
-> frozen CLIP text encoder
-> sampled triplet text features
-> aggregate to teacher
```

其中 condition 第一版建议用 MTM 的 relation feature：

```text
c = x_mtm.detach()
```

也就是当前已经实现的：

```text
image -> CLIP patch feature map -> RoIAlign -> MTMUnionGateFusion -> x_mtm
```

这里要 detach，避免 text teacher 的 posterior loss 反向破坏 MTM / baseline 视觉特征。

## 7. 在当前任务中的 prompt 形式

原 BPL 是 image classification，class name 是单个类别。

当前 SGG 应改成 triplet label：

```text
"a photo of a {subject} {predicate} a {object}"
```

更适合的 token 结构：

```text
[SOS] [ctx + r] [triplet label tokens] [EOS]
```

第一版不要加入整图 entity scene prior，避免文本被实体共现重新主导。

## 8. Teacher 聚合方式

如果采样 `L` 次，可以得到：

```text
t_1, ..., t_L
```

第一版建议：

```text
t_teacher = normalize(mean_l normalize(t_l))
```

MTM align：

```text
L_align = 1 - cos(q_rel, t_teacher.detach())
```

后续如果要用分布信息，可以记录：

```text
sigma_text = std_l(t_l)
```

但第一版不建议直接让 align loss 对齐方差。

## 9. SVD 应该怎么处理

当前 SVD basis 是从原始 triplet template 上统计的：

```text
"a photo of a subject predicate a object"
```

BPL 加入后，文本 embedding 分布变成：

```text
soft ctx + sampled residual + triplet label
```

因此旧 SVD basis 不一定可靠。

建议第一版配置上允许关闭 BPL teacher 的 SVD：

```text
MTM.BPL_USE_SVD = False
```

如果要使用 SVD，正确方式是：

```text
BPL sampled prompt -> CLIP text encoder -> SVD refine with fixed basis
```

但更严谨的做法是用新模板重新统计 SVD basis。第一版先不做。

推荐第一版：

```text
BPL teacher 不走 SVD
原始 raw triplet teacher / debug 可以继续保留 SVD
```

## 10. 推理方式

推理时对每个 candidate triplet：

```text
condition = x_mtm for current pair
posterior -> mu/logvar
sample or use mean bias
prompt = ctx + bias + candidate triplet label
text feature = CLIP text encoder(prompt)
score_mtm = cos(q_rel, text_feature)
```

为控制开销，第一版推理建议：

```text
use posterior mean only, no multi-sample
```

即：

```text
r = mu
```

训练可采样 `L=2` 或 `L=4`。

## 11. 与 MTM structure loss 的关系

BPL 只改变 text teacher。

MTM 仍然是：

```text
x_mtm -> relationMtm -> q_rel
```

结构保持仍然计算：

```text
S_visual = cos(adapted_visual_features)
S_pred   = cos(q_rel)
L_visual_structure = D(S_visual, S_pred)
```

不要让 BPL 替代 MTM structure loss。

## 12. 不建议做的事

- 不要在最终 text embedding 上直接加 Gaussian noise。
- 不要一开始做 token-wise `[n_ctx, ctx_dim]` posterior，参数和不稳定性都更高。
- 不要让 BPL condition 的视觉特征不 detach。
- 不要直接复用旧 triplet-only SVD basis 作为强约束。
- 不要第一版加入整图所有实体 scene prior。
- 不要同时改 baseline MVA。

## 13. 推荐最小实现

第一版实现：

```text
BPL_ENABLED
BPL_CTX_LEN
BPL_SAMPLE_NUM
BPL_KL_WEIGHT
BPL_USE_SVD = False
```

模块：

```text
BayesianTripletPromptLearner
  - learnable ctx: [n_ctx, 512]
  - posterior net: x_mtm.detach() -> mu/logvar [B, 512]
  - sample shared bias r: [L, B, 512]
  - ctx_shifted = ctx + r
  - concatenate with triplet token embeddings
  - frozen CLIP text encoder
  - aggregate sampled text features
```

训练 loss：

```text
L_total =
  existing MTM losses with BPL teacher
+ lambda_kl * KL(q(r|x_mtm.detach()) || N(0,I))
```

推理：

```text
use posterior mean r=mu
candidate triplet prompt -> BPL text feature
cos(q_rel, text_feature)
```

## 14. 一句话定位

BPL 在当前工作中的作用是：

```text
用 relation visual condition 生成 prompt-token residual distribution，
让 triplet text teacher 具有样本自适应和不确定性，
但仍通过 frozen CLIP text encoder 保持语言空间约束。
```

