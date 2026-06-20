# Primitive-Based Relation Generation Notes

## Motivation

The core assumption is that predicates are not independent labels. Many predicates share latent semantic factors:

- `on`, `standing on`, `sitting on`, `laying on`: contact, support, spatial layout.
- `riding`, `mounted on`, `sitting on`: contact, support, posture/motion.
- `wearing`, `attached to`, `covering`: contact, attachment, coverage.

The method should therefore model predicates as compositions of shared semantic primitives, then transfer the learned primitive-to-visual mapping from base predicates to novel predicates.

## Related Directions

### Predicate Correlation Learning

PCL and PCPL argue that SGG predicates have semantic overlap and long-tailed distributions. They model predicate-predicate correlation instead of treating each predicate as independent.

Useful idea:

- Build a predicate correlation matrix from semantic similarity, co-occurrence, or primitive overlap.
- Use it as a training constraint so related predicates share information and unrelated predicates stay separable.

For this project:

```text
target_sim(predicate_i, predicate_j) = Jaccard(primitive_slots_i, primitive_slots_j)
L_graph = MSE(cos(e_pred_i, e_pred_j), target_sim)
```

This directly expresses the hypothesis that predicate relations come from shared primitives.

### Predicate Similarity / Fine-Grained Groups

Some SGG work observes that rare predicates are often confused with similar frequent predicates, e.g. `parked on` vs `on`, `covered in` vs `in`. They group similar predicates and focus on fine-grained classification within each group.

Useful idea:

- Treat primitive sharing as a way to form related-predicate groups.
- Add hard negatives inside each group.

For this project:

```text
hard negatives = same subject-object + different predicate
or
hard negatives = high primitive-overlap predicates
```

This should improve predicate discrimination, which current reconstruction-only training lacks.

### Language Priors and Knowledge Distillation

VRD work uses language priors or external linguistic knowledge to regularize visual relation prediction. The main insight is that subject-object-predicate combinations have strong semantic regularities, and these priors help zero-shot or low-shot relationships.

Useful idea:

- Use CLIP/LLM text as semantic anchors, not random primitive vectors.
- Keep primitive semantics tied to natural-language descriptions.

For this project:

```text
primitive_anchor_k = CLIP_text("one object supports another")
primitive_embed_k = normalize(primitive_anchor_k + learnable_delta_k)
L_anchor = 1 - cos(primitive_embed_k, primitive_anchor_k)
```

This makes primitive labels more defensible and improves open-vocabulary transfer.

### Composite Visual Cues

RECODE decomposes visual relations into multiple cues such as subject, object, and spatial/relation descriptions, then fuses them. This is close to the desired primitive idea.

Useful idea:

- Do not use a single predicate word only.
- Decompose relation semantics into multiple primitive/cue components.

For this project:

```text
predicate -> primitive slots
subject/object -> object semantics
primitive slots -> relation-factor semantics
generator condition = fuse(subject_object_cond, primitive_cond)
```

### HOIGen-Style Feature Generation

HOIGen trains a VAE-like generator to reconstruct CLIP image features through frozen CLIP text encoder prompts. It uses class labels in the prompt and trains:

```text
image crop -> CLIP image feature
feature -> Encoder -> z
z -> Generator -> prompt bias
prompt bias + learnable context + class name -> CLIP text encoder
loss = feature reconstruction + KL
```

Useful idea:

- Generate CLIP-compatible visual prototypes.
- Use Stage II MLP if the downstream detector feature space differs from CLIP space.

Limitation for this project:

- HOIGen does not explicitly model predicate semantic factorization.
- Directly copying it gives class-conditioned generation, not primitive-conditioned generation.

## Recommended Method

Use primitives as explicit generator conditions, not only as prompt-prefix tokens.

### 1. Primitive Semantic Bank

Define 16 primitives with text anchors:

```text
contact, support, containment, attachment, motion, wearing/body, gaze,
part-whole, coverage, direction/path, proximity, vertical layout, etc.
```

Represent each primitive as:

```text
a_k = frozen CLIP_text(anchor_k)
d_k = learnable residual
p_k = normalize(a_k + d_k)
```

This gives both semantic anchoring and learnable task adaptation.

### 2. Predicate as Primitive Composition

Each predicate maps to a small set of primitive slots:

```text
on -> contact + support + vertical
riding -> contact + support + motion + agent
wearing -> contact + attachment + body
```

Predicate condition:

```text
e_pred = mean({p_k | k in slots(predicate)})
```

Mean pooling is enough for the first version. Attention pooling can be added later.

### 3. Subject-Object Condition

Use object semantics separately:

```text
e_so = CLIP_text("subject object")
```

This keeps relation semantics in primitives and object semantics in subject/object text.

### 4. Conditional Generator

First version should be deterministic or lightly variational:

```text
c = MLP([e_pred, e_so])
x_pred = Decoder([z, c])      # VAE version
or
x_pred = Decoder(c)           # deterministic baseline
```

Target:

```text
x_u = CLIP_image(subject-object union crop)
```

The decoder predicts a CLIP union visual feature:

```text
x_pred: [512]
```

This avoids making primitive tokens compete with natural-language tokens in the CLIP text prompt.

### 5. Losses

Use reconstruction, contrastive discrimination, and semantic-structure regularization:

```text
L = L_recon
  + alpha * L_global_nce
  + alpha_hard * L_hard_nce
  + gamma * L_graph
  + lambda_anchor * L_anchor
  + beta * L_kl       # only if VAE is used
```

Recommended first weights:

```text
alpha = 0.05
alpha_hard = 0.1
gamma = 0.05
lambda_anchor = 0.01
beta = 0.001
tau = 0.07
```

Important losses:

```text
L_recon = MSE(normalize(x_pred), normalize(x_u))
L_global_nce = CE((x_pred @ x_u.T) / tau, arange(B))
L_hard_nce = InfoNCE over same subject-object but different predicates
L_graph = MSE(cos(e_pred_i, e_pred_j), Jaccard(slots_i, slots_j))
L_anchor = 1 - cos(p_k, a_k)
```

## Evaluation

Use the following order:

1. Reconstruction:
   - base / novel cosine and MSE.

2. Predicate rank:
   - current union CLIP feature vs generated predicate prototypes.
   - report R@1/R@5/R@10/mean rank.
   - fix candidate set first: 15-class novel, then 50-class total.

3. Hard rank:
   - same subject-object candidate group.
   - high primitive-overlap candidate group.

4. Primitive controllability:
   - slot ablation: remove one primitive and measure rank/cosine change.
   - slot replacement: replace motion/support/contact and inspect top-k predicates.

5. Generation diversity:
   - pairwise cosine among samples for the same condition.
   - only important if claiming distributional generation; less important if using deterministic prototypes.

## Practical Recommendation

Do not continue expanding the current prompt-prefix VAE. Build a clean Stage1-v2:

```text
Primitive anchor bank
Predicate primitive composition
Subject-object condition
Conditional visual feature decoder
Reconstruction + hard contrastive + graph regularization
```

Start with a deterministic decoder before reintroducing VAE or normalizing flow. If deterministic prototypes improve predicate rank, then add VAE/Flow for diversity.

## Sources

- Predicate Correlation Learning for Scene Graph Generation, arXiv:2107.02713.
- PCPL: Predicate-Correlation Perception Learning for Unbiased Scene Graph Generation, arXiv:2009.00893.
- Unbiased Scene Graph Generation using Predicate Similarities, arXiv:2210.00920.
- Visual Relationship Detection with Language Priors, arXiv:1608.00187.
- Visual Relationship Detection with Internal and External Linguistic Knowledge Distillation, arXiv:1707.09423.
- Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models, arXiv:2305.12476.
- Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning, arXiv:2208.08165.
- Scene Graph Generation with Role-Playing Large Language Models, arXiv:2410.15364.
- HOIGen codebase: https://github.com/soberguo/HOIGen
