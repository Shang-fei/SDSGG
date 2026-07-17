# ICASSP 2027 Draft

## Working Title

Semantic-Structural Adaptation for Novel Predicate Recognition in Scene Graph Generation

Alternative titles:

1. Low-Rank Semantic-Structural Adaptation for Novel Predicate Scene Graph Prediction
2. Improving Novel Predicate Recognition with Semantic Structure Alignment and Feature Generation
3. Structure-Preserving Semantic Adaptation for Predicate Classification in Scene Graph Generation

## Current Positioning

This paper studies novel predicate recognition under the PredCls setting of scene graph generation. The method is built on SDSGG, but does not incorporate APT. APT is used as an external recent SOTA comparison. The current contribution should be stated conservatively: the proposed modules improve SDSGG and outperform the reported APT numbers on the evaluated Visual Genome PredCls setting, with stronger evidence on novel predicates than on general SGG settings.

## Core Claim

We show that low-rank semantic adaptation and structure-preserving visual-text alignment can improve novel predicate recognition in SDSGG-based PredCls scene graph generation, and that SHIP-based novel predicate generation further improves novel recall with a measurable base-novel trade-off.

## Contributions

1. We propose a semantic-structural adaptation framework built on SDSGG for novel predicate recognition under the PredCls setting.
2. We introduce a low-rank semantic adaptation module and a structure-preserving alignment objective to improve transfer from base predicates to novel predicates.
3. We incorporate SHIP-based novel predicate feature generation and provide ablation analysis showing that it mainly benefits novel predicates while introducing a base-novel trade-off.
4. Experiments on Visual Genome show that the proposed method improves over SDSGG and outperforms the reported APT results in the evaluated setting.

## Abstract Draft

Scene graph generation requires models to recognize visual relationships between object pairs, yet predicate distributions are highly long-tailed and novel predicates remain difficult to predict from limited supervision. This paper studies novel predicate recognition under the PredCls setting and proposes a semantic-structural adaptation framework built on SDSGG. The proposed method combines low-rank semantic adaptation, structure-preserving visual-text alignment, and SHIP-based novel predicate feature generation. Low-rank adaptation reduces redundancy in predicate semantics, structure alignment encourages visual relation features to preserve semantic neighborhood structure, and novel feature generation provides additional support for under-observed predicates. Experiments on Visual Genome show that the proposed method improves over the SDSGG baseline and outperforms the reported APT results in the evaluated setting. Ablation studies further show that SVD-based semantic adaptation provides a strong initial improvement, structure alignment strengthens the learned representation, and SHIP-based generation mainly improves novel predicate recall while introducing a measurable base-novel trade-off. These results suggest that explicitly modeling semantic structure is a practical direction for improving novel predicate recognition in scene graph generation.

## 1. Introduction Draft

Scene graph generation (SGG) aims to represent an image as a structured graph of objects and their pairwise relationships. By converting visual content into subject-predicate-object triplets, SGG provides an intermediate representation for image understanding, visual reasoning, image retrieval, and downstream vision-language tasks. Despite substantial progress, predicate recognition remains a central challenge. Unlike object categories, visual predicates are often abstract, context-dependent, and highly imbalanced. Frequent relations such as "on" or "near" dominate training data, while many semantically meaningful predicates appear rarely or are absent from the training split. As a result, models often learn biased decision boundaries that perform well on base predicates but generalize poorly to novel predicates.

Recent methods have attempted to address this issue by introducing language priors, semantic descriptions, or open-vocabulary predicate representations. SDSGG uses scene-specific descriptions to improve predicate recognition by injecting textual knowledge into the relation prediction process. However, text-guided predicate representations alone do not fully solve the transfer problem. Predicate semantics are not only determined by individual word meanings, but also by the structural relationships among predicates and by their compatibility with subject-object visual contexts. If the learned visual-text mapping fails to preserve this semantic structure, improvements on base predicates may not translate into reliable novel predicate recognition.

This paper focuses on improving novel predicate recognition under the PredCls setting, where object labels and boxes are given and the model predicts predicates between object pairs. We build on SDSGG and propose a semantic-structural adaptation framework. The framework contains three components. First, we apply low-rank semantic adaptation based on singular value decomposition (SVD) to obtain a more compact predicate semantic space. Second, we introduce a structure-preserving alignment objective that encourages visual relation features and textual predicate features to maintain consistent neighborhood structure. Third, we use SHIP-based novel predicate feature generation to provide additional training signals for under-observed predicates.

Our experiments on Visual Genome show that the proposed method improves over SDSGG and outperforms the reported APT results in the evaluated PredCls setting. More importantly, the ablation results reveal distinct roles of the proposed components. SVD-only adaptation already improves the baseline, suggesting that low-rank semantic modeling is useful for predicate transfer. Adding structure alignment further improves representation quality, while SHIP-based novel generation provides the strongest gains on novel predicates but can slightly reduce base predicate performance. This indicates a meaningful base-novel trade-off rather than a uniform gain across all predicate categories.

The contributions of this work are summarized as follows:

- We propose a semantic-structural adaptation framework for novel predicate recognition in SDSGG-based scene graph generation.
- We introduce low-rank semantic adaptation and structure-preserving visual-text alignment to improve base-to-novel predicate transfer.
- We incorporate SHIP-based novel predicate feature generation and analyze its effect on base and novel predicates.
- We report improvements over SDSGG and the reported APT results on Visual Genome PredCls, with ablation studies clarifying the contribution of each component.

## 2. Related Work Outline

### 2.1 Scene Graph Generation

Need to cover classic SGG formulation and evaluation protocols, especially PredCls. Keep this short because ICASSP page budget is limited.

Key points to write:

- SGG predicts subject-predicate-object triplets.
- PredCls isolates predicate classification by assuming ground-truth object boxes and labels.
- Long-tailed predicate distribution remains a major challenge.

### 2.2 Long-Tailed and Novel Predicate Recognition

Key points to write:

- Predicate imbalance causes models to overfit frequent/base predicates.
- Novel predicate recognition requires transfer from semantic or contextual information.
- Current paper focuses on base-to-novel predicate transfer rather than full SGCls/SGDet generalization.

### 2.3 Text-Guided and Open-Vocabulary SGG

Key points to write:

- Language priors and textual descriptions provide semantic knowledge.
- SDSGG is the direct baseline because it uses scene-specific descriptions.
- APT is a recent SOTA comparison, but our method does not use APT.

## 3. Method Draft Outline

### 3.1 Problem Formulation

Given an image with ground-truth object boxes and object labels, the PredCls task predicts a predicate label for each candidate subject-object pair. Let a relation instance be represented by visual relation feature \(v_{ij}\) for object pair \((i, j)\), and let predicate textual features be denoted as \(t_p\) for predicate class \(p\). The goal is to improve prediction over both base and novel predicate categories, with particular emphasis on novel predicate recall.

### 3.2 Low-Rank Semantic Adaptation

The first component applies SVD-based low-rank adaptation to predicate semantic representations. The motivation is that predicate textual features may contain redundant or noisy dimensions, while the transferable structure among predicates can be captured in a lower-dimensional semantic subspace. By projecting predicate representations into this low-rank space, the model obtains a compact semantic basis for base-to-novel transfer.

To be completed with exact notation from implementation:

- input semantic matrix;
- SVD decomposition;
- retained rank or selected components;
- how adapted semantic features are used in relation prediction.

### 3.3 Structure-Preserving Alignment Loss

The second component constrains visual relation features and textual predicate features to preserve consistent structure. Instead of only aligning individual visual features to their corresponding text targets, the structure loss encourages pairwise or neighborhood relationships in the visual feature space to match those in the semantic feature space. This is intended to reduce distortions in the learned mapping and improve transfer to novel predicates.

To be completed with exact loss definition:

- visual structure term;
- text structure term;
- normalization and distance/similarity function;
- loss weight and warmup schedule.

### 3.4 SHIP-Based Novel Predicate Generation

The third component introduces SHIP-based novel predicate feature generation. During training, generated features are sampled for novel predicates using subject-object compatibility and textual predicate information. These generated features provide additional supervision for predicates that are under-observed or absent from the base training set. The expected effect is stronger novel predicate recall, although it may introduce a trade-off with base predicate stability.

To be completed with implementation details:

- how novel predicate candidates are sampled;
- generator input and output;
- reconstruction / KL / alignment losses;
- pseudo ratio and ramp schedule.

### 3.5 Training Objective

The full objective combines the original SDSGG relation prediction loss with the proposed adaptation and generation losses:

\[
\mathcal{L} = \mathcal{L}_{rel} + \lambda_s \mathcal{L}_{structure} + \lambda_g \mathcal{L}_{ship}.
\]

For ablation, the SVD-only setting removes both structure and SHIP losses; the w/o SHIP setting keeps SVD and structure loss but disables SHIP-based novel generation; the full model enables all components.

## 4. Experiments Draft Outline

### 4.1 Dataset and Evaluation Setting

We evaluate on Visual Genome under the PredCls setting. Following the base/novel split used in our experimental setup, predicates are divided into base and novel categories. We report Recall@20, Recall@50, and Recall@100 for base and novel predicates.

Need to fill:

- exact VG split;
- number of base and novel predicates;
- whether evaluation includes left and right table groups from the current results;
- implementation details and training schedule.

### 4.2 Baselines

We compare with:

- SDSGG: the original baseline used as the foundation of our implementation.
- APT: a recent SOTA method reported as an external comparison.
- SVD only: our method with only SVD-based semantic adaptation, removing structure loss and SHIP loss.
- SVD + Structure: our method without SHIP-based novel generation.
- Full Model: SVD + structure alignment + SHIP-based novel predicate generation.

### 4.3 Main Results

Current result interpretation:

- The full model improves over SDSGG and reported APT on most novel predicate metrics.
- SVD-only already improves over SDSGG, indicating that low-rank semantic adaptation is useful.
- SVD + Structure improves base performance strongly, suggesting that structure alignment stabilizes learned representations.
- The full model improves novel performance most clearly, showing the benefit of SHIP-based novel generation.

### 4.4 Ablation Study

Recommended table naming:

| Method | SVD | Structure Loss | SHIP / Novel Generation |
|---|---|---|---|
| SDSGG | No | No | No |
| SVD only | Yes | No | No |
| SVD + Structure | Yes | Yes | No |
| Full Model | Yes | Yes | Yes |

Need to include the actual base and novel R@20/R@50/R@100 values from the current table.

### 4.5 Discussion and Limitations

Important limitations to state honestly:

- Current experiments focus on VG PredCls.
- SGCls and SGDet are not yet validated.
- Cross-dataset generalization is not fully established unless GQA overlap experiments are completed.
- SHIP-based generation improves novel recall but may reduce base predicate performance, showing a base-novel trade-off.

## 5. Conclusion Draft

This paper presented a semantic-structural adaptation framework for novel predicate recognition in PredCls scene graph generation. Built on SDSGG, the method combines low-rank semantic adaptation, structure-preserving visual-text alignment, and SHIP-based novel predicate generation. Experiments on Visual Genome show that the proposed method improves over SDSGG and outperforms the reported APT results in the evaluated setting. Ablation studies indicate that SVD provides a useful semantic basis, structure alignment improves representation consistency, and SHIP-based generation mainly benefits novel predicate recall. Future work will extend the evaluation to SGCls, SGDet, and broader cross-dataset generalization settings.

## Writing TODO

- [ ] Fill exact method notation from implementation.
- [ ] Add current result table.
- [ ] Decide whether to include GQA overlap or alternative VG split.
- [ ] Verify APT citation and exact reported setting.
- [ ] Add citations for SDSGG, APT, Visual Genome, and SGG evaluation metrics.
- [ ] Compress to ICASSP 4-page format after results are finalized.
