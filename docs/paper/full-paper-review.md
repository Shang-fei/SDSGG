# ICASSP Full-Paper Adversarial Review

## Reverse Outline

### Abstract

- **Task and gap:** OVSGG must recognize unseen predicates, while pointwise
  transfer may alter visual geometry and novel predicates lack visual support.
- **Method:** relation-preserving transfer combines a refined triplet teacher,
  VSP, and relation-conditioned pseudo visual features.
- **Evidence:** matched VG/GQA gains, APT comparison, and scoped ablation
  findings.

### Introduction

1. **Task and difficulty:** define OVSGG and explain why relations require
   fine-grained subject--object interaction cues.
2. **Challenge:** richer semantic prompts improve transfer targets but leave
   visual geometry and missing novel instances unresolved.
3. **Technical view:** formulate relation recognition as geometry-preserving
   cross-modal distribution completion and define its two concrete requirements.
4. **Framework:** DCR triplet teacher, projector/VSP, novel synthesis, and score
   fusion.
5. **Evidence and contribution:** report only matched measurements and connect
   the ablations to the two stated requirements.

### Related Work

1. **Language-guided OVSGG:** synthesize prompt/description methods and identify
   their semantic-target focus.
2. **Structured transfer and synthesis:** distinguish ACC's interaction mismatch
   and SHIP's category synthesis from relation-level geometry preservation.

### Method

1. **Setting and overview:** define PredCls inputs, SDSGG logits, and the added
   branch.
2. **Triplet teacher:** define contextual prompts and dominant-component
   removal.
3. **Pointwise transfer:** define adapter/projector and cosine alignment.
4. **VSP:** preserve off-diagonal pairwise cosine similarities from source
   visual features.
5. **Novel synthesis:** reconstruct base visual features and sample
   relation-conditioned pseudo features.
6. **Optimization/inference:** combine implemented losses and add triplet scores
   to SDSGG logits.

### Experiments

1. **Reproducibility:** state datasets, splits, metrics, and active
   hyperparameters.
2. **Primary comparison:** compare SDSGG, APT, and Ours on matched VG
   base/novel splits.
3. **Harder settings:** evaluate VG semantic transfer and GQA base/novel splits.
4. **Component evidence:** isolate VSP and NS, including the base--novel
   trade-off.
5. **Scope and pending evidence:** reserve DCR, seed, sensitivity, efficiency,
   and matched APT-GQA measurements without relying on them.

### Conclusion

- **Takeaway:** source-geometry preservation plus novel synthesis improves
  SDSGG-based open-vocabulary PredCls.
- **Boundary:** detection-dependent tasks and cross-baseline transfer are future
  scope, not established results.

## Claim--Evidence Map

Claim: The method improves SDSGG by 2.06 percentage points on average across 30
matched entries. | Evidence: six VG base, six VG novel, six VG semantic, six
GQA base, and six GQA novel R/mR measurements. | Status: supported

Claim: The method outperforms APT on all 12 matched VG base/novel metrics. |
Evidence: official APT values and Ours values in Table 1, including corrected
APT novel mR@100 of 32.3. | Status: supported

Claim: VSP benefits the completed VG base/novel evaluation. | Evidence: full
model exceeds w/o VSP on all 12 stored measurements, by 1.28 points on average.
| Status: supported

Claim: Novel synthesis improves novel predicates but creates a base--novel
trade-off. | Evidence: full model exceeds w/o NS on all six novel entries and
is lower on all six base entries. | Status: supported

Claim: DCR independently improves performance. | Evidence: the reserved w/o-DCR
row is empty. | Status: needs evidence

Claim: The method is robust to hyperparameters and random seeds. | Evidence:
reserved sensitivity and three-seed measurements are empty. | Status: needs
evidence

Claim: The added branch is efficient relative to APT. | Evidence: no matched
parameter/runtime measurement is available. | Status: needs evidence

Claim: VSP directly reduces measured geometric distortion. | Evidence: the
removal study supports performance, but the pairwise-similarity diagnostic is
not measured. The Introduction therefore states that pointwise alignment leaves
relative arrangement unconstrained and describes VSP by its optimization
objective; it does not claim a measured reduction in distortion. | Status:
needs evidence for a stronger causal claim

## Five-Dimension Review

### 1. Contribution

- **What is new?** A relation-level transfer objective that retains source
  visual geometry, coupled with relation-conditioned novel synthesis inside an
  SDSGG-based PredCls system. **Status: pass.**
- **Is the story merely a list of modules?** No; DCR, VSP, and NS are organized
  around two coupled transfer failures. **Status: pass.**
- **Could DCR appear unsubstantiated?** Yes; its independent row is empty and no
  claim attributes gains specifically to DCR. **Status: needs new experiment.**

### 2. Writing Clarity

- **Is the method reproducible?** The paper defines inputs, teacher refinement,
  alignment, VSP, generator objective, total loss, inference fusion, dimensions,
  layer counts, and weights. **Status: pass.**
- **Are terms stable?** DCR, VSP, NS, triplet teacher, and PredCls retain one
  meaning throughout. **Status: pass.**
- **Does each paragraph have one role?** The reverse outline maps every
  paragraph to a section thesis. **Status: pass.**

### 3. Experimental Strength

- **Are gains consistent?** All 30 SDSGG comparisons are positive across two
  datasets and multiple splits. **Status: pass.**
- **Are strong baselines included fairly?** APT is compared only on its matched
  VG protocol; incompatible GQA cells remain empty. **Status: pass.**
- **Is uncertainty reported?** Not yet; the reserved three-seed result must be
  completed before submission. **Status: needs new experiment.**

### 4. Evaluation Completeness

- **Are central modules ablated?** VSP and NS are; DCR is reserved but empty.
  **Status: needs new experiment.**
- **Are metrics and harder settings sufficient?** R/mR at three cutoffs, VG
  semantic transfer, and GQA novel transfer provide relevant evidence within
  PredCls. **Status: pass for the stated scope.**
- **Are practical cost and qualitative failure cases covered?** No common
  hardware cost or qualitative panel is yet available. **Status: needs new
  experiment, but no current claim depends on either.**

### 5. Method Design Soundness

- **Is the setting internally valid?** Ground-truth boxes/labels and novel-label
  exclusion match PredCls. **Status: pass.**
- **Are disabled components accidentally claimed?** Text structure, negative
  alignment, base replay, and relationness are excluded. **Status: pass.**
- **Does NS have a cost?** Yes; it improves novel results while lowering base
  results, and the paper discloses this trade-off. **Status: pass with stated
  limitation.**
- **Is robustness established?** No. The sensitivity rows remain reserved.
  **Status: needs new experiment.**

## Submission Risks and Required Actions

1. Fill the w/o-DCR row before making an independent DCR effectiveness claim.
2. Report three-seed mean and standard deviation for the full model and central
   removals.
3. Fill the matched APT-GQA row only if exactly the same split and SDSGG protocol
   can be reproduced; otherwise keep the explanation and omit numerical claims.
4. Add one compact sensitivity or efficiency result if page budget permits.
5. Replace the funding reservation with the authors' final statement.
6. Compile and visually inspect the paper; page-count compliance is not verified
   until page 4 ends with technical content and page 5 contains only references,
   Funding, and Compliance with Ethical Standards.
