# ICASSP 2027 Full-Paper Design

## Objective

Produce a complete, reviewer-facing English ICASSP manuscript in the supplied
LaTeX template. Technical content must end by page 4. Page 5 may contain only
references, funding acknowledgements, and the Compliance with Ethical
Standards statement.

## Paper Story

The paper studies open-vocabulary predicate classification rather than the
full SGCls or SGDet pipelines. Existing language-guided OVSGG methods improve
semantic targets but do not explicitly preserve the relational geometry of
visual instances during visual-to-text transfer, and novel predicates lack
annotated visual support. The proposed SDSGG-based framework addresses these
two coupled gaps through relation-preserving modality transfer and
relation-conditioned novel feature synthesis.

The method is presented as one coherent transfer framework rather than a set
of incremental patches:

1. A frozen triplet-text teacher removes its dominant shared component.
2. A learnable projector aligns union-region features with triplet semantics.
3. Visual structure preservation constrains pairwise cosine geometry before
   and after projection; text-structure loss remains disabled everywhere.
4. Relation-conditioned synthesis supplies pseudo visual support for unseen
   predicates.
5. Transfer scores complement the original SDSGG logits at inference.

## Four-Page Technical Budget

- **Page 1:** title, 100--150-word abstract, introduction, and compact related
  work. Introduction establishes task, gap, solution, evidence, and three
  contributions without claiming SGCls, SGDet, or cross-baseline transfer.
- **Page 2:** method overview and complete mathematical formulation. A compact
  pipeline figure is included only if an existing or newly created figure is
  readable within the budget; otherwise a single algorithm-flow equation is
  used.
- **Pages 3--4:** experimental setup, compact comparison tables, core
  ablation, challenging-setting results, limitations, and conclusion.
- **Page 5:** references followed only by funding acknowledgements and the
  Compliance with Ethical Standards statement. No technical prose, figures,
  tables, or conclusion may spill onto this page.

The page allocation is a target rather than an artificial page break after
each section. Final rendering decides the exact column breaks.

## Section Responsibilities

### Abstract

State the PredCls problem, the geometry/support gaps, the complete method, and
the strongest supported result. Keep it within 100--150 words and avoid
citations or unsupported causal language.

### Introduction

Use five compact paragraph roles: task and importance; prior progress and
remaining gap; technical hypothesis; proposed framework; evidence and
contributions. The evidence paragraph may report the verified 2.06-point
average improvement and matched VG comparison with APT.

### Related Work

Use two compact themes: language-guided OVSGG and structure-aware
cross-modal/feature transfer. Differentiate the present work by its explicit
preservation of source visual-relation geometry and relation-conditioned
novel support within PredCls.

### Method

Define notation and baseline first, then formalize triplet teacher refinement,
pointwise alignment, visual structure preservation, novel feature synthesis,
training loss, and inference fusion. Every symbol must be defined before use.
Do not introduce a text-structure objective as an active component.

### Experiments

Retain only information needed to reproduce and evaluate the central claims.
Comparison evidence covers VG base/novel, VG semantic, and the
SDSGG-compatible GQA base/novel setting. Core ablation covers visual structure
preservation and novel synthesis. DCR isolation, seed statistics, focused
hyperparameter sweeps, balanced metrics, efficiency, and qualitative analyses
remain visibly reserved where space permits, using empty cells or a compact
pending block rather than invented values.

### Conclusion

Summarize the scoped contribution and supported findings in one paragraph.
Explicitly retain the PredCls boundary and avoid promising universal transfer
to other baselines.

## Tables and Pending Results

The rendered draft prioritizes one compact VG comparison table, one compact
GQA/semantic table, and one core ablation table. Completed values are copied
exactly from verified sources or user-provided experiments. Missing required
measurements remain empty and are explained in captions as pending. If a
pending table causes technical content to exceed page 4, it is condensed into
reserved rows inside an existing table; it is not silently deleted.

Captions specify task, split, metrics, notation, and the meaning of empty
cells. Captions do not duplicate the full result discussion.

## Claim--Evidence Constraints

- **Overall improvement:** supported by 30 matched Ours--SDSGG entries, average
  absolute gain 2.06 percentage points.
- **Matched APT comparison:** supported only on the official VG base/novel
  PredCls protocol, where all 12 reported entries improve.
- **Visual structure preservation:** supported by the completed removal study
  on all 12 VG base/novel entries.
- **Novel synthesis:** supported as a novel-predicate improvement with an
  explicitly disclosed base--novel trade-off.
- **DCR, robustness, efficiency, balanced metrics, and qualitative mechanism
  evidence:** pending; reserve space but make no affirmative performance claim.
- **Scope:** no claims for SGCls, SGDet, multiple backbones, or transfer to
  other baselines.

## Formatting and Verification

Use the supplied `spconf` template, two columns, at least 9-point text, clean
booktabs tables, and stable terminology. The final workflow must:

1. run a reverse outline after each section;
2. validate every Abstract/Introduction claim against measured evidence;
3. run the five-dimension adversarial paper review;
4. compile the LaTeX when an engine is available;
5. render and visually inspect every page;
6. verify that technical content ends on page 4 and page 5 contains only the
   three permitted back-matter categories.

If a local LaTeX engine remains unavailable, static checks are reported as
such and page-count compliance is not claimed until a compiled PDF is
inspected.

## Self-Review

- No `TODO` or unspecified writing decision remains in this design.
- The visible-pending-results requirement is reconciled with the four-page
  limit by reserving compact rows rather than rendering multiple sparse tables.
- The paper story, method terminology, evidence scope, and experimental claims
  are consistent.
- The work is scoped to one manuscript and one verification workflow.
