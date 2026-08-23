# ICASSP Experiments and LaTeX Structure Design

## Objective

Refactor the ICASSP manuscript from one monolithic `Template.tex` into section- and table-level files, then write a reviewer-facing Experiments section that supports three questions: whether the method is stronger than comparable prior work, which components cause the gains, and how the method behaves in harder open-vocabulary cases.

The manuscript remains strictly scoped to open-vocabulary Predicate Classification (PredCls). It must not imply completed SGCls/SGDet experiments or portability across multiple unrelated baselines.

## LaTeX file organization

`ICASSP2026_Paper_Templates/Template.tex` will contain only the preamble, title/author block, document structure, section inputs, and bibliography declaration.

```text
ICASSP2026_Paper_Templates/
├── Template.tex
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── experiments.tex
│   └── conclusion.tex
├── tables/
│   ├── vg_base_novel.tex
│   ├── vg_semantic.tex
│   ├── gqa_base_novel.tex
│   ├── core_ablation.tex
│   └── focused_ablations.tex
└── figures/
    └── challenging_cases_placeholder.tex
```

Each section file will contain its `\section` command. Table files will contain complete floating environments so captions, labels, and table formatting remain colocated.

## Experiments outline

### 1. Experimental setup

State that the evaluation follows the SDSGG PredCls protocol. Ground-truth boxes and object labels are given. Visual Genome uses 35 base and 15 novel predicates and a 24-predicate semantic subset; GQA uses its SDSGG-compatible base/novel split. Report `R@K` and `mR@K` for `K \in \{20,50,100\}`. Training details that cannot be verified from the current run configuration will be explicitly marked in LaTeX comments rather than guessed.

### 2. Comparison experiments

Five comparison groups will be represented:

1. VG base/novel: CLS, Epic, SDSGG, APT, and Ours.
2. VG semantic: CLS, CLS-DE, RECODE variants, SDSGG, and Ours.
3. GQA base/novel: CLS, SDSGG, Ours, plus an empty APT row pending a matched-protocol reproduction.
4. Balanced base/novel performance: planned `F@K` columns or table entries left empty until the exact definition and values are confirmed.
5. Efficiency: trainable parameters, training time, and inference time left empty pending measurement.

Published values will come from the official SDSGG and APT papers. The APT VG-novel `mR@100` value will be 32.3, correcting the 31.1 transcription in the uploaded screenshot. APT's published GQA values will not be mixed with the current project because the reproduced SDSGG reference numbers indicate a protocol mismatch.

### 3. Ablation studies

The core ablation table will distinguish the actual configurations:

- SDSGG baseline;
- Ours without dominant-component removal (planned; empty);
- Ours without visual structure preservation (available);
- Ours without novel synthesis (available);
- full method (available).

Text-structure preservation will not appear as an enabled component because its loss weight has always been zero. Current rows will not be renamed as “SVD only” or “SVD + Structure,” because they do not isolate those factors.

Focused ablations will list but leave unmeasured entries empty:

- number of removed dominant components: 0, 1, 2, 4, 8;
- visual-structure loss weight and L1/L2 distance;
- pseudo-novel ratio and warm-up/ramp schedule;
- uniform versus low-recall novel-predicate sampling;
- inference fusion weight;
- triplet-text versus predicate-only teacher.

### 4. Challenging evaluation and demos

No new SGG task is assumed. Harder evidence will stay within the current paper scope:

- VG semantic predicates;
- GQA novel predicates;
- rare-predicate and near-synonym confusion subsets;
- visual-geometry preservation diagnostics;
- qualitative comparisons of SDSGG, ablations, and the full method.

A figure placeholder will reserve the qualitative layout without fabricating examples. Its caption will define row/column meanings and state the PredCls setting.

## Table and figure rules

- Use `booktabs`; do not use vertical rules.
- Place table captions above tables.
- Put metric direction arrows in headers.
- Use consistent one-decimal precision for reported means.
- Bold only the best completed value in a comparison group.
- Empty cells denote required experiments whose values are not yet available; each caption will state this explicitly.
- A dash denotes a metric not reported by the cited method, not an unfinished experiment.
- Captions state dataset, split, PredCls protocol, metrics, and notation, but do not duplicate result analysis.

## Result-writing policy

The prose will make only evidence-backed claims:

- Ours improves SDSGG on all 30 currently reported R/mR entries, with a 2.06-point average absolute gain.
- On matched VG base/novel evaluation, Ours exceeds official APT values on all 12 reported R/mR entries.
- Visual structure preservation improves all 12 corresponding VG base/novel entries over `w/o visual structure`.
- Novel synthesis improves all six novel entries over `w/o novel synthesis` but lowers all six base entries, so it is described as a base--novel trade-off rather than a uniform gain.
- No independent SVD-effect claim will be made until the corresponding ablation is completed.

## Validation

After implementation:

1. Verify every citation key used by the section exists in `refs.bib`.
2. Verify every `\input` target exists.
3. Run `git diff --check` and static scans for vertical table rules, undefined labels, and accidental text-structure claims.
4. Compile with an available LaTeX engine; if no engine exists, report that limitation and complete all static checks.
5. Run a reverse outline and a claim-evidence audit for the Experiments prose.

## Out of scope

- SGCls and SGDet experiments.
- Transplanting the method to multiple baselines.
- Backbone-swap experiments as evidence of generality.
- Direct numerical comparison with methods using incompatible splits or evaluation protocols.
