# Experimental Setup Revision Design

## Scope

Revise only the opening material of `ICASSP2027_Paper_Templates/sections/experiments.tex` under `Experimental Setup`. Leave `Compared methods` and `Implementation` unchanged.

## Structure

Use three compact inline blocks:

1. **Dataset.** Introduce Visual Genome and GQA as the two benchmarks without mixing in split details.
2. **Split.** State that the work follows established OVSGG partitions, define the base/novel ratios and visibility condition, and explain that the VG semantic set contains 24 semantically richer predicates.
3. **Evaluation Metrics.** Define R@K and mR@K for K in {20, 50, 100}. Mention ground-truth object inputs only as a short protocol clause, without foregrounding PredCls.

## Style Constraints

- Preserve all verified dataset counts and citations.
- Use one message per block and keep transitions explicit.
- Avoid defensive language about evaluating only PredCls.
- Do not alter claims, tables, comparison methods, implementation details, or numerical results.
- Keep the revision compact for the four-page ICASSP technical-content limit.

## Acceptance Checks

- The subsection begins with exactly the three requested bold labels.
- Dataset information is separated from split information.
- Base, novel, and semantic settings are defined without redundant notation.
- R@K and mR@K are defined, including all reported K values.
- `Compared methods` and `Implementation` remain byte-for-byte unchanged.
