# ICASSP 2027 Full-Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete English ICASSP manuscript whose technical content ends by page 4 and whose fifth page contains only references, funding acknowledgements, and the Compliance with Ethical Standards statement.

**Architecture:** Keep the manuscript modular through `sections/`, `tables/`, and `figures/`. Build the scientific story from the implemented modality-transfer code, express it with stable notation, then compress the evidence into three readable tables. Treat page count, claim support, and the visible reservation of pending experiments as verification constraints rather than final cosmetic edits.

**Tech Stack:** ICASSP `spconf` LaTeX, BibTeX, booktabs tables, TikZ or LaTeX-native method diagram, repository Python/PyTorch implementation, Poppler PDF rendering.

## Global Constraints

- Technical content must end by page 4.
- Page 5 may contain only references, funding acknowledgements, and the Compliance with Ethical Standards statement.
- The paper covers PredCls only; it must not imply completed SGCls or SGDet results.
- Text-structure loss is disabled in every experiment and must not be presented as an active component.
- Missing measurements remain visibly reserved with empty cells or compact pending rows; no value may be invented.
- The paper must not claim transferability to multiple baselines.
- Completed numerical values must match the verified SDSGG, APT, and user-provided results.
- Abstract length must remain between 100 and 150 words.
- Body and caption fonts must remain at least 9 point under the supplied template.

---

### Task 1: Establish the implemented method specification

**Files:**
- Inspect: `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`
- Inspect: `maskrcnn_benchmark/config/defaults.py`
- Inspect: `configs/**/*.yaml`
- Create: `docs/paper/method-evidence-map.md`

**Interfaces:**
- Consumes: the existing SDSGG implementation and active experiment configurations.
- Produces: a symbol-level method specification used by `sections/method.tex` and the paper-wide terminology list.

- [ ] **Step 1: Locate every active modality-transfer component**

Run:

```bash
rg -n "STRUCTURE|ALIGN|PSEUDO|NOVEL|SYNTH|FUSION|REMOVE|COMPONENT|TRANSFORMER|MTM" maskrcnn_benchmark configs tools
```

Expected: definitions and call sites for dominant-component removal, alignment, visual structure preservation, novel synthesis, and inference fusion.

- [ ] **Step 2: Trace the complete training and inference data flow**

Read the located constructors, forward paths, losses, configuration defaults, and active YAML overrides. Record tensor roles, dimensions, normalization, similarity functions, loss weights, warm-up/ramp behavior, and score fusion.

- [ ] **Step 3: Write the evidence map**

Create `docs/paper/method-evidence-map.md` with one row per manuscript claim:

```markdown
| Manuscript term | Implemented operation | Code/config evidence | Allowed claim |
|---|---|---|---|
| Dominant-component refinement | ... | `path:line` | ... |
| Pointwise alignment | ... | `path:line` | ... |
| Visual structure preservation | ... | `path:line` | ... |
| Novel feature synthesis | ... | `path:line` | ... |
| Score fusion | ... | `path:line` | ... |
```

- [ ] **Step 4: Verify the map contains no unsupported component**

Run:

```bash
rg -n "text.structure|SGCls|SGDet|other baseline|low-rank projection" docs/paper/method-evidence-map.md
```

Expected: no affirmative claim that text structure, SGCls, SGDet, cross-baseline transfer, or a low-rank projection is used.

- [ ] **Step 5: Commit the method specification**

```bash
git add docs/paper/method-evidence-map.md
git commit -m "docs: map ICASSP method to implementation"
```

### Task 2: Write the complete Method section

**Files:**
- Modify: `ICASSP2026_Paper_Templates/sections/method.tex`
- Create: `ICASSP2026_Paper_Templates/figures/method_overview.tex`
- Modify: `ICASSP2026_Paper_Templates/Template.tex`

**Interfaces:**
- Consumes: `docs/paper/method-evidence-map.md`.
- Produces: defined notation, complete objective, inference rule, and a compact method overview that Introduction and Experiments can reference.

- [ ] **Step 1: Load the Method section guide**

Read `/Users/shangfei/.codex/skills/research-paper-writing/references/method.md` completely before drafting.

- [ ] **Step 2: Define problem notation and the SDSGG starting point**

Write the first Method subsection with image/object-pair notation, base and novel predicate sets, union-region feature, SDSGG score, and CLIP triplet embedding. Define every symbol before reuse.

- [ ] **Step 3: Formalize relation-preserving modality transfer**

Write equations for triplet-bank component removal, projector output, cosine alignment, pairwise visual similarity matrices, and the visual structure-preservation loss. State explicitly that source visual geometry is the preserved structure and text-structure loss is not used.

- [ ] **Step 4: Formalize relation-conditioned novel synthesis and inference**

Write the generator inputs, reconstruction/KL objectives, pseudo-novel sampling, transfer score, SDSGG-score fusion, and total training loss using only implemented operations.

- [ ] **Step 5: Create a compact method overview**

Use a LaTeX-native one-column or two-column figure showing visual feature, triplet teacher, DCR, projector, VSP, novel synthesis, and fused prediction. The caption must define abbreviations without discussing results.

- [ ] **Step 6: Run the Method reverse outline**

Record each paragraph's first-sentence role and verify the mapping: notation -> gap -> transfer -> structure -> synthesis -> optimization/inference.

- [ ] **Step 7: Run static notation checks**

Run:

```bash
rg -n "\\label\{|\\ref\{|\\mathcal|\\mathbf|DCR|VSP|novel" ICASSP2026_Paper_Templates/sections/method.tex ICASSP2026_Paper_Templates/figures/method_overview.tex
rg -n "text.structure.*(enabled|active)|low-rank" ICASSP2026_Paper_Templates/sections/method.tex
```

Expected: all method blocks and labels exist; the second command returns no unsupported wording.

- [ ] **Step 8: Commit the Method section**

```bash
git add ICASSP2026_Paper_Templates/sections/method.tex ICASSP2026_Paper_Templates/figures/method_overview.tex ICASSP2026_Paper_Templates/Template.tex
git commit -m "docs: write ICASSP method section"
```

### Task 3: Write Related Work and revise Introduction

**Files:**
- Modify: `ICASSP2026_Paper_Templates/sections/related_work.tex`
- Modify: `ICASSP2026_Paper_Templates/sections/introduction.tex`
- Modify: `ICASSP2026_Paper_Templates/refs.bib`

**Interfaces:**
- Consumes: the completed Method terminology and verified primary literature.
- Produces: a concise novelty argument whose claims are reused in Abstract and Conclusion.

- [ ] **Step 1: Load the Related Work guide**

Read `/Users/shangfei/.codex/skills/research-paper-writing/references/related-work.md` completely.

- [ ] **Step 2: Verify cited prior work from primary sources**

Check the official publications for CLIP, Epic, RECODE, RAHP, SDSGG, ACC, APT, and SHIP. Add or correct only the BibTeX entries used by the final prose.

- [ ] **Step 3: Write two compact Related Work themes**

Write one paragraph on language-guided OVSGG and one on structure-aware transfer/generative feature synthesis. End each paragraph with a direct distinction from this paper.

- [ ] **Step 4: Run the Related Work reverse outline**

Verify that each paragraph synthesizes a research direction rather than listing papers and that novelty claims match the implemented method.

- [ ] **Step 5: Load the Introduction guide**

Read `/Users/shangfei/.codex/skills/research-paper-writing/references/introduction.md` completely.

- [ ] **Step 6: Rewrite Introduction to the five-role outline**

Use compact paragraphs for task/importance, prior progress/gap, hypothesis, framework, and evidence/contributions. Preserve the strict PredCls boundary and the verified 2.06-point average gain.

- [ ] **Step 7: Run Introduction reverse outline and claim check**

Check every performance, novelty, and causal statement against `docs/paper/method-evidence-map.md` and completed tables. Weaken statements whose mechanism evidence remains pending.

- [ ] **Step 8: Commit Related Work and Introduction**

```bash
git add ICASSP2026_Paper_Templates/sections/related_work.tex ICASSP2026_Paper_Templates/sections/introduction.tex ICASSP2026_Paper_Templates/refs.bib
git commit -m "docs: write ICASSP introduction and related work"
```

### Task 4: Consolidate Experiments into the four-page evidence budget

**Files:**
- Modify: `ICASSP2026_Paper_Templates/sections/experiments.tex`
- Create: `ICASSP2026_Paper_Templates/tables/main_comparison.tex`
- Create: `ICASSP2026_Paper_Templates/tables/transfer_analysis.tex`
- Modify: `ICASSP2026_Paper_Templates/tables/core_ablation.tex`
- Modify: `ICASSP2026_Paper_Templates/Template.tex`

**Interfaces:**
- Consumes: all completed comparison/ablation values and the visible-pending requirement.
- Produces: three compact tables that support the paper's central claims without exceeding the technical-page budget.

- [ ] **Step 1: Load the Experiments guide**

Read `/Users/shangfei/.codex/skills/research-paper-writing/references/experiments.md` completely.

- [ ] **Step 2: Build the consolidated comparison table**

Combine VG base/novel results into one readable table containing SDSGG, APT, and Ours. Preserve official APT novel mR@100 as 32.3. Use a caption that defines PredCls, splits, R/mR, and empty cells.

- [ ] **Step 3: Build the transfer-analysis table**

Combine VG semantic and GQA base/novel evidence. Retain a visibly empty matched-protocol APT GQA row and explain why it is pending.

- [ ] **Step 4: Compress the core ablation table**

Keep full, w/o VSP, w/o novel synthesis, and a visibly empty w/o DCR row. Reserve seed variability and efficiency through compact empty columns/rows only if they remain readable at 9 point.

- [ ] **Step 5: Rewrite experimental prose**

Use setup, comparison, ablation, and limitations paragraphs. Report only supported values; disclose the base--novel trade-off and pending DCR/robustness/efficiency evidence.

- [ ] **Step 6: Verify table values and pending markers**

Run:

```bash
rg -n "18\.7|26\.5|31\.6|19\.4|26\.6|32\.3|21\.3|27\.5|33\.9|35\.3|46\.0|51\.2" ICASSP2026_Paper_Templates/tables
rg -n "pending|empty cell|not yet" ICASSP2026_Paper_Templates/sections/experiments.tex ICASSP2026_Paper_Templates/tables/main_comparison.tex ICASSP2026_Paper_Templates/tables/transfer_analysis.tex ICASSP2026_Paper_Templates/tables/core_ablation.tex
```

Expected: all key completed values and explicit pending-result explanations are present.

- [ ] **Step 7: Run the Experiments reverse outline**

Verify the paragraph chain: reproducibility -> primary comparison -> harder settings -> component causality -> disclosed limitations.

- [ ] **Step 8: Commit the consolidated Experiments section**

```bash
git add ICASSP2026_Paper_Templates/sections/experiments.tex ICASSP2026_Paper_Templates/tables/main_comparison.tex ICASSP2026_Paper_Templates/tables/transfer_analysis.tex ICASSP2026_Paper_Templates/tables/core_ablation.tex ICASSP2026_Paper_Templates/Template.tex
git commit -m "docs: consolidate ICASSP experimental evidence"
```

### Task 5: Write Abstract, Conclusion, and permitted fifth-page material

**Files:**
- Modify: `ICASSP2026_Paper_Templates/sections/abstract.tex`
- Modify: `ICASSP2026_Paper_Templates/sections/conclusion.tex`
- Create: `ICASSP2026_Paper_Templates/sections/backmatter.tex`
- Modify: `ICASSP2026_Paper_Templates/Template.tex`

**Interfaces:**
- Consumes: final Introduction claims, Method, and completed experimental evidence.
- Produces: the manuscript opening/closing and compliant page-5 material.

- [ ] **Step 1: Load the Abstract guide**

Read `/Users/shangfei/.codex/skills/research-paper-writing/references/abstract.md` completely.

- [ ] **Step 2: Rewrite the Abstract**

Use problem, gap, method, and evidence roles in 100--150 words. Include no citation and no claim unsupported by the comparison or ablation tables.

- [ ] **Step 3: Load the Conclusion guide**

Read `/Users/shangfei/.codex/skills/research-paper-writing/references/conclusion.md` completely.

- [ ] **Step 4: Write the Conclusion**

Use one compact paragraph summarizing the scoped contribution, supported result, and PredCls limitation without introducing new evidence.

- [ ] **Step 5: Add compliant back matter**

After the bibliography, add unnumbered `Funding` and `Compliance with Ethical Standards` headings. Use neutral placeholders that clearly require author confirmation but contain no technical content:

```latex
\section*{Funding}
Funding information will be inserted after author confirmation.

\section*{Compliance with Ethical Standards}
This research did not involve human subjects or animals.
```

- [ ] **Step 6: Verify abstract length and back-matter order**

Run:

```bash
sed -n '/\\begin{abstract}/,/\\end{abstract}/p' ICASSP2026_Paper_Templates/sections/abstract.tex | wc -w
rg -n "bibliography|backmatter" ICASSP2026_Paper_Templates/Template.tex
```

Expected: abstract body is 100--150 words; bibliography precedes back matter.

- [ ] **Step 7: Commit the opening and closing sections**

```bash
git add ICASSP2026_Paper_Templates/sections/abstract.tex ICASSP2026_Paper_Templates/sections/conclusion.tex ICASSP2026_Paper_Templates/sections/backmatter.tex ICASSP2026_Paper_Templates/Template.tex
git commit -m "docs: complete ICASSP manuscript narrative"
```

### Task 6: Compile, enforce page boundaries, and perform adversarial review

**Files:**
- Modify as needed: `ICASSP2026_Paper_Templates/Template.tex`
- Modify as needed: `ICASSP2026_Paper_Templates/sections/*.tex`
- Modify as needed: `ICASSP2026_Paper_Templates/tables/*.tex`
- Create: `docs/paper/full-paper-review.md`
- Output: `ICASSP2026_Paper_Templates/Template.pdf`

**Interfaces:**
- Consumes: the complete manuscript.
- Produces: a rendered five-page paper and a claim/review audit.

- [ ] **Step 1: Load the paper-review guide**

Read `/Users/shangfei/.codex/skills/research-paper-writing/references/paper-review.md` completely.

- [ ] **Step 2: Compile twice with BibTeX**

Run from `ICASSP2026_Paper_Templates/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error Template.tex
bibtex Template
pdflatex -interaction=nonstopmode -halt-on-error Template.tex
pdflatex -interaction=nonstopmode -halt-on-error Template.tex
```

Expected: exit code 0, resolved citations/references, and exactly five PDF pages. If `pdflatex` is unavailable, install or use a local `tectonic` engine before claiming page compliance.

- [ ] **Step 3: Render all pages for visual inspection**

Run:

```bash
mkdir -p /tmp/icassp-paper-render
pdftoppm -png -r 160 Template.pdf /tmp/icassp-paper-render/page
pdfinfo Template.pdf | rg "Pages|Page size"
```

Expected: five letter-size pages and five readable PNG renders.

- [ ] **Step 4: Enforce the page-4/page-5 boundary**

Inspect page 4 and page 5. Page 4 must contain the end of Conclusion. Page 5 must contain only references, Funding, and Compliance with Ethical Standards. Tighten prose, captions, and table spacing without reducing text below 9 point until this condition holds.

- [ ] **Step 5: Run the five-dimension adversarial review**

Write `docs/paper/full-paper-review.md` with answered questions for contribution, writing clarity, experimental strength, evaluation completeness, and method design soundness. Add a claim--evidence map using:

```markdown
Claim: ... | Evidence: ... | Status: supported/needs evidence
```

Revise any high-risk unsupported statement in the manuscript.

- [ ] **Step 6: Run final static checks**

Run:

```bash
git diff --check
rg -n "TODO|TBD|SGCls|SGDet|text.structure.*(enabled|active)" ICASSP2026_Paper_Templates/sections ICASSP2026_Paper_Templates/tables
rg -n "\\cite\{|\\ref\{" ICASSP2026_Paper_Templates/sections
```

Expected: no formatting errors, no drafting placeholders, no out-of-scope affirmative claims, and only resolved citation/reference commands.

- [ ] **Step 7: Commit the verified manuscript**

```bash
git add ICASSP2026_Paper_Templates docs/paper/full-paper-review.md
git commit -m "docs: finalize four-page ICASSP manuscript"
```

## Plan Self-Review

- Every design requirement maps to a concrete task and verification step.
- Pending experiments are represented as an intentional publication-layout
  requirement, not as unfinished writing instructions.
- Method terminology is grounded in code before prose is drafted.
- Section guides are loaded sequentially, never all at once.
- Page compliance requires compiled-PDF inspection and cannot be inferred from
  word count alone.
- No implementation step refers to an undefined manuscript file or symbol.
