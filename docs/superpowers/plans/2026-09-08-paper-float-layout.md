# ICASSP Paper Float Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a visually balanced manuscript with the method figure at the top of page 2, Table 1 at the top of page 3, simplified one-row headers for Tables 2 and 3, and all remaining tables adjacent to their discussions.

**Architecture:** Control wide-float placement primarily through source declaration order: queue the method figure at the end of the Introduction and queue Table 1 late enough on page 2 to place it on page 3. Keep single-column tables local to their result paragraphs and use float barriers only where later-section drift must be prevented. Standardize table typography and metric grouping without changing values.

**Tech Stack:** LaTeX, ICASSP `spconf`, `booktabs`, `multirow`, `xcolor`, Poppler (`pdfinfo`, `pdftoppm`).

## Global Constraints

- Visual polish is the primary criterion.
- Page 1 contains the motivation figure; page 2 top contains the method figure; page 3 top contains Table 1.
- Tables 2 and 3 use one header row with full metric names.
- Experimental values and technical claims must not change.
- Avoid forced `[H]` placement and conspicuous white space.

---

### Task 1: Establish deterministic wide-float order

**Files:**
- Modify: `ICASSP2027_Paper_Templates/sections/introduction.tex`
- Modify: `ICASSP2027_Paper_Templates/sections/method.tex`
- Modify: `ICASSP2027_Paper_Templates/sections/experiments.tex`
- Modify: `ICASSP2027_Paper_Templates/Template.tex`

**Interfaces:**
- Consumes: `figures/method_overview.tex`, `tables/main_comparison.tex`
- Produces: source order in which the method figure precedes Table 1 and each can occupy one full-width top-float slot.

- [ ] **Step 1: Record current float declarations**

Run:
```bash
rg -n 'input\{figures|input\{tables|begin\{figure|begin\{table' ICASSP2027_Paper_Templates --glob '*.tex'
```
Expected: the method figure appears before all result tables.

- [ ] **Step 2: Move the Table 1 declaration to the latest source position that still precedes the Experiments discussion**

Declare `\input{tables/main_comparison}` near the Method-to-Experiments transition so it enters the queue after the page-2 method figure and before Tables 2--4.

- [ ] **Step 3: Keep wide-float capacity to one per page**

Retain `\setcounter{dbltopnumber}{1}` and top-float fractions that permit a large figure or table without forcing a float page.

- [ ] **Step 4: Compile and inspect float order**

Run:
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error Template.tex
pdfinfo Template.pdf
```
Expected: compilation succeeds; the method figure precedes Table 1.

### Task 2: Simplify and harmonize result tables

**Files:**
- Modify: `ICASSP2027_Paper_Templates/tables/main_comparison.tex`
- Modify: `ICASSP2027_Paper_Templates/tables/vg_semantic.tex`
- Modify: `ICASSP2027_Paper_Templates/tables/gqa_base_novel.tex`
- Modify: `ICASSP2027_Paper_Templates/tables/core_ablation.tex`

**Interfaces:**
- Consumes: existing numerical values and captions.
- Produces: compact tables with consistent typography, row spacing, rules, and visible R/mR separation.

- [ ] **Step 1: Preserve Table 1 grouping while separating metric families**

Use Base/Novel group labels, light vertical group rules, and explicit whitespace between each `R@K` block and `mR@K` block.

- [ ] **Step 2: Replace the two-row Table 2 header**

Use exactly:
```latex
Method & R@20 & R@50 & R@100 & mR@20 & mR@50 & mR@100 \\
```

- [ ] **Step 3: Replace the two-row Table 3 header**

Use exactly:
```latex
Method & R@20 & R@50 & R@100 & mR@20 & mR@50 & mR@100 \\
```

- [ ] **Step 4: Harmonize styling**

Use `\scriptsize`, restrained `\tabcolsep`, consistent `\arraystretch`, `booktabs` horizontal rules, and one subtle highlight style for RPMT rows.

- [ ] **Step 5: Verify unchanged data**

Run:
```bash
git diff --word-diff=porcelain -- ICASSP2027_Paper_Templates/tables
```
Expected: header and layout tokens change; all numeric values remain unchanged.

### Task 3: Keep Tables 2--4 with their discussions

**Files:**
- Modify: `ICASSP2027_Paper_Templates/sections/experiments.tex`
- Modify: `ICASSP2027_Paper_Templates/tables/vg_semantic.tex`
- Modify: `ICASSP2027_Paper_Templates/tables/gqa_base_novel.tex`
- Modify: `ICASSP2027_Paper_Templates/tables/core_ablation.tex`

**Interfaces:**
- Consumes: Quantitative Comparison and Ablation subsection boundaries.
- Produces: local top floats that do not drift into the Conclusion or references.

- [ ] **Step 1: Place each input immediately before its interpretation paragraph**

Keep Table 2 with the VG semantic paragraph, Table 3 with the GQA paragraph, and Table 4 at the start of Ablation Studies.

- [ ] **Step 2: Add the minimum necessary float boundary**

Use a barrier only before the Conclusion if compilation shows an experimental table crossing the section boundary.

- [ ] **Step 3: Compile and render all pages**

Run:
```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error Template.tex
mkdir -p tmp/pdfs
pdftoppm -png -r 150 Template.pdf tmp/pdfs/template
```
Expected: one PNG per PDF page with no clipped or detached float.

### Task 4: Visual quality gate

**Files:**
- Inspect: `ICASSP2027_Paper_Templates/tmp/pdfs/template-*.png`
- Modify as needed: the LaTeX files listed above.

**Interfaces:**
- Consumes: rendered manuscript pages.
- Produces: final balanced layout.

- [ ] **Step 1: Inspect pages 1--4**

Confirm the target page assignment, balanced column endings, readable captions, aligned rules, and sufficient R/mR separation.

- [ ] **Step 2: Iterate only on observed defects**

Adjust declaration position, float specifier, `\tabcolsep`, or `\arraystretch`; recompile and rerender after each meaningful change.

- [ ] **Step 3: Run final source checks**

Run:
```bash
rg -n 'afterpage|\[H\]|FloatBarrier|input\{tables|input\{figures' ICASSP2027_Paper_Templates --glob '*.tex'
```
Expected: no accidental `[H]`; float inputs follow the approved order; any barrier is intentional.
