# ICASSP Experiments LaTeX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the ICASSP manuscript into focused LaTeX section/table files and produce an evidence-backed Experiments section with completed values, visibly empty pending-value cells, concise captions, and an explicit PredCls scope.

**Architecture:** `Template.tex` becomes the manuscript entry point and includes six section files. The Experiments section includes five self-contained table files plus one qualitative-figure placeholder; prose separates setup, comparisons, ablations, and challenging evaluation so every claim maps to a reported or explicitly pending experiment.

**Tech Stack:** LaTeX, ICASSP `spconf` style, BibTeX with `IEEEbib`, `booktabs`, `xcolor`, shell-based static validation.

## Global Constraints

- Evaluate open-vocabulary Predicate Classification only; do not imply SGCls or SGDet results.
- Do not claim portability across unrelated SGG baselines or add backbone-swap evidence.
- Use official SDSGG and APT values only under matched protocols.
- Use APT VG-novel `mR@100 = 32.3`; do not copy the screenshot's `31.1` transcription.
- Keep APT's GQA row empty until a matched-protocol result exists.
- `w/o visual structure` disables only visual structure preservation; text-structure loss is disabled in every reported configuration.
- Do not attribute an independent gain to dominant-component removal until its isolated ablation is completed.
- Use captions above tables, `booktabs`, no vertical rules, metric-direction arrows, and one-decimal precision.
- Empty cells mean required results are pending; a dash means the cited method did not report that metric.

---

### Task 1: Split the manuscript into section files

**Files:**
- Modify: `ICASSP2026_Paper_Templates/Template.tex`
- Create: `ICASSP2026_Paper_Templates/sections/abstract.tex`
- Create: `ICASSP2026_Paper_Templates/sections/introduction.tex`
- Create: `ICASSP2026_Paper_Templates/sections/related_work.tex`
- Create: `ICASSP2026_Paper_Templates/sections/method.tex`
- Create: `ICASSP2026_Paper_Templates/sections/experiments.tex`
- Create: `ICASSP2026_Paper_Templates/sections/conclusion.tex`

**Interfaces:**
- Consumes: Current abstract and Introduction in `Template.tex`.
- Produces: Six `\input{sections/...}` targets consumed by `Template.tex`; `experiments.tex` will consume table and figure inputs created in later tasks.

- [ ] **Step 1: Record the expected input targets before the refactor**

Run:

```bash
rg -n '\\section|\\begin\{abstract\}|\\bibliography' ICASSP2026_Paper_Templates/Template.tex
```

Expected: the abstract and all five manuscript sections currently appear directly in `Template.tex`.

- [ ] **Step 2: Add formatting packages and replace inline sections with inputs**

Use this document-body structure in `Template.tex`:

```latex
\usepackage{spconf,amsmath,graphicx,hyperref}
\usepackage{booktabs}
\usepackage[table]{xcolor}

\begin{document}
\maketitle
\input{sections/abstract}
\input{sections/introduction}
\input{sections/related_work}
\input{sections/method}
\input{sections/experiments}
\input{sections/conclusion}
\bibliographystyle{IEEEbib}
\bibliography{strings,refs}
\end{document}
```

- [ ] **Step 3: Move existing content into focused section files**

`abstract.tex` must contain the `abstract` and `keywords` environments. `introduction.tex` must contain the existing Introduction and contribution list. The remaining section files must contain their section heading and a neutral comment describing the future content; these comments must not render in the manuscript.

- [ ] **Step 4: Verify all section inputs resolve**

Run:

```bash
for f in abstract introduction related_work method experiments conclusion; do test -f "ICASSP2026_Paper_Templates/sections/$f.tex" || exit 1; done
rg -n '\\input\{sections/' ICASSP2026_Paper_Templates/Template.tex
```

Expected: exit code 0 and exactly six section inputs.

- [ ] **Step 5: Commit the section split**

```bash
git add ICASSP2026_Paper_Templates/Template.tex ICASSP2026_Paper_Templates/sections
git commit -m "docs: split ICASSP manuscript sections"
```

### Task 2: Add matched-protocol comparison tables

**Files:**
- Create: `ICASSP2026_Paper_Templates/tables/vg_base_novel.tex`
- Create: `ICASSP2026_Paper_Templates/tables/vg_semantic.tex`
- Create: `ICASSP2026_Paper_Templates/tables/gqa_base_novel.tex`
- Modify: `ICASSP2026_Paper_Templates/sections/experiments.tex`
- Modify: `ICASSP2026_Paper_Templates/refs.bib`

**Interfaces:**
- Consumes: Official SDSGG Tables 1--3, official APT VG row, and the user's uploaded Ours results.
- Produces: labels `tab:vg_base_novel`, `tab:vg_semantic`, and `tab:gqa_base_novel`, referenced by comparison prose.

- [ ] **Step 1: Create the VG base/novel table with explicit missing-value semantics**

Use a two-part `table*` with one row per method and split. Populate CLS, Epic, SDSGG, APT, and Ours using one-decimal precision. Use `--` only for Epic metrics absent from its paper. Caption text:

```latex
\caption{PredCls results (\%) on the Visual Genome base and novel splits under the SDSGG protocol. $\mathrm{R}@K$ and $\mathrm{mR}@K$ denote Recall and mean Recall among the top-$K$ predictions. A dash indicates a metric not reported by the cited method; an empty cell indicates a planned result that is not yet available.}
```

The APT novel row must be `19.4, 26.6, 31.1, 18.6, 26.7, 32.3`; the Ours rows must be the uploaded full-model values.

- [ ] **Step 2: Create the VG semantic table**

Populate CLS, CLS-DE, RECODE variants, SDSGG, and Ours. Use the official SDSGG Table 2 values for published methods and `24.3/31.7/36.5` R and `18.6/24.5/29.1` mR for Ours. State in the caption that the semantic subset contains 24 predicates and uses the base-trained checkpoint.

- [ ] **Step 3: Create the GQA base/novel table**

Populate CLS and SDSGG from official SDSGG Table 3 and Ours from the uploaded results. Include an APT row with six empty metric cells for each split, labeled `APT (matched protocol)`, because the published APT GQA protocol does not reproduce the same SDSGG reference values.

- [ ] **Step 4: Write comparison prose with one claim per paragraph**

Create one paragraph for VG base/novel, one for VG semantic, and one for GQA. The first sentences must respectively state: (a) the full method consistently improves the direct SDSGG baseline and the recent APT comparator on matched VG splits; (b) the gain extends to semantically richer VG relations; and (c) the gain transfers to the SDSGG-compatible GQA split. Do not describe absent APT-GQA values as a comparison.

- [ ] **Step 5: Verify table values and references**

Run:

```bash
rg -n '32\.3|24\.3|31\.7|36\.5|35\.3|46\.0|51\.2|31\.4|42\.0|45\.6' ICASSP2026_Paper_Templates/tables
rg -n '\\label\{tab:(vg_base_novel|vg_semantic|gqa_base_novel)\}' ICASSP2026_Paper_Templates/tables
```

Expected: every listed value and all three labels appear.

- [ ] **Step 6: Commit comparison tables and prose**

```bash
git add ICASSP2026_Paper_Templates/tables ICASSP2026_Paper_Templates/sections/experiments.tex ICASSP2026_Paper_Templates/refs.bib
git commit -m "docs: add OVSGG comparison experiments"
```

### Task 3: Add core and focused ablations

**Files:**
- Create: `ICASSP2026_Paper_Templates/tables/core_ablation.tex`
- Create: `ICASSP2026_Paper_Templates/tables/focused_ablations.tex`
- Modify: `ICASSP2026_Paper_Templates/sections/experiments.tex`

**Interfaces:**
- Consumes: VG base/novel results for `w/o visual structure`, `w/o novel synthesis`, and the full model; project defaults for SVD, structure loss, SHIP, and inference fusion.
- Produces: labels `tab:core_ablation` and `tab:focused_ablations`, referenced by ablation prose.

- [ ] **Step 1: Build the component table with truthful switches**

Use columns `DCR`, `Align`, `VSP`, and `NS`, where DCR is dominant-component removal, VSP is visual structure preservation, and NS is novel synthesis. Include rows for SDSGG, `Ours w/o DCR` with empty metrics, `Ours w/o VSP`, `Ours w/o NS`, and full Ours. Do not include a text-structure switch.

- [ ] **Step 2: Populate the available core-ablation metrics**

Use all six R/mR values for VG base and all six for VG novel in each available row. Preserve these exact meanings:

```text
w/o VSP: teacher + alignment + novel synthesis enabled; visual structure disabled
w/o NS: teacher + alignment + visual structure enabled; novel synthesis disabled
Ours: teacher + alignment + visual structure + novel synthesis enabled
```

- [ ] **Step 3: Add focused pending-value tables**

Create grouped mini-tables for removed-component count `{0,1,2,4,8}`, structure distance `{L1,L2}` and weight, pseudo-novel ratio, sampling policy, and fusion weight. Metric columns must be present but empty. The caption must say that the configurations are predeclared analyses and empty cells are pending measurements.

- [ ] **Step 4: Write ablation prose without unsupported causality**

The first paragraph states that visual structure preservation improves all 12 completed VG base/novel entries and gives the average 1.28-point gain. The second paragraph states that novel synthesis improves all six novel entries while lowering all six base entries, identifying a base--novel trade-off. The final paragraph explains that the empty focused ablations are required before claiming an independent DCR effect or hyperparameter robustness.

- [ ] **Step 5: Run terminology and value checks**

Run:

```bash
rg -n 'w/o (VSP|NS)|DCR|Align|VSP|NS' ICASSP2026_Paper_Templates/tables/core_ablation.tex
! rg -n 'text structure.*(enabled|on)|SVD only|SVD \+ Structure' ICASSP2026_Paper_Templates/sections/experiments.tex ICASSP2026_Paper_Templates/tables
```

Expected: the component notation exists and the forbidden descriptions do not.

- [ ] **Step 6: Commit ablation tables and prose**

```bash
git add ICASSP2026_Paper_Templates/tables/core_ablation.tex ICASSP2026_Paper_Templates/tables/focused_ablations.tex ICASSP2026_Paper_Templates/sections/experiments.tex
git commit -m "docs: add modality-transfer ablations"
```

### Task 4: Add challenging evaluation and complete the setup

**Files:**
- Create: `ICASSP2026_Paper_Templates/figures/challenging_cases_placeholder.tex`
- Modify: `ICASSP2026_Paper_Templates/sections/experiments.tex`

**Interfaces:**
- Consumes: Existing VG-semantic and GQA-novel evidence plus planned rare-predicate, confusion, geometry, and qualitative analyses.
- Produces: label `fig:challenging_cases` and a complete four-subsection Experiments section.

- [ ] **Step 1: Write the experimental setup**

Define PredCls, the VG and GQA splits, `R@K`, and `mR@K`. State that novel predicate annotations are removed from training. Describe only verified configuration values: CLIP ViT-B/32; DCR count 1; alignment weight 1.0; VSP weight 1.0 with L1 distance; text-structure weight 0; SHIP pseudo-novel ratio 0.25; low-recall sampling; reconstruction/KL weights 1.0/0.01; SHIP warm-up/ramp 2000/6000; inference score weight 1.0. Leave machine, total training time, and repeated-run statistics as commented pending fields if they are not verified.

- [ ] **Step 2: Add a non-fabricated qualitative placeholder**

Use a `figure*` containing a framed box whose visible text says `Qualitative examples will be inserted after case selection.` Caption text:

```latex
\caption{Planned PredCls comparison on challenging novel relations. Columns will show the input subject--object pair, ground truth, SDSGG prediction, predictions without VSP and NS, and the full-model prediction; color will distinguish correct and incorrect predicates.}
```

- [ ] **Step 3: Write challenging-evaluation prose**

Explain that VG semantic and GQA novel are the current harder evaluations. Predeclare three analyses without claiming results: performance by predicate frequency, near-synonym confusion matrices, and correlation between pre/post-projection pairwise similarities. Keep the qualitative figure reference conditional on future case insertion.

- [ ] **Step 4: Verify subsection and input structure**

Run:

```bash
rg -n '^\\subsection\{' ICASSP2026_Paper_Templates/sections/experiments.tex
rg -n '\\input\{tables/|\\input\{figures/' ICASSP2026_Paper_Templates/sections/experiments.tex
```

Expected: four subsections and six input targets.

- [ ] **Step 5: Commit setup and challenging evaluation**

```bash
git add ICASSP2026_Paper_Templates/sections/experiments.tex ICASSP2026_Paper_Templates/figures/challenging_cases_placeholder.tex
git commit -m "docs: complete ICASSP experiments section"
```

### Task 5: Perform static QA and claim-evidence review

**Files:**
- Modify when required by checks: `ICASSP2026_Paper_Templates/Template.tex`
- Modify when required by checks: `ICASSP2026_Paper_Templates/sections/experiments.tex`
- Modify when required by checks: `ICASSP2026_Paper_Templates/tables/*.tex`

**Interfaces:**
- Consumes: Complete LaTeX manuscript structure and Experiments artifacts.
- Produces: A statically consistent draft whose completed claims map to filled results and whose pending claims remain explicitly non-assertive.

- [ ] **Step 1: Check input targets and citation keys**

Run a shell loop that extracts every `\input{...}` path from `Template.tex` and `experiments.tex`, appends `.tex`, and verifies each target exists relative to `ICASSP2026_Paper_Templates`. Extract citation keys and verify each appears as an entry key in `refs.bib`.

Expected: no missing file or citation key.

- [ ] **Step 2: Check table style and whitespace**

Run:

```bash
! rg -n '\\begin\{tabular\}\{[^}]*\|' ICASSP2026_Paper_Templates/tables
! rg -n '\\hline' ICASSP2026_Paper_Templates/tables
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Run the reverse outline**

Confirm the paragraph-topic sequence is: protocol and metrics; implementation; VG base/novel evidence; semantic evidence; GQA evidence; VSP causality; NS trade-off; pending focused ablations; current hard settings; planned diagnostics and demos. Revise any paragraph whose first sentence does not state its role.

- [ ] **Step 4: Run the claim-evidence audit**

Verify these mappings:

```text
Claim: stronger than SDSGG across reported entries
Evidence: filled VG and GQA comparison tables

Claim: stronger than APT on matched VG metrics
Evidence: filled VG base/novel table using official APT values

Claim: VSP helps consistently
Evidence: full versus w/o-VSP rows across 12 VG entries

Claim: NS favors novel performance with a base trade-off
Evidence: full versus w/o-NS rows across 12 VG entries

Claim: DCR independently helps
Evidence: empty; claim must not appear as a finding
```

- [ ] **Step 5: Attempt compilation and report environment limits**

Run the first available command among `latexmk -pdf`, `pdflatex` plus `bibtex`, or `tectonic`. If none exists, record that compilation could not be performed and retain the successful static checks as verification evidence.

- [ ] **Step 6: Commit QA corrections**

```bash
git add ICASSP2026_Paper_Templates
git commit -m "docs: verify ICASSP experiment evidence"
```
