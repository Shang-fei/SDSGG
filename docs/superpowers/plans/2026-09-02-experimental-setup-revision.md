# Experimental Setup Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite only the Dataset, Split, and Evaluation Metrics blocks in the Experimental Setup subsection.

**Architecture:** Replace the current combined dataset-and-metrics paragraph with three compact bold-labeled blocks. Preserve the following Compared methods and Implementation blocks exactly.

**Tech Stack:** LaTeX, ICASSP 2027 two-column template.

## Global Constraints

- Modify only `ICASSP2027_Paper_Templates/sections/experiments.tex` before `\textbf{Compared methods.}`.
- Use the labels `\textbf{Dataset.}`, `\textbf{Split.}`, and `\textbf{Evaluation Metrics.}`.
- Mention the evaluation protocol through ground-truth object inputs without foregrounding the term PredCls.
- Preserve citations, verified split counts, and the four-page technical-content constraint.

---

### Task 1: Rewrite Experimental Setup Opening

**Files:**
- Modify: `ICASSP2027_Paper_Templates/sections/experiments.tex:7-18`
- Test: source-level LaTeX and scope checks on the same file

**Interfaces:**
- Consumes: verified VG/GQA split definitions and existing BibTeX keys.
- Produces: three LaTeX prose blocks ending immediately before `\textbf{Compared methods.}`.

- [ ] **Step 1: Capture the unchanged suffix**

Run:

```bash
sed -n '/\\textbf{Compared methods\.}/,$p' ICASSP2027_Paper_Templates/sections/experiments.tex
```

Expected: the existing Compared methods and Implementation blocks are visible for post-edit comparison.

- [ ] **Step 2: Replace the opening with the approved prose**

Insert three blocks: Dataset introduces VG and GQA; Split gives 35/15 and 21/10 base/novel partitions plus the 24-predicate VG semantic split; Evaluation Metrics defines R@K and mR@K and states that ground-truth objects isolate relation recognition.

- [ ] **Step 3: Verify structure, citations, and scope**

Run:

```bash
rg -n '\\textbf\{(Dataset|Split|Evaluation Metrics|Compared methods|Implementation)\.\}' ICASSP2027_Paper_Templates/sections/experiments.tex
```

Expected: all five labels occur once and in the intended order.

Run:

```bash
git diff --check -- ICASSP2027_Paper_Templates/sections/experiments.tex
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Review paragraph logic**

Check that each block has one message, Dataset and Split information are not mixed, PredCls is not foregrounded, and no claims extend beyond the cited protocols.
