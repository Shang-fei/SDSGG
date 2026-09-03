# RPMT Motivation Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate an original, compact RPMT motivation figure for the ICASSP 2027 Introduction.

**Architecture:** Use GPT-Image-2 to produce one wide two-panel scientific infographic from the approved design. Inspect the result for scientific meaning, text accuracy, and originality, then copy the accepted image into the paper's figure directory without overwriting the pipeline figure.

**Tech Stack:** GPT-Image-2 built-in image generation, local image inspection, PNG.

## Global Constraints

- Preserve the approved two-panel `Independent Semantic Alignment` versus `Relation-Preserving Modality Transfer` narrative.
- Do not reproduce MASK's three-dimensional grids, image columns, star prototypes, word lists, or bidirectional curved-arrow composition.
- Use circles for visual instances, diamonds for semantic anchors, and dotted circles for synthesized unseen features.
- Use a white background, flat vector-like conference styling, concise English labels, and a wide two-column-paper aspect ratio.
- Do not overwrite any existing method pipeline figure.

---

### Task 1: Generate, inspect, and save the motivation figure

**Files:**
- Read: `docs/superpowers/specs/2026-09-03-rpmt-motivation-figure-design.md`
- Create: `ICASSP2027_Paper_Templates/figures/rpmt_motivation_concept.png`

**Interfaces:**
- Consumes: the approved design specification and the current Introduction narrative.
- Produces: one high-resolution PNG suitable for inclusion with `\\includegraphics`.

- [ ] **Step 1: Generate a high-resolution candidate with GPT-Image-2**

  Use the built-in image generator with the complete visual specification, exact labels, shape semantics, originality constraints, and a wide scientific-figure composition.

- [ ] **Step 2: Inspect the generated candidate**

  Verify that the left panel communicates independent alignment, structural distortion, and missing unseen visual support; verify that the right panel communicates preserved organization and unseen feature synthesis. Check all visible text character by character and reject any result that visually imitates MASK's composition.

- [ ] **Step 3: Iterate only if an acceptance check fails**

  If needed, regenerate with one targeted correction: either text accuracy, arrow/data-flow clarity, shape consistency, or layout density. Preserve every element that already satisfies the design.

- [ ] **Step 4: Save the accepted output in the paper project**

  Copy the final generated PNG to `ICASSP2027_Paper_Templates/figures/rpmt_motivation_concept.png`. Confirm that the file is a valid image and report its pixel dimensions.

- [ ] **Step 5: Commit the figure asset**

```bash
git add ICASSP2027_Paper_Templates/figures/rpmt_motivation_concept.png
git commit -m "docs: add RPMT motivation figure"
```
