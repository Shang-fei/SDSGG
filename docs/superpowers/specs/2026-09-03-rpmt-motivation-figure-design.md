# RPMT Introduction Motivation Figure Design

## Purpose

Create a compact, publication-style conceptual figure for the Introduction of
the ICASSP 2027 paper. The figure explains the two limitations identified in
the Introduction and how RPMT addresses them, without reproducing the visual
composition of the MASK motivation figure.

## Composition

Use a wide two-panel layout with a central contrast arrow.

### (a) Independent Semantic Alignment

- Show three observed visual-relation clusters in a pale visual-space panel.
- Connect individual visual samples independently to their corresponding text
  anchors in a neighboring semantic-space panel.
- Use visibly distorted cluster neighborhoods after projection to convey that
  individual matching alone does not retain the organization among relations.
- Place an unseen-predicate anchor in gray outside the learned visual support,
  with a short dashed connection and the annotation `No visual support`.
- Use the concise callouts `Independent alignment` and
  `Relation structure is not retained`.

### (b) Relation-Preserving Modality Transfer

- Show the same observed relation clusters and one synthesized unseen cluster.
- Connect the visual space to a shared semantic space through a compact module
  labeled `Relation-Preserving Mapping`.
- Draw corresponding within-space neighborhood links before and after mapping
  with consistent topology, expressing preservation without equations.
- Show the unseen semantic anchor conditioning a small synthesized visual
  cluster, which then joins the same alignment space.
- Use the concise callouts `Structure-preserving alignment`,
  `Unseen feature synthesis`, and `Observed + unseen relations`.

## Visual Language

- Flat vector-like scientific infographic, white background, no 3D axes.
- Muted conference palette: blue, orange, green for predicate groups; purple
  for unseen predicates; charcoal for typography and arrows.
- Visual instances are filled circles; text anchors are diamonds; synthesized
  samples are circles with a subtle dotted fill.
- Use consistent colors and shapes across both panels and include a tiny legend.
- Horizontal aspect ratio suitable for a two-column ICASSP figure.
- Typography should resemble clean LaTeX sans-serif labels; no decorative
  shadows, gradients, stock imagery, watermark, or long prose.

## Required Text

- `(a) Independent Semantic Alignment`
- `(b) Relation-Preserving Modality Transfer`
- `Visual Relation Space`
- `Semantic Space`
- `Seen`
- `Unseen`
- `Independent alignment`
- `Relation structure is not retained`
- `No visual support`
- `Relation-Preserving Mapping`
- `Structure-preserving alignment`
- `Unseen feature synthesis`
- `Observed + unseen relations`

## Originality Constraints

Do not copy MASK's perspective grids, object-photo columns, word lists,
bidirectional curved arrows, star prototypes, or matching panel geometry. The
figure must be organized around OVSGG relation triplets and RPMT's asymmetric
visual-to-text transfer plus unseen-relation synthesis.

## Deliverables

- High-resolution PNG for inspection and direct LaTeX use.
- Preserve the generated version separately; do not overwrite the existing
  method pipeline figure.
- Target project path:
  `ICASSP2027_Paper_Templates/figures/rpmt_motivation_concept.png`.

## Acceptance Checks

- The two limitations and two RPMT responses are understandable at column-width
  scale without reading the caption.
- Shapes, colors, and arrows remain consistent across panels.
- All required text is spelled correctly.
- The layout is visually distinct from the MASK reference.
