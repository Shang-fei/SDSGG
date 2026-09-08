# ICASSP Paper Float and Table Layout Design

## Goal

Improve the visual rhythm of the four-page main paper by making figure and
table placement predictable, keeping each result table close to its discussion,
and simplifying dense table headers. Visual polish is the primary criterion.

## Page-level layout

- Page 1: keep the motivation figure with the Introduction.
- Page 2 top: place the full-width method overview before the Method text.
- Page 3 top: place the full-width main comparison table before the detailed
  experimental discussion.
- Keep the VG semantic, GQA, and ablation tables close to the paragraphs that
  interpret them; prevent them from drifting into the Conclusion or references.
- Prefer standard top floats and controlled source order over forced `[H]`
  placement, which can create conspicuous white space.

## Table styling

- Table 1 retains its Base/Novel grouping and light vertical separators.
- Within each Base/Novel group, add visible whitespace between the R and mR
  metric blocks.
- Tables 2 and 3 use a single header row:
  `Method | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100`.
- Use consistent font size, row height, rule weight, numeric alignment, and
  highlighting across all result tables.
- Table 4 remains compact and appears adjacent to Ablation Studies.

## Verification

- Compile the current manuscript after float declarations are reordered.
- Render every page to PNG and inspect page balance, whitespace, table width,
  header clarity, float order, and proximity to the corresponding discussion.
- Iterate until the main figure is on page 2, Table 1 is on page 3, and no
  remaining table is detached from its discussion or visibly overcrowded.

## Scope

This revision changes only layout and table presentation. It does not change
experimental values, claims, captions' technical meaning, or method content.
