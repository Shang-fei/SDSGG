# Configurable Structure Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible configuration switch between the existing visual-to-projected structure loss and a visual-to-text structure loss.

**Architecture:** `structure_losses()` continues to build the three pairwise cosine-similarity matrices once, then selects either the projected or text matrix as the reference for the visual-structure term. The plugin passes the configured choice to every training and diagnostic call.

**Tech Stack:** Python, PyTorch, yacs configuration, unittest/pytest.

## Global Constraints

- `STRUCTURE_REFERENCE="projected"` must exactly preserve current behavior.
- `STRUCTURE_REFERENCE="text"` must compute `|cos(v_i,v_j)-cos(t_i,t_j)|`.
- Unsupported values must raise `ValueError`.
- Do not change existing loss keys or weights.

---

### Task 1: Add and verify the structure-reference switch

**Files:**
- Create: `tests/test_mtm_structure_reference.py`
- Modify: `maskrcnn_benchmark/modeling/roi_heads/relation_head/mtm/losses.py`
- Modify: `maskrcnn_benchmark/modeling/roi_heads/relation_head/mtm/plugin.py`
- Modify: `maskrcnn_benchmark/config/defaults.py`

**Interfaces:**
- Consumes: normalized `visual_features`, `projected_features`, and `text_features` tensors.
- Produces: `structure_losses(..., reference="projected"|"text") -> (visual_structure, text_structure)`.

- [x] **Step 1: Write the failing test**

Create a focused test that imports `losses.py` directly, verifies the numerical
visual-structure loss for `projected` and `text`, and verifies invalid-value
rejection.

- [x] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_mtm_structure_reference.py`

Expected: FAIL because `structure_losses()` does not yet accept `reference`.

- [x] **Step 3: Implement the minimal switch**

Add `reference="projected"` to `structure_losses()`. Select
`projected_similarity` for `"projected"` and `text_similarity` for `"text"`;
raise `ValueError` otherwise. Add the default config:

```python
_C.MODEL.ROI_RELATION_HEAD.MTM.LOSS.STRUCTURE_REFERENCE = "projected"
```

Pass `loss_config.STRUCTURE_REFERENCE` to all three calls in `plugin.py`.

- [x] **Step 4: Run focused and regression tests and verify GREEN**

Run:

```bash
pytest -q tests/test_mtm_structure_reference.py tests/test_mtm_plugin_structure.py
```

Expected: all tests pass.

- [x] **Step 5: Compile-check modified Python files**

Run:

```bash
python3 -m py_compile \
  maskrcnn_benchmark/modeling/roi_heads/relation_head/mtm/losses.py \
  maskrcnn_benchmark/modeling/roi_heads/relation_head/mtm/plugin.py \
  maskrcnn_benchmark/config/defaults.py
```

Expected: exit code 0 with no output.

- [x] **Step 6: Commit the implementation**

```bash
git add tests/test_mtm_structure_reference.py \
  maskrcnn_benchmark/modeling/roi_heads/relation_head/mtm/losses.py \
  maskrcnn_benchmark/modeling/roi_heads/relation_head/mtm/plugin.py \
  maskrcnn_benchmark/config/defaults.py \
  docs/superpowers/plans/2026-09-09-structure-reference-option.md
git commit -m "feat: make MTM structure reference configurable"
```
