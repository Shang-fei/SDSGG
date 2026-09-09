# Configurable Structure Reference

## Goal

Allow the pairwise structure loss to use either the projected representations
or the refined text embeddings as its reference, without changing the default
behavior of existing experiments.

## Configuration

Add `MODEL.ROI_RELATION_HEAD.MTM.LOSS.STRUCTURE_REFERENCE` with two accepted
values:

- `"projected"` (default): compare pairwise similarities of visual relation
  representations and projected representations, preserving the current loss.
- `"text"`: compare pairwise similarities of visual relation representations
  and refined text embeddings.

Any other value raises a clear `ValueError`.

## Implementation

`structure_losses()` will compute the visual, projected, and text similarity
matrices once. It will select either the projected or text matrix as the
reference for the existing visual-structure loss. The separately returned
projected-to-text diagnostic loss remains unchanged.

## Compatibility

The default value is `"projected"`, so existing configurations and results are
unchanged. Selecting `"text"` implements

`|cos(v_i, v_j) - cos(t_i, t_j)|`.

## Verification

A focused unit test will verify the numerical result for both modes and confirm
that an unsupported configuration value is rejected.
