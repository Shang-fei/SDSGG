# VISOR-PRISM: Verified Sparse Primitive Bottlenecks for Inductive OVSGG

## Research Question

Can a sparse, anchored primitive bottleneck produce novel predicate prototypes that remain discriminative under text-only, object-prior, and shortcut-controlled evaluations, indicating transferable primitive structure beyond isolated CLIP predicate prototypes?

The claim is intentionally bounded: VISOR-PRISM does not assert that every primitive is fully visually grounded. It operationalizes primitive grounding through stratified verification protocols and tests whether primitive-based transfer survives strong text-only and subject-object shortcut controls.

## Method Summary

VISOR-PRISM trains an offline standalone prototype generator:

```text
predicate description -> sparse primitive weights -> primitive condition
subject/object context + geometry -> low-rank gate
primitive condition + gate -> CLIP-compatible relation prototype
```

The first implementation uses frozen CLIP union crop features as `f_rel`. It does not use VAE, flow, diffusion, or direct residual subtraction.

Training losses:

```text
L = L_rec + L_cls + L_comp + L_visprim + L_adv_obj
```

- `L_rec`: cosine alignment between generated prototype and CLIP union feature.
- `L_cls`: standalone predicate ranking loss.
- `L_comp`: sparse composer supervision from fixed primitive mapping.
- `L_visprim`: auxiliary primitive probe from visual relation feature.
- `L_adv_obj`: object-pair adversarial control for shortcut reduction.

## Verification Protocol

Primitive verification is stratified:

- Geometry-verifiable primitives: vertical layout, containment, proximity, overlap/contact proxy. These use box-derived pseudo labels.
- Appearance/action-verifiable primitives: holding, wearing, riding, covering, looking-at. These require a future human-audited VLM-assisted validation set.
- Semantic/function-heavy primitives: using, made-of, part-of, belonging-to. These are evaluated through ranking gains and shortcut-controlled gaps, not claimed as fully visually causal.

The first code version implements geometry-derived pseudo labels and standalone shortcut baselines. Human-audited verification is defined here but not used in training.

## Required Evidence

Standalone results must show:

- full VISOR-PRISM > CLIP predicate text.
- full VISOR-PRISM > text-only primitive composition.
- full VISOR-PRISM > c_so-only subject-object prior.
- full VISOR-PRISM > random primitive anchors.

Shortcut checks:

- `no_c_so` should remain non-trivial.
- `c_so_only` should not explain all novel gains.
- raw same-SO diagnostic should be reported; a clean same-SO subset is future work.

Fusion into SDSGG is intentionally postponed until standalone evidence is positive.

## Experiment Log

Append new runs below this line.

<!-- VISOR_PRISM_RESULTS_START -->
