# ICASSP Method Revision Design

## Goal

Rewrite the Method section as a self-contained, contribution-first description of relation-preserving modality transfer. The section must align with the current Introduction and architecture figure while remaining compact enough for a four-page ICASSP technical manuscript.

## Narrative Structure

The Method contains exactly four subsections:

1. **Task Description and Framework Overview**: define open-vocabulary predicate classification and summarize the three-stage framework.
2. **Debiased Triplet Semantic Anchors**: construct subject--predicate--object embeddings and remove their dominant shared components.
3. **Structure-Preserving Cross-Modal Alignment**: align each visual relation with its semantic anchor while retaining pairwise geometry from the visual relation space.
4. **Relation-Conditioned Novel Synthesis**: learn the distribution of base visual relations and synthesize visual support for novel triplets.

The central claim is that reliable base-to-novel transfer requires both semantic correspondence and visual-structure preservation; synthesized novel support complements this alignment when visual observations are unavailable.

## Symbol System

- Input image: $I$
- Subject--object pair: $(s_i,o_i)$
- Visual relation feature: $\mathbf v_i$
- Raw triplet embedding: $\mathbf a_i$
- Debiased semantic anchor: $\mathbf t_i$
- Aligned relation feature: $\mathbf z_i$
- Synthesized relation feature: $\tilde{\mathbf v}_i$
- Relation latent: $\mathbf r_i$
- Latent statistics: $\boldsymbol\mu_i,\boldsymbol\sigma_i$
- Visual/text encoders: $E_V,E_T$
- Cross-modal projector: $P_\theta$
- Latent encoder and prompt-conditioned decoder: $q_\phi,D_\psi$

No symbol may change meaning across the text, equations, caption, and pipeline figure.

## Core Equations

The section uses only three principal displayed equations:

1. Dominant-component removal for the semantic anchor.
2. A unified structure-preserving alignment objective containing:
   - instance-level cosine alignment between $\mathbf z_i$ and $\mathbf t_i$;
   - pairwise geometry preservation between $\mathbf v_i$ and $\mathbf z_i$.
3. Reconstruction and KL regularization for relation-conditioned synthesis.

The alignment equation follows the user's requested two-term presentation but retains the implementation's visual-to-aligned structure constraint and distance choice.

## Figure-to-Text Mapping

The pipeline figure contains three explicitly named spaces:

- **Visual Relation Space** $\mathcal V$ (upper left)
- **Debiased Semantic Space** $\mathcal T$ (upper right)
- **Structure-Preserved Aligned Space** $\mathcal Z$ (lower middle)

Recommended module labels:

- Frozen Visual Encoder $E_V$
- Frozen Text Encoder $E_T$
- Cross-Modal Projector $P_\theta$
- Novel Support Synthesizer $G_\psi$
- Latent Relation Encoder $q_\phi$
- Prompt-Conditioned Decoder $D_\psi$

The caption explains the visual/text/aligned spaces and the roles of structure preservation and novel synthesis. It does not repeat experimental claims.

## Editorial and Layout Constraints

- Do not frame the method as a patch to a named baseline.
- Do not include score-fusion equations or baseline-specific inference details.
- Omit layer counts, sampling ratios, warm-up schedules, disabled losses, and other implementation settings from Method.
- Each module subsection follows motivation, design, then technical advantage.
- Each paragraph carries one message and opens with its main claim.
- Prefer one compact equation per technical subsection and avoid multi-line prose formulas.
- Target approximately 450--500 words for Method.
- Use restrained terminology: one framework name and no unnecessary acronyms beyond established module abbreviations.
- Figure typography, colors, line weights, and variable names must remain consistent; use a minimal pastel palette with clear visual/semantic/synthesis grouping and no decorative effects that reduce print readability.

## Accuracy Boundary

The revision may omit non-core implementation details and avoid emphasizing limitations, but it must not claim that the method performs operations absent from the implementation. Claims of superiority remain in Experiments and require measured evidence.

## Verification

- Confirm exactly four Method subsections.
- Confirm every figure variable is defined in Method.
- Confirm equations match the implemented data flow.
- Run LaTeX/static checks and undefined-reference scans.
- Recount manuscript words and, when a LaTeX engine is available, render the PDF to verify the four-page technical-content limit.
