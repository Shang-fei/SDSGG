# MTM SVD and Predicate Indexing Notes

## Current ClipPredictor Switching Logic

`ClipPredictor.updata(mode)` switches the active predicate space by slicing the CSV-driven relation tables:

- `base`: `description_relation.iloc[self.base, 1:]`
- `novel`: `description_relation.iloc[self.novel, 1:]`
- `semantic`: `description_relation.iloc[self.semantic, 1:]`
- `total`: `description_relation.iloc[:, 1:]`

The sliced table row order becomes the model output order for that mode. During training, `RelationLossComputation` applies the same kind of slicing to `description_relation_loss`, then indexes it directly with `rel_labels`. That means relation labels used by this branch are local to the active output space, not global VG predicate ids.

Example: in `base` mode, label `1` means the first non-background row in the base-sliced table, not necessarily global predicate id `1` unless the sliced ordering happens to place it there.

## Important Constraint for MTM

MTM target text lookup must follow the active local predicate order:

```python
relationName = self.activeRelNames[int(relationLabel)]
```

It should not convert training labels to global predicate ids unless the sampler/loss pipeline is also changed to emit global ids. The current pipeline does not do that.

The previous conditional

```python
self.activeRelNames = self.trainRelNames if mode == self.trainPart else activeRelNames
```

is unsafe for MTM because it makes training use the full dataset relation list while the relation logits/loss still use the active sliced output space. That can silently supervise a base-local label with the wrong predicate name.

## How SVD Should Be Applied

The SVD/CDP step should not be computed on each mini-batch. It should be computed over a complete triplet text bank whose ordering matches the active predicate space used for supervision.

For the current code, the implemented design is:

1. Iterate the global filtered triplet space from the full `filter_total.csv`:
   `foreground subject x filtered non-background relations for subject x foreground object`.
2. Encode all triplet texts with CLIP in chunks.
3. Accumulate only the `512 x 512` text-space covariance.
4. Compute the top principal direction from that covariance.
5. Refine target/candidate triplet embeddings by removing the projection on that direction.

This is still not the dense `predicate x subject x object` space: `filter_total.csv` removes invalid subject-relation combinations before SVD statistics are computed.
This avoids using mini-batch SVD and also avoids global/local label mismatch.
It also avoids storing the full dense triplet bank on GPU.

If we later want a single global SVD bank, we need an explicit mapping from each active local predicate index to the corresponding global predicate id. That mapping must be used only for bank lookup, while relation logits and `rel_labels` remain local. Without that explicit mapping, a global bank is too easy to index incorrectly.

## Recommended Implementation Direction

The MTM helpers keep indexing explicit:

- `iterFilteredTripletTextChunks()`: streams filtered global triplet texts without materializing the full bank.
- `updateMtmTextSvdBasis()`: computes or reuses the filtered-global principal direction.
- `refineTripletTextFeatures(embeddings)`: removes the active mode principal direction.
- `encodeTripletTexts(texts)`: CLIP-encodes triplet texts and applies the refinement.

`updata(mode)` clears filtered candidate caches but keeps using the same filtered-global SVD basis. The active mode still controls relation-logit order and candidate scoring; it does not redefine the SVD statistics.

The behavior is controlled by:

- `MODEL.ROI_RELATION_HEAD.MTM.TEXT_TEACHER.SVD_ENABLED`
- `MODEL.ROI_RELATION_HEAD.MTM.TEXT_TEACHER.NUM_SVD_COMPONENTS`
- `MODEL.ROI_RELATION_HEAD.MTM.TEXT_TEACHER.ENCODE_BATCH_SIZE`
