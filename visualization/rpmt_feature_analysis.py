"""Utilities for reproducible RPMT feature-space analysis.

The numerical helpers intentionally depend only on NumPy. Heavy project and
plotting dependencies are imported lazily so this module can be unit-tested in
lightweight environments and reused from a notebook running the SGG stack.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import numpy as np


class ProjectorCapture:
    """Forward hook that snapshots every invocation of ``MTMProjector``."""

    def __init__(self):
        self.calls = []

    def clear(self):
        self.calls.clear()

    def __call__(self, _module, inputs, output):
        raw = _to_numpy(inputs[0]).copy()
        visual, projected = output
        self.calls.append(
            {
                "raw": raw,
                "visual": _to_numpy(visual).copy(),
                "projected": _to_numpy(projected).copy(),
            }
        )


def l2_normalize(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


def positive_relation_records(relation_matrix):
    """Return ``(subject, object, predicate, pair_position)`` records.

    ``pair_position`` follows ``RelationSampling.prepare_test_pairs``: all
    directed non-diagonal pairs in row-major order. This index is what aligns a
    GT relation with the corresponding projector output during PredCls testing.
    """
    matrix = np.asarray(relation_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("relation_matrix must be square")
    records = []
    pair_position = 0
    for subject in range(matrix.shape[0]):
        for object_index in range(matrix.shape[1]):
            if subject == object_index:
                continue
            predicate = int(matrix[subject, object_index])
            if predicate > 0:
                records.append(
                    (subject, object_index, predicate, pair_position)
                )
            pair_position += 1
    return records


class BalancedReservoir:
    """Independent fixed-size reservoir for every predicate class."""

    def __init__(self, max_per_class, seed=2027):
        if int(max_per_class) <= 0:
            raise ValueError("max_per_class must be positive")
        self.max_per_class = int(max_per_class)
        self._rng = random.Random(int(seed))
        self._records = defaultdict(list)
        self.counts_seen = {}

    def add(self, predicate_id, record):
        predicate_id = int(predicate_id)
        seen = self.counts_seen.get(predicate_id, 0) + 1
        self.counts_seen[predicate_id] = seen
        bucket = self._records[predicate_id]
        if len(bucket) < self.max_per_class:
            bucket.append(record)
            return
        replacement = self._rng.randrange(seen)
        if replacement < self.max_per_class:
            bucket[replacement] = record

    def by_class(self):
        return {key: list(self._records[key]) for key in sorted(self._records)}

    def records(self):
        flattened = []
        for predicate_id in sorted(self._records):
            flattened.extend(self._records[predicate_id])
        return flattened


def cosine_similarity_matrix(features):
    normalized = l2_normalize(features)
    return normalized @ normalized.T


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left, right):
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator <= 1e-12:
        return 1.0 if np.allclose(left, right) else float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def _knn_overlap(left_similarity, right_similarity, k):
    size = left_similarity.shape[0]
    if size < 2:
        return float("nan"), 0
    effective_k = min(max(int(k), 1), size - 1)
    left = left_similarity.copy()
    right = right_similarity.copy()
    np.fill_diagonal(left, -np.inf)
    np.fill_diagonal(right, -np.inf)
    left_neighbors = np.argpartition(left, -effective_k, axis=1)[:, -effective_k:]
    right_neighbors = np.argpartition(right, -effective_k, axis=1)[:, -effective_k:]
    overlaps = [
        len(set(a.tolist()).intersection(b.tolist())) / float(effective_k)
        for a, b in zip(left_neighbors, right_neighbors)
    ]
    return float(np.mean(overlaps)), effective_k


def structure_metrics(visual, projected, k=10):
    visual = np.asarray(visual)
    projected = np.asarray(projected)
    if visual.shape != projected.shape:
        raise ValueError("visual and projected features must have identical shapes")
    if visual.ndim != 2 or visual.shape[0] < 2:
        raise ValueError("at least two feature vectors are required")
    visual_similarity = cosine_similarity_matrix(visual)
    projected_similarity = cosine_similarity_matrix(projected)
    upper = np.triu_indices(visual.shape[0], k=1)
    visual_pairs = visual_similarity[upper]
    projected_pairs = projected_similarity[upper]
    overlap, effective_k = _knn_overlap(
        visual_similarity, projected_similarity, k
    )
    return {
        "structure_mae": float(np.mean(np.abs(visual_pairs - projected_pairs))),
        "structure_spearman": _spearman(visual_pairs, projected_pairs),
        "knn_overlap@{}".format(effective_k): overlap,
    }


def paired_cosine(left, right):
    left = l2_normalize(left)
    right = l2_normalize(right)
    if left.shape != right.shape:
        raise ValueError("paired feature arrays must have identical shapes")
    return float(np.mean(np.sum(left * right, axis=1)))


def anchor_interclass_similarity(anchors, predicate_ids):
    anchors = np.asarray(anchors)
    predicate_ids = np.asarray(predicate_ids)
    centroids = []
    for predicate_id in np.unique(predicate_ids):
        centroids.append(l2_normalize(anchors[predicate_ids == predicate_id].mean(0)))
    if len(centroids) < 2:
        return float("nan")
    similarity = cosine_similarity_matrix(np.stack(centroids))
    upper = np.triu_indices(len(centroids), k=1)
    return float(similarity[upper].mean())


def feature_arrays(records):
    if not records:
        raise ValueError("no feature records were collected")
    vector_keys = ("visual", "projected", "anchor_raw", "anchor_refined")
    arrays = {key: np.stack([record[key] for record in records]) for key in vector_keys}
    for key in (
        "predicate_id",
        "subject_id",
        "object_id",
        "image_id",
        "pair_position",
    ):
        arrays[key] = np.asarray([record[key] for record in records])
    arrays["split"] = np.asarray([record["split"] for record in records])
    arrays["predicate_name"] = np.asarray(
        [record["predicate_name"] for record in records]
    )
    arrays["sample_key"] = np.asarray([record["sample_key"] for record in records])
    return arrays


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def sample_dataset_occurrences(
    dataset,
    selected_predicates,
    max_per_class=80,
    seed=2027,
):
    """Count every predicate and sample selected GT relation occurrences."""
    selected = set(selected_predicates)
    predicate_names = list(dataset.ind_to_predicates)
    support = defaultdict(int)
    reservoir = BalancedReservoir(max_per_class=max_per_class, seed=seed)
    random_state = random.getstate()
    random.seed(int(seed))
    try:
        for image_index in range(len(dataset)):
            target = dataset.get_groundtruth(image_index)
            relation_matrix = _to_numpy(target.get_field("relation"))
            for subject, object_index, predicate_id, pair_position in positive_relation_records(
                relation_matrix
            ):
                predicate_name = predicate_names[predicate_id]
                support[predicate_name] += 1
                if predicate_name in selected:
                    reservoir.add(
                        predicate_id,
                        {
                            "image_index": image_index,
                            "subject_index": subject,
                            "object_index": object_index,
                            "predicate_id": predicate_id,
                            "predicate_name": predicate_name,
                            "pair_position": pair_position,
                        },
                    )
    finally:
        random.setstate(random_state)
    return dict(sorted(support.items())), reservoir.records()


def align_feature_arrays(full, ablation):
    full_keys = np.asarray(full["sample_key"])
    ablation_keys = np.asarray(ablation["sample_key"])
    if len(set(full_keys.tolist())) != len(full_keys):
        raise ValueError("full-model sample keys are not unique")
    if len(set(ablation_keys.tolist())) != len(ablation_keys):
        raise ValueError("ablation sample keys are not unique")
    ablation_index = {key: index for index, key in enumerate(ablation_keys.tolist())}
    missing = [key for key in full_keys.tolist() if key not in ablation_index]
    if missing:
        raise ValueError("ablation cache is missing {} paired samples".format(len(missing)))
    order = np.asarray([ablation_index[key] for key in full_keys.tolist()])
    aligned_ablation = {
        key: np.asarray(value)[order]
        for key, value in ablation.items()
        if len(np.asarray(value)) == len(ablation_keys)
    }
    aligned_full = {
        key: np.asarray(value)
        for key, value in full.items()
        if len(np.asarray(value)) == len(full_keys)
    }
    return aligned_full, aligned_ablation


def summarize_metrics(arrays, k=10):
    rows = []
    for split_name in ("overall", "seen", "unseen"):
        mask = (
            np.ones(len(arrays["predicate_id"]), dtype=bool)
            if split_name == "overall"
            else arrays["split"] == split_name
        )
        if mask.sum() < 2:
            continue
        visual = arrays["visual"][mask]
        projected = arrays["projected"][mask]
        refined = arrays["anchor_refined"][mask]
        row = {"split": split_name, "samples": int(mask.sum())}
        row["alignment_cosine"] = paired_cosine(projected, refined)
        row.update(structure_metrics(visual, projected, k=k))
        row["raw_anchor_interclass_cosine"] = anchor_interclass_similarity(
            arrays["anchor_raw"][mask], arrays["predicate_id"][mask]
        )
        row["refined_anchor_interclass_cosine"] = anchor_interclass_similarity(
            refined, arrays["predicate_id"][mask]
        )
        rows.append(row)
    return rows


def joint_reduce(named_arrays, method="pca", seed=2027, perplexity=30.0):
    """Fit one reducer to all named arrays and split the coordinates back."""
    if not named_arrays:
        raise ValueError("named_arrays cannot be empty")
    names = list(named_arrays)
    normalized = [l2_normalize(named_arrays[name]) for name in names]
    widths = {array.shape[1] for array in normalized}
    if len(widths) != 1:
        raise ValueError("all feature sets must have the same dimensionality")
    combined = np.concatenate(normalized, axis=0)
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=int(seed))
    elif method in ("tsne", "t-sne"):
        from sklearn.manifold import TSNE

        if combined.shape[0] < 3:
            raise ValueError("t-SNE requires at least three samples")
        effective_perplexity = min(float(perplexity), combined.shape[0] - 1.0)
        reducer = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=int(seed),
        )
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    coordinates = reducer.fit_transform(combined)
    result = {}
    offset = 0
    for name, array in zip(names, normalized):
        result[name] = coordinates[offset : offset + len(array)]
        offset += len(array)
    return result


def filter_feature_arrays(arrays, selected_predicates):
    selected = tuple(selected_predicates)
    if not selected:
        raise ValueError(
            "SELECTED_PREDICATES is empty. Inspect the support table and "
            "choose predicates before producing a paper figure."
        )
    available = set(np.asarray(arrays["predicate_name"]).tolist())
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError("selected predicates are absent from the cache: {}".format(missing))
    mask = np.isin(arrays["predicate_name"], selected)
    return {
        key: np.asarray(value)[mask]
        for key, value in arrays.items()
        if len(np.asarray(value)) == len(mask)
    }


def export_figure(figure, output_stem, dpi=600):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    return (pdf_path, png_path)


def paper_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.7,
            "legend.fontsize": 7,
            "figure.dpi": 140,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def predicate_palette(predicate_names):
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(predicate_names))
    colormap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    return {name: colormap(index % colormap.N) for index, name in enumerate(names)}


def _clean_axis(axis, title):
    axis.set_title(title, pad=5)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#B8BEC7")
        spine.set_linewidth(0.65)
    axis.set_facecolor("#FAFBFC")


def _add_modality_legend(axis):
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor="#6B7280", markeredgecolor="white",
            label="Projected relation",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=4.5,
            markerfacecolor="none", markeredgecolor="#4B5563",
            label="Triplet anchor",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#D1D5DB",
        borderpad=0.35,
        handletextpad=0.35,
    )


def _scatter_by_predicate(
    axis,
    coordinates,
    predicate_names,
    palette,
    marker="o",
    alpha=0.68,
    size=16,
    label_suffix="",
    hollow=False,
):
    predicate_names = np.asarray(predicate_names)
    for predicate in dict.fromkeys(predicate_names.tolist()):
        mask = predicate_names == predicate
        kwargs = {
            "s": size,
            "marker": marker,
            "alpha": alpha,
            "linewidths": 0.7,
            "label": "{}{}".format(predicate, label_suffix),
        }
        if hollow:
            kwargs.update(facecolors="none", edgecolors=[palette[predicate]])
        else:
            kwargs.update(c=[palette[predicate]], edgecolors="white")
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], **kwargs)


def _connect_class_centroids(axis, left, right, predicate_names, palette):
    predicate_names = np.asarray(predicate_names)
    for predicate in dict.fromkeys(predicate_names.tolist()):
        mask = predicate_names == predicate
        start = np.asarray(left)[mask].mean(axis=0)
        end = np.asarray(right)[mask].mean(axis=0)
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=palette[predicate],
            alpha=0.55,
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            zorder=0,
        )


def make_three_stage_figure(
    arrays,
    selected_predicates,
    method="pca",
    seed=2027,
    perplexity=30.0,
):
    """Visual space, refined semantic space, and their aligned shared space."""
    import matplotlib.pyplot as plt

    paper_style()
    selected = filter_feature_arrays(arrays, selected_predicates)
    names = selected["predicate_name"]
    palette = predicate_palette(selected_predicates)
    visual_xy = joint_reduce(
        {"visual": selected["visual"]}, method, seed, perplexity
    )["visual"]
    semantic_xy = joint_reduce(
        {"semantic": selected["anchor_refined"]}, method, seed, perplexity
    )["semantic"]
    aligned_xy = joint_reduce(
        {
            "projected": selected["projected"],
            "semantic": selected["anchor_refined"],
        },
        method,
        seed,
        perplexity,
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.05))
    _scatter_by_predicate(axes[0], visual_xy, names, palette)
    _clean_axis(axes[0], "Visual relation space")
    _scatter_by_predicate(axes[1], semantic_xy, names, palette, marker="D", size=18)
    _clean_axis(axes[1], "Triplet semantic space")
    _scatter_by_predicate(axes[2], aligned_xy["projected"], names, palette)
    _scatter_by_predicate(
        axes[2],
        aligned_xy["semantic"],
        names,
        palette,
        marker="D",
        alpha=0.9,
        size=19,
        hollow=True,
    )
    _connect_class_centroids(
        axes[2], aligned_xy["projected"], aligned_xy["semantic"], names, palette
    )
    _clean_axis(axes[2], "Aligned shared space")
    _add_modality_legend(axes[2])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.subplots_adjust(bottom=0.24, left=0.02, right=0.995, wspace=0.12)
    return figure


def make_split_comparison_figure(
    arrays,
    selected_predicates,
    method="pca",
    seed=2027,
    perplexity=30.0,
    compact=False,
):
    """Compare input geometry and aligned space, separated by seen status."""
    import matplotlib.pyplot as plt

    paper_style()
    selected = filter_feature_arrays(arrays, selected_predicates)
    palette = predicate_palette(selected_predicates)
    row_splits = ["overall"] if compact else ["seen", "unseen"]
    figure, axes = plt.subplots(
        len(row_splits),
        2,
        squeeze=False,
        figsize=((3.45, 1.65) if compact else (7.05, 3.45)),
    )
    for row, split_name in enumerate(row_splits):
        mask = (
            np.ones(len(selected["split"]), dtype=bool)
            if split_name == "overall"
            else selected["split"] == split_name
        )
        if mask.sum() < 2:
            for axis in axes[row]:
                axis.text(0.5, 0.5, "No {} samples".format(split_name), ha="center")
                _clean_axis(axis, split_name.title())
            continue
        names = selected["predicate_name"][mask]
        before = joint_reduce(
            {"visual": selected["visual"][mask]}, method, seed, perplexity
        )["visual"]
        after = joint_reduce(
            {
                "projected": selected["projected"][mask],
                "semantic": selected["anchor_refined"][mask],
            },
            method,
            seed,
            perplexity,
        )
        _scatter_by_predicate(axes[row, 0], before, names, palette)
        _scatter_by_predicate(axes[row, 1], after["projected"], names, palette)
        _scatter_by_predicate(
            axes[row, 1],
            after["semantic"],
            names,
            palette,
            marker="D",
            size=18,
            hollow=True,
        )
        _connect_class_centroids(
            axes[row, 1], after["projected"], after["semantic"], names, palette
        )
        prefix = "" if compact else "{}: ".format(split_name.title())
        _clean_axis(axes[row, 0], prefix + "visual geometry")
        _clean_axis(axes[row, 1], prefix + "aligned space")
        _add_modality_legend(axes[row, 1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(3 if compact else 5, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(
        bottom=0.17 if compact else 0.12,
        left=0.02,
        right=0.995,
        hspace=0.22,
        wspace=0.10,
    )
    return figure


def make_anchor_refinement_figure(
    arrays,
    selected_predicates,
    method="pca",
    seed=2027,
    perplexity=30.0,
):
    import matplotlib.pyplot as plt

    paper_style()
    selected = filter_feature_arrays(arrays, selected_predicates)
    names = selected["predicate_name"]
    palette = predicate_palette(selected_predicates)
    coordinates = joint_reduce(
        {
            "raw": selected["anchor_raw"],
            "refined": selected["anchor_refined"],
        },
        method,
        seed,
        perplexity,
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.0))
    for axis, key, title in zip(
        axes,
        ("raw", "refined"),
        ("Raw triplet anchors", "Debiased triplet anchors"),
    ):
        _scatter_by_predicate(axis, coordinates[key], names, palette, marker="D", size=18)
        _clean_axis(axis, title)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(bottom=0.22, left=0.02, right=0.995, wspace=0.10)
    return figure


def make_ablation_figure(
    full_arrays,
    ablation_arrays,
    selected_predicates,
    method="pca",
    seed=2027,
    perplexity=30.0,
):
    """Direct paired comparison against a w/o-structure checkpoint."""
    import matplotlib.pyplot as plt

    paper_style()
    full, ablation = align_feature_arrays(full_arrays, ablation_arrays)
    full = filter_feature_arrays(full, selected_predicates)
    ablation = filter_feature_arrays(ablation, selected_predicates)
    names = full["predicate_name"]
    palette = predicate_palette(selected_predicates)
    coordinates = joint_reduce(
        {
            "ablation": ablation["projected"],
            "full": full["projected"],
            "semantic": full["anchor_refined"],
        },
        method,
        seed,
        perplexity,
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.0))
    for axis, key, title in zip(
        axes,
        ("ablation", "full"),
        ("w/o structure preservation", "RPMT"),
    ):
        _scatter_by_predicate(axis, coordinates[key], names, palette)
        _scatter_by_predicate(
            axis,
            coordinates["semantic"],
            names,
            palette,
            marker="D",
            alpha=0.9,
            size=19,
            hollow=True,
        )
        _connect_class_centroids(
            axis, coordinates[key], coordinates["semantic"], names, palette
        )
        _clean_axis(axis, title)
        _add_modality_legend(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    keep = [index for index, label in enumerate(labels) if label in selected_predicates]
    figure.legend(
        [handles[index] for index in keep],
        [labels[index] for index in keep],
        loc="lower center",
        ncol=min(5, len(keep)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(bottom=0.22, left=0.02, right=0.995, wspace=0.10)
    return figure


def _trusted_torch_load(path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def checkpoint_compatibility_report(model, checkpoint_path):
    checkpoint = _trusted_torch_load(checkpoint_path)
    state = checkpoint.get("model", checkpoint)
    model_keys = set(model.state_dict())
    checkpoint_keys = set(state)
    mtm_keys = {key for key in checkpoint_keys if ".mtm." in key or key.startswith("mtm.")}
    return {
        "checkpoint_parameters": len(checkpoint_keys),
        "model_parameters": len(model_keys),
        "exact_key_matches": len(model_keys.intersection(checkpoint_keys)),
        "missing_exact_keys": len(model_keys.difference(checkpoint_keys)),
        "unexpected_exact_keys": len(checkpoint_keys.difference(model_keys)),
        "mtm_checkpoint_keys": len(mtm_keys),
    }


def load_analysis_runtime(
    config_path,
    checkpoint_path,
    dataset_name="VG",
    device=None,
):
    """Build the exact project model and test dataset for feature collection.

    Checkpoints are executable pickle files in this legacy stack; only load
    checkpoints from a trusted source.
    """
    import torch
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.data import make_data_loader
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError("CONFIG_PATH does not exist: {}".format(config_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("CHECKPOINT_PATH does not exist: {}".format(checkpoint_path))

    cfg.defrost()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHT = str(checkpoint_path)
    cfg.MODEL.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.OV_SETTING.TEST_PART = "total"
    cfg.MODEL.ROI_RELATION_HEAD.MTM.INFERENCE.ENABLED = True
    if float(cfg.MODEL.ROI_RELATION_HEAD.MTM.INFERENCE.SCORE_WEIGHT) == 0.0:
        cfg.MODEL.ROI_RELATION_HEAD.MTM.INFERENCE.SCORE_WEIGHT = 1.0
    cfg.freeze()

    if not cfg.MODEL.ROI_RELATION_HEAD.MTM.ENABLED:
        raise ValueError("the supplied configuration does not enable RPMT/MTM")
    if not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX or not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
        raise ValueError("feature analysis requires the PredCls protocol")

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.updata("total")
    compatibility = checkpoint_compatibility_report(model, checkpoint_path)
    if compatibility["mtm_checkpoint_keys"] == 0:
        raise ValueError("checkpoint contains no RPMT/MTM parameters")
    checkpointer = DetectronCheckpointer(cfg, model, save_dir="")
    checkpointer.load(str(checkpoint_path), with_optim=False)
    model.eval()

    loaders = make_data_loader(
        cfg=cfg,
        mode="test",
        is_distributed=False,
        dataset_to_test="test",
    )
    if len(loaders) != 1:
        raise ValueError(
            "expected one test dataset, found {}. Select one DATASETS.TEST entry in the YAML.".format(
                len(loaders)
            )
        )
    predictor = model.roi_heads.relation.predictor
    if getattr(predictor, "mtm", None) is None:
        raise ValueError("the constructed predictor has no RPMT/MTM plugin")
    return {
        "cfg": cfg,
        "model": model,
        "dataset": loaders[0].dataset,
        "predictor": predictor,
        "mtm": predictor.mtm,
        "device": torch.device(cfg.MODEL.DEVICE),
        "dataset_name": str(dataset_name),
        "compatibility": compatibility,
    }


def collect_checkpoint_features(runtime, occurrences, progress=True):
    """Run selected images and collect real GT-positive RPMT features."""
    import torch
    from torch.nn import functional as functional
    from maskrcnn_benchmark.data.collate_batch import BatchCollator

    if not occurrences:
        raise ValueError(
            "no selected occurrences; set SELECTED_PREDICATES after inspecting support"
        )
    grouped = defaultdict(list)
    for occurrence in occurrences:
        grouped[int(occurrence["image_index"])].append(occurrence)
    image_indices = sorted(grouped)
    if progress:
        try:
            from tqdm.auto import tqdm

            image_indices = tqdm(image_indices, desc="Extracting RPMT features")
        except ImportError:
            pass

    model = runtime["model"]
    dataset = runtime["dataset"]
    mtm = runtime["mtm"]
    predictor = runtime["predictor"]
    device = runtime["device"]
    collator = BatchCollator(runtime["cfg"].DATALOADER.SIZE_DIVISIBILITY)
    capture = ProjectorCapture()
    hook = mtm.projector.register_forward_hook(capture)
    records = []
    try:
        for image_index in image_indices:
            image, target, returned_image_id = dataset[image_index]
            images, targets, _ = collator([(image, target, returned_image_id)])
            target_on_device = target.to(device)
            capture.clear()
            with torch.no_grad():
                model(images.to(device), [target_on_device])
            if len(capture.calls) != 1:
                raise RuntimeError(
                    "expected one projector call for image {}, observed {}".format(
                        image_index, len(capture.calls)
                    )
                )
            captured = capture.calls[0]
            object_count = len(target)
            expected_pairs = object_count * max(object_count - 1, 0)
            if len(captured["visual"]) != expected_pairs:
                raise RuntimeError(
                    "projector/pair mismatch for image {}: {} features vs {} pairs".format(
                        image_index, len(captured["visual"]), expected_pairs
                    )
                )

            labels = _to_numpy(target.get_field("labels")).astype(np.int64)
            selected_occurrences = grouped[int(image_index)]
            texts = [
                mtm.teacher.format_triplet(
                    mtm.teacher.object_names[int(labels[item["subject_index"]])],
                    item["predicate_name"],
                    mtm.teacher.object_names[int(labels[item["object_index"]])],
                )
                for item in selected_occurrences
            ]
            with torch.no_grad():
                raw_anchors = mtm.teacher.encode_raw(texts).float()
                components = mtm.teacher.principal_components
                if components is None:
                    refined_anchors = raw_anchors
                else:
                    components = components.to(raw_anchors)
                    refined_anchors = functional.normalize(
                        raw_anchors - (raw_anchors @ components) @ components.t(),
                        dim=-1,
                    )
            raw_anchors = _to_numpy(raw_anchors)
            refined_anchors = _to_numpy(refined_anchors)
            seen_ids = set(int(value) for value in predictor.base)
            unseen_ids = set(int(value) for value in predictor.novel)

            for local_index, occurrence in enumerate(selected_occurrences):
                pair_position = int(occurrence["pair_position"])
                predicate_id = int(occurrence["predicate_id"])
                if predicate_id in seen_ids:
                    split = "seen"
                elif predicate_id in unseen_ids:
                    split = "unseen"
                else:
                    split = "other"
                subject = int(occurrence["subject_index"])
                object_index = int(occurrence["object_index"])
                sample_key = "{}:{}:{}:{}".format(
                    image_index, subject, object_index, predicate_id
                )
                records.append(
                    {
                        "sample_key": sample_key,
                        "image_id": int(returned_image_id),
                        "pair_position": pair_position,
                        "subject_id": int(labels[subject]),
                        "object_id": int(labels[object_index]),
                        "predicate_id": predicate_id,
                        "predicate_name": occurrence["predicate_name"],
                        "split": split,
                        "visual": captured["visual"][pair_position],
                        "projected": captured["projected"][pair_position],
                        "anchor_raw": raw_anchors[local_index],
                        "anchor_refined": refined_anchors[local_index],
                    }
                )
    finally:
        hook.remove()
    arrays = feature_arrays(records)
    for key in ("visual", "projected", "anchor_raw", "anchor_refined"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError("{} contains NaN or Inf".format(key))
        if np.any(np.linalg.norm(arrays[key], axis=1) <= 1e-12):
            raise ValueError("{} contains zero-norm vectors".format(key))
    return arrays


def save_feature_cache(path, arrays, metadata):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": {key: value for key, value in arrays.items()},
        "metadata": dict(metadata),
    }
    torch.save(payload, str(path))


def load_feature_cache(path):
    payload = _trusted_torch_load(path)
    if "features" not in payload or "metadata" not in payload:
        raise ValueError("invalid RPMT feature cache")
    return payload["features"], payload["metadata"]
