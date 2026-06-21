import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.roi_heads.relation_head.primitive_stage1_dataset import (
    PrimitiveStage1Dataset,
    build_raw_sgg_dataset,
    predicate_names_for_part,
    primitive_stage1_collate,
)
from maskrcnn_benchmark.modeling.roi_heads.relation_head.primitive_stage1_vae import (
    build_clip_model,
    default_mapping_path,
    load_primitive_mapping,
)
from maskrcnn_benchmark.modeling.roi_heads.relation_head.primitive_visor_prism import (
    VISORPRISM,
    slot_labels,
)
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VISOR-PRISM standalone prototypes")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--predicate-part", default="novel", choices=["base", "novel", "semantic", "total"])
    parser.add_argument("--candidate-part", default="total", choices=["base", "novel", "semantic", "total"])
    parser.add_argument("--predicate-split-file", default=None)
    parser.add_argument("--mapping-file", default=default_mapping_path())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-encode-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def pad_slots(mapping, slots):
    max_slots = int(mapping["max_slots_per_predicate"])
    fallback = mapping["fallback_slots"]
    slots = [int(s) for s in slots[:max_slots]]
    if len(slots) < max_slots:
        slots += [int(fallback[0])] * (max_slots - len(slots))
    return slots


def candidate_slot_tensor(mapping, candidate_names, device):
    rows = []
    for name in candidate_names:
        rows.append(pad_slots(mapping, mapping["predicate_to_slots"].get(name, mapping["fallback_slots"])))
    return torch.tensor(rows, dtype=torch.long, device=device)


def build_candidate_predicates(raw_dataset, allowed_names):
    if allowed_names is not None:
        name_to_id = {name: pred_id for pred_id, name in enumerate(raw_dataset.ind_to_predicates) if pred_id > 0}
        pairs = []
        for idx, name in enumerate(sorted(allowed_names)):
            pairs.append((name_to_id.get(name, -100000 - idx), name))
        return pairs
    pairs = []
    for pred_id, name in enumerate(raw_dataset.ind_to_predicates):
        if pred_id == 0:
            continue
        pairs.append((pred_id, name))
    return pairs


def subject_object_texts(batch):
    return ["{} {}".format(s, o) for s, o in zip(batch["subject_name"], batch["object_name"])]


def summarize_ranks(ranks):
    if not ranks:
        return {"r@1": None, "r@5": None, "r@10": None, "mean_rank": None, "count": 0}
    ranks = torch.tensor(ranks, dtype=torch.float)
    return {
        "r@1": float((ranks <= 1).float().mean().item()),
        "r@5": float((ranks <= 5).float().mean().item()),
        "r@10": float((ranks <= 10).float().mean().item()),
        "mean_rank": float(ranks.mean().item()),
        "count": int(ranks.numel()),
    }


def metric_from_logits(logits, labels):
    pred = logits.sigmoid() >= 0.5
    gold = labels >= 0.5
    tp = (pred & gold).sum().float()
    fp = (pred & ~gold).sum().float()
    fn = (~pred & gold).sum().float()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(precision.item()), float(recall.item()), float(f1.item())


def geometry_pseudo_labels(geometry, num_slots):
    labels = torch.zeros(geometry.shape[0], num_slots, device=geometry.device)
    sub = geometry[:, 0:4]
    obj = geometry[:, 4:8]
    delta = geometry[:, 8:10]
    iou = geometry[:, 10]
    center_dist = torch.sqrt(delta[:, 0].pow(2) + delta[:, 1].pow(2))

    labels[:, 2] = (delta[:, 1].abs() > 0.12).float()  # vertical layout
    labels[:, 3] = (center_dist < 0.30).float()  # proximity
    labels[:, 0] = (iou > 0.02).float()  # contact/overlap proxy
    sub_contains_obj = (sub[:, 0] <= obj[:, 0]) & (sub[:, 1] <= obj[:, 1]) & (sub[:, 2] >= obj[:, 2]) & (sub[:, 3] >= obj[:, 3])
    obj_contains_sub = (obj[:, 0] <= sub[:, 0]) & (obj[:, 1] <= sub[:, 1]) & (obj[:, 2] >= sub[:, 2]) & (obj[:, 3] >= sub[:, 3])
    labels[:, 4] = (sub_contains_obj | obj_contains_sub).float()
    mask = torch.zeros_like(labels)
    mask[:, [0, 2, 3, 4]] = 1.0
    return labels, mask


def encode_text_chunked(model, texts, device, chunk_size):
    features = []
    for start in range(0, len(texts), chunk_size):
        features.append(model.encode_text(texts[start:start + chunk_size], device).cpu())
    return torch.cat(features, dim=0).to(device)


def build_scores(
    model,
    mode,
    target,
    candidate_features,
    candidate_names,
    batch,
    geometry,
    so_features,
    random_primitives,
    text_encode_batch_size,
):
    device = target.device
    bsz = target.shape[0]
    num_candidates = len(candidate_names)
    if mode == "clip_text":
        texts = []
        for subject, obj in zip(batch["subject_name"], batch["object_name"]):
            for predicate in candidate_names:
                texts.append("{} {} {}".format(subject, predicate, obj))
        proto = encode_text_chunked(model, texts, device, text_encode_batch_size).view(bsz, num_candidates, -1)
    elif mode == "text_only_primitive":
        condition, _, _ = model.compose(candidate_features)
        proto = condition.unsqueeze(0).expand(bsz, -1, -1)
    elif mode == "random_anchors":
        condition, _, _ = model.compose(candidate_features, primitive_override=random_primitives)
        proto = condition.unsqueeze(0).expand(bsz, -1, -1)
    elif mode == "c_so_only":
        proto = model.generate_so_only(so_features, geometry).unsqueeze(1).expand(-1, num_candidates, -1)
    else:
        use_so = mode != "no_c_so"
        cand_pred = candidate_features.unsqueeze(0).expand(bsz, -1, -1).reshape(-1, model.feature_dim)
        cand_so = so_features.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, model.feature_dim)
        cand_geom = geometry.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, geometry.shape[-1])
        proto, _, _ = model.generate(cand_pred, cand_so, cand_geom, so_dropout=0.0, use_so=use_so)
        proto = proto.view(bsz, num_candidates, -1)
    return torch.einsum("bd,bcd->bc", target, proto) / 1.0


def main():
    args = parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    mapping = load_primitive_mapping(args.mapping_file)
    device = torch.device(args.device or cfg.MODEL.DEVICE)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval_{}".format(args.predicate_part))
    mkdir(output_dir)

    clip_model, preprocess = build_clip_model(checkpoint.get("clip_model", "ViT-B/32"), device)
    raw_dataset = build_raw_sgg_dataset(cfg, args.split)
    eval_predicates = predicate_names_for_part(cfg, args.predicate_part, args.predicate_split_file)
    candidate_predicates = predicate_names_for_part(cfg, args.candidate_part, args.predicate_split_file)
    candidate_pairs = build_candidate_predicates(raw_dataset, candidate_predicates)
    candidate_ids = [pred_id for pred_id, _ in candidate_pairs]
    candidate_names = [name for _, name in candidate_pairs]
    candidate_id_to_col = {pred_id: idx for idx, pred_id in enumerate(candidate_ids)}
    if not candidate_names:
        raise RuntimeError("No candidate predicates found for {}".format(args.candidate_part))

    dataset = PrimitiveStage1Dataset(
        raw_dataset,
        preprocess,
        mapping_path=args.mapping_file,
        max_samples=args.max_samples,
        allowed_predicates=eval_predicates,
    )
    if len(dataset) == 0:
        raise RuntimeError("No VISOR-PRISM samples found for predicate part {}".format(args.predicate_part))
    pair_to_predicates = defaultdict(set)
    for sample in dataset.samples:
        pair_to_predicates[(sample["subject_name"], sample["object_name"])].add(sample["predicate_name"])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=primitive_stage1_collate,
        pin_memory=True,
        drop_last=False,
    )

    model = VISORPRISM(
        clip_model,
        mapping=mapping,
        num_obj_classes=len(raw_dataset.ind_to_classes),
        top_m=checkpoint.get("top_m", 4),
        geom_dim=11,
    ).to(device)
    model.load_trainable_state_dict(checkpoint["model"])
    model.eval()

    candidate_features = model.encode_text(candidate_names, device)
    candidate_slots = candidate_slot_tensor(mapping, candidate_names, device)
    candidate_labels = slot_labels(candidate_slots, model.num_slots)
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(2027)
    random_primitives = F.normalize(
        torch.randn(model.num_slots, model.feature_dim, generator=random_generator, device=device),
        dim=-1,
    )

    modes = ["full", "clip_text", "text_only_primitive", "c_so_only", "no_c_so", "random_anchors"]
    ranks = {mode: [] for mode in modes}
    same_so_ranks = {mode: [] for mode in modes}
    per_predicate = {mode: defaultdict(list) for mode in modes}
    probe_logits = []
    probe_labels = []
    geom_logits = []
    geom_labels = []
    geom_masks = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating VISOR-PRISM", dynamic_ncols=True):
            images = batch["union_image"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            slot_ids = batch["slot_ids"].to(device, non_blocking=True)
            target = model.encode_image(images)
            so_features = model.encode_text(subject_object_texts(batch), device)

            primitive_labels = slot_labels(slot_ids, model.num_slots)
            vis_logits = model.activation_head(target)
            probe_logits.append(vis_logits.cpu())
            probe_labels.append(primitive_labels.cpu())
            g_labels, g_masks = geometry_pseudo_labels(geometry, model.num_slots)
            geom_logits.append(vis_logits.cpu())
            geom_labels.append(g_labels.cpu())
            geom_masks.append(g_masks.cpu())

            score_by_mode = {
                mode: build_scores(
                    model,
                    mode,
                    target,
                    candidate_features,
                    candidate_names,
                    batch,
                    geometry,
                    so_features,
                    random_primitives,
                    args.text_encode_batch_size,
                )
                for mode in modes
            }

            for i, pred_id_tensor in enumerate(batch["predicate_id"]):
                pred_id = int(pred_id_tensor.item())
                if pred_id not in candidate_id_to_col:
                    continue
                gt_col = candidate_id_to_col[pred_id]
                is_same_so = len(pair_to_predicates[(batch["subject_name"][i], batch["object_name"][i])]) > 1
                for mode in modes:
                    scores = score_by_mode[mode][i] / args.temperature
                    order = torch.argsort(scores, descending=True)
                    rank = int(torch.nonzero(order == gt_col, as_tuple=False)[0, 0]) + 1
                    ranks[mode].append(rank)
                    per_predicate[mode][batch["predicate_name"][i]].append(rank)
                    if is_same_so:
                        same_so_ranks[mode].append(rank)

    summary = {
        "split": args.split,
        "predicate_part": args.predicate_part,
        "candidate_part": args.candidate_part,
        "num_candidates": len(candidate_names),
        "num_samples": len(dataset),
        "modes": {mode: summarize_ranks(ranks[mode]) for mode in modes},
        "same_so_raw": {mode: summarize_ranks(same_so_ranks[mode]) for mode in modes},
    }

    probe_logits = torch.cat(probe_logits, dim=0)
    probe_labels = torch.cat(probe_labels, dim=0)
    p, r, f1 = metric_from_logits(probe_logits, probe_labels)
    summary["primitive_probe"] = {"precision": p, "recall": r, "f1": f1}

    geom_logits = torch.cat(geom_logits, dim=0)
    geom_labels = torch.cat(geom_labels, dim=0)
    geom_masks = torch.cat(geom_masks, dim=0)
    geom_rows = []
    for slot_id, name in [(0, "overlap_contact_proxy"), (2, "vertical_layout"), (3, "proximity"), (4, "containment")]:
        mask = geom_masks[:, slot_id] > 0
        if mask.any():
            pp, rr, ff = metric_from_logits(geom_logits[mask, slot_id:slot_id + 1], geom_labels[mask, slot_id:slot_id + 1])
            geom_rows.append({"slot": slot_id, "name": name, "precision": pp, "recall": rr, "f1": ff})
    summary["geometry_probe"] = geom_rows

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "candidate_predicates.json"), "w") as f:
        json.dump(candidate_names, f, indent=2)

    with open(os.path.join(output_dir, "per_predicate.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "predicate", "count", "r@1", "r@5", "r@10", "mean_rank"])
        writer.writeheader()
        for mode in modes:
            for predicate, pred_ranks in sorted(per_predicate[mode].items()):
                row = summarize_ranks(pred_ranks)
                writer.writerow({"mode": mode, "predicate": predicate, **row})

    with open(os.path.join(output_dir, "primitive_probe.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slot", "name", "precision", "recall", "f1"])
        writer.writeheader()
        for row in geom_rows:
            writer.writerow(row)

    with open(os.path.join(output_dir, "same_so.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "count", "r@1", "r@5", "r@10", "mean_rank"])
        writer.writeheader()
        for mode in modes:
            writer.writerow({"mode": mode, **summary["same_so_raw"][mode]})

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
