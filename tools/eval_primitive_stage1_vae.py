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

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.roi_heads.relation_head.primitive_stage1_dataset import (
    PrimitiveStage1Dataset,
    build_raw_sgg_dataset,
    predicate_names_for_part,
    primitive_stage1_collate,
)
from maskrcnn_benchmark.modeling.roi_heads.relation_head.primitive_stage1_vae import (
    PrimitiveStage1VAE,
    build_clip_model,
    default_mapping_path,
    reconstruction_loss,
)
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate primitive triplet VAE Stage 1")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--predicate-part", default="total", choices=["base", "novel", "semantic", "total"])
    parser.add_argument("--predicate-split-file", default=None)
    parser.add_argument(
        "--prompt-mode",
        default="full",
        choices=["full", "primitive_only", "triplet_only"],
        help="full: primitive + VAE bias + triplet; primitive_only: primitive + triplet; triplet_only: raw triplet text.",
    )
    parser.add_argument("--mapping-file", default=default_mapping_path())
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hard-retrieval-min-candidates", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def retrieval_metrics(query, target):
    sim = query @ target.t()
    return summarize_ranks(ranks_from_similarity(sim))


def ranks_from_similarity(sim, candidate_mask=None, min_candidates=1):
    ranks = []
    candidate_counts = []
    for i in range(sim.shape[0]):
        if candidate_mask is None:
            candidates = torch.ones(sim.shape[1], dtype=torch.bool, device=sim.device)
        else:
            candidates = candidate_mask[i]
        if not bool(candidates[i]):
            candidates = candidates.clone()
            candidates[i] = True
        candidate_count = int(candidates.sum().item())
        if candidate_count < min_candidates:
            continue
        candidate_indices = torch.nonzero(candidates, as_tuple=False).squeeze(1)
        scores = sim[i, candidate_indices]
        order = candidate_indices[torch.argsort(scores, descending=True)]
        rank = int(torch.nonzero(order == i, as_tuple=False)[0, 0]) + 1
        ranks.append(rank)
        candidate_counts.append(candidate_count)
    if not ranks:
        return None, None
    return torch.tensor(ranks, device=sim.device), torch.tensor(candidate_counts, device=sim.device)


def summarize_ranks(rank_result, candidate_counts=None):
    if isinstance(rank_result, tuple):
        ranks, candidate_counts = rank_result
    else:
        ranks = rank_result
    if ranks is None or ranks.numel() == 0:
        return {
            "r@1": None,
            "r@5": None,
            "r@10": None,
            "mean_rank": None,
            "count": 0,
            "mean_candidates": None,
        }
    if candidate_counts is None:
        candidate_counts = torch.full_like(ranks, int(ranks.numel()))
    return {
        "r@1": float((ranks <= 1).float().mean().item()),
        "r@5": float((ranks <= 5).float().mean().item()),
        "r@10": float((ranks <= 10).float().mean().item()),
        "mean_rank": float(ranks.float().mean().item()),
        "count": int(ranks.numel()),
        "mean_candidates": float(candidate_counts.float().mean().item()),
    }


def hard_retrieval_metrics(query, target, groups, min_candidates=2):
    sim = query @ target.t()
    result = {}
    for name, labels in groups.items():
        masks = []
        for label in labels:
            masks.append(torch.tensor([other == label for other in labels], device=sim.device))
        candidate_mask = torch.stack(masks, dim=0)
        result[name] = summarize_ranks(
            ranks_from_similarity(
                sim,
                candidate_mask=candidate_mask,
                min_candidates=min_candidates,
            )
        )
    return result


def update_retrieval_totals(totals, metrics):
    if metrics["count"] == 0:
        return
    count = metrics["count"]
    for key in ["r@1", "r@5", "r@10", "mean_rank", "mean_candidates"]:
        totals[key] += metrics[key] * count
    totals["count"] += count


def finalize_retrieval_totals(totals):
    if totals["count"] == 0:
        return {
            "r@1": None,
            "r@5": None,
            "r@10": None,
            "mean_rank": None,
            "count": 0,
            "mean_candidates": None,
        }
    result = {
        key: totals[key] / totals["count"]
        for key in ["r@1", "r@5", "r@10", "mean_rank", "mean_candidates"]
    }
    result["count"] = totals["count"]
    return result


def empty_retrieval_totals():
    return {"r@1": 0.0, "r@5": 0.0, "r@10": 0.0, "mean_rank": 0.0, "mean_candidates": 0.0, "count": 0}


def main():
    args = parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device or cfg.MODEL.DEVICE)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    mkdir(output_dir)

    clip_model, preprocess = build_clip_model(checkpoint.get("clip_model", "ViT-B/32"), device)
    raw_dataset = build_raw_sgg_dataset(cfg, args.split)
    allowed_predicates = predicate_names_for_part(
        cfg,
        part=args.predicate_part,
        split_file=args.predicate_split_file,
    )
    dataset = PrimitiveStage1Dataset(
        raw_dataset,
        preprocess,
        mapping_path=args.mapping_file,
        max_samples=args.max_samples,
        allowed_predicates=allowed_predicates,
    )
    if len(dataset) == 0:
        raise RuntimeError("No Stage 1 samples found for predicate part {}".format(args.predicate_part))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=primitive_stage1_collate,
        pin_memory=True,
    )

    model = PrimitiveStage1VAE(
        clip_model,
        num_slots=dataset.mapping["num_slots"],
        n_ctx=4,
        max_slots_per_predicate=dataset.max_slots,
    ).to(device)
    model.load_trainable_state_dict(checkpoint["model"])
    model.eval()

    totals = {"mse": 0.0, "cos": 0.0, "count": 0}
    per_pred = defaultdict(lambda: {"mse": 0.0, "cos": 0.0, "count": 0})
    retrieval_totals = empty_retrieval_totals()
    hard_retrieval_totals = {
        "same_predicate": empty_retrieval_totals(),
        "same_subject_object": empty_retrieval_totals(),
        "same_triplet": empty_retrieval_totals(),
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["union_image"].to(device, non_blocking=True)
            slot_ids = batch["slot_ids"].to(device, non_blocking=True)
            target = model.encode_image(images)
            recon, _, _ = model(
                target,
                slot_ids,
                batch["triplet_text"],
                sample=False,
                prompt_mode=args.prompt_mode,
            )
            mse_vec = (F.normalize(recon, dim=-1) - F.normalize(target, dim=-1)).pow(2).mean(dim=1)
            cos_vec = F.cosine_similarity(recon, target, dim=-1)
            query = F.normalize(recon, dim=-1)
            target = F.normalize(target, dim=-1)
            metrics = retrieval_metrics(query, target)
            update_retrieval_totals(retrieval_totals, metrics)
            groups = {
                "same_predicate": batch["predicate_name"],
                "same_subject_object": [
                    "{}|{}".format(subject, obj)
                    for subject, obj in zip(batch["subject_name"], batch["object_name"])
                ],
                "same_triplet": batch["triplet_text"],
            }
            hard_metrics = hard_retrieval_metrics(
                query,
                target,
                groups,
                min_candidates=args.hard_retrieval_min_candidates,
            )
            for name, row in hard_metrics.items():
                update_retrieval_totals(hard_retrieval_totals[name], row)
            for i, pred in enumerate(batch["predicate_name"]):
                per_pred[pred]["mse"] += float(mse_vec[i].item())
                per_pred[pred]["cos"] += float(cos_vec[i].item())
                per_pred[pred]["count"] += 1
            totals["mse"] += float(mse_vec.sum().item())
            totals["cos"] += float(cos_vec.sum().item())
            totals["count"] += images.shape[0]

    sim = model.prompt_learner.similarity_matrix().cpu()
    off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
    slot_usage = defaultdict(list)
    for pred, slots in dataset.mapping["predicate_to_slots"].items():
        for slot in slots:
            slot_usage[int(slot)].append(pred)

    summary = {
        "predicate_part": args.predicate_part,
        "predicate_split_file": args.predicate_split_file,
        "prompt_mode": args.prompt_mode,
        "allowed_predicates": sorted(allowed_predicates) if allowed_predicates is not None else None,
        "mse": totals["mse"] / totals["count"],
        "cosine": totals["cos"] / totals["count"],
        "retrieval": finalize_retrieval_totals(retrieval_totals),
        "hard_retrieval": {
            name: finalize_retrieval_totals(row)
            for name, row in hard_retrieval_totals.items()
        },
        "hard_retrieval_min_candidates": args.hard_retrieval_min_candidates,
        "primitive_similarity_offdiag_mean": float(off_diag.mean().item()),
        "primitive_similarity_offdiag_max": float(off_diag.max().item()),
        "slot_coverage": len(slot_usage),
        "num_slots": dataset.mapping["num_slots"],
        "num_samples": len(dataset),
    }
    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "unmapped_predicates.json"), "w") as f:
        json.dump(sorted(dataset.unmapped_predicates), f, indent=2)
    with open(os.path.join(output_dir, "skipped_predicates.json"), "w") as f:
        json.dump(sorted(dataset.skipped_predicates), f, indent=2)
    with open(os.path.join(output_dir, "predicate_split_info.json"), "w") as f:
        json.dump(
            {
                "predicate_part": args.predicate_part,
                "predicate_split_file": args.predicate_split_file,
                "prompt_mode": args.prompt_mode,
                "hard_retrieval_min_candidates": args.hard_retrieval_min_candidates,
                "allowed_predicates": sorted(allowed_predicates) if allowed_predicates is not None else None,
                "skipped_predicates": sorted(dataset.skipped_predicates),
                "num_samples": len(dataset),
            },
            f,
            indent=2,
            )

    with open(os.path.join(output_dir, "hard_retrieval_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "count",
                "mean_candidates",
                "r@1",
                "r@5",
                "r@10",
                "mean_rank",
            ],
        )
        writer.writeheader()
        for name, row in sorted(summary["hard_retrieval"].items()):
            writer.writerow(
                {
                    "group": name,
                    "count": row["count"],
                    "mean_candidates": row["mean_candidates"],
                    "r@1": row["r@1"],
                    "r@5": row["r@5"],
                    "r@10": row["r@10"],
                    "mean_rank": row["mean_rank"],
                }
            )

    with open(os.path.join(output_dir, "per_predicate_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["predicate", "count", "mse", "cosine"])
        writer.writeheader()
        for pred, row in sorted(per_pred.items()):
            writer.writerow(
                {
                    "predicate": pred,
                    "count": row["count"],
                    "mse": row["mse"] / row["count"],
                    "cosine": row["cos"] / row["count"],
                }
            )

    with open(os.path.join(output_dir, "primitive_slot_usage.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slot", "name", "count", "predicates"])
        writer.writeheader()
        for slot in range(dataset.mapping["num_slots"]):
            info = dataset.mapping["slots"].get(str(slot), {})
            preds = sorted(slot_usage.get(slot, []))
            writer.writerow(
                {
                    "slot": slot,
                    "name": info.get("name", ""),
                    "count": len(preds),
                    "predicates": "|".join(preds),
                }
            )

    with open(os.path.join(output_dir, "primitive_similarity.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sim.tolist())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
