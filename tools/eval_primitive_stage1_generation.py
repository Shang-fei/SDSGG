import argparse
import csv
import json
import os
import random
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
    PrimitiveStage1VAE,
    build_clip_model,
    default_mapping_path,
)
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate primitive Stage 1 prior generation")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--predicate-part", default="novel", choices=["base", "novel", "semantic", "total"])
    parser.add_argument("--predicate-split-file", default=None)
    parser.add_argument("--mapping-file", default=default_mapping_path())
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-triplets", type=int, default=200)
    parser.add_argument("--min-real-per-triplet", type=int, default=2)
    parser.add_argument("--num-gen-per-triplet", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def pairwise_offdiag_cosine(features):
    if features.shape[0] < 2:
        return None
    sim = features @ features.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    vals = sim[mask]
    return {
        "mean": float(vals.mean().item()),
        "max": float(vals.max().item()),
        "min": float(vals.min().item()),
        "std": float(vals.std(unbiased=False).item()),
    }


def cosine_stats(query, target):
    sim = query @ target.t()
    return {
        "mean": float(sim.mean().item()),
        "max": float(sim.max(dim=1).values.mean().item()),
    }


def main():
    args = parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device or cfg.MODEL.DEVICE)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "generation_eval")
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

    real_features = []
    metadata = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="encoding real union features", dynamic_ncols=True):
            images = batch["union_image"].to(device, non_blocking=True)
            features = model.encode_image(images).cpu()
            real_features.append(features)
            for i in range(features.shape[0]):
                metadata.append(
                    {
                        "triplet_text": batch["triplet_text"][i],
                        "predicate_name": batch["predicate_name"][i],
                        "subject_name": batch["subject_name"][i],
                        "object_name": batch["object_name"][i],
                        "slot_ids": batch["slot_ids"][i].tolist(),
                    }
                )

    real_features = F.normalize(torch.cat(real_features, dim=0), dim=-1).to(device)
    by_triplet = defaultdict(list)
    by_predicate = defaultdict(list)
    for idx, item in enumerate(metadata):
        by_triplet[item["triplet_text"]].append(idx)
        by_predicate[item["predicate_name"]].append(idx)

    candidate_triplets = [
        triplet for triplet, indices in by_triplet.items()
        if len(indices) >= args.min_real_per_triplet
    ]
    if args.max_triplets is not None and len(candidate_triplets) > args.max_triplets:
        candidate_triplets = random.sample(candidate_triplets, args.max_triplets)
    candidate_triplets = sorted(candidate_triplets)

    rows = []
    predicate_rows = defaultdict(lambda: {
        "count": 0,
        "same_triplet_mean": 0.0,
        "same_triplet_max": 0.0,
        "same_predicate_mean": 0.0,
        "same_predicate_max": 0.0,
        "coverage": 0.0,
        "diversity_mean": 0.0,
        "diversity_std": 0.0,
    })

    with torch.no_grad():
        for triplet in tqdm(candidate_triplets, desc="generating triplets", dynamic_ncols=True):
            indices = by_triplet[triplet]
            first = metadata[indices[0]]
            slot_ids = torch.tensor(first["slot_ids"], dtype=torch.long, device=device)
            generated = model.generate(slot_ids, [triplet], args.num_gen_per_triplet)
            generated = F.normalize(generated, dim=-1)

            same_triplet_real = real_features[torch.tensor(indices, dtype=torch.long, device=device)]
            same_predicate_indices = by_predicate[first["predicate_name"]]
            same_predicate_real = real_features[
                torch.tensor(same_predicate_indices, dtype=torch.long, device=device)
            ]
            same_triplet_stats = cosine_stats(generated, same_triplet_real)
            same_predicate_stats = cosine_stats(generated, same_predicate_real)
            coverage = float((same_triplet_real @ generated.t()).max(dim=1).values.mean().item())
            diversity = pairwise_offdiag_cosine(generated)

            row = {
                "triplet_text": triplet,
                "predicate": first["predicate_name"],
                "real_count": len(indices),
                "generated_count": args.num_gen_per_triplet,
                "same_triplet_mean_cos": same_triplet_stats["mean"],
                "same_triplet_max_cos": same_triplet_stats["max"],
                "same_predicate_mean_cos": same_predicate_stats["mean"],
                "same_predicate_max_cos": same_predicate_stats["max"],
                "coverage_real_to_generated": coverage,
                "generated_pairwise_cos_mean": diversity["mean"],
                "generated_pairwise_cos_std": diversity["std"],
                "generated_pairwise_cos_min": diversity["min"],
                "generated_pairwise_cos_max": diversity["max"],
            }
            rows.append(row)

            pred_row = predicate_rows[first["predicate_name"]]
            pred_row["count"] += 1
            pred_row["same_triplet_mean"] += row["same_triplet_mean_cos"]
            pred_row["same_triplet_max"] += row["same_triplet_max_cos"]
            pred_row["same_predicate_mean"] += row["same_predicate_mean_cos"]
            pred_row["same_predicate_max"] += row["same_predicate_max_cos"]
            pred_row["coverage"] += row["coverage_real_to_generated"]
            pred_row["diversity_mean"] += row["generated_pairwise_cos_mean"]
            pred_row["diversity_std"] += row["generated_pairwise_cos_std"]

    if not rows:
        raise RuntimeError("No triplets have at least {} real samples".format(args.min_real_per_triplet))

    summary = {
        "split": args.split,
        "predicate_part": args.predicate_part,
        "num_real_samples": len(metadata),
        "num_eval_triplets": len(rows),
        "num_gen_per_triplet": args.num_gen_per_triplet,
        "min_real_per_triplet": args.min_real_per_triplet,
        "same_triplet_mean_cos": sum(row["same_triplet_mean_cos"] for row in rows) / len(rows),
        "same_triplet_max_cos": sum(row["same_triplet_max_cos"] for row in rows) / len(rows),
        "same_predicate_mean_cos": sum(row["same_predicate_mean_cos"] for row in rows) / len(rows),
        "same_predicate_max_cos": sum(row["same_predicate_max_cos"] for row in rows) / len(rows),
        "coverage_real_to_generated": sum(row["coverage_real_to_generated"] for row in rows) / len(rows),
        "generated_pairwise_cos_mean": sum(row["generated_pairwise_cos_mean"] for row in rows) / len(rows),
        "generated_pairwise_cos_std": sum(row["generated_pairwise_cos_std"] for row in rows) / len(rows),
    }

    with open(os.path.join(output_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "generation_triplet_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(output_dir, "generation_predicate_metrics.csv"), "w", newline="") as f:
        fieldnames = [
            "predicate",
            "triplet_count",
            "same_triplet_mean_cos",
            "same_triplet_max_cos",
            "same_predicate_mean_cos",
            "same_predicate_max_cos",
            "coverage_real_to_generated",
            "generated_pairwise_cos_mean",
            "generated_pairwise_cos_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for predicate, row in sorted(predicate_rows.items()):
            count = row["count"]
            writer.writerow(
                {
                    "predicate": predicate,
                    "triplet_count": count,
                    "same_triplet_mean_cos": row["same_triplet_mean"] / count,
                    "same_triplet_max_cos": row["same_triplet_max"] / count,
                    "same_predicate_mean_cos": row["same_predicate_mean"] / count,
                    "same_predicate_max_cos": row["same_predicate_max"] / count,
                    "coverage_real_to_generated": row["coverage"] / count,
                    "generated_pairwise_cos_mean": row["diversity_mean"] / count,
                    "generated_pairwise_cos_std": row["diversity_std"] / count,
                }
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
