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
    PrimitiveStage1VAE,
    build_clip_model,
    default_mapping_path,
    load_primitive_mapping,
)
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="Rank predicates with primitive Stage 1 prototypes")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--predicate-part", default="novel", choices=["base", "novel", "semantic", "total"])
    parser.add_argument("--candidate-part", default="total", choices=["base", "novel", "semantic", "total"])
    parser.add_argument("--predicate-split-file", default=None)
    parser.add_argument("--mapping-file", default=default_mapping_path())
    parser.add_argument("--text-mode", default=None, choices=["subject_object", "triplet"])
    parser.add_argument("--n-ctx", type=int, default=None)
    parser.add_argument("--bias-mode", default=None, choices=["global", "token_wise"])
    parser.add_argument("--num-prototypes", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
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


def build_candidate_predicates(raw_dataset, allowed_names):
    pairs = []
    for pred_id, name in enumerate(raw_dataset.ind_to_predicates):
        if pred_id == 0:
            continue
        if allowed_names is not None and name not in allowed_names:
            continue
        pairs.append((pred_id, name))
    return pairs


def build_prompt_text(subject, predicate, obj, text_mode):
    if text_mode == "triplet":
        return "{} {} {}".format(subject, predicate, obj)
    if text_mode == "subject_object":
        return "{} {}".format(subject, obj)
    raise ValueError("Unknown text_mode: {}".format(text_mode))


def build_batch_prototypes(model, mapping, candidate_names, subjects, objects, text_mode, num_prototypes, device):
    slot_rows = []
    prompt_texts = []
    for subject, obj in zip(subjects, objects):
        for predicate in candidate_names:
            slots = mapping["predicate_to_slots"].get(predicate, mapping["fallback_slots"])
            slot_rows.append(pad_slots(mapping, slots))
            prompt_texts.append(build_prompt_text(subject, predicate, obj, text_mode))

    slot_ids = torch.tensor(slot_rows, dtype=torch.long, device=device)
    if num_prototypes == 1:
        z = torch.randn(len(prompt_texts), model.latent_dim, device=device)
        bias = model.generator(z)
        prompts, tokenized = model.build_prompt(slot_ids, bias, prompt_texts)
        features = model.text_encoder(prompts, tokenized)
        features = F.normalize(features.float(), dim=-1)
        return features.view(len(subjects), len(candidate_names), -1)

    slot_ids = slot_ids.repeat_interleave(num_prototypes, dim=0)
    repeated_texts = []
    for text in prompt_texts:
        repeated_texts.extend([text] * num_prototypes)
    z = torch.randn(len(repeated_texts), model.latent_dim, device=device)
    bias = model.generator(z)
    prompts, tokenized = model.build_prompt(slot_ids, bias, repeated_texts)
    features = model.text_encoder(prompts, tokenized)
    features = F.normalize(features.float(), dim=-1)
    return features.view(len(subjects), len(candidate_names), num_prototypes, -1)


def summarize_ranks(ranks):
    ranks = torch.tensor(ranks, dtype=torch.float)
    return {
        "r@1": float((ranks <= 1).float().mean().item()),
        "r@5": float((ranks <= 5).float().mean().item()),
        "r@10": float((ranks <= 10).float().mean().item()),
        "mean_rank": float(ranks.mean().item()),
        "count": int(ranks.numel()),
    }


def main():
    args = parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    text_mode = args.text_mode or checkpoint.get("text_mode") or ckpt_args.get("text_mode", "triplet")
    n_ctx = args.n_ctx or checkpoint.get("n_ctx") or ckpt_args.get("n_ctx", 4)
    bias_mode = args.bias_mode or checkpoint.get("bias_mode") or ckpt_args.get("bias_mode", "token_wise")
    device = torch.device(args.device or cfg.MODEL.DEVICE)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "predicate_rank")
    mkdir(output_dir)

    mapping = load_primitive_mapping(args.mapping_file)
    clip_model, preprocess = build_clip_model(checkpoint.get("clip_model", "ViT-B/32"), device)
    raw_dataset = build_raw_sgg_dataset(cfg, args.split)
    eval_predicates = predicate_names_for_part(
        cfg,
        part=args.predicate_part,
        split_file=args.predicate_split_file,
    )
    candidate_predicates = predicate_names_for_part(
        cfg,
        part=args.candidate_part,
        split_file=args.predicate_split_file,
    )
    candidate_pairs = build_candidate_predicates(raw_dataset, candidate_predicates)
    candidate_ids = [pred_id for pred_id, _ in candidate_pairs]
    candidate_names = [name for _, name in candidate_pairs]
    candidate_id_to_col = {pred_id: idx for idx, pred_id in enumerate(candidate_ids)}
    if not candidate_pairs:
        raise RuntimeError("No candidate predicates found for {}".format(args.candidate_part))

    dataset = PrimitiveStage1Dataset(
        raw_dataset,
        preprocess,
        mapping_path=args.mapping_file,
        max_samples=args.max_samples,
        allowed_predicates=eval_predicates,
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
        num_slots=mapping["num_slots"],
        n_ctx=n_ctx,
        max_slots_per_predicate=mapping["max_slots_per_predicate"],
        bias_token_count=1 if bias_mode == "global" else mapping["max_slots_per_predicate"] * n_ctx,
    ).to(device)
    model.load_trainable_state_dict(checkpoint["model"])
    model.eval()

    ranks = []
    per_predicate = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="ranking predicates", dynamic_ncols=True):
            images = batch["union_image"].to(device, non_blocking=True)
            target = model.encode_image(images)
            prototypes = build_batch_prototypes(
                model,
                mapping,
                candidate_names,
                batch["subject_name"],
                batch["object_name"],
                text_mode,
                args.num_prototypes,
                device,
            )
            if args.num_prototypes == 1:
                scores = (target.unsqueeze(1) * prototypes).sum(dim=-1)
            else:
                scores = (target.unsqueeze(1).unsqueeze(2) * prototypes).sum(dim=-1).max(dim=2).values
            scores = scores / args.temperature

            for i, pred_id_tensor in enumerate(batch["predicate_id"]):
                pred_id = int(pred_id_tensor.item())
                if pred_id not in candidate_id_to_col:
                    continue
                gt_col = candidate_id_to_col[pred_id]
                order = torch.argsort(scores[i], descending=True)
                rank = int(torch.nonzero(order == gt_col, as_tuple=False)[0, 0]) + 1
                ranks.append(rank)
                per_predicate[batch["predicate_name"][i]].append(rank)

    if not ranks:
        raise RuntimeError("No evaluated predicates are included in candidate-part {}".format(args.candidate_part))

    summary = {
        "split": args.split,
        "predicate_part": args.predicate_part,
        "candidate_part": args.candidate_part,
        "text_mode": text_mode,
        "n_ctx": n_ctx,
        "bias_mode": bias_mode,
        "num_candidates": len(candidate_names),
        "num_prototypes": args.num_prototypes,
        **summarize_ranks(ranks),
    }
    with open(os.path.join(output_dir, "predicate_rank_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "candidate_predicates.json"), "w") as f:
        json.dump(candidate_names, f, indent=2)

    with open(os.path.join(output_dir, "predicate_rank_per_predicate.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["predicate", "count", "r@1", "r@5", "r@10", "mean_rank"])
        writer.writeheader()
        for predicate, pred_ranks in sorted(per_predicate.items()):
            row = summarize_ranks(pred_ranks)
            writer.writerow(
                {
                    "predicate": predicate,
                    "count": row["count"],
                    "r@1": row["r@1"],
                    "r@5": row["r@5"],
                    "r@10": row["r@10"],
                    "mean_rank": row["mean_rank"],
                }
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
