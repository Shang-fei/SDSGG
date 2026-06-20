import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

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
    parser = argparse.ArgumentParser(description="Train VISOR-PRISM offline prototype generator")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--predicate-part", default="base", choices=["base", "novel", "semantic", "total"])
    parser.add_argument("--predicate-split-file", default=None)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--mapping-file", default=default_mapping_path())
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-cls", type=float, default=2.0)
    parser.add_argument("--lambda-rec", type=float, default=0.3)
    parser.add_argument("--lambda-comp", type=float, default=0.1)
    parser.add_argument("--lambda-visprim", type=float, default=0.2)
    parser.add_argument("--lambda-adv-obj", type=float, default=0.05)
    parser.add_argument("--lambda-so-only", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.01)
    parser.add_argument("--lambda-div", type=float, default=0.001)
    parser.add_argument("--so-dropout", type=float, default=0.3)
    parser.add_argument("--disable-adv-obj", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def build_candidate_predicates(raw_dataset, allowed_names):
    pairs = []
    for pred_id, name in enumerate(raw_dataset.ind_to_predicates):
        if pred_id == 0:
            continue
        if allowed_names is not None and name not in allowed_names:
            continue
        pairs.append((pred_id, name))
    return pairs


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


def subject_object_texts(batch):
    return ["{} {}".format(s, o) for s, o in zip(batch["subject_name"], batch["object_name"])]


def save_checkpoint(path, model, args, mapping, candidate_names, epoch, clip_model_name):
    torch.save(
        {
            "model": model.trainable_state_dict(),
            "epoch": epoch,
            "clip_model": clip_model_name,
            "mapping": mapping,
            "candidate_names": candidate_names,
            "top_m": args.top_m,
            "args": vars(args),
            "arch": "visor_prism_v1",
        },
        path,
    )


def main():
    args = parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(args.device or cfg.MODEL.DEVICE)
    output_root = args.output_dir or cfg.OUTPUT_DIR
    output_dir = os.path.join(output_root, "visor_prism")
    mkdir(output_dir)

    mapping = load_primitive_mapping(args.mapping_file)
    clip_model, preprocess = build_clip_model(args.clip_model, device)
    raw_dataset = build_raw_sgg_dataset(cfg, args.split)
    allowed_predicates = predicate_names_for_part(
        cfg,
        part=args.predicate_part,
        split_file=args.predicate_split_file,
    )
    candidate_pairs = build_candidate_predicates(raw_dataset, allowed_predicates)
    candidate_names = [name for _, name in candidate_pairs]
    if not candidate_names:
        raise RuntimeError("No candidate predicates found for {}".format(args.predicate_part))
    candidate_name_to_col = {name: idx for idx, name in enumerate(candidate_names)}

    dataset = PrimitiveStage1Dataset(
        raw_dataset,
        preprocess,
        mapping_path=args.mapping_file,
        max_samples=args.max_samples,
        allowed_predicates=allowed_predicates,
    )
    if len(dataset) == 0:
        raise RuntimeError("No VISOR-PRISM samples found for predicate part {}".format(args.predicate_part))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=primitive_stage1_collate,
        pin_memory=True,
        drop_last=False,
    )

    model = VISORPRISM(
        clip_model,
        mapping=mapping,
        num_obj_classes=len(raw_dataset.ind_to_classes),
        top_m=args.top_m,
        geom_dim=11,
    ).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    candidate_features = model.encode_text(candidate_names, device)
    candidate_slots = candidate_slot_tensor(mapping, candidate_names, device)
    candidate_labels = slot_labels(candidate_slots, model.num_slots)

    shutil.copyfile(args.mapping_file, os.path.join(output_dir, "primitive_mapping.json"))
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    with open(os.path.join(output_dir, "visor_prism_args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)
    with open(os.path.join(output_dir, "predicate_split_info.json"), "w") as f:
        json.dump(
            {
                "predicate_part": args.predicate_part,
                "predicate_split_file": args.predicate_split_file,
                "allowed_predicates": sorted(allowed_predicates) if allowed_predicates is not None else None,
                "candidate_names": candidate_names,
                "num_samples": len(dataset),
            },
            f,
            indent=2,
        )

    train_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "cls": 0.0,
            "rec": 0.0,
            "comp": 0.0,
            "visprim": 0.0,
            "adv_obj": 0.0,
            "so_only": 0.0,
            "anchor": 0.0,
            "div": 0.0,
            "count": 0,
        }
        progress = tqdm(
            loader,
            desc="epoch {}/{}".format(epoch, args.epochs),
            dynamic_ncols=True,
            disable=args.no_progress,
        )
        for batch in progress:
            images = batch["union_image"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            slot_ids = batch["slot_ids"].to(device, non_blocking=True)
            subject_ids = batch["subject_id"].to(device, non_blocking=True)
            object_ids = batch["object_id"].to(device, non_blocking=True)

            with torch.no_grad():
                target = model.encode_image(images)
                so_features = model.encode_text(subject_object_texts(batch), device)
                gt_pred_features = model.encode_text(batch["predicate_name"], device)

            gt_cols = torch.tensor(
                [candidate_name_to_col[name] for name in batch["predicate_name"]],
                dtype=torch.long,
                device=device,
            )
            primitive_labels = slot_labels(slot_ids, model.num_slots)

            q_gt, weights, comp_logits = model.generate(
                gt_pred_features,
                so_features,
                geometry,
                so_dropout=args.so_dropout,
                use_so=True,
            )
            q_so = model.generate_so_only(so_features, geometry)

            bsz = target.shape[0]
            cand_pred = candidate_features.unsqueeze(0).expand(bsz, -1, -1).reshape(-1, model.feature_dim)
            cand_so = so_features.unsqueeze(1).expand(-1, len(candidate_names), -1).reshape(-1, model.feature_dim)
            cand_geom = geometry.unsqueeze(1).expand(-1, len(candidate_names), -1).reshape(-1, geometry.shape[-1])
            q_cand, _, _ = model.generate(cand_pred, cand_so, cand_geom, so_dropout=0.0, use_so=True)
            q_cand = q_cand.view(bsz, len(candidate_names), -1)

            scores = torch.einsum("bd,bcd->bc", target, q_cand) / args.temperature
            loss_cls = F.cross_entropy(scores, gt_cols)
            loss_rec = 1.0 - (q_gt * target).sum(dim=-1).mean()
            loss_comp = F.binary_cross_entropy_with_logits(comp_logits, primitive_labels)
            vis_logits = model.activation_head(target.detach())
            loss_visprim = F.binary_cross_entropy_with_logits(vis_logits, primitive_labels)
            loss_so_only = 1.0 - (q_so * target).sum(dim=-1).mean()
            loss_anchor = model.primitive_bank.anchor_loss()
            loss_div = model.primitive_bank.diversity_loss()

            if args.disable_adv_obj or args.lambda_adv_obj <= 0:
                loss_adv_obj = target.new_tensor(0.0)
            else:
                sub_logits, obj_logits = model.adversary(q_gt, grl_weight=1.0)
                loss_adv_obj = 0.5 * (
                    F.cross_entropy(sub_logits, subject_ids) + F.cross_entropy(obj_logits, object_ids)
                )

            loss = (
                args.lambda_cls * loss_cls
                + args.lambda_rec * loss_rec
                + args.lambda_comp * loss_comp
                + args.lambda_visprim * loss_visprim
                + args.lambda_adv_obj * loss_adv_obj
                + args.lambda_so_only * loss_so_only
                + args.lambda_anchor * loss_anchor
                + args.lambda_div * loss_div
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = images.shape[0]
            for key, value in [
                ("loss", loss),
                ("cls", loss_cls),
                ("rec", loss_rec),
                ("comp", loss_comp),
                ("visprim", loss_visprim),
                ("adv_obj", loss_adv_obj),
                ("so_only", loss_so_only),
                ("anchor", loss_anchor),
                ("div", loss_div),
            ]:
                totals[key] += float(value.item()) * n
            totals["count"] += n
            progress.set_postfix(
                loss="{:.4f}".format(totals["loss"] / totals["count"]),
                cls="{:.4f}".format(totals["cls"] / totals["count"]),
                rec="{:.4f}".format(totals["rec"] / totals["count"]),
            )

        row = {
            "epoch": epoch,
            "num_samples": len(dataset),
        }
        row.update({key: totals[key] / totals["count"] for key in totals if key != "count"})
        train_log.append(row)
        print(json.dumps(row, sort_keys=True))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                os.path.join(output_dir, "model.pth"),
                model,
                args,
                mapping,
                candidate_names,
                epoch,
                args.clip_model,
            )
            with open(os.path.join(output_dir, "train_log.json"), "w") as f:
                json.dump(train_log, f, indent=2)


if __name__ == "__main__":
    main()
