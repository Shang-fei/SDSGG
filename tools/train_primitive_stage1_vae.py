import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401

import torch
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
    PrimitiveStage1VAE,
    build_clip_model,
    default_mapping_path,
    kl_loss,
    reconstruction_loss,
)
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="Train primitive triplet VAE Stage 1")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--predicate-part", default="total", choices=["base", "novel", "semantic", "total"])
    parser.add_argument("--predicate-split-file", default=None)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--mapping-file", default=default_mapping_path())
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lambda-orth", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(args.device or cfg.MODEL.DEVICE)
    output_root = args.output_dir or cfg.OUTPUT_DIR
    output_dir = os.path.join(output_root, "primitive_stage1")
    mkdir(output_dir)

    clip_model, preprocess = build_clip_model(args.clip_model, device)
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
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=primitive_stage1_collate,
        pin_memory=True,
        drop_last=False,
    )

    model = PrimitiveStage1VAE(
        clip_model,
        num_slots=dataset.mapping["num_slots"],
        n_ctx=4,
        max_slots_per_predicate=dataset.max_slots,
    ).to(device)
    print(
        json.dumps(
            {
                "clip_feature_dim": model.clip_feature_dim,
                "clip_token_dim": model.clip_token_dim,
                "generator_out_dim": model.generator.net[-1].out_features,
                "primitive_prompt_dim": model.prompt_learner.primitive_prompt_bank.shape[-1],
                "target_feature_mode": "cls-token" if getattr(clip_model.visual, "proj", None) is not None else "pooled",
            },
            sort_keys=True,
        )
    )
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters())
        + list(model.generator.parameters())
        + list(model.prompt_learner.parameters()),
        lr=args.lr,
    )

    shutil.copyfile(args.mapping_file, os.path.join(output_dir, "primitive_mapping.json"))
    with open(os.path.join(output_dir, "unmapped_predicates.json"), "w") as f:
        json.dump(sorted(dataset.unmapped_predicates), f, indent=2)
    with open(os.path.join(output_dir, "skipped_predicates.json"), "w") as f:
        json.dump(sorted(dataset.skipped_predicates), f, indent=2)
    with open(os.path.join(output_dir, "predicate_split_info.json"), "w") as f:
        json.dump(
            {
                "predicate_part": args.predicate_part,
                "predicate_split_file": args.predicate_split_file,
                "allowed_predicates": sorted(allowed_predicates) if allowed_predicates is not None else None,
                "skipped_predicates": sorted(dataset.skipped_predicates),
                "num_samples": len(dataset),
            },
            f,
            indent=2,
        )
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    with open(os.path.join(output_dir, "stage1_args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    train_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "orth": 0.0, "count": 0}
        progress = tqdm(
            loader,
            desc="epoch {}/{}".format(epoch, args.epochs),
            dynamic_ncols=True,
            disable=args.no_progress,
        )
        for batch in progress:
            images = batch["union_image"].to(device, non_blocking=True)
            slot_ids = batch["slot_ids"].to(device, non_blocking=True)
            with torch.no_grad():
                target_features = model.encode_image(images)

            recon, mean, log_var = model(target_features, slot_ids, batch["triplet_text"])
            loss_recon = reconstruction_loss(recon, target_features)
            loss_kl = kl_loss(mean, log_var)
            loss_orth = model.prompt_learner.orthogonal_loss()
            loss = loss_recon + args.beta * loss_kl + args.lambda_orth * loss_orth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = images.shape[0]
            totals["loss"] += float(loss.item()) * n
            totals["recon"] += float(loss_recon.item()) * n
            totals["kl"] += float(loss_kl.item()) * n
            totals["orth"] += float(loss_orth.item()) * n
            totals["count"] += n
            progress.set_postfix(
                loss="{:.4f}".format(totals["loss"] / totals["count"]),
                recon="{:.4f}".format(totals["recon"] / totals["count"]),
                kl="{:.4f}".format(totals["kl"] / totals["count"]),
                orth="{:.4f}".format(totals["orth"] / totals["count"]),
            )

        row = {
            "epoch": epoch,
            "loss": totals["loss"] / totals["count"],
            "loss_recon": totals["recon"] / totals["count"],
            "loss_kl": totals["kl"] / totals["count"],
            "loss_orth": totals["orth"] / totals["count"],
            "num_samples": len(dataset),
        }
        train_log.append(row)
        print(json.dumps(row, sort_keys=True))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "model": model.trainable_state_dict(),
                    "epoch": epoch,
                    "clip_model": args.clip_model,
                    "mapping": dataset.mapping,
                    "predicate_part": args.predicate_part,
                    "allowed_predicates": sorted(allowed_predicates) if allowed_predicates is not None else None,
                    "args": vars(args),
                },
                os.path.join(output_dir, "model.pth"),
            )
            with open(os.path.join(output_dir, "train_log.json"), "w") as f:
                json.dump(train_log, f, indent=2)


if __name__ == "__main__":
    main()
