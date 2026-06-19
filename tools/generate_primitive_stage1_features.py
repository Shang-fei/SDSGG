import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401

import torch

from maskrcnn_benchmark.modeling.roi_heads.relation_head.primitive_stage1_vae import (
    PrimitiveStage1VAE,
    build_clip_model,
    default_mapping_path,
    load_primitive_mapping,
)
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate primitive Stage 1 features")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--predicate", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--mapping-file", default=default_mapping_path())
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def pad_slots(mapping, slots):
    max_slots = int(mapping["max_slots_per_predicate"])
    fallback = mapping["fallback_slots"]
    slots = [int(s) for s in slots[:max_slots]]
    if len(slots) < max_slots:
        slots += [int(fallback[0])] * (max_slots - len(slots))
    return slots


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    mapping = load_primitive_mapping(args.mapping_file)
    slots = mapping["predicate_to_slots"].get(args.predicate, mapping["fallback_slots"])
    slots = pad_slots(mapping, slots)

    device = torch.device(args.device)
    clip_model, _ = build_clip_model(checkpoint.get("clip_model", "ViT-B/32"), device)
    model = PrimitiveStage1VAE(
        clip_model,
        num_slots=mapping["num_slots"],
        n_ctx=4,
        max_slots_per_predicate=mapping["max_slots_per_predicate"],
    ).to(device)
    model.load_trainable_state_dict(checkpoint["model"])
    model.eval()

    triplet = "{} {} {}".format(args.subject, args.predicate, args.object)
    slot_ids = torch.tensor(slots, dtype=torch.long, device=device)
    with torch.no_grad():
        features = model.generate(slot_ids, [triplet], args.num_samples).cpu()

    mkdir(os.path.dirname(args.output) or ".")
    torch.save(
        {
            "features": features,
            "subject": args.subject,
            "predicate": args.predicate,
            "object": args.object,
            "triplet_text": triplet,
            "slot_ids": slots,
            "checkpoint": args.checkpoint,
        },
        args.output,
    )
    print(json.dumps({"output": args.output, "shape": list(features.shape), "slot_ids": slots}))


if __name__ == "__main__":
    main()
