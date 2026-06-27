# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.roi_heads.relation_head.roi_relation_predictors import crop_and_resize
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


class PredicatePrototypeCollector(object):
    def __init__(self, predicate_names, obj_names, max_per_predicate, min_per_predicate, max_per_subobj_pair):
        self.predicate_names = list(predicate_names)
        self.obj_names = list(obj_names)
        self.max_per_predicate = int(max_per_predicate)
        self.min_per_predicate = int(min_per_predicate)
        self.max_per_subobj_pair = int(max_per_subobj_pair)
        self.features = defaultdict(list)
        self.subobj_counts = defaultdict(Counter)
        self.total_seen = Counter()
        self.skipped_by_subobj_cap = Counter()

    def can_accept(self, predicate_name, subj_label, obj_label):
        if predicate_name == "__background__":
            return False
        self.total_seen[predicate_name] += 1
        if len(self.features[predicate_name]) >= self.max_per_predicate:
            return False
        subj_name = self.obj_names[int(subj_label)]
        obj_name = self.obj_names[int(obj_label)]
        pair_key = (subj_name, obj_name)
        if self.subobj_counts[predicate_name][pair_key] >= self.max_per_subobj_pair:
            self.skipped_by_subobj_cap[predicate_name] += 1
            return False
        return True

    def add(self, predicate_name, subj_label, obj_label, feature):
        subj_name = self.obj_names[int(subj_label)]
        obj_name = self.obj_names[int(obj_label)]
        pair_key = (subj_name, obj_name)
        self.subobj_counts[predicate_name][pair_key] += 1
        self.features[predicate_name].append(feature.detach().cpu())

    def is_full(self):
        for name in self.predicate_names:
            if name != "__background__" and len(self.features[name]) < self.max_per_predicate:
                return False
        return True

    def build(self, feature_dim):
        prototypes = torch.zeros((len(self.predicate_names), feature_dim), dtype=torch.float32)
        counts = torch.zeros((len(self.predicate_names),), dtype=torch.long)
        metadata = {}
        unreliable = []
        for idx, name in enumerate(self.predicate_names):
            rel_features = self.features[name]
            counts[idx] = len(rel_features)
            if len(rel_features) > 0:
                stacked = torch.stack(rel_features, dim=0).float()
                prototypes[idx] = F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            if name != "__background__" and len(rel_features) < self.min_per_predicate:
                unreliable.append(name)
            top_pairs = [
                {"subject": pair[0], "object": pair[1], "count": count}
                for pair, count in self.subobj_counts[name].most_common(10)
            ]
            metadata[name] = {
                "num_instances": int(len(rel_features)),
                "num_seen_before_caps": int(self.total_seen[name]),
                "skipped_by_subobj_cap": int(self.skipped_by_subobj_cap[name]),
                "top_subject_object_pairs": top_pairs,
            }
        return prototypes, counts, metadata, unreliable


def get_relation_predictor(model):
    return model.roi_heads.relation.predictor


def build_relation_crops(image_tensor, target, pair_indices, predictor):
    crops = []
    for pair_index in pair_indices:
        subj_index = int(pair_index[0])
        obj_index = int(pair_index[1])
        union_img = crop_and_resize(
            image_tensor.unsqueeze(0),
            target.bbox[subj_index],
            target.bbox[obj_index],
        )
        pil_img = union_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        pil_img = Image.fromarray(np.uint8(pil_img))
        crops.append(predictor.clip_preprocess(pil_img).unsqueeze(0).to(predictor.device))
    if len(crops) == 0:
        return None
    return torch.cat(crops, dim=0)


def collect_prototypes(cfg, model, data_loader, logger):
    predictor = get_relation_predictor(model)
    prototype_cfg = cfg.MODEL.ROI_RELATION_HEAD.MTM.PROTOTYPE
    collector = PredicatePrototypeCollector(
        predictor.activeRelNames,
        predictor.obj_names,
        prototype_cfg.MAX_INSTANCES_PER_PREDICATE,
        prototype_cfg.MIN_INSTANCES_PER_PREDICATE,
        prototype_cfg.MAX_INSTANCES_PER_SUBOBJ_PAIR,
    )

    model.eval()
    predictor.clip_model.eval()
    predictor.relationMtm.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    feature_dim = cfg.MODEL.ROI_RELATION_HEAD.MTM.EMBED_DIM

    with torch.no_grad():
        for iteration, (images, targets, _) in enumerate(data_loader):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            for image_index, target in enumerate(targets):
                relation_matrix = target.get_field("relation")
                positive_pairs = torch.nonzero(relation_matrix > 0)
                if positive_pairs.numel() == 0:
                    continue

                selected_pairs = []
                selected_predicates = []
                selected_subj_labels = []
                selected_obj_labels = []
                labels = target.get_field("labels").long()
                for pair_index in positive_pairs:
                    rel_label = int(relation_matrix[pair_index[0], pair_index[1]].item())
                    if rel_label <= 0 or rel_label >= len(predictor.activeRelNames):
                        continue
                    predicate_name = predictor.activeRelNames[rel_label]
                    subj_label = int(labels[pair_index[0]].item())
                    obj_label = int(labels[pair_index[1]].item())
                    if not collector.can_accept(predicate_name, subj_label, obj_label):
                        continue
                    selected_pairs.append(pair_index)
                    selected_predicates.append(predicate_name)
                    selected_subj_labels.append(subj_label)
                    selected_obj_labels.append(obj_label)

                if len(selected_pairs) == 0:
                    continue

                pair_tensor = torch.stack(selected_pairs, dim=0)
                crop_tensor = build_relation_crops(images.tensors[image_index], target, pair_tensor, predictor)
                if crop_tensor is None:
                    continue
                clip_features = predictor.clip_model.encode_image(crop_tensor)
                if clip_features.dim() == 3:
                    clip_features = clip_features[:, 0, :]
                adapted_features = predictor.relationMtm.encode_visual(clip_features)
                adapted_features = F.normalize(adapted_features.float(), dim=-1)
                for row, predicate_name in enumerate(selected_predicates):
                    collector.add(
                        predicate_name,
                        selected_subj_labels[row],
                        selected_obj_labels[row],
                        adapted_features[row],
                    )

            if (iteration + 1) % 100 == 0:
                logger.info("Processed {} batches for MTM prototypes".format(iteration + 1))
            if collector.is_full():
                logger.info("All predicates reached the prototype instance cap.")
                break

    return collector.build(feature_dim)


def main():
    parser = argparse.ArgumentParser(description="Build MTM predicate visual prototypes from the training split.")
    parser.add_argument("--config-file", required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger("maskrcnn_benchmark", output_dir, 0)
    logger.info("Building MTM prototypes with config: {}".format(args.config_file))

    device = torch.device(cfg.MODEL.DEVICE)
    model = build_detection_model(cfg)
    model.to(device)
    model.updata(cfg.OV_SETTING.TRAIN_PART)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    data_loaders = make_data_loader(cfg=cfg, mode="test", is_distributed=False, dataset_to_test="train")
    if len(data_loaders) != 1:
        raise RuntimeError("MTM prototype building expects exactly one training data loader.")

    prototypes, counts, metadata, unreliable = collect_prototypes(cfg, model, data_loaders[0], logger)
    prototype_cfg = cfg.MODEL.ROI_RELATION_HEAD.MTM.PROTOTYPE
    output_path = prototype_cfg.OUTPUT_PATH
    if not output_path:
        output_path = os.path.join(cfg.OUTPUT_DIR, "mtm_base_visual_prototypes.pth")

    predictor = get_relation_predictor(model)
    output = {
        "space": "mtm_adapted_visual",
        "predicate_names": list(predictor.activeRelNames),
        "prototypes": prototypes.cpu(),
        "counts": counts.cpu(),
        "metadata": metadata,
        "unreliable_predicates": unreliable,
        "config": {
            "train_part": cfg.OV_SETTING.TRAIN_PART,
            "max_instances_per_predicate": prototype_cfg.MAX_INSTANCES_PER_PREDICATE,
            "min_instances_per_predicate": prototype_cfg.MIN_INSTANCES_PER_PREDICATE,
            "max_instances_per_subobj_pair": prototype_cfg.MAX_INSTANCES_PER_SUBOBJ_PAIR,
        },
    }
    torch.save(output, output_path)
    logger.info("Saved MTM base visual prototypes to {}".format(output_path))
    logger.info("Unreliable predicates (< min instances): {}".format(unreliable))


if __name__ == "__main__":
    main()
