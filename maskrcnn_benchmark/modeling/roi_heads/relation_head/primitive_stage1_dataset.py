import copy
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.data import datasets as D

from .primitive_stage1_vae import default_mapping_path, load_primitive_mapping


BOX_SCALE = 1024


def build_raw_sgg_dataset(cfg, split="train"):
    dataset_names = getattr(cfg.DATASETS, split.upper())
    if len(dataset_names) != 1:
        raise ValueError("Primitive Stage 1 expects exactly one dataset for {}".format(split))
    data = DatasetCatalog.get(dataset_names[0], cfg)
    factory = getattr(D, data["factory"])
    args = copy.deepcopy(data["args"])
    args.pop("capgraphs_file", None)
    args["transforms"] = None
    args["filter_duplicate_rels"] = False
    return factory(**args)


def predicate_names_for_part(cfg, part="total", split_file=None):
    if split_file:
        with open(split_file, "r") as f:
            split_data = json.load(f)
        if part not in split_data:
            raise KeyError("{} is not defined in {}".format(part, split_file))
        return set(split_data[part])

    part = part.lower()
    if part == "total":
        return None
    if part == "base":
        return set(cfg.OV_SETTING.PRDCS_BASE)
    if part == "novel":
        return set(cfg.OV_SETTING.PRDCS_NOVEL)
    if part == "semantic":
        return set(cfg.OV_SETTING.SEMAN)
    raise ValueError("Unknown predicate part: {}".format(part))


class PrimitiveStage1Dataset(Dataset):
    def __init__(
        self,
        base_dataset,
        preprocess,
        mapping_path=None,
        max_samples=None,
        min_box_size=2,
        allowed_predicates=None,
    ):
        self.base_dataset = base_dataset
        self.preprocess = preprocess
        self.mapping_path = mapping_path or default_mapping_path()
        self.mapping = load_primitive_mapping(self.mapping_path)
        self.max_slots = int(self.mapping["max_slots_per_predicate"])
        self.fallback_slots = self._pad_slots(self.mapping["fallback_slots"])
        self.min_box_size = min_box_size
        self.allowed_predicates = set(allowed_predicates) if allowed_predicates is not None else None
        self.unmapped_predicates = set()
        self.skipped_predicates = set()
        self.samples = self._build_index(max_samples)

    def _pad_slots(self, slots):
        slots = [int(s) for s in slots[:self.max_slots]]
        if len(slots) < self.max_slots:
            slots = slots + [int(self.mapping["fallback_slots"][0])] * (self.max_slots - len(slots))
        return slots

    def _slots_for_predicate(self, predicate_name):
        slots = self.mapping["predicate_to_slots"].get(predicate_name)
        if slots is None:
            self.unmapped_predicates.add(predicate_name)
            slots = self.fallback_slots
        return self._pad_slots(slots)

    def _image_path(self, image_index):
        filename = self.base_dataset.filenames[image_index]
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.base_dataset.img_dir, filename)

    def _recover_boxes(self, image_index):
        boxes = self.base_dataset.gt_boxes[image_index].astype(np.float32)
        if self.base_dataset.__class__.__name__ == "VGDataset":
            info = self.base_dataset.img_info[image_index]
            boxes = boxes / BOX_SCALE * max(info["width"], info["height"])
        return boxes

    def _build_index(self, max_samples):
        samples = []
        for image_index, relations in enumerate(self.base_dataset.relationships):
            gt_classes = self.base_dataset.gt_classes[image_index]
            for sub_idx, obj_idx, pred_id in relations:
                pred_id = int(pred_id)
                if pred_id <= 0:
                    continue
                predicate_name = self.base_dataset.ind_to_predicates[pred_id]
                if self.allowed_predicates is not None and predicate_name not in self.allowed_predicates:
                    self.skipped_predicates.add(predicate_name)
                    continue
                subject_name = self.base_dataset.ind_to_classes[int(gt_classes[int(sub_idx)])]
                object_name = self.base_dataset.ind_to_classes[int(gt_classes[int(obj_idx)])]
                samples.append(
                    {
                        "image_index": image_index,
                        "subject_index": int(sub_idx),
                        "object_index": int(obj_idx),
                        "predicate_id": pred_id,
                        "subject_name": subject_name,
                        "predicate_name": predicate_name,
                        "object_name": object_name,
                        "slot_ids": self._slots_for_predicate(predicate_name),
                    }
                )
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
        return samples

    def __len__(self):
        return len(self.samples)

    def _union_crop(self, image, boxes, subject_index, object_index):
        sub_box = boxes[subject_index]
        obj_box = boxes[object_index]
        x1 = max(0, float(min(sub_box[0], obj_box[0])))
        y1 = max(0, float(min(sub_box[1], obj_box[1])))
        x2 = min(float(image.width), float(max(sub_box[2], obj_box[2])))
        y2 = min(float(image.height), float(max(sub_box[3], obj_box[3])))
        if x2 - x1 < self.min_box_size or y2 - y1 < self.min_box_size:
            x1, y1, x2, y2 = 0, 0, image.width, image.height
        return image.crop((x1, y1, x2, y2))

    def _pair_geometry(self, image, boxes, subject_index, object_index):
        sub = boxes[subject_index].astype(np.float32)
        obj = boxes[object_index].astype(np.float32)
        width = max(float(image.width), 1.0)
        height = max(float(image.height), 1.0)
        scale = np.array([width, height, width, height], dtype=np.float32)
        sub_n = sub / scale
        obj_n = obj / scale

        sub_wh = np.maximum(sub[2:] - sub[:2], 1.0)
        obj_wh = np.maximum(obj[2:] - obj[:2], 1.0)
        sub_ctr = sub[:2] + 0.5 * sub_wh
        obj_ctr = obj[:2] + 0.5 * obj_wh
        delta = (obj_ctr - sub_ctr) / np.array([width, height], dtype=np.float32)

        inter_x1 = max(sub[0], obj[0])
        inter_y1 = max(sub[1], obj[1])
        inter_x2 = min(sub[2], obj[2])
        inter_y2 = min(sub[3], obj[3])
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        sub_area = sub_wh[0] * sub_wh[1]
        obj_area = obj_wh[0] * obj_wh[1]
        union = max(sub_area + obj_area - inter, 1.0)
        iou = np.array([inter / union], dtype=np.float32)

        return np.concatenate([sub_n, obj_n, delta.astype(np.float32), iou], axis=0)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(self._image_path(sample["image_index"])).convert("RGB")
        boxes = self._recover_boxes(sample["image_index"])
        crop = self._union_crop(image, boxes, sample["subject_index"], sample["object_index"])
        triplet_text = "{} {} {}".format(
            sample["subject_name"],
            sample["predicate_name"],
            sample["object_name"],
        )
        return {
            "union_image": self.preprocess(crop),
            "slot_ids": torch.tensor(sample["slot_ids"], dtype=torch.long),
            "predicate_id": torch.tensor(sample["predicate_id"], dtype=torch.long),
            "subject_id": torch.tensor(int(self.base_dataset.gt_classes[sample["image_index"]][sample["subject_index"]]), dtype=torch.long),
            "object_id": torch.tensor(int(self.base_dataset.gt_classes[sample["image_index"]][sample["object_index"]]), dtype=torch.long),
            "geometry": torch.tensor(
                self._pair_geometry(image, boxes, sample["subject_index"], sample["object_index"]),
                dtype=torch.float32,
            ),
            "predicate_name": sample["predicate_name"],
            "subject_name": sample["subject_name"],
            "object_name": sample["object_name"],
            "triplet_text": triplet_text,
        }


def primitive_stage1_collate(batch):
    return {
        "union_image": torch.stack([item["union_image"] for item in batch], dim=0),
        "slot_ids": torch.stack([item["slot_ids"] for item in batch], dim=0),
        "predicate_id": torch.stack([item["predicate_id"] for item in batch], dim=0),
        "subject_id": torch.stack([item["subject_id"] for item in batch], dim=0),
        "object_id": torch.stack([item["object_id"] for item in batch], dim=0),
        "geometry": torch.stack([item["geometry"] for item in batch], dim=0),
        "predicate_name": [item["predicate_name"] for item in batch],
        "subject_name": [item["subject_name"] for item in batch],
        "object_name": [item["object_name"] for item in batch],
        "triplet_text": [item["triplet_text"] for item in batch],
    }
