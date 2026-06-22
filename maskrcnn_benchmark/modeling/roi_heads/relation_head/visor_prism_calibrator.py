import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from .CLIP import clip
from .primitive_stage1_dataset import predicate_names_for_part
from .primitive_visor_prism import VISORPRISM


class VisorPrismCalibrator(nn.Module):
    def __init__(self, cfg, obj_classes, rel_classes):
        super().__init__()
        visor_cfg = cfg.MODEL.ROI_RELATION_HEAD.VISOR_PRISM
        self.enabled = bool(visor_cfg.ENABLED)
        self.apply_in_train = bool(visor_cfg.APPLY_IN_TRAIN)
        self.alpha = float(visor_cfg.ALPHA)
        self.temperature = float(visor_cfg.TEMPERATURE)
        self.crop_batch_size = int(visor_cfg.CROP_BATCH_SIZE)
        self.prototype_batch_size = int(visor_cfg.PROTOTYPE_BATCH_SIZE)
        self.device_name = cfg.MODEL.DEVICE
        self.obj_classes = list(obj_classes)
        self.rel_classes = list(rel_classes)
        self.model = None
        self.clip_preprocess = None
        self.candidate_ids = []
        self.candidate_names = []
        self.register_buffer("candidate_features", torch.empty(0))

        if not self.enabled:
            return
        if not visor_cfg.CKPT:
            raise ValueError("VISOR_PRISM.ENABLED=True requires VISOR_PRISM.CKPT")
        if not os.path.exists(visor_cfg.CKPT):
            raise FileNotFoundError("VISOR-PRISM checkpoint not found: {}".format(visor_cfg.CKPT))

        checkpoint = torch.load(visor_cfg.CKPT, map_location="cpu")
        clip_model, self.clip_preprocess = clip.load(checkpoint.get("clip_model", "ViT-B/32"), device=self.device_name)
        clip_model = clip_model.float().eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        mapping = checkpoint["mapping"]
        self.model = VISORPRISM(
            clip_model,
            mapping=mapping,
            num_obj_classes=len(self.obj_classes),
            top_m=checkpoint.get("top_m", 4),
            geom_dim=11,
        ).to(self.device_name)
        self.model.load_trainable_state_dict(checkpoint["model"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        allowed_names = predicate_names_for_part(cfg, visor_cfg.CANDIDATE_PART)
        name_to_id = {name: idx for idx, name in enumerate(self.rel_classes) if idx > 0}
        self.candidate_names = [name for name in self.rel_classes[1:] if allowed_names is None or name in allowed_names]
        self.candidate_ids = [name_to_id[name] for name in self.candidate_names if name in name_to_id]
        if not self.candidate_ids:
            raise ValueError("VISOR-PRISM found no candidate predicates for {}".format(visor_cfg.CANDIDATE_PART))
        with torch.no_grad():
            candidate_features = self.model.encode_text(self.candidate_names, torch.device(self.device_name))
        self.candidate_features = candidate_features

    def should_apply(self):
        return self.enabled and self.model is not None and (not self.training or self.apply_in_train)

    def forward(self, rel_dists, proposals, rel_pair_idxs, obj_preds, img, logit_indices=None):
        if not self.should_apply() or img is None:
            return rel_dists
        with torch.no_grad():
            obj_preds = self._split_obj_preds(obj_preds, proposals)
            additions = self.compute_logits(proposals, rel_pair_idxs, obj_preds, img)
        fused = []
        for rel_logit, add_logit in zip(rel_dists, additions):
            if rel_logit.numel() == 0:
                fused.append(rel_logit)
                continue
            add_logit = add_logit.to(device=rel_logit.device, dtype=rel_logit.dtype)
            if rel_logit.shape[-1] != add_logit.shape[-1]:
                if logit_indices is None or len(logit_indices) != rel_logit.shape[-1]:
                    fused.append(rel_logit)
                    continue
                index = torch.as_tensor(logit_indices, device=rel_logit.device, dtype=torch.long)
                add_logit = add_logit.index_select(1, index)
            fused.append(rel_logit + self.alpha * add_logit / max(self.temperature, 1e-6))
        return fused

    def _split_obj_preds(self, obj_preds, proposals):
        if isinstance(obj_preds, torch.Tensor):
            if obj_preds.dim() == 0:
                return [obj_preds.view(1)]
            num_objs = [len(proposal) for proposal in proposals]
            if obj_preds.dim() == 1 and int(obj_preds.numel()) == sum(num_objs):
                return list(obj_preds.split(num_objs, dim=0))
            return [obj_preds]
        if isinstance(obj_preds, (list, tuple)):
            return list(obj_preds)
        raise TypeError("Unsupported obj_preds type for VISOR-PRISM: {}".format(type(obj_preds)))

    def compute_logits(self, proposals, rel_pair_idxs, obj_preds, img):
        logits = []
        for image_index, (proposal, pair_idx, pred_labels) in enumerate(zip(proposals, rel_pair_idxs, obj_preds)):
            num_rel = int(pair_idx.shape[0])
            full_logits = torch.zeros(num_rel, len(self.rel_classes), device=self.candidate_features.device)
            if num_rel == 0:
                logits.append(full_logits)
                continue
            image_tensor = img[image_index]
            target_features = self._encode_union_crops(image_tensor, proposal, pair_idx)
            so_features, geometry = self._relation_conditions(proposal, pair_idx, pred_labels)
            scores = self._prototype_scores(target_features, so_features, geometry)
            full_logits[:, self.candidate_ids] = scores
            logits.append(full_logits)
        return logits

    def _encode_union_crops(self, image_tensor, proposal, pair_idx):
        crops = []
        for rel in pair_idx:
            sub_box = proposal.bbox[int(rel[0])]
            obj_box = proposal.bbox[int(rel[1])]
            crop = self._crop_union(image_tensor, sub_box, obj_box)
            crops.append(crop)
        features = []
        for start in range(0, len(crops), self.crop_batch_size):
            batch = torch.cat(crops[start:start + self.crop_batch_size], dim=0)
            features.append(self.model.encode_image(batch))
        return torch.cat(features, dim=0)

    def _crop_union(self, image_tensor, sub_box, obj_box):
        x1 = int(torch.min(sub_box[0], obj_box[0]).floor().clamp(min=0).item())
        y1 = int(torch.min(sub_box[1], obj_box[1]).floor().clamp(min=0).item())
        x2 = int(torch.max(sub_box[2], obj_box[2]).ceil().clamp(max=image_tensor.shape[-1]).item())
        y2 = int(torch.max(sub_box[3], obj_box[3]).ceil().clamp(max=image_tensor.shape[-2]).item())
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, image_tensor.shape[-1], image_tensor.shape[-2]
        crop = TF.crop(image_tensor.detach().cpu(), y1, x1, y2 - y1, x2 - x1)
        array = crop.permute(1, 2, 0).numpy()
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
        return self.clip_preprocess(pil).unsqueeze(0).to(self.candidate_features.device)

    def _relation_conditions(self, proposal, pair_idx, pred_labels):
        subjects = []
        objects = []
        for rel in pair_idx:
            sub_id = int(pred_labels[int(rel[0])].item())
            obj_id = int(pred_labels[int(rel[1])].item())
            subjects.append(self.obj_classes[sub_id] if 0 <= sub_id < len(self.obj_classes) else "object")
            objects.append(self.obj_classes[obj_id] if 0 <= obj_id < len(self.obj_classes) else "object")
        texts = ["{} {}".format(s, o) for s, o in zip(subjects, objects)]
        so_features = self._encode_unique_texts(texts)
        geometry = self._pair_geometry(proposal, pair_idx).to(self.candidate_features.device)
        return so_features, geometry

    def _encode_unique_texts(self, texts):
        unique = {}
        ordered = []
        for text in texts:
            if text not in unique:
                unique[text] = len(ordered)
                ordered.append(text)
        features = self.model.encode_text(ordered, self.candidate_features.device)
        rows = [features[unique[text]].unsqueeze(0) for text in texts]
        return torch.cat(rows, dim=0)

    def _pair_geometry(self, proposal, pair_idx):
        boxes = proposal.bbox.float()
        width, height = float(proposal.size[0]), float(proposal.size[1])
        scale = boxes.new_tensor([max(width, 1.0), max(height, 1.0), max(width, 1.0), max(height, 1.0)])
        rows = []
        for rel in pair_idx:
            sub = boxes[int(rel[0])]
            obj = boxes[int(rel[1])]
            sub_n = sub / scale
            obj_n = obj / scale
            sub_wh = (sub[2:] - sub[:2]).clamp(min=1.0)
            obj_wh = (obj[2:] - obj[:2]).clamp(min=1.0)
            sub_ctr = sub[:2] + 0.5 * sub_wh
            obj_ctr = obj[:2] + 0.5 * obj_wh
            delta = (obj_ctr - sub_ctr) / scale[:2]
            inter_x1 = torch.max(sub[0], obj[0])
            inter_y1 = torch.max(sub[1], obj[1])
            inter_x2 = torch.min(sub[2], obj[2])
            inter_y2 = torch.min(sub[3], obj[3])
            inter = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)
            area_sub = sub_wh[0] * sub_wh[1]
            area_obj = obj_wh[0] * obj_wh[1]
            iou = inter / (area_sub + area_obj - inter).clamp(min=1.0)
            rows.append(torch.cat([sub_n, obj_n, delta, iou.view(1)], dim=0).unsqueeze(0))
        return torch.cat(rows, dim=0)

    def _prototype_scores(self, target_features, so_features, geometry):
        num_rel = target_features.shape[0]
        num_cand = len(self.candidate_names)
        scores = []
        flat_pred = self.candidate_features.unsqueeze(0).expand(num_rel, -1, -1).reshape(-1, self.candidate_features.shape[-1])
        flat_so = so_features.unsqueeze(1).expand(-1, num_cand, -1).reshape(-1, so_features.shape[-1])
        flat_geom = geometry.unsqueeze(1).expand(-1, num_cand, -1).reshape(-1, geometry.shape[-1])
        proto_rows = []
        for start in range(0, flat_pred.shape[0], self.prototype_batch_size):
            end = start + self.prototype_batch_size
            proto, _, _ = self.model.generate(flat_pred[start:end], flat_so[start:end], flat_geom[start:end])
            proto_rows.append(proto)
        prototypes = torch.cat(proto_rows, dim=0).view(num_rel, num_cand, -1)
        scores = torch.einsum("bd,bcd->bc", F.normalize(target_features.float(), dim=-1), prototypes.float())
        return scores
