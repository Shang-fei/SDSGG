import os
import sys

sys.path.append("../../../config")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .low_rank_text import build_predicate_splits


curpath = os.path.dirname(__file__)


class ClipDescriptionRegressionLoss(nn.Module):
    def __init__(self, cfg, device=None):
        super(ClipDescriptionRegressionLoss, self).__init__()
        self.description_relation_loss = pd.read_csv(curpath + "/description_relation_loss.csv")
        train_part = cfg.OV_SETTING.TRAIN_PART
        splits = build_predicate_splits(cfg)
        if cfg.OV_SETTING.DATA_OPTION == "gqa" and train_part == "base":
            active_indices = [0, 31, 48, 30, 29, 22, 8, 23, 21, 1, 50, 40, 43, 38, 41, 11, 46, 6, 13, 35, 47, 12]
            self.description_relation_loss = self.description_relation_loss.iloc[active_indices, 1:]
        elif train_part == "total":
            self.description_relation_loss = self.description_relation_loss.iloc[:, 1:]
        elif train_part in splits:
            self.description_relation_loss = self.description_relation_loss.iloc[splits[train_part], 1:]
        else:
            raise ValueError("Unsupported OV relation split: {}".format(train_part))

        self.description_relation_loss = self.description_relation_loss.applymap(
            lambda x: [int(s) for s in x.split(",")]
        )
        self.description_relation_loss = np.array(self.description_relation_loss)
        self.description_relation_loss = np.array(
            [[np.array(item) for item in inner_list] for inner_list in self.description_relation_loss]
        )
        self.description_relation_loss = torch.Tensor(self.description_relation_loss).to(device)

    def forward(self, relation_logits, rel_labels):
        rel_labels = rel_labels.view(-1)
        valid = rel_labels > 0
        if valid.sum().item() == 0:
            return relation_logits.sum() * 0
        relation_logits = relation_logits[valid].half()
        rel_labels = rel_labels[valid]

        teacher_logits = relation_logits[:, 1]
        relation_logits = relation_logits[:, 0]

        target = torch.cat(
            [(self.description_relation_loss[label, 2]).unsqueeze(0) for label in rel_labels]
        ).half()
        target[:, 21] = target[:, 21] * 0.5
        target = target * 2 + teacher_logits * 0.01 - 0.03

        return F.mse_loss(relation_logits, target, reduction="mean").half()


class ForegroundPredicateAlignmentLoss(nn.Module):
    def __init__(self, cfg, predicate_proportion, device):
        super(ForegroundPredicateAlignmentLoss, self).__init__()
        loss_cfg = cfg.MODEL.ROI_RELATION_HEAD.PRIMITIVE_TEXT
        self.focal_gamma = loss_cfg.FOCAL_GAMMA
        self.focal_alpha = loss_cfg.FOCAL_ALPHA
        self.class_weight_power = loss_cfg.CLASS_WEIGHT_POWER
        self.class_weight_min = loss_cfg.CLASS_WEIGHT_MIN
        self.class_weight_max = loss_cfg.CLASS_WEIGHT_MAX

        pred_weight = 1.0 / torch.FloatTensor([0.5] + predicate_proportion).to(device)
        self.register_buffer("pred_weight", self._build_active_pred_weight(cfg, pred_weight))

    def _build_active_pred_weight(self, cfg, pred_weight):
        train_part = cfg.OV_SETTING.TRAIN_PART
        splits = build_predicate_splits(cfg)
        if train_part not in splits:
            raise ValueError("Unsupported OV relation split: {}".format(train_part))

        active_indices = torch.as_tensor(
            splits[train_part],
            dtype=torch.long,
            device=pred_weight.device,
        )
        active_weight = pred_weight[active_indices].float()
        active_weight = active_weight.clamp_min(1e-6).pow(self.class_weight_power)
        if active_weight.numel() > 1:
            fg_weight = active_weight[1:]
            active_weight[1:] = fg_weight / fg_weight.mean().clamp_min(1e-6)
        return active_weight.clamp(
            min=self.class_weight_min,
            max=self.class_weight_max,
        )

    def forward(self, relation_logits, rel_labels):
        if self.pred_weight.numel() != relation_logits.size(1):
            raise ValueError(
                "Relation loss weight size {} does not match relation logits size {}.".format(
                    self.pred_weight.numel(), relation_logits.size(1)
                )
            )

        logits = relation_logits.float()
        labels = rel_labels.long()
        valid = labels > 0
        if valid.sum().item() == 0:
            return logits.sum() * 0

        # Background is a sampler placeholder, not a decomposed text predicate.
        logits = logits[valid, 1:]
        labels = labels[valid] - 1
        pred_weight = self.pred_weight[1:]

        if labels.numel() > 0 and labels.max().item() >= logits.size(1):
            raise ValueError(
                "Relation label {} is out of range for {} foreground logits.".format(
                    labels.max().item() + 1, logits.size(1)
                )
            )

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
        target_probs = target_log_probs.exp()
        target_weight = pred_weight.gather(0, labels)
        ce_loss = -target_log_probs
        focal_weight = torch.pow(1.0 - target_probs, self.focal_gamma)
        loss = self.focal_alpha * target_weight * focal_weight * ce_loss
        return loss.sum() / target_weight.sum().clamp_min(1e-6)
