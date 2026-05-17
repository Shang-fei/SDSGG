# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
import os

sys.path.append("../../../config")

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
from torch.autograd import Variable
from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from .low_rank_text import build_full_predicate_names, build_split_indices
import pandas as pd

from  defaults import PRDCS_BASE ,PRDCS_NOVEL,SEMAN,TRAIN_PART,DATA_OPTION
curpath=os.path.dirname(__file__)


class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, ignore_index=-1):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        valid = targets != self.ignore_index
        logits = logits[valid].float()
        targets = targets[valid].long()
        if targets.numel() == 0:
            return logits.sum() * 0

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        loss = -self.alpha * torch.pow(1 - target_probs, self.gamma) * target_log_probs
        return loss.mean()


class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        device,
        cfg=None
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        self.use_low_rank_text = (
            cfg is not None
            and (
                cfg.MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.ENABLED
                or cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "LowRankClipPredictor"
            )
        )
        self.low_rank_pred_weight = None
        self.low_rank_focal_gamma = 2.0
        self.low_rank_focal_alpha = 1.0
        self.low_rank_class_weight_power = 0.5
        self.low_rank_class_weight_min = 0.2
        self.low_rank_class_weight_max = 2.0
        if self.use_low_rank_text:
            low_rank_cfg = cfg.MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT
            self.low_rank_focal_gamma = low_rank_cfg.FOCAL_GAMMA
            self.low_rank_focal_alpha = low_rank_cfg.FOCAL_ALPHA
            self.low_rank_class_weight_power = low_rank_cfg.CLASS_WEIGHT_POWER
            self.low_rank_class_weight_min = low_rank_cfg.CLASS_WEIGHT_MIN
            self.low_rank_class_weight_max = low_rank_cfg.CLASS_WEIGHT_MAX
            self.low_rank_pred_weight = self._build_low_rank_pred_weight(cfg)
        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()
        self.loss=Loss(gamma=0.0, alpha=1, size_average=True,device=device)
        #self.focal_loss=MultiCEFocalLoss(class_num=25,device=device)

    def _build_low_rank_active_indices(self, cfg):
        predicate_names = build_full_predicate_names(cfg)
        active_indices_by_mode = build_split_indices(cfg, predicate_names)
        if cfg.OV_SETTING.TRAIN_PART not in active_indices_by_mode:
            raise ValueError("Unsupported OV relation split: {}".format(cfg.OV_SETTING.TRAIN_PART))

        return torch.as_tensor(
            active_indices_by_mode[cfg.OV_SETTING.TRAIN_PART],
            dtype=torch.long,
            device=self.pred_weight.device,
        )

    def _build_low_rank_pred_weight(self, cfg):
        active_indices = self._build_low_rank_active_indices(cfg)
        pred_weight = self.pred_weight[active_indices].float()
        pred_weight = pred_weight.clamp_min(1e-6).pow(self.low_rank_class_weight_power)
        if pred_weight.numel() > 1:
            fg_weight = pred_weight[1:]
            pred_weight[1:] = fg_weight / fg_weight.mean().clamp_min(1e-6)
        return pred_weight.clamp(
            min=self.low_rank_class_weight_min,
            max=self.low_rank_class_weight_max,
        )

    def _get_low_rank_pred_weight(self, relation_logits):
        pred_weight = self.low_rank_pred_weight.to(relation_logits.device)
        if pred_weight.numel() != relation_logits.size(1):
            raise ValueError(
                "Low-rank loss weight size {} does not match relation logits size {}.".format(
                    pred_weight.numel(), relation_logits.size(1)
                )
            )
        return pred_weight

    def _low_rank_focal_loss(self, relation_logits, rel_labels):
        pred_weight = self._get_low_rank_pred_weight(relation_logits)
        logits = relation_logits.float()
        labels = rel_labels.long()

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
        target_probs = target_log_probs.exp()
        target_weight = pred_weight.gather(0, labels)
        ce_loss = -target_log_probs
        focal_weight = torch.pow(1.0 - target_probs, self.low_rank_focal_gamma)
        loss = self.low_rank_focal_alpha * target_weight * focal_weight * ce_loss
        return loss.sum() / target_weight.sum().clamp_min(1e-6)

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        if self.use_low_rank_text:
            loss_relation = self._low_rank_focal_loss(relation_logits, rel_labels)
        else:
            loss_relation = self.loss(relation_logits, rel_labels.long())
        #loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss





class Loss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, device=None):
        super(Loss, self).__init__()
        self.gamma = gamma
        global PRDCS_BASE,PRDCS_NOVEL,SEMAN,datamode

        self.alpha = alpha
        self.size_average = size_average

        self.description_relation_loss = pd.read_csv(curpath+"/description_relation_loss.csv")
        self.id_dict = {'__background__': 0, 'above': 1, 'across': 2, 'against': 3, 'along': 4, 'and': 5, 'at': 6,
                        'attached to': 7, 'behind': 8, 'belonging to': 9, 'between': 10, 'carrying': 11,
                        'covered in': 12, 'covering': 13, 'eating': 14, 'flying in': 15, 'for': 16, 'from': 17,
                        'growing on': 18, 'hanging from': 19, 'has': 20, 'holding': 21, 'in': 22, 'in front of': 23,
                        'laying on': 24, 'looking at': 25, 'lying on': 26, 'made of': 27, 'mounted on': 28, 'near': 29,
                        'of': 30, 'on': 31, 'on back of': 32, 'over': 33, 'painted on': 34, 'parked on': 35,
                        'part of': 36, 'playing': 37, 'riding': 38, 'says': 39, 'sitting on': 40, 'standing on': 41,
                        'to': 42, 'under': 43, 'using': 44, 'walking in': 45, 'walking on': 46, 'watching': 47,
                        'wearing': 48, 'wears': 49, 'with': 50}

        if DATA_OPTION == "vg":
            self.base=[0,1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 23, 25, 27, 29, 30, 31, 33, 35, 37, 38, 40, 41, 42, 43, 46, 47, 48, 49, 50]
        elif DATA_OPTION == "gqa":
            self.base = [0, 31, 48, 30, 29, 22, 8, 23, 21, 1, 50, 40, 43, 38, 41, 11, 46, 6, 13, 35, 47, 12]  # gqa

        if TRAIN_PART=="base":
            self.description_relation_loss = self.description_relation_loss.iloc[self.base, 1:]
        elif TRAIN_PART=="total":
            self.description_relation_loss = self.description_relation_loss.iloc[:, 1:]


        self.description_relation_loss = self.description_relation_loss.applymap(lambda x: [int(s) for s in x.split(',')])
        self.description_relation_loss = np.array(self.description_relation_loss)
        self.description_relation_loss = np.array([[np.array(item) for item in inner_list] for inner_list in self.description_relation_loss])
        self.description_relation_loss = torch.Tensor(self.description_relation_loss).to(device)

    def forward(self, input, target):

        target = target.view(-1)

        zz = torch.where(target > 0, True, False)
        input = input[zz].half()
        target = target[zz]

        totarget=input[:,1]
        input=input[:,0]

        target = torch.cat([(self.description_relation_loss[x, 2]).unsqueeze(0) for x in target]).half()
        target[:, 21] = target[:, 21] * 0.5

        target=target*2+totarget*0.01-0.03

        loss = F.mse_loss(input, target, reduction="mean", reduce=True).half()
        return loss



class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num,device, gamma=10, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1)).to(device)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        predict=predict[target!=0]
        target=target[target!=0]
        pt = F.softmax(predict, dim=-1) # softmmax获取预测概率
        
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        cfg.MODEL.DEVICE,
        cfg=cfg
    )

    return loss_evaluator
