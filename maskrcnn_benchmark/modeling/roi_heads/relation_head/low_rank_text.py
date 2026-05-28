import json

import torch
from torch import nn
from torch.nn import functional as F

from .hola_low_rank import HOLaLowRankDecomposer


VG_PREDICATES = [
    "__background__", "above", "across", "against", "along", "and", "at", "attached to",
    "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating",
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in",
    "in front of", "laying on", "looking at", "lying on", "made of", "mounted on",
    "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of",
    "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using",
    "walking in", "walking on", "watching", "wearing", "wears", "with",
]
VG_PREDICATE_ID = {name: idx for idx, name in enumerate(VG_PREDICATES)}

CORE_PROMPT_FIELD = "core_prompt"


def _predicate_ids(names):
    missing = [name for name in names if name not in VG_PREDICATE_ID]
    if missing:
        raise KeyError("Unknown VG predicates: {}".format(missing))
    return sorted(VG_PREDICATE_ID[name] for name in set(names))


def build_predicate_splits(cfg):
    return {
        "base": [0] + _predicate_ids(cfg.OV_SETTING.PRDCS_BASE),
        "novel": [0] + _predicate_ids(cfg.OV_SETTING.PRDCS_NOVEL),
        "semantic": [0] + _predicate_ids(cfg.OV_SETTING.SEMAN),
        "total": list(range(len(VG_PREDICATES))),
    }


def build_full_predicate_names(cfg):
    return list(VG_PREDICATES)


def load_relation_prompt_texts(prompt_json, predicate_names, field=CORE_PROMPT_FIELD):
    if not prompt_json:
        raise ValueError("prompt_json must point to a predicate prompt JSON.")
    with open(prompt_json, "r") as f:
        prompt_data = json.load(f)

    texts = []
    for name in predicate_names[1:]:
        if name not in prompt_data:
            raise KeyError("Missing predicate '{}' in {}".format(name, prompt_json))
        item = prompt_data[name]
        if field not in item:
            raise KeyError("Missing field '{}.{}' in {}".format(name, field, prompt_json))
        texts.append(item[field])
    return texts


class CoreRelationTextAdapter(nn.Module):
    """Low-rank predicate text classifier.

    The adapter owns only the text decomposition path:
      text_features ~= mean + W B

    It deliberately avoids visual transport/gating experiments so the predictor has
    a clean, auditable text-decomposition baseline.
    """

    def __init__(
        self,
        text_features,
        rank,
        init_method="pca",
        train_basis=False,
        train_mode="none",
        logit_temperature=0.07,
        recon_loss_weight=0.0,
        sparsity_weight=0.0,
        basis_decorr_weight=0.0,
        weight_decorr_weight=0.0,
        weight_delta_scale=0.0,
        weight_anchor_weight=0.0,
        w_active_threshold=0.05,
    ):
        super(CoreRelationTextAdapter, self).__init__()
        self.decomposer = HOLaLowRankDecomposer(
            text_features,
            rank=rank,
            init_method=init_method,
            train_basis=train_basis,
            train_mode=train_mode,
            recon_loss_weight=recon_loss_weight,
            sparsity_weight=sparsity_weight,
            basis_decorr_weight=basis_decorr_weight,
            weight_decorr_weight=weight_decorr_weight,
            weight_delta_scale=weight_delta_scale,
            weight_anchor_weight=weight_anchor_weight,
        )
        self.logit_temperature = float(logit_temperature)
        self.w_active_threshold = float(w_active_threshold)

    def class_weights_by_indices(self, indices, normalize=False, initial=False):
        if initial:
            weights = self.decomposer.initial_class_weights
        else:
            weights = self.decomposer.effective_class_weights()
        weights = weights[indices.to(weights.device).long()].float()
        if normalize:
            weights = F.normalize(weights, dim=-1)
        return weights

    def active_class_weights(self, fg_ids, normalize=False):
        return self.class_weights_by_indices(fg_ids, normalize=normalize)

    def active_classifier(self, fg_ids, normalize=True):
        classifier = self.decomposer.classifier_features()
        classifier = classifier[fg_ids.to(classifier.device).long()].float()
        if normalize:
            classifier = F.normalize(classifier, dim=-1)
        return classifier

    def basis_logits(self, visual_features):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        basis = F.normalize(self.decomposer.classifier_basis().float(), dim=-1)
        return visual_features @ basis.t()

    def logits(self, visual_features, fg_ids, raw_visual_features=None):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        classifier = self.active_classifier(fg_ids)
        temperature = max(self.logit_temperature, 1e-6)
        text_logits = visual_features @ classifier.t() / temperature

        basis_source = visual_features if raw_visual_features is None else raw_visual_features
        basis_logits = self.basis_logits(basis_source)
        return text_logits, basis_logits

    def losses(self):
        losses = {}
        factor_losses = self.decomposer.losses()
        if "recon" in factor_losses:
            losses["recon_loss2"] = factor_losses["recon"]
        if "sparse" in factor_losses:
            losses["loss_w_sparse"] = factor_losses["sparse"]
        if "weight_decorr" in factor_losses:
            losses["loss_w_unique"] = factor_losses["weight_decorr"]
        if "basis_decorr" in factor_losses:
            losses["disentangle_basis"] = factor_losses["basis_decorr"]
        if "weight_anchor" in factor_losses:
            losses["loss_lr_anchor"] = factor_losses["weight_anchor"]
        return losses

    def debug_stats(self):
        return self.decomposer.debug_stats()

    def weight_usage_stats(self, fg_ids, threshold=None):
        threshold = self.w_active_threshold if threshold is None else float(threshold)
        with torch.no_grad():
            weights = self.active_class_weights(fg_ids).float()
            abs_weights = weights.abs()
            active = (abs_weights > threshold).float().sum(dim=1)
            total = abs_weights.sum(dim=1).clamp_min(1e-6)
            max_share = abs_weights.max(dim=1)[0] / total
            return {
                "W_active": active.mean(),
                "W_max_share": max_share.mean(),
            }
