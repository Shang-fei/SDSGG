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
    base = [0] + _predicate_ids(cfg.OV_SETTING.PRDCS_BASE)
    novel = [0] + _predicate_ids(cfg.OV_SETTING.PRDCS_NOVEL)
    semantic = [0] + _predicate_ids(cfg.OV_SETTING.SEMAN)
    return {
        "base": base,
        "novel": novel,
        "semantic": semantic,
        "total": list(range(len(VG_PREDICATES))),
    }


def build_full_predicate_names(cfg):
    return list(VG_PREDICATES)


def load_relation_prompt_texts(prompt_json, predicate_names, field=CORE_PROMPT_FIELD):
    if not prompt_json:
        raise ValueError("LOW_RANK_TEXT.PROMPT_JSON must point to an LLM prompt JSON.")
    with open(prompt_json, "r") as f:
        prompt_data = json.load(f)

    texts = []
    for name in predicate_names:
        if name == "__background__":
            texts.append("")
            continue
        if name not in prompt_data:
            raise KeyError("Missing predicate '{}' in {}".format(name, prompt_json))
        item = prompt_data[name]
        if field not in item:
            raise KeyError("Missing field '{}.{}' in {}".format(name, field, prompt_json))
        texts.append(item[field])
    return texts[1:]


class CoreRelationTextAdapter(nn.Module):
    def __init__(
        self,
        text_features,
        rank,
        init_method="pca",
        recon_loss_weight=0.1,
        sparsity_weight=0.1,
        basis_decorr_weight=0.001,
        weight_decorr_weight=0.001,
        train_basis=True,
        train_mode="w",
        logit_temperature=0.05,
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
        )
        self.logit_temperature = logit_temperature

    def active_class_weights(self, active_fg_indices, normalize=False):
        weights = self.decomposer.class_weights
        if self.decomposer.train_mode in ("b", "none"):
            weights = weights.detach()
        weights = weights[active_fg_indices].float()
        if normalize:
            weights = F.normalize(weights, dim=-1)
        return weights

    def basis_logits(self, visual_features):
        basis = self.decomposer.basis_feat
        if self.decomposer.train_mode in ("w", "none"):
            basis = basis.detach()
        basis = F.normalize(basis.float(), dim=-1)
        return visual_features @ basis.t()

    def active_classifier(self, active_fg_indices):
        classifier = self.decomposer.classifier_features()[active_fg_indices].float()
        return F.normalize(classifier, dim=-1)

    def logits(self, visual_features, active_fg_indices):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        basis_logits = self.basis_logits(visual_features)
        classifier = self.active_classifier(active_fg_indices)
        logits = visual_features @ classifier.t()
        return logits / self.logit_temperature, basis_logits

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
        return losses

    def debug_stats(self):
        return self.decomposer.debug_stats()

    def weight_usage_stats(self, active_fg_indices, threshold=0.05):
        with torch.no_grad():
            weights = self.active_class_weights(active_fg_indices).float()
            abs_weights = weights.abs()
            active = (abs_weights > threshold).float().sum(dim=1)
            total = abs_weights.sum(dim=1).clamp_min(1e-6)
            max_share = abs_weights.max(dim=1)[0] / total
            return {
                "W_active": active.mean(),
                "W_max_share": max_share.mean(),
            }
