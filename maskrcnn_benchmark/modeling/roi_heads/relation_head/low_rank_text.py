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

CORE_PROMPT_FIELD = "core_prompt"


def build_full_predicate_names(cfg):
    predicate_names = ["__background__"]
    allowed = set(cfg.OV_SETTING.PRDCS_BASE + cfg.OV_SETTING.PRDCS_NOVEL)
    for name in VG_PREDICATES:
        if name in allowed and name not in predicate_names:
            predicate_names.append(name)
    for name in cfg.OV_SETTING.PRDCS_BASE + cfg.OV_SETTING.PRDCS_NOVEL:
        if name not in predicate_names:
            predicate_names.append(name)
    return predicate_names


def build_split_indices(cfg, predicate_names):
    def ids_for(names):
        selected = set(names)
        return [idx for idx, name in enumerate(predicate_names) if idx == 0 or name in selected]

    return {
        "base": ids_for(cfg.OV_SETTING.PRDCS_BASE),
        "novel": ids_for(cfg.OV_SETTING.PRDCS_NOVEL),
        "semantic": ids_for(cfg.OV_SETTING.SEMAN),
        "total": list(range(len(predicate_names))),
    }


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
    return texts


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
        factor_comp_weight=0.1,
        factor_comp_tau=0.2,
        diff_neg_weight=0.1,
        diff_neg_topk=5,
        diff_neg_margin=0.2,
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
        self.factor_comp_weight = factor_comp_weight
        self.factor_comp_tau = factor_comp_tau
        self.diff_neg_weight = diff_neg_weight
        self.diff_neg_topk = diff_neg_topk
        self.diff_neg_margin = diff_neg_margin
        self._last_factor_comp_stats = None
        self._last_diff_neg_stats = None

    def active_class_weights(self, active_indices, normalize=False):
        weights = self.decomposer.class_weights
        if self.decomposer.train_mode in ("b", "none"):
            weights = weights.detach()
        weights = weights[active_indices].float()
        if active_indices.numel() > 0 and active_indices[0].item() == 0:
            weights = weights.clone()
            weights[0].zero_()
        if normalize:
            weights = F.normalize(weights, dim=-1)
        return weights

    def basis_logits(self, visual_features):
        basis = self.decomposer.basis_feat
        if self.decomposer.train_mode in ("w", "none"):
            basis = basis.detach()
        basis = F.normalize(basis.float(), dim=-1)
        visual_features = F.normalize(visual_features.float(), dim=-1)
        return visual_features @ basis.t()

    def logits(self, visual_features, active_indices):
        basis_logits = self.basis_logits(visual_features)
        weights = self.active_class_weights(active_indices)
        logits = basis_logits @ weights.t()
        return logits / self.logit_temperature, basis_logits

    def factor_composition_loss(self, basis_logits, labels, active_indices):
        self._last_factor_comp_stats = None
        if self.factor_comp_weight <= 0 or labels is None:
            return {}

        labels = labels.long()
        valid = (labels > 0) & (labels < len(active_indices))
        if valid.sum().item() == 0:
            return {}

        basis_logits = basis_logits[valid].float()
        labels = labels[valid]
        weights = self.active_class_weights(active_indices).detach().float()
        target_weights = weights[labels]

        tau = max(float(self.factor_comp_tau), 1e-6)
        target_dist = F.softmax(target_weights / tau, dim=-1)
        pred_log_dist = F.log_softmax(basis_logits / tau, dim=-1)
        loss = F.kl_div(pred_log_dist, target_dist, reduction="batchmean")

        self._last_factor_comp_stats = {
            "kl": loss.detach(),
            "cos": F.cosine_similarity(
                F.normalize(basis_logits, dim=-1),
                F.normalize(target_weights, dim=-1),
                dim=-1,
            ).detach().mean(),
            "entropy": (-(target_dist * target_dist.clamp_min(1e-12).log()).sum(dim=-1))
            .detach()
            .mean(),
        }
        return {"loss_factor_comp": loss * self.factor_comp_weight}

    def factor_difference_loss(self, basis_logits, relation_logits, labels, active_indices):
        self._last_diff_neg_stats = None
        if self.diff_neg_weight <= 0 or labels is None:
            return {}

        labels = labels.long()
        valid = (labels > 0) & (labels < relation_logits.size(1))
        if valid.sum().item() == 0:
            return {}

        basis_logits = basis_logits[valid].float()
        relation_logits = relation_logits[valid].float()
        labels = labels[valid]
        weights = self.active_class_weights(active_indices).detach()
        norm_weights = F.normalize(weights, dim=-1)

        target_weights = weights[labels]
        target_norm_weights = norm_weights[labels]
        semantic_reward = (target_norm_weights @ norm_weights.t()).clamp_min(0)

        target_logits = relation_logits.gather(1, labels.view(-1, 1))
        confusion_reward = torch.sigmoid(
            relation_logits.detach() - target_logits.detach() + self.diff_neg_margin
        )
        reward = semantic_reward * confusion_reward
        reward.scatter_(1, labels.view(-1, 1), 0)
        if reward.size(1) > 0:
            reward[:, 0] = 0

        topk = min(int(self.diff_neg_topk), max(reward.size(1) - 1, 1))
        neg_reward, neg_idx = reward.topk(topk, dim=1)
        if neg_reward.sum().item() <= 0:
            return {}

        neg_weights = weights[neg_idx]
        diff = F.normalize(target_weights.unsqueeze(1) - neg_weights, dim=-1)
        diff_score = (basis_logits.unsqueeze(1) * diff).sum(-1)
        loss = F.softplus(self.diff_neg_margin - diff_score)
        loss = (loss * neg_reward.detach()).sum() / neg_reward.detach().sum().clamp_min(1e-6)
        selected_sem = semantic_reward.gather(1, neg_idx)
        selected_conf = confusion_reward.gather(1, neg_idx)
        self._last_diff_neg_stats = {
            "reward": neg_reward.detach().mean(),
            "semantic": selected_sem.detach().mean(),
            "confusion": selected_conf.detach().mean(),
            "score": diff_score.detach().mean(),
        }
        return {"loss_factor_diff_neg": loss * self.diff_neg_weight}

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
        stats = self.decomposer.debug_stats()
        stats["diff_neg_weight"] = torch.as_tensor(
            self.diff_neg_weight,
            device=self.decomposer.basis_feat.device,
        )
        stats["factor_comp_weight"] = torch.as_tensor(
            self.factor_comp_weight,
            device=self.decomposer.basis_feat.device,
        )
        if self._last_factor_comp_stats is not None:
            for name, value in self._last_factor_comp_stats.items():
                stats["factor_comp_{}".format(name)] = value
        if self._last_diff_neg_stats is not None:
            for name, value in self._last_diff_neg_stats.items():
                stats["diff_neg_{}".format(name)] = value
        return stats

    def weight_usage_stats(self, active_indices, threshold=0.05):
        with torch.no_grad():
            weights = self.active_class_weights(active_indices).float()
            if weights.size(0) > 1:
                weights = weights[1:]
            abs_weights = weights.abs()
            active = (abs_weights > threshold).float().sum(dim=1)
            total = abs_weights.sum(dim=1).clamp_min(1e-6)
            max_share = abs_weights.max(dim=1)[0] / total
            return {
                "W_active": active.mean(),
                "W_max_share": max_share.mean(),
            }
