import json
import math

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


class CommonRelationExtractor(nn.Module):
    def __init__(self, feature_dim, geometry_dim=32, prompt_scale=0.1, enabled=True):
        super(CommonRelationExtractor, self).__init__()
        self.enabled = bool(enabled)
        self.prompt_scale = float(prompt_scale)
        self.common_prompt_delta = nn.Parameter(torch.zeros(feature_dim))
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def prompt(self, text_mean):
        return text_mean.float() + self.prompt_scale * self.common_prompt_delta.float()

    def forward(self, pair_features, raw_pair_features, subject_features, object_features, geometry_features, text_mean):
        common_prompt = self.prompt(text_mean).to(pair_features.device)
        if not self.enabled:
            return pair_features, common_prompt

        geometry_token = F.normalize(self.geometry_proj(geometry_features.float()), dim=-1)
        tokens = torch.stack(
            (
                pair_features.float(),
                raw_pair_features.float(),
                subject_features.float(),
                object_features.float(),
                geometry_token.float(),
            ),
            dim=1,
        )
        query = F.normalize(common_prompt, dim=-1)
        attn = torch.matmul(tokens, query.view(-1, 1)).squeeze(-1)
        attn = F.softmax(attn / math.sqrt(tokens.size(-1)), dim=-1)
        common_feature = (attn.unsqueeze(-1) * tokens).sum(dim=1)
        common_feature = self.output_norm(common_feature + pair_features.float())
        return F.normalize(common_feature, dim=-1), common_prompt


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
        address_anchor_weight=0.0,
        address_graph_weight=0.0,
        address_graph_topk=5,
        train_basis=True,
        train_mode="w",
        logit_temperature=0.05,
        address_logit_weight=0.25,
        address_logit_temperature=0.05,
        visual_gate_scale=0.5,
        common_prompt_enabled=True,
        common_prompt_scale=0.1,
        common_loss_weight=0.02,
        residual_logit_weight=0.45,
        text_logit_weight=0.35,
        delta_logit_weight=0.20,
        residual_align_weight=0.10,
        delta_transfer_enabled=True,
        delta_topk_anchor=5,
        delta_aux_topk=3,
        delta_loss_weight=0.10,
        delta_dir_weight=0.02,
        weight_delta_scale=0.1,
        weight_anchor_weight=0.005,
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
            address_anchor_weight=address_anchor_weight,
            address_graph_weight=address_graph_weight,
            address_graph_topk=address_graph_topk,
            weight_delta_scale=weight_delta_scale,
            weight_anchor_weight=weight_anchor_weight,
        )
        self.logit_temperature = logit_temperature
        self.lowrank_temperature = address_logit_temperature
        self.residual_logit_weight = float(residual_logit_weight)
        self.text_logit_weight = float(text_logit_weight)
        self.delta_logit_weight = float(delta_logit_weight)
        self.common_loss_weight = float(common_loss_weight)
        self.residual_align_weight = float(residual_align_weight)
        self.delta_transfer_enabled = bool(delta_transfer_enabled)
        self.delta_topk_anchor = int(delta_topk_anchor)
        self.delta_aux_topk = int(delta_aux_topk)
        self.delta_loss_weight = float(delta_loss_weight)
        self.delta_dir_weight = float(delta_dir_weight)
        self.visual_gate_scale = float(visual_gate_scale)
        self.w_active_threshold = float(w_active_threshold)
        rank_dim = self.decomposer.basis_feat.size(0)
        feature_dim = text_features.size(-1)

        self.common_extractor = CommonRelationExtractor(
            feature_dim,
            geometry_dim=32,
            prompt_scale=common_prompt_scale,
            enabled=common_prompt_enabled,
        )
        visual_context_dim = feature_dim * 5 + 32
        self.visual_lowrank_encoder = nn.Sequential(
            nn.LayerNorm(visual_context_dim),
            nn.Linear(visual_context_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, rank_dim),
        )
        self.visual_lowrank_gate = nn.Sequential(
            nn.LayerNorm(rank_dim),
            nn.Linear(rank_dim, rank_dim),
            nn.ReLU(inplace=True),
            nn.Linear(rank_dim, rank_dim),
        )
        self.delta_gate = nn.Sequential(
            nn.LayerNorm(rank_dim * 2),
            nn.Linear(rank_dim * 2, rank_dim),
            nn.ReLU(inplace=True),
            nn.Linear(rank_dim, rank_dim),
        )
        nn.init.zeros_(self.visual_lowrank_encoder[-1].weight)
        nn.init.zeros_(self.visual_lowrank_encoder[-1].bias)
        nn.init.zeros_(self.visual_lowrank_gate[-1].weight)
        nn.init.zeros_(self.visual_lowrank_gate[-1].bias)
        nn.init.zeros_(self.delta_gate[-1].weight)
        nn.init.zeros_(self.delta_gate[-1].bias)

        # Kept for old configs; the new path uses residual/text/delta weights.
        self.address_logit_weight = float(address_logit_weight)

    def active_class_weights(self, active_fg_indices, normalize=False):
        weights = self.class_weights_by_indices(active_fg_indices)
        if normalize:
            weights = F.normalize(weights, dim=-1)
        return weights

    def class_weights_by_indices(self, indices, normalize=False, initial=False):
        if initial:
            weights = self.decomposer.initial_class_weights
        else:
            weights = self.decomposer.effective_class_weights()
        indices = indices.to(weights.device).long()
        weights = weights[indices].float()
        if normalize:
            weights = F.normalize(weights, dim=-1)
        return weights

    def basis_logits(self, visual_features):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        basis = self.decomposer.classifier_basis()
        basis = F.normalize(basis.float(), dim=-1)
        return visual_features @ basis.t()

    def _basis_for_classifier(self):
        return self.decomposer.classifier_basis().float()

    def _weight_active_mask(self, active_fg_indices):
        weights = self.class_weights_by_indices(active_fg_indices, initial=True).abs()
        if self.w_active_threshold <= 0:
            return torch.ones_like(weights)
        return (weights > self.w_active_threshold).float()

    def _visual_gate(self, visual_lowrank, active_fg_indices):
        if self.visual_gate_scale <= 0:
            return None
        gate_delta = torch.tanh(self.visual_lowrank_gate(visual_lowrank.float()))
        gate = 1 + self.visual_gate_scale * gate_delta
        active_mask = self._weight_active_mask(active_fg_indices)
        return 1 + (gate.unsqueeze(1) - 1) * active_mask.unsqueeze(0)

    def active_classifier(self, active_fg_indices, common_prompt, visual_gate=None):
        weights = self.active_class_weights(active_fg_indices).float()
        basis = self._basis_for_classifier()
        if visual_gate is None:
            classifier = common_prompt.float().view(1, -1) + weights @ basis
            return F.normalize(classifier, dim=-1)

        gated_weights = weights.unsqueeze(0) * visual_gate
        residual = torch.matmul(gated_weights, basis)
        classifier = common_prompt.float().view(1, 1, -1) + residual
        return F.normalize(classifier, dim=-1)

    def lowrank_logits(self, visual_lowrank, active_fg_indices):
        visual_lowrank = F.normalize(visual_lowrank.float(), dim=-1)
        weights = self.active_class_weights(active_fg_indices, normalize=True).float()
        temperature = max(float(self.lowrank_temperature), 1e-6)
        return visual_lowrank @ weights.t() / temperature

    def _prepare_visual_inputs(
        self,
        visual_features,
        raw_visual_features=None,
        subject_features=None,
        object_features=None,
        geometry_features=None,
    ):
        pair = F.normalize(visual_features.float(), dim=-1)
        raw = pair if raw_visual_features is None else F.normalize(raw_visual_features.float(), dim=-1)
        subject = pair if subject_features is None else F.normalize(subject_features.float(), dim=-1)
        obj = pair if object_features is None else F.normalize(object_features.float(), dim=-1)
        if geometry_features is None:
            geometry = pair.new_zeros((pair.size(0), 32))
        else:
            geometry = geometry_features.float().to(pair.device)
        return pair, raw, subject, obj, geometry

    def _delta_shift(self, visual_lowrank, delta):
        gate_input = torch.cat((visual_lowrank.float(), delta.float()), dim=-1)
        gate = torch.sigmoid(self.delta_gate(gate_input))
        return visual_lowrank + gate * delta, gate

    def delta_transfer_logits(self, visual_lowrank, active_fg_indices, base_anchor_fg_indices):
        if (
            not self.delta_transfer_enabled
            or self.delta_logit_weight <= 0
            or base_anchor_fg_indices is None
            or base_anchor_fg_indices.numel() == 0
        ):
            return visual_lowrank.new_zeros((visual_lowrank.size(0), active_fg_indices.numel()))

        anchor_weights = self.class_weights_by_indices(base_anchor_fg_indices)
        candidate_weights = self.active_class_weights(active_fg_indices)
        if anchor_weights.size(0) == 0 or candidate_weights.size(0) == 0:
            return visual_lowrank.new_zeros((visual_lowrank.size(0), active_fg_indices.numel()))

        anchor_logits = self.lowrank_logits(visual_lowrank, base_anchor_fg_indices)
        k = min(max(self.delta_topk_anchor, 1), anchor_weights.size(0))
        top_logits, top_indices = anchor_logits.topk(k=k, dim=-1)
        top_anchor_weights = anchor_weights[top_indices]
        anchor_prob = F.softmax(top_logits, dim=-1)

        delta = candidate_weights.view(1, 1, candidate_weights.size(0), -1)
        delta = delta - top_anchor_weights.unsqueeze(2)
        visual = visual_lowrank.view(visual_lowrank.size(0), 1, 1, -1).expand_as(delta)
        shifted, _ = self._delta_shift(visual, delta)

        candidate_norm = F.normalize(candidate_weights, dim=-1).view(1, 1, candidate_weights.size(0), -1)
        shifted = F.normalize(shifted, dim=-1)
        temperature = max(float(self.lowrank_temperature), 1e-6)
        scores = (shifted * candidate_norm).sum(dim=-1) / temperature
        return (anchor_prob.unsqueeze(-1) * scores).sum(dim=1)

    def logits(
        self,
        visual_features,
        active_fg_indices,
        raw_visual_features=None,
        subject_features=None,
        object_features=None,
        geometry_features=None,
        base_anchor_fg_indices=None,
    ):
        pair, raw, subject, obj, geometry = self._prepare_visual_inputs(
            visual_features,
            raw_visual_features=raw_visual_features,
            subject_features=subject_features,
            object_features=object_features,
            geometry_features=geometry_features,
        )
        common_feature, common_prompt = self.common_extractor(
            pair,
            raw,
            subject,
            obj,
            geometry,
            self.decomposer.text_mean,
        )

        base_lowrank = self.basis_logits(pair)
        visual_context = torch.cat((common_feature, pair, raw, subject, obj, geometry), dim=-1)
        visual_lowrank = base_lowrank + self.visual_lowrank_encoder(visual_context.float())

        visual_gate = self._visual_gate(visual_lowrank, active_fg_indices)
        classifier = self.active_classifier(active_fg_indices, common_prompt, visual_gate)
        if classifier.dim() == 3:
            text_logits = torch.einsum("nd,ncd->nc", pair, classifier)
        else:
            text_logits = pair @ classifier.t()
        text_logits = text_logits / max(float(self.logit_temperature), 1e-6)

        residual_logits = self.lowrank_logits(visual_lowrank, active_fg_indices)
        delta_logits = self.delta_transfer_logits(visual_lowrank, active_fg_indices, base_anchor_fg_indices)

        logit_parts = [
            (self.residual_logit_weight, residual_logits),
            (self.text_logit_weight, text_logits),
            (self.delta_logit_weight if self.delta_transfer_enabled else 0.0, delta_logits),
        ]
        total_weight = sum(max(weight, 0.0) for weight, _ in logit_parts)
        if total_weight <= 0:
            total_weight = 1.0
        logits = sum(max(weight, 0.0) * part for weight, part in logit_parts) / total_weight

        aux_cache = {
            "common_features": common_feature,
            "common_prompt": common_prompt,
            "visual_lowrank": visual_lowrank,
            "active_fg_indices": active_fg_indices.detach(),
            "base_anchor_fg_indices": None if base_anchor_fg_indices is None else base_anchor_fg_indices.detach(),
        }
        return logits, visual_lowrank, aux_cache

    def _merge_aux_cache(self, aux_cache):
        if not aux_cache:
            return None
        return {
            "common_features": torch.cat([item["common_features"] for item in aux_cache], dim=0),
            "visual_lowrank": torch.cat([item["visual_lowrank"] for item in aux_cache], dim=0),
            "common_prompt": aux_cache[0]["common_prompt"],
            "active_fg_indices": aux_cache[0]["active_fg_indices"],
            "base_anchor_fg_indices": aux_cache[0]["base_anchor_fg_indices"],
        }

    def _lowrank_aux_losses(self, aux_cache, labels):
        losses = {}
        cache = self._merge_aux_cache(aux_cache)
        if cache is None or labels is None:
            return losses

        visual_lowrank = cache["visual_lowrank"]
        labels = labels.to(visual_lowrank.device).long().view(-1)
        if labels.numel() != visual_lowrank.size(0):
            raise ValueError(
                "Low-rank auxiliary labels size {} does not match visual features size {}.".format(
                    labels.numel(), visual_lowrank.size(0)
                )
            )

        active_fg_indices = cache["active_fg_indices"].to(visual_lowrank.device)
        valid = (labels > 0) & (labels <= active_fg_indices.numel())
        zero = visual_lowrank.sum() * 0

        if self.common_extractor.enabled and self.common_loss_weight > 0:
            if valid.any():
                target = F.normalize(cache["common_prompt"].view(1, -1), dim=-1)
                common = F.normalize(cache["common_features"][valid], dim=-1)
                common_loss = 1 - (common * target).sum(dim=-1)
                losses["loss_lr_common"] = common_loss.mean() * self.common_loss_weight
            else:
                losses["loss_lr_common"] = zero

        if self.residual_align_weight > 0:
            if valid.any():
                local_labels = labels[valid] - 1
                target = self.active_class_weights(active_fg_indices)[local_labels].detach()
                residual_loss = 1 - F.cosine_similarity(visual_lowrank[valid], target, dim=-1)
                losses["loss_lr_residual_align"] = residual_loss.mean() * self.residual_align_weight
            else:
                losses["loss_lr_residual_align"] = zero

        if self.delta_transfer_enabled and (self.delta_loss_weight > 0 or self.delta_dir_weight > 0):
            delta_cls, delta_dir = self._delta_auxiliary_losses(
                visual_lowrank,
                labels,
                valid,
                active_fg_indices,
                cache["base_anchor_fg_indices"],
            )
            if self.delta_loss_weight > 0:
                losses["loss_lr_delta_cls"] = delta_cls * self.delta_loss_weight
            if self.delta_dir_weight > 0:
                losses["loss_lr_delta_dir"] = delta_dir * self.delta_dir_weight

        return losses

    def _delta_auxiliary_losses(self, visual_lowrank, labels, valid, active_fg_indices, base_anchor_fg_indices):
        zero = visual_lowrank.sum() * 0
        if not valid.any() or base_anchor_fg_indices is None or base_anchor_fg_indices.numel() == 0:
            return zero, zero

        base_anchor_fg_indices = base_anchor_fg_indices.to(visual_lowrank.device).long()
        anchor_weights = self.class_weights_by_indices(base_anchor_fg_indices)
        if anchor_weights.size(0) <= 1:
            return zero, zero

        local_labels = labels[valid] - 1
        gt_fg_indices = active_fg_indices[local_labels].to(visual_lowrank.device)
        gt_weights = self.class_weights_by_indices(gt_fg_indices)
        gt_visual = visual_lowrank[valid]

        sim = F.normalize(gt_weights, dim=-1) @ F.normalize(anchor_weights, dim=-1).t()
        same_predicate = gt_fg_indices.view(-1, 1) == base_anchor_fg_indices.view(1, -1)
        sim = sim.masked_fill(same_predicate, float("-inf"))
        k = min(max(self.delta_aux_topk, 1), anchor_weights.size(0))
        top_sim, target_indices = sim.topk(k=k, dim=-1)
        finite = torch.isfinite(top_sim)
        if not finite.any():
            return zero, zero

        target_weights = anchor_weights[target_indices]
        delta = target_weights - gt_weights.unsqueeze(1)
        visual = gt_visual.unsqueeze(1).expand_as(delta)
        shifted, _ = self._delta_shift(visual, delta)

        shifted_norm = F.normalize(shifted, dim=-1)
        anchor_norm = F.normalize(anchor_weights, dim=-1)
        temperature = max(float(self.lowrank_temperature), 1e-6)
        logits = torch.matmul(shifted_norm, anchor_norm.t()) / temperature

        flat_valid = finite.view(-1)
        cls_loss = F.cross_entropy(
            logits.view(-1, anchor_weights.size(0))[flat_valid],
            target_indices.reshape(-1)[flat_valid],
        )
        direction = shifted - visual
        dir_loss = 1 - F.cosine_similarity(direction[finite], delta[finite], dim=-1)
        return cls_loss, dir_loss.mean()

    def losses(self, aux_cache=None, labels=None):
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
        if "lowrank_anchor" in factor_losses:
            losses["loss_lr_anchor"] = factor_losses["lowrank_anchor"]
        if "address_graph" in factor_losses:
            losses["loss_lr_graph"] = factor_losses["address_graph"]
        losses.update(self._lowrank_aux_losses(aux_cache, labels))
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
