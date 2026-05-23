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


def build_relation_texts(prompt_json, predicate_names, field=CORE_PROMPT_FIELD, source="decomposed"):
    source = (source or "decomposed").lower()
    if source in ("decomposed", "prompt", "prompt_json", "description"):
        return load_relation_prompt_texts(prompt_json, predicate_names, field)
    if source in ("relation_of", "a_relation_of"):
        return ["a relation of {}".format(name) for name in predicate_names[1:]]
    raise ValueError("Unsupported LOW_RANK_TEXT.RELATION_TEXT_SOURCE: {}".format(source))


class RelationQueryBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super(RelationQueryBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        x = query
        q = self.norm1(x)
        x = x + self.dropout(self.self_attn(q, q, q, need_weights=False)[0])
        q = self.norm2(x)
        memory = memory.to(q.dtype)
        x = x + self.dropout(self.cross_attn(q, memory, memory, need_weights=False)[0])
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class RelationQueryTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        geometry_dim=32,
        prompt_scale=0.1,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        enabled=True,
    ):
        super(RelationQueryTransformer, self).__init__()
        self.enabled = bool(enabled)
        self.prompt_scale = float(prompt_scale)
        self.common_prompt_delta = nn.Parameter(torch.zeros(feature_dim))
        self.rel_prompt_delta = nn.Parameter(torch.zeros(feature_dim))
        self.geometry_query = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.geometry_token = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.memory_norm = nn.LayerNorm(feature_dim)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.blocks = nn.ModuleList(
            [RelationQueryBlock(feature_dim, num_heads, dropout) for _ in range(max(int(num_layers), 1))]
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def common_prompt(self, text_mean):
        return text_mean.float() + self.prompt_scale * self.common_prompt_delta.float()

    def relation_prompt(self, text_mean):
        return text_mean.float() + self.prompt_scale * self.rel_prompt_delta.float()

    def _as_tokens(self, features):
        if features.dim() == 2:
            return features.unsqueeze(1)
        return features

    def forward(
        self,
        pair_feature,
        subject_tokens,
        object_tokens,
        mva_s2o_tokens,
        mva_o2s_tokens,
        geometry_features,
        text_mean,
    ):
        pair_feature = F.normalize(pair_feature.float(), dim=-1)
        subject_tokens = F.normalize(self._as_tokens(subject_tokens).float(), dim=-1)
        object_tokens = F.normalize(self._as_tokens(object_tokens).float(), dim=-1)
        mva_s2o_tokens = F.normalize(self._as_tokens(mva_s2o_tokens).float(), dim=-1)
        mva_o2s_tokens = F.normalize(self._as_tokens(mva_o2s_tokens).float(), dim=-1)
        geometry_features = geometry_features.float().to(pair_feature.device)

        common_prompt = self.common_prompt(text_mean).to(pair_feature.device)
        rel_prompt = self.relation_prompt(text_mean).to(pair_feature.device)
        if not self.enabled:
            return pair_feature, pair_feature, common_prompt

        geo_query = self.geometry_query(geometry_features)
        geo_token = F.normalize(self.geometry_token(geometry_features), dim=-1).unsqueeze(1)
        subject_cls = subject_tokens[:, :1]
        object_cls = object_tokens[:, :1]
        pair_cls = pair_feature.unsqueeze(1)
        memory = torch.cat(
            (
                subject_tokens,
                object_tokens,
                mva_s2o_tokens,
                mva_o2s_tokens,
                subject_cls,
                object_cls,
                pair_cls,
                geo_token,
            ),
            dim=1,
        )
        memory = self.memory_norm(memory)

        query = torch.stack(
            (
                common_prompt.view(1, -1).expand(pair_feature.size(0), -1) + geo_query,
                rel_prompt.view(1, -1).expand(pair_feature.size(0), -1) + geo_query,
            ),
            dim=1,
        )
        query = self.query_norm(query)
        for block in self.blocks:
            query = block(query, memory)

        z_common = F.normalize(self.output_norm(query[:, 0] + pair_feature), dim=-1)
        z_rel = F.normalize(self.output_norm(query[:, 1] + pair_feature), dim=-1)
        return z_common, z_rel, common_prompt


class SharedWAdapter(nn.Module):
    def __init__(self, feature_dim, rank_dim, geometry_dim=32, scale=0.25, enabled=True):
        super(SharedWAdapter, self).__init__()
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim + geometry_dim),
            nn.Linear(feature_dim + geometry_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, rank_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, z_rel, geometry, weights, active_mask):
        if not self.enabled or self.scale <= 0:
            return weights.unsqueeze(0).expand(z_rel.size(0), -1, -1)
        gate_input = torch.cat((z_rel.float(), geometry.float()), dim=-1)
        gate = torch.tanh(self.gate(gate_input))
        factor = 1 + self.scale * gate.unsqueeze(1) * active_mask.unsqueeze(0)
        return weights.unsqueeze(0) * factor


class SemanticTransport(nn.Module):
    def __init__(self, feature_dim, geometry_dim=32, step_scale=0.2, enabled=True):
        super(SemanticTransport, self).__init__()
        self.enabled = bool(enabled)
        self.step_scale = float(step_scale)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim * 2 + geometry_dim),
            nn.Linear(feature_dim * 2 + geometry_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_rel, delta_text, geometry):
        if not self.enabled or self.step_scale <= 0:
            return z_rel
        geometry = geometry.float().to(z_rel.device)
        while geometry.dim() < z_rel.dim():
            geometry = geometry.unsqueeze(-2)
        geometry = geometry.expand(*z_rel.shape[:-1], geometry.size(-1))
        step = self.mlp(torch.cat((z_rel.float(), delta_text.float(), geometry), dim=-1))
        return F.normalize(z_rel.float() + self.step_scale * step, dim=-1)


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
        visual_lowrank_residual_scale=0.2,
        relation_query_layers=2,
        relation_query_heads=4,
        relation_query_dropout=0.1,
        shared_w_adapter_enabled=True,
        shared_w_adapter_scale=0.25,
        transport_logit_weight=None,
        semantic_transport_enabled=None,
        transport_topk_anchor=None,
        transport_loss_weight=None,
        transport_cycle_weight=0.02,
        transport_step_scale=0.2,
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
        self.transport_logit_weight = float(
            delta_logit_weight if transport_logit_weight is None else transport_logit_weight
        )
        self.common_loss_weight = float(common_loss_weight)
        self.residual_align_weight = float(residual_align_weight)
        self.semantic_transport_enabled = bool(
            delta_transfer_enabled if semantic_transport_enabled is None else semantic_transport_enabled
        )
        self.transport_topk_anchor = int(delta_topk_anchor if transport_topk_anchor is None else transport_topk_anchor)
        self.delta_aux_topk = int(delta_aux_topk)
        self.transport_loss_weight = float(delta_loss_weight if transport_loss_weight is None else transport_loss_weight)
        self.transport_cycle_weight = float(transport_cycle_weight)
        self.w_active_threshold = float(w_active_threshold)
        self.visual_lowrank_residual_scale = float(visual_lowrank_residual_scale)
        rank_dim = self.decomposer.basis_feat.size(0)
        feature_dim = text_features.size(-1)

        self.relation_queries = RelationQueryTransformer(
            feature_dim,
            geometry_dim=32,
            prompt_scale=common_prompt_scale,
            num_layers=relation_query_layers,
            num_heads=relation_query_heads,
            dropout=relation_query_dropout,
            enabled=common_prompt_enabled,
        )
        self.rank_projector = nn.Sequential(
            nn.LayerNorm(feature_dim * 2 + 32),
            nn.Linear(feature_dim * 2 + 32, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, rank_dim),
        )
        self.rank_norm = nn.LayerNorm(rank_dim)
        self.shared_w_adapter = SharedWAdapter(
            feature_dim,
            rank_dim,
            geometry_dim=32,
            scale=shared_w_adapter_scale,
            enabled=shared_w_adapter_enabled,
        )
        self.semantic_transport = SemanticTransport(
            feature_dim,
            geometry_dim=32,
            step_scale=transport_step_scale,
            enabled=self.semantic_transport_enabled,
        )
        nn.init.zeros_(self.rank_projector[-1].weight)
        nn.init.zeros_(self.rank_projector[-1].bias)

        # Kept for old configs; the new path uses residual/text/delta weights.
        self.address_logit_weight = float(address_logit_weight)
        self._debug_transport_abs = None
        self._debug_transport_std = None

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

    def adapted_class_weights(self, z_rel, geometry, indices):
        weights = self.class_weights_by_indices(indices).float()
        active_mask = self._weight_active_mask(indices).to(weights.device)
        return self.shared_w_adapter(z_rel, geometry, weights, active_mask)

    def active_classifier(self, adapted_weights, z_common):
        basis = self._basis_for_classifier()
        residual = torch.matmul(adapted_weights.float(), basis)
        classifier = z_common.float().unsqueeze(1) + residual
        return F.normalize(classifier, dim=-1)

    def lowrank_logits(self, visual_lowrank, active_fg_indices, adapted_weights=None):
        visual_lowrank = F.normalize(visual_lowrank.float(), dim=-1)
        if adapted_weights is None:
            weights = self.active_class_weights(active_fg_indices, normalize=True).float()
            weights = weights.unsqueeze(0).expand(visual_lowrank.size(0), -1, -1)
        else:
            weights = F.normalize(adapted_weights.float(), dim=-1)
        temperature = max(float(self.lowrank_temperature), 1e-6)
        return torch.einsum("nr,ncr->nc", visual_lowrank, weights) / temperature

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

    def _prepare_tokens(self, features, fallback):
        if features is None:
            return fallback.unsqueeze(1)
        if features.dim() == 2:
            return features.unsqueeze(1)
        return features

    def _rank_features(self, z_rel, z_common, geometry):
        base_rank = self.basis_logits(z_rel)
        rank_delta = self.rank_projector(torch.cat((z_rel.float(), z_common.float(), geometry.float()), dim=-1))
        return self.rank_norm(base_rank + self.visual_lowrank_residual_scale * rank_delta)

    def _delta_text(self, delta_rank):
        basis = self._basis_for_classifier()
        return torch.matmul(delta_rank.float(), basis)

    def transport_logits(
        self,
        z_rel,
        geometry,
        visual_lowrank,
        active_fg_indices,
        base_anchor_fg_indices,
        active_classifier,
    ):
        if (
            not self.semantic_transport_enabled
            or self.transport_logit_weight <= 0
            or base_anchor_fg_indices is None
            or base_anchor_fg_indices.numel() == 0
        ):
            return z_rel.new_zeros((z_rel.size(0), active_fg_indices.numel()))

        anchor_weights = self.class_weights_by_indices(base_anchor_fg_indices)
        candidate_weights = self.active_class_weights(active_fg_indices)
        if anchor_weights.size(0) == 0 or candidate_weights.size(0) == 0:
            return z_rel.new_zeros((z_rel.size(0), active_fg_indices.numel()))

        anchor_logits = self.lowrank_logits(visual_lowrank, base_anchor_fg_indices)
        k = min(max(self.transport_topk_anchor, 1), anchor_weights.size(0))
        top_logits, top_indices = anchor_logits.topk(k=k, dim=-1)
        top_anchor_weights = anchor_weights[top_indices]
        anchor_prob = F.softmax(top_logits, dim=-1)

        delta = candidate_weights.view(1, 1, candidate_weights.size(0), -1)
        delta = delta - top_anchor_weights.unsqueeze(2)
        delta_text = self._delta_text(delta)
        visual = z_rel.view(z_rel.size(0), 1, 1, -1).expand_as(delta_text)
        shifted = self.semantic_transport(visual, delta_text, geometry)

        classifier = active_classifier.unsqueeze(1)
        temperature = max(float(self.logit_temperature), 1e-6)
        scores = (F.normalize(shifted, dim=-1) * classifier).sum(dim=-1) / temperature
        logits = (anchor_prob.unsqueeze(-1) * scores).sum(dim=1)
        self._debug_transport_abs = logits.detach().float().abs().mean()
        self._debug_transport_std = logits.detach().float().std()
        return logits

    def logits(
        self,
        visual_features,
        active_fg_indices,
        raw_visual_features=None,
        subject_features=None,
        object_features=None,
        subject_tokens=None,
        object_tokens=None,
        mva_s2o_tokens=None,
        mva_o2s_tokens=None,
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
        subject_tokens = self._prepare_tokens(subject_tokens, subject)
        object_tokens = self._prepare_tokens(object_tokens, obj)
        mva_s2o_tokens = self._prepare_tokens(mva_s2o_tokens, pair)
        mva_o2s_tokens = self._prepare_tokens(mva_o2s_tokens, pair)

        z_common, z_rel, common_prompt = self.relation_queries(
            pair,
            subject_tokens,
            object_tokens,
            mva_s2o_tokens,
            mva_o2s_tokens,
            geometry,
            self.decomposer.text_mean,
        )

        visual_lowrank = self._rank_features(z_rel, z_common, geometry)

        adapted_weights = self.adapted_class_weights(z_rel, geometry, active_fg_indices)
        classifier = self.active_classifier(adapted_weights, z_common)
        text_logits = torch.einsum("nd,ncd->nc", z_rel, classifier)
        text_logits = text_logits / max(float(self.logit_temperature), 1e-6)

        residual_logits = self.lowrank_logits(visual_lowrank, active_fg_indices, adapted_weights)
        transport_logits = self.transport_logits(
            z_rel,
            geometry,
            visual_lowrank,
            active_fg_indices,
            base_anchor_fg_indices,
            classifier,
        )

        logit_parts = [
            (self.residual_logit_weight, residual_logits),
            (self.text_logit_weight, text_logits),
            (self.transport_logit_weight if self.semantic_transport_enabled else 0.0, transport_logits),
        ]
        total_weight = sum(max(weight, 0.0) for weight, _ in logit_parts)
        if total_weight <= 0:
            total_weight = 1.0
        logits = sum(max(weight, 0.0) * part for weight, part in logit_parts) / total_weight

        aux_cache = {
            "common_features": z_common,
            "common_prompt": common_prompt,
            "z_rel": z_rel,
            "visual_lowrank": visual_lowrank,
            "geometry": geometry,
            "active_fg_indices": active_fg_indices.detach(),
            "base_anchor_fg_indices": None if base_anchor_fg_indices is None else base_anchor_fg_indices.detach(),
        }
        return logits, visual_lowrank, aux_cache

    def _merge_aux_cache(self, aux_cache):
        if not aux_cache:
            return None
        return {
            "common_features": torch.cat([item["common_features"] for item in aux_cache], dim=0),
            "z_rel": torch.cat([item["z_rel"] for item in aux_cache], dim=0),
            "visual_lowrank": torch.cat([item["visual_lowrank"] for item in aux_cache], dim=0),
            "geometry": torch.cat([item["geometry"] for item in aux_cache], dim=0),
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
        z_rel = cache["z_rel"]
        geometry = cache["geometry"]
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

        if self.relation_queries.enabled and self.common_loss_weight > 0:
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
                losses["loss_lr_rank_align"] = residual_loss.mean() * self.residual_align_weight
            else:
                losses["loss_lr_rank_align"] = zero

        if self.semantic_transport_enabled and (
            self.transport_loss_weight > 0 or self.transport_cycle_weight > 0
        ):
            transport_cls, transport_cycle = self._transport_auxiliary_losses(
                z_rel,
                cache["common_features"],
                geometry,
                labels,
                valid,
                active_fg_indices,
                cache["base_anchor_fg_indices"],
            )
            if self.transport_loss_weight > 0:
                losses["loss_lr_transport_cls"] = transport_cls * self.transport_loss_weight
            if self.transport_cycle_weight > 0:
                losses["loss_lr_transport_cycle"] = transport_cycle * self.transport_cycle_weight

        return losses

    def _transport_auxiliary_losses(
        self,
        z_rel,
        z_common,
        geometry,
        labels,
        valid,
        active_fg_indices,
        base_anchor_fg_indices,
    ):
        zero = z_rel.sum() * 0
        if not valid.any() or base_anchor_fg_indices is None or base_anchor_fg_indices.numel() == 0:
            return zero, zero

        base_anchor_fg_indices = base_anchor_fg_indices.to(z_rel.device).long()
        anchor_weights = self.class_weights_by_indices(base_anchor_fg_indices)
        if anchor_weights.size(0) <= 1:
            return zero, zero

        local_labels = labels[valid] - 1
        gt_fg_indices = active_fg_indices[local_labels].to(z_rel.device)
        gt_weights = self.class_weights_by_indices(gt_fg_indices)
        gt_visual = z_rel[valid]
        gt_common = z_common[valid]
        gt_geometry = geometry[valid]

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
        delta_text = self._delta_text(delta)
        visual = gt_visual.unsqueeze(1).expand_as(delta_text)
        shifted = self.semantic_transport(visual, delta_text, gt_geometry)

        anchor_weights_per_pair = self.adapted_class_weights(gt_visual, gt_geometry, base_anchor_fg_indices)
        anchor_classifier = self.active_classifier(anchor_weights_per_pair, gt_common)
        shifted_norm = F.normalize(shifted, dim=-1)
        temperature = max(float(self.logit_temperature), 1e-6)
        logits = torch.einsum("nkd,nad->nka", shifted_norm, anchor_classifier) / temperature

        flat_valid = finite.view(-1)
        cls_loss = F.cross_entropy(
            logits.view(-1, anchor_weights.size(0))[flat_valid],
            target_indices.reshape(-1)[flat_valid],
        )
        cycle = self.semantic_transport(shifted, -delta_text, gt_geometry)
        cycle_loss = 1 - F.cosine_similarity(cycle[finite], visual[finite], dim=-1)
        return cls_loss, cycle_loss.mean()

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
        stats = self.decomposer.debug_stats()
        if self._debug_transport_abs is not None:
            stats["transport_abs"] = self._debug_transport_abs
        if self._debug_transport_std is not None:
            stats["transport_std"] = self._debug_transport_std
        return stats

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
