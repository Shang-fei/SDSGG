import json
import math

import torch
from torch import nn
from torch.nn import functional as F


VG_PREDICATES = [
    "__background__", "above", "across", "against", "along", "and", "at", "attached to",
    "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating",
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in",
    "in front of", "laying on", "looking at", "lying on", "made of", "mounted on",
    "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of",
    "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using",
    "walking in", "walking on", "watching", "wearing", "wears", "with",
]

FACTOR_NAMES = ("spatial", "interaction", "attribute")
PROMPT_FIELDS = {
    "spatial": "spatial_prompt",
    "interaction": "interaction_prompt",
    "attribute": "attribute_prompt",
}


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


def load_factor_prompt_texts(prompt_json, predicate_names):
    if not prompt_json:
        raise ValueError("LOW_RANK_TEXT.PROMPT_JSON must point to an LLM factor prompt JSON.")
    with open(prompt_json, "r") as f:
        prompt_data = json.load(f)

    texts = {factor: [] for factor in FACTOR_NAMES}
    for name in predicate_names:
        if name == "__background__":
            for factor in FACTOR_NAMES:
                texts[factor].append("")
            continue
        if name not in prompt_data:
            raise KeyError("Missing predicate '{}' in {}".format(name, prompt_json))
        item = prompt_data[name]
        for factor, field in PROMPT_FIELDS.items():
            if field not in item:
                raise KeyError("Missing field '{}.{}' in {}".format(name, field, prompt_json))
            texts[factor].append(item[field])
    return texts


def init_low_rank_with_pca(text_features, rank):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "LOW_RANK_TEXT uses sklearn.decomposition.PCA to match HOLa. "
            "Please install scikit-learn in the training environment."
        )

    temp_text = text_features.t().cpu().numpy()
    max_rank = min(temp_text.shape[0], temp_text.shape[1])
    rank = max(1, min(int(rank), max_rank))

    pca = PCA(n_components=rank)
    pca.fit(temp_text)
    basis = torch.tensor(pca.transform(temp_text).T).type_as(text_features)
    basis = F.normalize(basis, dim=-1)
    weights = torch.tensor(pca.components_.T).type_as(text_features)
    return basis, weights


class FactorizedRelationTextAdapter(nn.Module):
    def __init__(
        self,
        factor_text_features,
        rank,
        router_hidden_dim=256,
        temperature=0.07,
        recon_loss_weight=0.1,
        sparsity_weight=0.1,
        basis_decorr_weight=0.001,
        gate_entropy_weight=0.0,
        train_basis=False,
    ):
        super(FactorizedRelationTextAdapter, self).__init__()
        self.factors = FACTOR_NAMES
        self.temperature = float(temperature)
        self.recon_loss_weight = recon_loss_weight
        self.sparsity_weight = sparsity_weight
        self.basis_decorr_weight = basis_decorr_weight
        self.gate_entropy_weight = gate_entropy_weight

        self.class_weights = nn.ParameterDict()
        self.query_proj = nn.ModuleDict()
        self.key_proj = nn.ModuleDict()
        self.factor_to_idx = {factor: idx for idx, factor in enumerate(self.factors)}

        feature_dim = None
        for factor in self.factors:
            text_features = factor_text_features[factor].detach().float()
            text_features = F.normalize(text_features, dim=-1)
            if text_features.size(0) > 0:
                text_features = text_features.clone()
                text_features[0].zero_()
            if feature_dim is None:
                feature_dim = text_features.size(-1)

            basis, weights = init_low_rank_with_pca(text_features, rank)
            self.class_weights[factor] = nn.Parameter(weights)
            if train_basis:
                setattr(self, "{}_basis".format(factor), nn.Parameter(basis))
            else:
                self.register_buffer("{}_basis".format(factor), basis)
            self.register_buffer("{}_original_text_features".format(factor), text_features)

            self.query_proj[factor] = nn.Sequential(
                nn.Linear(feature_dim, router_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(router_hidden_dim, feature_dim),
            )
            self.key_proj[factor] = nn.Linear(feature_dim, feature_dim)

        self.factor_tokens = nn.Parameter(torch.zeros(len(self.factors), feature_dim))
        nn.init.normal_(self.factor_tokens, std=0.02)

    def basis(self, factor):
        return getattr(self, "{}_basis".format(factor))

    def original_text_features(self, factor):
        return getattr(self, "{}_original_text_features".format(factor))

    def reconstruct_factor(self, factor):
        text_features = self.class_weights[factor] @ self.basis(factor)
        if text_features.size(0) > 0:
            text_features = text_features.clone()
            text_features[0].zero_()
        return text_features

    def forward(self, active_indices, evidences):
        factor_texts = self.factor_classifiers(active_indices)
        gates = self.factor_gates(active_indices, evidences)
        texts = factor_texts.permute(1, 0, 2)
        pair_texts = torch.einsum("ncf,cfd->ncd", gates.float(), texts.float())
        if active_indices.numel() > 0 and active_indices[0].item() == 0:
            pair_texts = pair_texts.clone()
            pair_texts[:, 0].zero_()
        return F.normalize(pair_texts, dim=-1), gates

    def factor_classifiers(self, active_indices):
        factor_texts = []
        for factor in self.factors:
            reconstructed = F.normalize(self.reconstruct_factor(factor)[active_indices], dim=-1)
            factor_texts.append(reconstructed)
        factor_texts = torch.stack(factor_texts, dim=0)
        if active_indices.numel() > 0 and active_indices[0].item() == 0:
            factor_texts = factor_texts.clone()
            factor_texts[:, 0].zero_()
        return factor_texts

    def factor_gates(self, active_indices, evidences):
        factor_scores = []
        for factor in self.factors:
            anchors = self.original_text_features(factor)[active_indices]
            keys = self.key_proj[factor](anchors.float())
            keys = keys + self.factor_tokens[self.factor_to_idx[factor]].view(1, -1)
            keys = F.normalize(keys, dim=-1)

            queries = F.normalize(self.query_proj[factor](evidences[factor].float()), dim=-1)
            scores = queries @ keys.t()
            factor_scores.append(scores)

        scores = torch.stack(factor_scores, dim=-1) / self.temperature
        return F.softmax(scores, dim=-1)

    def losses(self, last_gates=None):
        losses = {}
        recon_losses = []
        sparse_losses = []
        decorr_losses = []

        for factor in self.factors:
            reconstructed = self.reconstruct_factor(factor)
            original = self.original_text_features(factor)
            if reconstructed.size(0) > 1:
                recon = 1 - F.cosine_similarity(
                    F.normalize(reconstructed[1:].float(), dim=-1),
                    F.normalize(original[1:].float(), dim=-1),
                    dim=-1,
                )
                recon_losses.append(recon.mean())

            sparse_losses.append(self.class_weights[factor].abs().mean())

            basis = F.normalize(self.basis(factor).float(), dim=-1)
            if basis.size(0) > 1:
                corr = basis @ basis.t()
                off_diag = corr - torch.eye(corr.size(0), device=corr.device).type_as(corr)
                decorr_losses.append(off_diag.abs().mean())

        if recon_losses and self.recon_loss_weight > 0:
            losses["loss_factor_recon"] = torch.stack(recon_losses).mean() * self.recon_loss_weight
        if sparse_losses and self.sparsity_weight > 0:
            losses["loss_factor_sparse"] = torch.stack(sparse_losses).mean() * self.sparsity_weight
        if decorr_losses and self.basis_decorr_weight > 0:
            losses["loss_factor_basis_decorr"] = torch.stack(decorr_losses).mean() * self.basis_decorr_weight
        if last_gates is not None and self.gate_entropy_weight > 0:
            entropy = -(last_gates * (last_gates + 1e-12).log()).sum(-1).mean()
            losses["loss_factor_gate_entropy"] = entropy * self.gate_entropy_weight
        return losses


class RelationRegionPrompt(nn.Module):
    def __init__(
        self,
        union_dim,
        feature_dim=512,
        geometry_dim=32,
        grid_size=7,
        num_prompt_tokens=1,
        num_heads=8,
        union_residual_init=0.0,
    ):
        super(RelationRegionPrompt, self).__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.num_prompt_tokens = num_prompt_tokens
        self.region_prompt = nn.Parameter(torch.zeros(num_prompt_tokens, feature_dim))
        nn.init.normal_(self.region_prompt, std=0.02)

        self.mask_proj = nn.Linear(3 * grid_size * grid_size, feature_dim)
        self.geometry_proj = nn.Linear(geometry_dim, feature_dim)
        self.union_proj = nn.Linear(union_dim, feature_dim)
        self.union_residual_scale = nn.Parameter(torch.tensor(float(union_residual_init)))
        self.attn = nn.MultiheadAttention(feature_dim, num_heads)
        self.out_norm = nn.LayerNorm(feature_dim)

    def _boxes_to_mask(self, boxes, image_size):
        width, height = image_size
        device = boxes.device
        masks = boxes.new_zeros((boxes.size(0), self.grid_size, self.grid_size))
        scale_x = float(self.grid_size) / max(float(width), 1.0)
        scale_y = float(self.grid_size) / max(float(height), 1.0)

        for idx, box in enumerate(boxes):
            x1 = int(torch.floor(box[0] * scale_x).clamp(0, self.grid_size - 1).item())
            y1 = int(torch.floor(box[1] * scale_y).clamp(0, self.grid_size - 1).item())
            x2 = int(torch.ceil(box[2] * scale_x).clamp(0, self.grid_size).item())
            y2 = int(torch.ceil(box[3] * scale_y).clamp(0, self.grid_size).item())
            masks[idx, y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)] = 1.0
        return masks.to(device=device)

    def build_region_masks(self, subject_boxes, object_boxes, image_size):
        union_boxes = torch.cat(
            (
                torch.min(subject_boxes[:, :2], object_boxes[:, :2]),
                torch.max(subject_boxes[:, 2:], object_boxes[:, 2:]),
            ),
            dim=-1,
        )
        masks = torch.stack(
            (
                self._boxes_to_mask(union_boxes, image_size),
                self._boxes_to_mask(subject_boxes, image_size),
                self._boxes_to_mask(object_boxes, image_size),
            ),
            dim=1,
        )
        return masks.flatten(1)

    def forward(self, full_image_tokens, union_features, geometry_features, region_masks):
        if full_image_tokens.dim() == 2:
            full_image_tokens = full_image_tokens.unsqueeze(0)
        patch_tokens = full_image_tokens[:, 1:, :].float()
        if patch_tokens.size(1) == 0:
            patch_tokens = full_image_tokens.float()

        num_rel = union_features.size(0)
        memory = patch_tokens.expand(num_rel, -1, -1).transpose(0, 1).contiguous()

        prompt = self.region_prompt.unsqueeze(0).expand(num_rel, -1, -1)
        prompt = prompt + self.mask_proj(region_masks.float()).unsqueeze(1)
        prompt = prompt + self.geometry_proj(geometry_features.float()).unsqueeze(1)

        attended, _ = self.attn(prompt.transpose(0, 1), memory, memory)
        region_evidence = attended.transpose(0, 1).mean(1)
        union_evidence = self.union_proj(union_features.float())
        return self.out_norm(region_evidence + self.union_residual_scale * union_evidence)
