import json

import torch
from torch import nn
from torch.nn import functional as F


def load_primitive_spec(json_path, predicate_names):
    with open(json_path, "r") as f:
        spec = json.load(f)

    primitive_bank = spec.get("primitive_bank", [])
    predicates = spec.get("predicates", {})
    if not primitive_bank:
        raise ValueError("primitive_bank is empty in {}".format(json_path))
    if len(primitive_bank) != len(set(primitive_bank)):
        raise ValueError("primitive_bank contains duplicates in {}".format(json_path))

    missing = [name for name in predicate_names[1:] if name not in predicates]
    if missing:
        raise KeyError("Missing predicate primitive entries: {}".format(missing))

    descriptions = []
    predicate_primitives = []
    primitive_set = set(primitive_bank)
    primitive_to_idx = {name: idx for idx, name in enumerate(primitive_bank)}
    primitive_mask = torch.zeros(len(predicate_names) - 1, len(primitive_bank), dtype=torch.float32)
    for pred_idx, name in enumerate(predicate_names[1:]):
        item = predicates[name]
        description = item.get("description", "")
        primitives = item.get("primitives", [])
        if not description:
            raise ValueError("Missing description for predicate '{}'".format(name))
        if not primitives:
            raise ValueError("Missing primitives for predicate '{}'".format(name))
        unknown = [p for p in primitives if p not in primitive_set]
        if unknown:
            raise ValueError("Unknown primitives for '{}': {}".format(name, unknown))
        descriptions.append(description)
        predicate_primitives.append(primitives)
        primitive_mask[pred_idx, [primitive_to_idx[p] for p in primitives]] = 1.0

    return primitive_bank, descriptions, predicate_primitives, primitive_mask


class PrimitiveLowRankTextAdapter(nn.Module):
    def __init__(
        self,
        primitive_features,
        predicate_features,
        predicate_primitive_mask=None,
        train_basis=True,
        train_weight=True,
        logit_temperature=0.07,
        recon_loss_weight=0.1,
        sparsity_weight=0.01,
        basis_decorr_weight=0.001,
        weight_decorr_weight=0.001,
        basis_anchor_weight=0.01,
    ):
        super(PrimitiveLowRankTextAdapter, self).__init__()
        primitive_features = F.normalize(primitive_features.float(), dim=-1)
        predicate_features = F.normalize(predicate_features.float(), dim=-1)

        text_mean = predicate_features.mean(0)
        centered_predicates = predicate_features - text_mean.unsqueeze(0)
        init_weight = self._initialize_weights(
            centered_predicates,
            primitive_features,
            predicate_primitive_mask,
        )

        if train_basis:
            self.basis_feat = nn.Parameter(primitive_features)
        else:
            self.register_buffer("basis_feat", primitive_features)
        if train_weight:
            self.class_weights = nn.Parameter(init_weight)
        else:
            self.register_buffer("class_weights", init_weight)

        self.register_buffer("basis_anchor", primitive_features.detach().clone())
        self.register_buffer("predicate_features", predicate_features.detach().clone())
        self.register_buffer("text_mean", text_mean.detach().clone())
        if predicate_primitive_mask is not None:
            self.register_buffer("predicate_primitive_mask", predicate_primitive_mask.float())

        self.logit_temperature = float(logit_temperature)
        self.recon_loss_weight = float(recon_loss_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.basis_decorr_weight = float(basis_decorr_weight)
        self.weight_decorr_weight = float(weight_decorr_weight)
        self.basis_anchor_weight = float(basis_anchor_weight)

    def _initialize_weights(self, centered_predicates, primitive_features, predicate_primitive_mask):
        if predicate_primitive_mask is None:
            return centered_predicates @ torch.pinverse(primitive_features)

        mask = predicate_primitive_mask.to(
            device=primitive_features.device,
            dtype=primitive_features.dtype,
        )
        if mask.shape != (centered_predicates.size(0), primitive_features.size(0)):
            raise ValueError(
                "predicate_primitive_mask shape {} does not match ({}, {})".format(
                    tuple(mask.shape),
                    centered_predicates.size(0),
                    primitive_features.size(0),
                )
            )

        init_weight = primitive_features.new_zeros(mask.shape)
        for pred_idx in range(mask.size(0)):
            primitive_idx = torch.nonzero(mask[pred_idx], as_tuple=False).flatten()
            selected_basis = primitive_features[primitive_idx]
            selected_weight = centered_predicates[pred_idx:pred_idx + 1] @ torch.pinverse(selected_basis)
            init_weight[pred_idx, primitive_idx] = selected_weight.squeeze(0)
        return init_weight

    def classifier_basis(self):
        return F.normalize(self.basis_feat.float(), dim=-1)

    def effective_class_weights(self):
        return self.class_weights.float()

    def reconstruct(self):
        return self.text_mean.unsqueeze(0) + self.effective_class_weights() @ self.classifier_basis()

    def active_weights(self, fg_ids):
        return self.effective_class_weights()[fg_ids.to(self.class_weights.device).long()].float()

    def logits(self, visual_features, fg_ids):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        primitive_logits = visual_features @ self.classifier_basis().t()
        weights = self.active_weights(fg_ids)
        logits = primitive_logits @ weights.t()
        return logits / max(self.logit_temperature, 1e-6), primitive_logits

    def debug_stats(self):
        with torch.no_grad():
            weights = self.effective_class_weights()
            basis = self.classifier_basis()
            reconstructed = F.normalize(self.reconstruct(), dim=-1)
            recon_cos = (reconstructed * self.predicate_features.float()).sum(dim=-1)
            basis_shift = (self.basis_feat.float() - self.basis_anchor.float()).norm(dim=-1)
            basis_anchor_norm = self.basis_anchor.float().norm(dim=-1).clamp_min(1e-6)
            basis_rel_shift = basis_shift / basis_anchor_norm
            basis_corr = basis @ basis.t()
            basis_mask = torch.ones_like(basis_corr) - torch.eye(
                basis_corr.size(0),
                device=basis_corr.device,
            ).type_as(basis_corr)

            return {
                "w_abs_mean": weights.abs().mean().item(),
                "w_abs_max": weights.abs().max().item(),
                "w_nonzero_005": (weights.abs() > 0.05).float().mean().item(),
                "basis_norm_mean": self.basis_feat.float().norm(dim=-1).mean().item(),
                "basis_rel_shift_mean": basis_rel_shift.mean().item(),
                "basis_rel_shift_max": basis_rel_shift.max().item(),
                "basis_corr_offdiag": (basis_corr * basis_mask).abs().mean().item(),
                "recon_cos_mean": recon_cos.mean().item(),
                "recon_cos_min": recon_cos.min().item(),
            }

    def losses(self):
        losses = {}
        weights = self.effective_class_weights()
        basis = self.classifier_basis()

        if self.recon_loss_weight > 0:
            reconstructed = F.normalize(self.reconstruct(), dim=-1)
            losses["loss_primitive_recon"] = (
                F.mse_loss(reconstructed, self.predicate_features.float(), reduction="mean")
                * self.recon_loss_weight
            )
        if self.sparsity_weight > 0:
            losses["loss_primitive_sparse"] = weights.abs().mean() * self.sparsity_weight
        if self.basis_decorr_weight > 0 and basis.size(0) > 1:
            corr = basis @ basis.t()
            mask = torch.ones_like(corr) - torch.eye(corr.size(0), device=corr.device).type_as(corr)
            losses["loss_primitive_basis_decorr"] = (corr * mask).abs().mean() * self.basis_decorr_weight
        if self.weight_decorr_weight > 0 and weights.size(1) > 1:
            weight_basis = F.normalize(weights.t(), dim=-1)
            corr = weight_basis @ weight_basis.t()
            mask = torch.ones_like(corr) - torch.eye(corr.size(0), device=corr.device).type_as(corr)
            losses["loss_primitive_weight_decorr"] = (corr * mask).abs().mean() * self.weight_decorr_weight
        if self.basis_anchor_weight > 0:
            losses["loss_primitive_basis_anchor"] = (
                F.mse_loss(self.basis_feat.float(), self.basis_anchor.float(), reduction="mean")
                * self.basis_anchor_weight
            )
        return losses
