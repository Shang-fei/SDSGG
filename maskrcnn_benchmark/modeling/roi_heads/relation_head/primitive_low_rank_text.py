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
        cosine_recon_loss_weight=0.0,
        sparsity_weight=0.01,
        mask_out_weight=0.0,
        basis_decorr_weight=0.001,
        weight_decorr_weight=0.001,
        basis_anchor_weight=0.01,
        distribution_enabled=False,
        distribution_context_dim=512,
        distribution_hidden_dim=128,
        distribution_rank=4,
        distribution_shift_scale=0.1,
        distribution_noise_scale=0.02,
        distribution_shift_loss_weight=0.001,
        distribution_var_loss_weight=0.001,
        distribution_sample=True,
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

        self.register_buffer("class_weight_anchor", init_weight.detach().clone())
        self.register_buffer("basis_anchor", primitive_features.detach().clone())
        self.register_buffer("predicate_features", predicate_features.detach().clone())
        self.register_buffer("text_mean", text_mean.detach().clone())
        if predicate_primitive_mask is not None:
            self.register_buffer("predicate_primitive_mask", predicate_primitive_mask.float())

        self.logit_temperature = float(logit_temperature)
        self.recon_loss_weight = float(recon_loss_weight)
        self.cosine_recon_loss_weight = float(cosine_recon_loss_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.mask_out_weight = float(mask_out_weight)
        self.basis_decorr_weight = float(basis_decorr_weight)
        self.weight_decorr_weight = float(weight_decorr_weight)
        self.basis_anchor_weight = float(basis_anchor_weight)
        self.distribution_enabled = bool(distribution_enabled)
        self.distribution_shift_scale = float(distribution_shift_scale)
        self.distribution_noise_scale = float(distribution_noise_scale)
        self.distribution_shift_loss_weight = float(distribution_shift_loss_weight)
        self.distribution_var_loss_weight = float(distribution_var_loss_weight)
        self.distribution_sample = bool(distribution_sample)
        self._distribution_shift_terms = []
        self._distribution_var_terms = []
        self._last_distribution_stats = {}

        if self.distribution_enabled:
            num_predicates, num_primitives = init_weight.shape
            self.dist_context = nn.Sequential(
                nn.Linear(distribution_context_dim, distribution_hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.dist_shift_proj = nn.Linear(distribution_hidden_dim, distribution_rank)
            self.dist_sigma_proj = nn.Linear(distribution_hidden_dim, distribution_rank)
            self.dist_rel_shift = nn.Parameter(
                torch.empty(num_predicates, distribution_rank, num_primitives)
            )
            self.dist_rel_sigma = nn.Parameter(
                torch.zeros(num_predicates, distribution_rank, num_primitives)
            )
            self.dist_log_sigma = nn.Parameter(torch.full((num_predicates, num_primitives), -5.0))
            nn.init.normal_(self.dist_rel_shift, mean=0.0, std=0.001)

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

    def classifier_basis(self, anchor_blend=0.0):
        basis = self.basis_feat.float()
        anchor_blend = float(anchor_blend)
        if anchor_blend > 0:
            anchor_blend = min(max(anchor_blend, 0.0), 1.0)
            basis = basis * (1.0 - anchor_blend) + self.basis_anchor.float() * anchor_blend
        return F.normalize(basis, dim=-1)

    def effective_class_weights(self, anchor_blend=0.0):
        weights = self.class_weights.float()
        anchor_blend = float(anchor_blend)
        if anchor_blend > 0:
            anchor_blend = min(max(anchor_blend, 0.0), 1.0)
            weights = weights * (1.0 - anchor_blend) + self.class_weight_anchor.float() * anchor_blend
        return weights

    def reconstruct(self):
        return self.text_mean.unsqueeze(0) + self.effective_class_weights() @ self.classifier_basis()

    def active_weights(self, fg_ids, anchor_blend=0.0):
        return self.effective_class_weights(anchor_blend=anchor_blend)[
            fg_ids.to(self.class_weights.device).long()
        ].float()

    def reset_distribution_losses(self):
        self._distribution_shift_terms = []
        self._distribution_var_terms = []

    def _distribution_logits(self, primitive_logits, fg_ids, condition_features, base_weights):
        condition_features = F.normalize(condition_features.float(), dim=-1)
        hidden = self.dist_context(condition_features)
        shift_context = torch.tanh(self.dist_shift_proj(hidden))
        sigma_context = torch.tanh(self.dist_sigma_proj(hidden))
        active_ids = fg_ids.to(self.class_weights.device).long()

        rel_shift = self.dist_rel_shift[active_ids].float()
        rel_sigma = self.dist_rel_sigma[active_ids].float()
        delta = torch.einsum("nr,crk->nck", shift_context, rel_shift)
        delta = torch.tanh(delta) * self.distribution_shift_scale

        log_sigma = self.dist_log_sigma[active_ids].float().unsqueeze(0)
        log_sigma = log_sigma + torch.einsum("nr,crk->nck", sigma_context, rel_sigma)
        sigma = F.softplus(log_sigma.clamp(-8.0, 2.0)) * self.distribution_noise_scale

        weights = base_weights.unsqueeze(0) + delta
        if self.training and self.distribution_sample and self.distribution_noise_scale > 0:
            weights = weights + torch.randn_like(weights) * sigma

        logits = torch.einsum("nk,nck->nc", primitive_logits, weights)
        self._record_distribution_terms(delta, sigma)
        return logits

    def _record_distribution_terms(self, delta, sigma):
        if self.training:
            self._distribution_shift_terms.append(delta.pow(2).mean())
            self._distribution_var_terms.append(sigma.mean())
        with torch.no_grad():
            self._last_distribution_stats = {
                "dist_shift_abs_mean": delta.detach().abs().mean().item(),
                "dist_shift_abs_max": delta.detach().abs().max().item(),
                "dist_sigma_mean": sigma.detach().mean().item(),
                "dist_sigma_max": sigma.detach().max().item(),
            }

    def logits(
        self,
        visual_features,
        fg_ids,
        weight_anchor_blend=0.0,
        basis_anchor_blend=0.0,
        condition_features=None,
    ):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        primitive_logits = visual_features @ self.classifier_basis(
            anchor_blend=basis_anchor_blend
        ).t()
        weights = self.active_weights(fg_ids, anchor_blend=weight_anchor_blend)
        if self.distribution_enabled and condition_features is not None:
            logits = self._distribution_logits(primitive_logits, fg_ids, condition_features, weights)
        else:
            logits = primitive_logits @ weights.t()
        return logits / max(self.logit_temperature, 1e-6), primitive_logits

    def debug_stats(self):
        with torch.no_grad():
            weights = self.effective_class_weights()
            basis = self.classifier_basis()
            reconstructed = F.normalize(self.reconstruct(), dim=-1)
            recon_cos = (reconstructed * self.predicate_features.float()).sum(dim=-1)
            mask = getattr(self, "predicate_primitive_mask", None)
            if mask is not None:
                mask = mask.to(device=weights.device, dtype=weights.dtype)
                mask_out_abs = (weights * (1.0 - mask)).abs().mean()
                mask_in_count = mask.sum().clamp_min(1.0)
                mask_in_abs = (weights * mask).abs().sum() / mask_in_count
            else:
                mask_out_abs = weights.new_zeros(())
                mask_in_abs = weights.abs().mean()
            basis_shift = (self.basis_feat.float() - self.basis_anchor.float()).norm(dim=-1)
            basis_anchor_norm = self.basis_anchor.float().norm(dim=-1).clamp_min(1e-6)
            basis_rel_shift = basis_shift / basis_anchor_norm
            basis_corr = basis @ basis.t()
            basis_mask = torch.ones_like(basis_corr) - torch.eye(
                basis_corr.size(0),
                device=basis_corr.device,
            ).type_as(basis_corr)

            stats = {
                "w_abs_mean": weights.abs().mean().item(),
                "w_abs_max": weights.abs().max().item(),
                "w_nonzero_005": (weights.abs() > 0.05).float().mean().item(),
                "w_mask_in_abs": mask_in_abs.item(),
                "w_mask_out_abs": mask_out_abs.item(),
                "basis_norm_mean": self.basis_feat.float().norm(dim=-1).mean().item(),
                "basis_rel_shift_mean": basis_rel_shift.mean().item(),
                "basis_rel_shift_max": basis_rel_shift.max().item(),
                "basis_corr_offdiag": (basis_corr * basis_mask).abs().mean().item(),
                "recon_cos_mean": recon_cos.mean().item(),
                "recon_cos_min": recon_cos.min().item(),
                "recon_cos_neg_frac": (recon_cos < 0).float().mean().item(),
            }
            stats.update({
                "dist_shift_abs_mean": 0.0,
                "dist_shift_abs_max": 0.0,
                "dist_sigma_mean": 0.0,
                "dist_sigma_max": 0.0,
            })
            stats.update(self._last_distribution_stats)
            return stats

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
        if self.cosine_recon_loss_weight > 0:
            reconstructed = F.normalize(self.reconstruct(), dim=-1)
            target = self.predicate_features.float()
            recon_cos = (reconstructed * target).sum(dim=-1)
            losses["loss_primitive_recon_cos"] = (
                (1.0 - recon_cos).mean() * self.cosine_recon_loss_weight
            )
        if self.sparsity_weight > 0:
            losses["loss_primitive_sparse"] = weights.abs().mean() * self.sparsity_weight
        if self.mask_out_weight > 0 and hasattr(self, "predicate_primitive_mask"):
            mask = self.predicate_primitive_mask.to(device=weights.device, dtype=weights.dtype)
            losses["loss_primitive_mask_out"] = (
                (weights * (1.0 - mask)).abs().mean() * self.mask_out_weight
            )
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
        if self.distribution_enabled and self._distribution_shift_terms:
            if self.distribution_shift_loss_weight > 0:
                losses["loss_primitive_dist_shift"] = (
                    torch.stack(self._distribution_shift_terms).mean()
                    * self.distribution_shift_loss_weight
                )
            if self.distribution_var_loss_weight > 0:
                losses["loss_primitive_dist_var"] = (
                    torch.stack(self._distribution_var_terms).mean()
                    * self.distribution_var_loss_weight
                )
        return losses
