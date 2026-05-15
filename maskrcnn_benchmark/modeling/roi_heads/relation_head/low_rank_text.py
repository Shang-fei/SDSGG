import torch
from torch import nn
from torch.nn import functional as F


class PredicateNameTextProvider(object):
    def __init__(self, template):
        self.template = template

    def __call__(self, predicate_names):
        texts = []
        for name in predicate_names:
            if name == "__background__":
                texts.append("")
            else:
                texts.append(self.template.format(name))
        return texts


def build_split_indices(cfg, predicate_names):
    name_to_idx = {name: idx for idx, name in enumerate(predicate_names)}

    def ids_for(names):
        return [0] + [name_to_idx[name] for name in sorted(names) if name in name_to_idx]

    return {
        "base": ids_for(cfg.OV_SETTING.PRDCS_BASE),
        "novel": ids_for(cfg.OV_SETTING.PRDCS_NOVEL),
        "semantic": ids_for(cfg.OV_SETTING.SEMAN),
        "total": list(range(len(predicate_names))),
    }


class LowRankRelationTextAdapter(nn.Module):
    def __init__(
        self,
        text_features,
        rank,
        recon_loss_weight=0.1,
        sparsity_weight=0.1,
        basis_decorr_weight=0.001,
        train_basis=False,
    ):
        super(LowRankRelationTextAdapter, self).__init__()
        text_features = text_features.detach().float()
        text_features = F.normalize(text_features, dim=-1)
        if text_features.size(0) > 0:
            text_features[0].zero_()

        basis, weights = self._init_low_rank(text_features, rank)
        self.class_weights = nn.Parameter(weights)
        if train_basis:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer("basis", basis)
        self.register_buffer("original_text_features", text_features)

        self.recon_loss_weight = recon_loss_weight
        self.sparsity_weight = sparsity_weight
        self.basis_decorr_weight = basis_decorr_weight

    @staticmethod
    def _init_low_rank(text_features, rank):
        centered = text_features - text_features.mean(dim=0, keepdim=True)
        max_rank = min(centered.size(0), centered.size(1))
        rank = max(1, min(int(rank), max_rank))

        _, _, v = torch.svd(centered)
        basis = F.normalize(v[:, :rank].t(), dim=-1)
        weights = text_features @ torch.pinverse(basis)
        return basis, weights

    def reconstruct(self):
        text_features = self.class_weights @ self.basis
        if text_features.size(0) > 0:
            text_features = text_features.clone()
            text_features[0].zero_()
        return text_features

    def forward(self, active_indices=None):
        text_features = F.normalize(self.reconstruct(), dim=-1)
        if active_indices is None:
            return text_features
        return text_features[active_indices]

    def losses(self):
        losses = {}
        reconstructed = self.reconstruct()
        if self.recon_loss_weight > 0:
            losses["loss_lr_recon"] = F.mse_loss(
                reconstructed, self.original_text_features
            ) * self.recon_loss_weight
        if self.sparsity_weight > 0:
            losses["loss_lr_sparse"] = self.class_weights.abs().mean() * self.sparsity_weight
        if self.basis_decorr_weight > 0 and self.basis.size(0) > 1:
            basis = F.normalize(self.basis.float(), dim=-1)
            corr = basis @ basis.t()
            off_diag = corr - torch.eye(corr.size(0), device=corr.device).type_as(corr)
            losses["loss_lr_basis_decorr"] = off_diag.abs().mean() * self.basis_decorr_weight
        return losses
