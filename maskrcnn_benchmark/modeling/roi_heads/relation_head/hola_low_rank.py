import torch
from torch import nn
from torch.nn import functional as F


def init_hola_low_rank_with_pca(text_features, rank):
    from sklearn.decomposition import PCA

    features_np = text_features.detach().cpu().float().numpy()

    if isinstance(rank, float) and rank <= 1:
        n_components = rank
    else:
        n_components = min(int(rank), min(features_np.shape[0], features_np.shape[1]))
        n_components = max(1, n_components)

    pca = PCA(n_components=n_components)

    weights = pca.fit_transform(features_np)
    basis = pca.components_
    mean = pca.mean_            # (d,)

    weights = torch.tensor(weights, device=text_features.device, dtype=text_features.dtype)
    basis = torch.tensor(basis, device=text_features.device, dtype=text_features.dtype)
    mean = torch.tensor(mean, device=text_features.device, dtype=text_features.dtype)
    basis = F.normalize(basis, dim=-1)
    return basis, weights, mean


class HOLaLowRankDecomposer(nn.Module):
    """HOLa-style low-rank text feature decomposition.

    This module is intentionally independent from relation routing or visual evidence:
    it receives text features and returns reconstructed classifier features plus
    HOLa-style regularization losses.
    """

    def __init__(
        self,
        text_features,
        rank=71,
        init_method="pca",
        train_basis=True,
        train_mode="w",
        recon_loss_weight=0.1,
        sparsity_weight=0.1,
        basis_decorr_weight=0.001,
        weight_decorr_weight=0.001,
        weight_delta_scale=1.0,
        weight_anchor_weight=0.0,
    ):
        super(HOLaLowRankDecomposer, self).__init__()
        if train_mode not in ("w", "b", "both", "none"):
            raise ValueError("Unsupported HOLa train_mode: {}".format(train_mode))

        if init_method == "pca":
            basis, weights, text_mean = init_hola_low_rank_with_pca(text_features, rank)
        elif init_method == "random":
            basis = torch.randn(int(rank), text_features.size(-1)).type_as(text_features)
            basis = F.normalize(basis, dim=-1)
            text_mean = text_features.mean(0)
            weights = (text_features - text_mean.unsqueeze(0)).float() @ torch.pinverse(basis.float())
            weights = weights.type_as(text_features)
        else:
            raise ValueError("Unsupported HOLa low-rank init_method: {}".format(init_method))

        self.class_weights = nn.Parameter(weights)
        if train_basis:
            self.basis_feat = nn.Parameter(basis)
        else:
            self.register_buffer("basis_feat", basis)
        self.register_buffer("original_text_features", text_features)
        self.register_buffer("text_mean", text_mean)
        self.register_buffer("initial_class_weights", weights.detach().clone())

        self.train_mode = train_mode
        self.recon_loss_weight = recon_loss_weight
        self.sparsity_weight = sparsity_weight
        self.basis_decorr_weight = basis_decorr_weight
        self.weight_decorr_weight = weight_decorr_weight
        self.weight_delta_scale = float(weight_delta_scale)
        self.weight_anchor_weight = float(weight_anchor_weight)
        self.recon_loss = nn.MSELoss(reduction="sum")

    def effective_class_weights(self):
        weights = self.class_weights
        if self.train_mode in ("b", "none"):
            weights = weights.detach()
        initial = self.initial_class_weights.to(weights.device).type_as(weights)
        return initial + self.weight_delta_scale * (weights - initial)

    def classifier_basis(self):
        basis = self.basis_feat
        if self.train_mode in ("w", "none"):
            basis = basis.detach()
        return basis

    def residual_features(self, weights=None, basis=None):
        if weights is None:
            weights = self.effective_class_weights()
        if basis is None:
            basis = self.classifier_basis()
        return weights @ basis

    def reconstruct(self):
        return self.text_mean.unsqueeze(0) + self.residual_features()

    def classifier_features(self):
        return self.text_mean.unsqueeze(0) + self.residual_features()

    def losses(self):
        losses = {}
        reconstructed = self.reconstruct()
        original = self.original_text_features.to(reconstructed.device)

        if self.recon_loss_weight > 0:
            losses["recon"] = self.recon_loss(reconstructed, original) * self.recon_loss_weight
        effective_weights = self.effective_class_weights()

        if self.sparsity_weight > 0:
            losses["sparse"] = effective_weights.abs().sum(-1).mean(0) * self.sparsity_weight
        if self.weight_decorr_weight > 0 and effective_weights.size(1) > 1:
            weight_basis = F.normalize(effective_weights.t().float(), dim=-1)
            corr = weight_basis @ weight_basis.t()
            mask = torch.ones_like(corr) - torch.eye(corr.size(0), device=corr.device).type_as(corr)
            losses["weight_decorr"] = (corr * mask).abs().sum() * self.weight_decorr_weight
        if self.basis_decorr_weight > 0 and self.basis_feat.size(0) > 1:
            basis = F.normalize(self.basis_feat.float(), dim=-1)
            corr = basis @ basis.t()
            mask = torch.ones_like(corr) - torch.eye(corr.size(0), device=corr.device).type_as(corr)
            disentangle = (corr * mask) @ (corr * mask).t()
            losses["basis_decorr"] = disentangle.abs().sum() * self.basis_decorr_weight
        if self.weight_anchor_weight > 0:
            delta = self.class_weights.float() - self.initial_class_weights.float()
            losses["weight_anchor"] = delta.pow(2).mean() * self.weight_anchor_weight
        return losses

    def debug_stats(self):
        with torch.no_grad():
            reconstructed = self.reconstruct().float()
            original = self.original_text_features.float().to(reconstructed.device)
            effective_weights = self.effective_class_weights().float()
            if reconstructed.size(0) > 0:
                recon_cos = F.cosine_similarity(
                    F.normalize(reconstructed, dim=-1),
                    F.normalize(original, dim=-1),
                    dim=-1,
                ).mean()
            else:
                recon_cos = reconstructed.sum() * 0
            return {
                "W_abs_mean": effective_weights.abs().mean(),
                "W_delta_mean": (self.class_weights.float() - self.initial_class_weights.float()).abs().mean(),
                "B_abs_mean": self.basis_feat.float().abs().mean(),
                "recon_cos": recon_cos,
                "rank": torch.as_tensor(self.basis_feat.size(0), device=reconstructed.device),
            }
