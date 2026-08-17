import torch
from torch.nn import functional as F


LOSS_KEYS = (
    "loss_mtm_align",
    "loss_mtm_visual_structure",
    "loss_mtm_text_structure",
    "loss_mtm_ship_recon",
    "loss_mtm_ship_kl",
)


def linear_ramp(step, warmup_steps, ramp_steps):
    return min(
        max((step - warmup_steps) / float(ramp_steps), 0.0),
        1.0,
    )


def structure_losses(visual_features, projected_features, text_features):
    if projected_features.size(0) < 2:
        zero = projected_features.sum() * 0.0
        return zero, zero
    mask = ~torch.eye(
        projected_features.size(0),
        dtype=torch.bool,
        device=projected_features.device,
    )
    visual_similarity = visual_features @ visual_features.t()
    projected_similarity = projected_features @ projected_features.t()
    text_similarity = text_features @ text_features.t()
    return (
        (visual_similarity - projected_similarity).abs()[mask].mean(),
        (projected_similarity - text_similarity).abs()[mask].mean(),
    )


class MTMLossComputer:
    def __init__(self, config, teacher):
        self.config = config
        self.teacher = teacher

    def empty(self, reference):
        zero = reference.sum() * 0.0
        return {key: zero for key in LOSS_KEYS}

    def alignment(self, projected, targets, subjects, objects, predicates):
        positive = 1.0 - (projected * targets).sum(dim=-1).mean()
        candidates, indices = self.teacher.alignment_candidates(
            subjects, objects, predicates, projected.device
        )
        similarities = torch.bmm(
            candidates, projected.unsqueeze(-1)
        ).squeeze(-1)
        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask.scatter_(1, indices.unsqueeze(1), False)
        negative = F.relu(
            similarities[mask] - float(self.config.NEGATIVE_MARGIN)
        ).mean()
        total = positive + float(self.config.NEGATIVE_ALIGNMENT_WEIGHT) * negative
        return total, positive, negative

    def weighted(
        self,
        alignment,
        visual_structure,
        text_structure,
        structure_ramp,
        reconstruction,
        kl,
        ship_ramp,
        ship_config,
    ):
        return {
            "loss_mtm_align": float(self.config.ALIGNMENT_WEIGHT) * alignment,
            "loss_mtm_visual_structure": float(
                self.config.VISUAL_STRUCTURE_WEIGHT
            )
            * structure_ramp
            * visual_structure,
            "loss_mtm_text_structure": float(self.config.TEXT_STRUCTURE_WEIGHT)
            * structure_ramp
            * text_structure,
            "loss_mtm_ship_recon": float(ship_config.RECONSTRUCTION_WEIGHT)
            * reconstruction,
            "loss_mtm_ship_kl": float(ship_config.KL_WEIGHT) * ship_ramp * kl,
        }
