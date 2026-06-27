import json
import os

import torch

from maskrcnn_benchmark.utils.comm import is_main_process


class MTMDebugger(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir or "."
        self.output_path = os.path.join(self.output_dir, "mtm_debug_summary.json")
        self.stats = {}

    def _update(self, group, name, value, count=1):
        if value is None:
            return
        if group not in self.stats:
            self.stats[group] = {}
        if name not in self.stats[group]:
            self.stats[group][name] = {"sum": 0.0, "count": 0}
        self.stats[group][name]["sum"] += float(value) * int(count)
        self.stats[group][name]["count"] += int(count)

    def _std(self, tensor):
        if tensor.numel() <= 1:
            return 0.0
        values = tensor.detach().float().view(-1)
        return values.std(unbiased=False).item()

    def record_alignment(self, predicted_norm, target_norm, positive_mask=None):
        if predicted_norm.size(0) == 0:
            return
        with torch.no_grad():
            similarity = torch.matmul(predicted_norm.detach().float(), target_norm.detach().float().t())
            if positive_mask is None:
                positive_mask = torch.eye(
                    similarity.size(0),
                    dtype=torch.bool,
                    device=similarity.device,
                )
            else:
                positive_mask = positive_mask.to(device=similarity.device, dtype=torch.bool)
            pos_cos = similarity.diag()
            self._update("alignment", "pos_cos", pos_cos.mean().item(), pos_cos.numel())

            if similarity.size(0) > 1:
                negative_mask = ~positive_mask
                if negative_mask.any():
                    neg_cos = similarity[negative_mask]
                    neg_mean = neg_cos.mean().item()
                    self._update("alignment", "neg_cos", neg_mean, neg_cos.numel())
                    self._update("alignment", "margin", pos_cos.mean().item() - neg_mean, pos_cos.numel())

            top1 = similarity.argmax(dim=1)
            top1_hit = positive_mask.gather(1, top1.view(-1, 1)).squeeze(1).float()
            self._update("alignment", "top1_acc", top1_hit.mean().item(), top1_hit.numel())
            topk = min(5, similarity.size(1))
            top5 = similarity.topk(topk, dim=1)[1]
            top5_hit = positive_mask.gather(1, top5).any(dim=1).float()
            self._update("alignment", "top5_acc", top5_hit.mean().item(), top5_hit.numel())
        self.write()

    def record_structure(self, adapted_visual_norm, predicted_norm, target_norm):
        if predicted_norm.size(0) < 2:
            return
        with torch.no_grad():
            predicted_similarity = torch.matmul(predicted_norm.detach().float(), predicted_norm.detach().float().t())
            adapted_visual_similarity = torch.matmul(
                adapted_visual_norm.detach().float(),
                adapted_visual_norm.detach().float().t(),
            )
            target_similarity = torch.matmul(target_norm.detach().float(), target_norm.detach().float().t())
            off_diagonal = ~torch.eye(
                predicted_similarity.size(0),
                dtype=torch.bool,
                device=predicted_similarity.device,
            )
            visual_error = (adapted_visual_similarity - predicted_similarity).abs()[off_diagonal]
            text_error = (predicted_similarity - target_similarity).abs()[off_diagonal]
            self._update("structure", "visual_structure_error", visual_error.mean().item(), visual_error.numel())
            self._update("structure", "text_structure_error", text_error.mean().item(), text_error.numel())
        self.write()

    def record_fusion(self, base_scores, mtm_scores, calibrated_scores, fused_scores):
        if mtm_scores is None or mtm_scores.numel() == 0:
            return
        with torch.no_grad():
            mtm_values = mtm_scores.detach().float().view(-1)
            calibrated_values = calibrated_scores.detach().float().view(-1)
            self._update("fusion", "mtm_score_mean", mtm_values.mean().item(), mtm_values.numel())
            self._update("fusion", "mtm_score_std", self._std(mtm_values), 1)
            self._update(
                "fusion",
                "mtm_score_calibrated_mean",
                calibrated_values.mean().item(),
                calibrated_values.numel(),
            )
            self._update("fusion", "mtm_score_calibrated_std", self._std(calibrated_values), 1)
            if base_scores is not None and fused_scores is not None and base_scores.numel() > 0:
                base_top1 = base_scores.detach().float().argmax(dim=1)
                fused_top1 = fused_scores.detach().float().argmax(dim=1)
                changed = (base_top1 != fused_top1).float()
                self._update("fusion", "top1_changed", changed.mean().item(), changed.numel())
        self.write()

    def summary(self):
        output = {}
        for group, values in self.stats.items():
            output[group] = {}
            for name, meter in values.items():
                if meter["count"] == 0:
                    output[group][name] = 0.0
                else:
                    output[group][name] = meter["sum"] / meter["count"]
        return output

    def write(self):
        if not is_main_process():
            return
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.output_path, "w") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)
