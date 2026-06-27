import json
import os

import torch

from maskrcnn_benchmark.utils.comm import is_main_process


class MTMDebugger(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir or "."
        self.output_path = os.path.join(self.output_dir, "mtm_debug_summary.json")
        self.triplet_similarity_path = os.path.join(self.output_dir, "mtm_triplet_similarity.jsonl")
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

    def record_triplet_similarity(
        self,
        step,
        triplet_records,
        canonical_embeddings,
        variant_embeddings,
        variant_counts,
        selected_indices,
    ):
        if not is_main_process():
            return
        if canonical_embeddings.size(0) < 2 or len(selected_indices) == 0:
            return

        with torch.no_grad():
            canonical_similarity = torch.matmul(
                canonical_embeddings.detach().float(),
                canonical_embeddings.detach().float().t(),
            )
            selected_indices = [int(index) for index in selected_indices]
            selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=canonical_similarity.device)
            selected_similarity = canonical_similarity.index_select(0, selected_tensor).index_select(1, selected_tensor)
            selected_triplets = [
                {
                    "index": index,
                    "prompt": triplet_records[index]["prompt"],
                    "subject": triplet_records[index]["subject"],
                    "relation": triplet_records[index]["relation"],
                    "object": triplet_records[index]["object"],
                }
                for index in selected_indices
            ]
            entries = []
            variant_offset = 0
            for selected_index in selected_indices:
                row = canonical_similarity[selected_index]
                other_mask = torch.ones(row.size(0), dtype=torch.bool, device=row.device)
                other_mask[selected_index] = False
                other_scores = row[other_mask]
                other_indices = torch.nonzero(other_mask, as_tuple=False).view(-1)

                max_pos = other_scores.argmax().item()
                min_pos = other_scores.argmin().item()
                most_index = int(other_indices[max_pos].item())
                least_index = int(other_indices[min_pos].item())

                variant_count = int(variant_counts[selected_index])
                variant_slice = variant_embeddings[variant_offset: variant_offset + variant_count]
                variant_offset += variant_count
                canonical_vector = canonical_embeddings[selected_index:selected_index + 1].detach().float()
                variant_scores = torch.matmul(variant_slice.detach().float(), canonical_vector.t()).view(-1)
                variant_pair_similarity = torch.matmul(
                    variant_slice.detach().float(),
                    variant_slice.detach().float().t(),
                )
                if variant_count > 1:
                    variant_off_diagonal = ~torch.eye(
                        variant_count,
                        dtype=torch.bool,
                        device=variant_pair_similarity.device,
                    )
                    variant_pair_scores = variant_pair_similarity[variant_off_diagonal]
                    variant_pair_mean = variant_pair_scores.mean().item()
                    variant_pair_std = self._std(variant_pair_scores)
                else:
                    variant_pair_mean = 1.0
                    variant_pair_std = 0.0

                record = triplet_records[selected_index]
                entries.append({
                    "index": selected_index,
                    "subject": record["subject"],
                    "relation": record["relation"],
                    "object": record["object"],
                    "prompt": record["prompt"],
                    "other_prompt_similarity_mean": other_scores.mean().item(),
                    "other_prompt_similarity_std": self._std(other_scores),
                    "most_similar": {
                        "index": most_index,
                        "prompt": triplet_records[most_index]["prompt"],
                        "similarity": other_scores[max_pos].item(),
                    },
                    "least_similar": {
                        "index": least_index,
                        "prompt": triplet_records[least_index]["prompt"],
                        "similarity": other_scores[min_pos].item(),
                    },
                    "prompt_variants": [
                        {
                            "prompt": prompt,
                            "similarity_to_canonical": variant_scores[i].item(),
                        }
                        for i, prompt in enumerate(record["variants"])
                    ],
                    "variant_pair_similarity_mean": variant_pair_mean,
                    "variant_pair_similarity_std": variant_pair_std,
                })

            self._ensure_output_dir()
            with open(self.triplet_similarity_path, "a") as handle:
                handle.write(json.dumps({
                    "step": int(step),
                    "num_triplets": len(triplet_records),
                    "selected_triplets": selected_triplets,
                    "selected_triplet_similarity_matrix": selected_similarity.detach().cpu().tolist(),
                    "entries": entries,
                }, sort_keys=True) + "\n")

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
        self._ensure_output_dir()
        with open(self.output_path, "w") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)

    def _ensure_output_dir(self):
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
