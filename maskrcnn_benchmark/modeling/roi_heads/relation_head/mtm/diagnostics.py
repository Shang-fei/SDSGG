import os

import torch
from torch.nn import functional as F

from maskrcnn_benchmark.utils.comm import get_rank


class MTMDebugWriter:
    def __init__(self, config, output_dir):
        self.enabled = config.ENABLED
        self.interval = max(int(config.INTERVAL_STEPS), 1)
        self.path = os.path.join(output_dir, config.FILE)

    @staticmethod
    def _stats(features):
        if features.size(0) < 2:
            return (float("nan"),) * 4
        normalized = F.normalize(features.float(), dim=-1)
        similarity = normalized @ normalized.t()
        mask = ~torch.eye(
            features.size(0), dtype=torch.bool, device=features.device
        )
        values = similarity[mask]
        return (
            values.mean().item(),
            values.max().item(),
            values.min().item(),
            values.std(unbiased=False).item(),
        )

    @classmethod
    def _format_stats(cls, features):
        return "/".join("{:.4f}".format(value) for value in cls._stats(features))

    @staticmethod
    def _format_sampling(names, counts, difficulty=None, topk=5):
        counts = counts.detach().cpu()
        total = int(counts.sum().item())
        coverage = int((counts > 0).sum().item())
        count = min(int(topk), len(names))
        if total > 0 and count > 0:
            values, indices = torch.topk(counts, count)
            top = ",".join(
                "{}:{}".format(names[int(index)], int(value))
                for value, index in zip(values.tolist(), indices.tolist())
                if value > 0
            )
        else:
            top = "-"
        result = "total={}/seen={}/{}/top={}".format(
            total, coverage, len(names), top
        )
        if difficulty is not None and count > 0:
            difficulty = difficulty.detach().cpu()
            values, indices = torch.topk(difficulty, count)
            hard = ",".join(
                "{}:{:.3f}".format(names[int(index)], float(value))
                for value, index in zip(values.tolist(), indices.tolist())
            )
            result += "/hard={}".format(hard)
        return result

    def write(self, step, base, novel=None, sampling=None):
        if not self.enabled or get_rank() != 0 or step % self.interval != 0:
            return
        sampling = sampling or {}
        base_sampling = self._format_sampling(
            sampling.get("base_names", []),
            sampling.get("base_counts", torch.zeros(0)),
            sampling.get("base_difficulty"),
        )
        lines = [
            "i={} base raw={} h={} q={} t={} align(pos/neg)={:.4f}/{:.4f} "
            "err(v/t)={:.4f}/{:.4f} recon={:.4f} kl={:.4f} "
            "norm(raw/h)={:.3f}/{:.3f} sample_global({})".format(
                step,
                self._format_stats(base["raw"]),
                self._format_stats(base["h"]),
                self._format_stats(base["q"]),
                self._format_stats(base["t"]),
                base["align_pos"],
                base["align_neg"],
                base["visual_error"],
                base["text_error"],
                base["recon"],
                base["kl"],
                base["raw"].norm(dim=-1).mean().item(),
                base["h"].norm(dim=-1).mean().item(),
                base_sampling,
            )
        ]
        novel_sampling = self._format_sampling(
            sampling.get("novel_names", []),
            sampling.get("novel_counts", torch.zeros(0)),
        )
        if novel is not None and novel["raw"].size(0) > 0:
            nearest = (
                F.normalize(novel["raw"], dim=-1)
                @ F.normalize(base["raw"], dim=-1).t()
            ).max(dim=1)[0].mean().item()
            lines.append(
                "i={} novel raw={} h={} q={} t={} align={:.4f} "
                "nearest_base={:.4f} err(v/t)={:.4f}/{:.4f} ramp={:.3f} "
                "n={} norm(raw/h)={:.3f}/{:.3f} sample_global({})".format(
                    step,
                    self._format_stats(novel["raw"]),
                    self._format_stats(novel["h"]),
                    self._format_stats(novel["q"]),
                    self._format_stats(novel["t"]),
                    novel["align"],
                    nearest,
                    novel["visual_error"],
                    novel["text_error"],
                    novel["ramp"],
                    novel["raw"].size(0),
                    novel["raw"].norm(dim=-1).mean().item(),
                    novel["h"].norm(dim=-1).mean().item(),
                    novel_sampling,
                )
            )
        else:
            lines.append("i={} novel sample_global({})".format(step, novel_sampling))
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as stream:
            stream.write("\n".join(lines) + "\n")
