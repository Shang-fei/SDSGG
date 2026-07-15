import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize
from torchvision.transforms import functional as TF

from CLIP import clip
from maskrcnn_benchmark.utils.comm import get_rank


SHIP_INIT_STD = 0.02
LOW_RECALL_NOVEL_RELATIONS = frozenset({
    "across",
    "along",
    "and",
    "laying on",
    "mounted on",
    "on back of",
    "painted on",
    "part of",
})


def _init_ship_layer(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=SHIP_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TripletTextTeacher(nn.Module):
    """Frozen CLIP triplet teacher with one shared global SVD refinement."""

    def __init__(self, clip_model, object_names, relation_names, text_filter, cfg, device):
        super().__init__()
        object.__setattr__(self, "clip_model", clip_model)
        self.object_names = object_names
        self.relation_names = relation_names
        self.text_filter = text_filter
        self.device = device
        self.svd_enabled = cfg.TEXT_SVD_ENABLED
        self.svd_components = int(cfg.TEXT_SVD_COMPONENTS)
        self.chunk_size = max(int(cfg.TEXT_ENCODE_CHUNK_SIZE), 1)
        self.candidate_cache = {}
        self.filtered_cache = {}
        self.register_buffer("principal_components", None)
        self.update_svd_basis()

    @staticmethod
    def format_triplet(subject, predicate, object_):
        return "a photo of a {} {} a {}".format(subject, predicate, object_)

    def encode_raw(self, texts):
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(texts), self.chunk_size):
                tokens = clip.tokenize(texts[offset : offset + self.chunk_size]).to(self.device)
                chunks.append(self.clip_model.encode_text(tokens).float())
        return F.normalize(torch.cat(chunks, dim=0), dim=-1)

    def encode(self, texts):
        features = self.encode_raw(texts)
        if self.principal_components is None:
            return features
        components = self.principal_components.to(features)
        projection = (features @ components) @ components.t()
        return F.normalize(features - projection, dim=-1)

    def _svd_text_chunks(self):
        chunk = []
        names = self.object_names[1:] if len(self.object_names) > 1 else self.object_names
        for subject in names:
            predicates = (
                list(self.text_filter[subject])
                if subject in self.text_filter.columns
                else self.relation_names
            )
            for predicate in predicates:
                if predicate == "__background__":
                    continue
                for object_ in names:
                    chunk.append(self.format_triplet(subject, predicate, object_))
                    if len(chunk) == self.chunk_size:
                        yield chunk
                        chunk = []
        if chunk:
            yield chunk

    def update_svd_basis(self):
        if not self.svd_enabled or self.svd_components <= 0:
            self.principal_components = None
            return
        covariance = None
        count = 0
        with torch.no_grad():
            for texts in self._svd_text_chunks():
                features = self.encode_raw(texts).float()
                if covariance is None:
                    covariance = features.new_zeros(features.size(1), features.size(1))
                covariance.add_(features.t() @ features)
                count += features.size(0)
        if covariance is None or count < 2:
            self.principal_components = None
            return
        _, eigenvectors = torch.linalg.eigh(covariance / float(count))
        count = min(self.svd_components, eigenvectors.size(1))
        self.principal_components = F.normalize(eigenvectors[:, -count:], dim=0).detach()

    def build_targets(self, subject_labels, relation_labels, object_labels, active_relations):
        texts, predicates = [], []
        for subject, predicate, object_ in zip(
            subject_labels.detach().cpu().tolist(),
            relation_labels.detach().cpu().tolist(),
            object_labels.detach().cpu().tolist(),
        ):
            predicate_name = active_relations[int(predicate)]
            predicates.append(predicate_name)
            texts.append(self.format_triplet(
                self.object_names[int(subject)], predicate_name, self.object_names[int(object_)]
            ))
        return texts, predicates, self.encode(texts).detach()

    def all_predicates(self, subject_label, object_label):
        key = (int(subject_label), int(object_label))
        if key not in self.candidate_cache:
            subject = self.object_names[key[0]]
            object_ = self.object_names[key[1]]
            texts = [self.format_triplet(subject, predicate, object_) for predicate in self.relation_names]
            self.candidate_cache[key] = self.encode(texts).detach().cpu()
        return self.candidate_cache[key].to(self.device)

    def alignment_candidates(self, subject_labels, object_labels, predicates, device):
        candidates = torch.stack([
            self.all_predicates(subject, object_)
            for subject, object_ in zip(
                subject_labels.detach().cpu().tolist(), object_labels.detach().cpu().tolist()
            )
        ]).to(device=device, dtype=torch.float32)
        predicate_to_index = {name: index for index, name in enumerate(self.relation_names)}
        targets = torch.tensor(
            [predicate_to_index[name] for name in predicates], device=device, dtype=torch.long
        )
        return candidates, targets

    def inference_candidates(self, subject_labels, object_labels, active_relations, inference_filter):
        features = []
        for subject, object_ in zip(
            subject_labels.detach().cpu().tolist(), object_labels.detach().cpu().tolist()
        ):
            subject_name = self.object_names[int(subject)]
            candidate_relations = (
                tuple(inference_filter[subject_name])
                if subject_name in inference_filter.columns
                else tuple(active_relations)
            )
            key = (int(subject), int(object_), candidate_relations)
            if key not in self.filtered_cache:
                object_name = self.object_names[key[1]]
                texts = [
                    self.format_triplet(subject_name, predicate, object_name)
                    for predicate in candidate_relations
                ]
                self.filtered_cache[key] = self.encode(texts).detach().cpu()
            features.append(self.filtered_cache[key])
        return torch.stack(features).to(self.device)


class _VisualResidualAdapter(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout, reduction=4):
        super().__init__()
        bottleneck = max(embed_dim // reduction, 1)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck)
        self.up = nn.Linear(bottleneck, embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        residual = self.input_proj(features.float())
        normalized = self.norm(residual)
        delta = self.up(self.dropout(F.gelu(self.down(normalized))))
        return residual + torch.sigmoid(self.gate(normalized)) * delta


class MTMProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.visual_adapter = _VisualResidualAdapter(cfg.INPUT_DIM, cfg.EMBED_DIM, cfg.DROPOUT)
        layers = []
        for _ in range(max(int(cfg.FC_LAYERS) - 1, 0)):
            layers.extend([nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(cfg.DROPOUT)])
        self.fc = nn.Sequential(*layers)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.EMBED_DIM,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.EMBED_DIM * 4,
            dropout=cfg.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.self_attention = nn.TransformerEncoder(layer, num_layers=cfg.ATTENTION_LAYERS)
        self.norm = nn.LayerNorm(cfg.EMBED_DIM)

    def forward(self, raw_features, use_visual_adapter=True):
        visual_features = self.visual_adapter(raw_features) if use_visual_adapter else raw_features.float()
        projected = self.fc(visual_features).unsqueeze(0)
        projected = self.norm(self.self_attention(projected).squeeze(0))
        return visual_features, projected


class _ShipVisualEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, 2048), nn.ReLU(inplace=True))
        self.mean = nn.Linear(2048, dim)
        self.logvar = nn.Linear(2048, dim)
        self.apply(_init_ship_layer)

    def forward(self, features):
        hidden = self.body(features.float())
        return self.mean(hidden), self.logvar(hidden)


class _ShipTextAdapter(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.body = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim)
        )
        self.body.apply(_init_ship_layer)

    def forward(self, features):
        return F.normalize(features.float() + self.body(self.norm(features.float())), dim=-1)


class SHIPGenerator(nn.Module):
    def __init__(self, clip_model, cfg):
        super().__init__()
        object.__setattr__(self, "clip_model", clip_model)
        self.dim = cfg.EMBED_DIM
        self.ctx_len = cfg.SHIP_CTX_LEN
        self.ctx = nn.Parameter(torch.empty(self.ctx_len, self.dim))
        nn.init.normal_(self.ctx, std=SHIP_INIT_STD)
        self.encoder = _ShipVisualEncoder(self.dim)
        self.generator = nn.Sequential(
            nn.Linear(self.dim, 4096), nn.ReLU(inplace=True), nn.Linear(4096, self.dim)
        )
        self.generator.apply(_init_ship_layer)
        self.text_adapter = _ShipTextAdapter(
            self.dim, cfg.SHIP_TEXT_ADAPTER_HIDDEN_DIM, cfg.SHIP_TEXT_ADAPTER_DROPOUT
        ) if cfg.SHIP_TEXT_ADAPTER_ENABLED else None

    def _encode_prompts(self, texts, residuals):
        prefix = " ".join(["X"] * self.ctx_len)
        tokens = clip.tokenize([prefix + " " + text for text in texts]).to(residuals.device)
        with torch.no_grad():
            embeddings = self.clip_model.token_embedding(tokens).float()
        prompts = torch.cat([
            embeddings[:, :1],
            self.ctx.unsqueeze(0).expand(residuals.size(0), -1, -1) + residuals.unsqueeze(1),
            embeddings[:, 1 + self.ctx_len :],
        ], dim=1)
        dtype = self.clip_model.dtype
        encoded = prompts.to(dtype) + self.clip_model.positional_embedding.type(dtype)
        encoded = self.clip_model.transformer(encoded.permute(1, 0, 2)).permute(1, 0, 2)
        encoded = self.clip_model.ln_final(encoded).type(dtype)
        eos = tokens.argmax(dim=-1)
        encoded = (encoded[torch.arange(encoded.size(0), device=encoded.device), eos] @ self.clip_model.text_projection).float()
        encoded = F.normalize(encoded, dim=-1)
        return self.text_adapter(encoded) if self.text_adapter is not None else encoded

    def reconstruct(self, texts, raw_features):
        target = F.normalize(raw_features.detach().float(), dim=-1)
        mean, logvar = self.encoder(target)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return self._encode_prompts(texts, self.generator(latent)), mean, logvar

    def generate(self, texts, device):
        latent = torch.randn(len(texts), self.dim, device=device)
        return self._encode_prompts(texts, self.generator(latent))


class MTMDebugWriter:
    def __init__(self, enabled, interval, output_dir, filename):
        self.enabled = enabled
        self.interval = max(int(interval), 1)
        self.path = os.path.join(output_dir, filename)

    @staticmethod
    def _stats(features):
        if features.size(0) < 2:
            return (float("nan"),) * 4
        similarity = F.normalize(features.float(), dim=-1) @ F.normalize(features.float(), dim=-1).t()
        mask = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        values = similarity[mask]
        return values.mean().item(), values.max().item(), values.min().item(), values.std(unbiased=False).item()

    @staticmethod
    def _format_stats(features):
        return "/".join("{:.4f}".format(value) for value in MTMDebugWriter._stats(features))

    def write(self, step, base, novel=None):
        if not self.enabled or get_rank() != 0 or step % self.interval != 0:
            return
        lines = [
            "i={} base raw={} h={} q={} t={} align(pos/neg)={:.4f}/{:.4f} "
            "err(v/t)={:.4f}/{:.4f} recon={:.4f} kl={:.4f} norm(raw/h)={:.3f}/{:.3f}".format(
                step, self._format_stats(base["raw"]), self._format_stats(base["h"]),
                self._format_stats(base["q"]), self._format_stats(base["t"]),
                base["align_pos"], base["align_neg"], base["visual_error"], base["text_error"],
                base["recon"], base["kl"], base["raw"].norm(dim=-1).mean().item(),
                base["h"].norm(dim=-1).mean().item(),
            )
        ]
        if novel is not None and novel["raw"].size(0) > 0:
            nearest = (F.normalize(novel["raw"], dim=-1) @ F.normalize(base["raw"], dim=-1).t()).max(dim=1)[0].mean().item()
            lines.append(
                "i={} novel raw={} h={} q={} t={} align={:.4f} nearest_base={:.4f} "
                "err(v/t)={:.4f}/{:.4f} ramp={:.3f} n={} norm(raw/h)={:.3f}/{:.3f}".format(
                    step, self._format_stats(novel["raw"]), self._format_stats(novel["h"]),
                    self._format_stats(novel["q"]), self._format_stats(novel["t"]), novel["align"],
                    nearest, novel["visual_error"], novel["text_error"], novel["ramp"],
                    novel["raw"].size(0), novel["raw"].norm(dim=-1).mean().item(),
                    novel["h"].norm(dim=-1).mean().item(),
                )
            )
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as stream:
            stream.write("\n".join(lines) + "\n")


class MTMShipBranch(nn.Module):
    """Complete auxiliary branch. Its forward mirrors the training data flow."""

    def __init__(self, cfg, clip_model, object_names, relation_names, active_relations,
                 base_indices, novel_indices, text_filter, inference_filter, device, output_dir):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.active_relations = active_relations
        self.base_indices = base_indices
        self.novel_indices = novel_indices
        self.novel_relation_entries = [
            (index, relation_names[index]) for index in novel_indices if index != 0
        ]
        self.novel_relations = [name for _, name in self.novel_relation_entries]
        self.novel_candidates = {}
        self.inference_filter = inference_filter
        self.teacher = TripletTextTeacher(
            clip_model, object_names, relation_names[1:], text_filter, cfg, device
        )
        self.projector = MTMProjector(cfg)
        self.generator = SHIPGenerator(clip_model, cfg) if cfg.SHIP_ENABLED else None
        object.__setattr__(self, "clip_model", clip_model)
        self.clip_input_size = clip_model.visual.input_resolution
        self.image_normalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.debug = MTMDebugWriter(cfg.DEBUG_ENABLED, cfg.DEBUG_INTERVAL, output_dir, cfg.DEBUG_FILE)
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long))

    def set_mode(self, active_relations, inference_filter):
        self.active_relations = active_relations
        self.inference_filter = inference_filter
        self.teacher.filtered_cache.clear()

    def _empty_losses(self):
        zero = self.projector.norm.weight.sum() * 0.0
        return {
            "loss_mtm_align": zero,
            "loss_mtm_visual_structure": zero,
            "loss_mtm_text_structure": zero,
            "loss_mtm_ship_recon": zero,
            "loss_mtm_ship_kl": zero,
        }

    def _crop_union_features(self, image, proposal, pairs):
        boxes = proposal.bbox
        subjects = boxes.index_select(0, pairs[:, 0].to(boxes.device))
        objects = boxes.index_select(0, pairs[:, 1].to(boxes.device))
        union = torch.cat([torch.min(subjects[:, :2], objects[:, :2]), torch.max(subjects[:, 2:], objects[:, 2:])], dim=-1).long()
        unique, inverse = torch.unique(union, dim=0, sorted=False, return_inverse=True)
        image = image[:, : int(proposal.size[1]), : int(proposal.size[0])]
        _, height, width = image.shape
        chunks = []
        with torch.no_grad():
            for offset in range(0, unique.size(0), int(self.cfg.UNION_CROP_CHUNK_SIZE)):
                crops = []
                for box in unique[offset : offset + int(self.cfg.UNION_CROP_CHUNK_SIZE)]:
                    left = int(box[0].clamp(0, max(width - 1, 0)).item())
                    top = int(box[1].clamp(0, max(height - 1, 0)).item())
                    right = int(box[2].clamp(left + 1, width).item())
                    bottom = int(box[3].clamp(top + 1, height).item())
                    crop = TF.resized_crop(image, top, left, bottom - top, right - left,
                                           (self.clip_input_size, self.clip_input_size))
                    crop = (crop * 255.0).to(torch.uint8).to(crop.dtype) / 255.0
                    crops.append(self.image_normalize(crop))
                encoded = self.clip_model.encode_image(torch.stack(crops).to(self.device))
                chunks.append(encoded[:, 0] if encoded.dim() == 3 else encoded)
        return torch.cat(chunks).index_select(0, inverse.to(self.device)).float()

    @staticmethod
    def _structure(features, projected, text):
        if projected.size(0) < 2:
            zero = projected.sum() * 0.0
            return zero, zero
        mask = ~torch.eye(projected.size(0), dtype=torch.bool, device=projected.device)
        visual_similarity = features @ features.t()
        projected_similarity = projected @ projected.t()
        text_similarity = text @ text.t()
        return (
            (visual_similarity - projected_similarity).abs()[mask].mean(),
            (projected_similarity - text_similarity).abs()[mask].mean(),
        )

    def _alignment(self, projected, targets, subjects, objects, predicates):
        positive = 1.0 - (projected * targets).sum(dim=-1).mean()
        candidates, indices = self.teacher.alignment_candidates(
            subjects, objects, predicates, projected.device
        )
        similarities = torch.bmm(candidates, projected.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask.scatter_(1, indices.unsqueeze(1), False)
        negative = F.relu(similarities[mask] - float(self.cfg.ALIGN_NEG_MARGIN)).mean()
        return positive + float(self.cfg.ALIGN_NEG_WEIGHT) * negative, positive, negative

    def _sample_novel(self, subjects, objects, count):
        if count <= 0 or not self.novel_relations:
            return [], None, None, []
        indices = torch.randint(subjects.numel(), (count,), device=subjects.device)
        sampled_subjects = subjects.index_select(0, indices)
        sampled_objects = objects.index_select(0, indices)
        texts, predicates = [], []
        for subject, object_ in zip(sampled_subjects.cpu().tolist(), sampled_objects.cpu().tolist()):
            subject_name = self.teacher.object_names[int(subject)]
            object_name = self.teacher.object_names[int(object_)]
            compatible = self.novel_candidates.get(subject_name)
            if compatible is None:
                compatible = []
                if subject_name in self.teacher.text_filter.columns:
                    compatible = [name for index, name in self.novel_relation_entries
                                  if index < len(self.teacher.text_filter)
                                  and str(self.teacher.text_filter.iloc[index][subject_name]) == name]
                compatible = compatible or self.novel_relations
                self.novel_candidates[subject_name] = compatible
            sampling_pool = compatible
            if self.cfg.SHIP_LOW_RECALL_SAMPLING_ENABLED:
                low_recall = [
                    predicate for predicate in compatible
                    if predicate in LOW_RECALL_NOVEL_RELATIONS
                ]
                if low_recall:
                    sampling_pool = low_recall
            predicate = sampling_pool[int(torch.randint(len(sampling_pool), (1,)).item())]
            predicates.append(predicate)
            texts.append(self.teacher.format_triplet(subject_name, predicate, object_name))
        return texts, sampled_subjects, sampled_objects, predicates

    def compute_losses(self, raw, subjects, objects, relations):
        self.training_step.add_(1)
        step = int(self.training_step.item())
        if self.cfg.MAX_PAIRS > 0 and raw.size(0) > self.cfg.MAX_PAIRS:
            index = torch.linspace(0, raw.size(0) - 1, self.cfg.MAX_PAIRS, device=raw.device).long()
            raw, subjects, objects, relations = [value.index_select(0, index) for value in (raw, subjects, objects, relations)]

        base_texts, base_predicates, base_targets = self.teacher.build_targets(
            subjects, relations, objects, self.active_relations
        )
        base_targets = base_targets.to(raw.device)
        base_raw_normalized = F.normalize(raw.float(), dim=-1)
        reconstructed = mean = logvar = None
        zero = raw.sum() * 0.0
        reconstruction_loss = kl_loss = zero
        ship_ramp = min(max((step - self.cfg.SHIP_WARMUP_ITERS) / float(self.cfg.SHIP_RAMP_ITERS), 0.0), 1.0)

        novel_raw = None
        if self.generator is not None:
            reconstructed, mean, logvar = self.generator.reconstruct(base_texts, base_raw_normalized)
            reconstruction_loss = (reconstructed - base_raw_normalized.detach()).pow(2).sum(dim=-1).mean()
            kl_loss = (-0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1)).mean()
            novel_count = round(raw.size(0) * float(self.cfg.SHIP_PSEUDO_RATIO) * ship_ramp)
            novel_texts, novel_subjects, novel_objects, novel_predicates = self._sample_novel(subjects, objects, novel_count)
            if novel_texts:
                novel_targets = self.teacher.encode(novel_texts).detach().to(raw.device)
                novel_raw = self.generator.generate(novel_texts, raw.device)
                novel_raw = novel_raw * raw.detach().float().norm(dim=-1).mean().clamp(min=1e-6)
        if novel_raw is None:
            novel_texts, novel_predicates, novel_targets = [], [], None
            novel_subjects = novel_objects = None

        all_raw = torch.cat([raw, novel_raw], dim=0) if novel_raw is not None else raw
        all_targets = torch.cat([base_targets, novel_targets], dim=0) if novel_targets is not None else base_targets
        all_subjects = torch.cat([subjects, novel_subjects]) if novel_subjects is not None else subjects
        all_objects = torch.cat([objects, novel_objects]) if novel_objects is not None else objects
        all_predicates = base_predicates + novel_predicates

        visual, projected = self.projector(all_raw, self.cfg.VISUAL_ADAPTER_ENABLED)
        visual_normalized = F.normalize(visual.float(), dim=-1)
        projected_normalized = F.normalize(projected.float(), dim=-1)
        align, align_pos, align_neg = self._alignment(
            projected_normalized, all_targets, all_subjects, all_objects, all_predicates
        )
        visual_structure, text_structure = self._structure(
            visual_normalized, projected_normalized, all_targets
        )
        structure_ramp = min(max((step - self.cfg.STRUCTURE_WARMUP_ITERS) / float(self.cfg.STRUCTURE_RAMP_ITERS), 0.0), 1.0)

        base_count = raw.size(0)
        base_visual_error, base_text_error = self._structure(
            visual_normalized[:base_count], projected_normalized[:base_count], base_targets
        )
        base_debug = {
            "raw": raw, "h": visual[:base_count], "q": projected_normalized[:base_count], "t": base_targets,
            "align_pos": align_pos.item(), "align_neg": align_neg.item(),
            "visual_error": base_visual_error.item(), "text_error": base_text_error.item(),
            "recon": (reconstructed * base_raw_normalized).sum(-1).mean().item() if reconstructed is not None else float("nan"),
            "kl": kl_loss.item(),
        }
        novel_debug = None
        if novel_raw is not None:
            novel_visual_error, novel_text_error = self._structure(
                visual_normalized[base_count:], projected_normalized[base_count:], novel_targets
            )
            novel_debug = {
                "raw": novel_raw, "h": visual[base_count:], "q": projected_normalized[base_count:], "t": novel_targets,
                "align": (projected_normalized[base_count:] * novel_targets).sum(-1).mean().item(),
                "visual_error": novel_visual_error.item(), "text_error": novel_text_error.item(), "ramp": ship_ramp,
            }
        with torch.no_grad():
            self.debug.write(step, base_debug, novel_debug)

        return {
            "loss_mtm_align": float(self.cfg.ALIGN_WEIGHT) * align,
            "loss_mtm_visual_structure": float(self.cfg.VISUAL_STRUCTURE_WEIGHT) * structure_ramp * visual_structure,
            "loss_mtm_text_structure": float(self.cfg.TEXT_STRUCTURE_WEIGHT) * structure_ramp * text_structure,
            "loss_mtm_ship_recon": float(self.cfg.SHIP_RECON_WEIGHT) * reconstruction_loss,
            "loss_mtm_ship_kl": float(self.cfg.SHIP_KL_WEIGHT) * ship_ramp * kl_loss,
        }

    def forward(self, images, proposals, pair_indices, relation_labels, object_predictions):
        raw_features, subject_labels, object_labels, target_relations = [], [], [], []
        inference_scores = []
        labels_per_image = (
            relation_labels if relation_labels is not None else [None] * len(pair_indices)
        )
        for image, proposal, pairs, predicted_objects, labels in zip(
            images, proposals, pair_indices, object_predictions, labels_per_image
        ):
            pairs = pairs.long()
            if self.training:
                positive = torch.nonzero(labels.to(pairs.device).view(-1) > 0, as_tuple=False).view(-1)
                selected_pairs = pairs.index_select(0, positive)
            else:
                positive = None
                selected_pairs = pairs
            if selected_pairs.numel() == 0:
                if not self.training:
                    inference_scores.append(image.new_zeros((pairs.size(0), len(self.active_relations))))
                continue
            raw = self._crop_union_features(image, proposal, selected_pairs)
            subjects = predicted_objects.index_select(0, selected_pairs[:, 0]).long().to(raw.device)
            objects = predicted_objects.index_select(0, selected_pairs[:, 1]).long().to(raw.device)
            if self.training:
                if proposal.has_field("labels"):
                    ground_truth = proposal.get_field("labels").long().to(raw.device)
                    subjects = ground_truth.index_select(0, selected_pairs[:, 0])
                    objects = ground_truth.index_select(0, selected_pairs[:, 1])
                raw_features.append(raw)
                subject_labels.append(subjects)
                object_labels.append(objects)
                target_relations.append(labels.to(raw.device).index_select(0, positive))
            else:
                _, projected = self.projector(raw, self.cfg.VISUAL_ADAPTER_ENABLED)
                projected = F.normalize(projected.float(), dim=-1)
                score_chunks = []
                chunk_size = self.teacher.chunk_size
                for offset in range(0, projected.size(0), chunk_size):
                    candidates = self.teacher.inference_candidates(
                        subjects[offset : offset + chunk_size],
                        objects[offset : offset + chunk_size],
                        self.active_relations,
                        self.inference_filter,
                    )
                    score_chunks.append(
                        (projected[offset : offset + chunk_size].unsqueeze(1) * candidates).sum(dim=-1)
                    )
                inference_scores.append(torch.cat(score_chunks, dim=0))

        if not self.training:
            return {"scores": inference_scores, "losses": {}}
        if not raw_features:
            return {"scores": None, "losses": self._empty_losses()}
        return {
            "scores": None,
            "losses": self.compute_losses(
                torch.cat(raw_features), torch.cat(subject_labels), torch.cat(object_labels), torch.cat(target_relations)
            ),
        }
