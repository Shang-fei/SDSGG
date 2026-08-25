from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize
from torchvision.transforms import functional as TF

from .diagnostics import MTMDebugWriter
from .losses import MTMLossComputer, linear_ramp, structure_losses
from .projector import MTMProjector
from .ship import PredicateSampler, SHIPGenerator
from .text_teacher import TripletTextTeacher


@dataclass
class MTMOutput:
    score_deltas: object = None
    losses: dict = field(default_factory=dict)
    score_weight: float = 0.0

    def fuse(self, relation_logits):
        if self.score_deltas is None:
            return relation_logits
        if len(relation_logits) != len(self.score_deltas):
            raise ValueError("MTM score batch size does not match relation logits")
        fused = []
        for logits, scores in zip(relation_logits, self.score_deltas):
            if scores is None:
                fused.append(logits)
                continue
            if logits.shape != scores.shape:
                raise ValueError(
                    "MTM score shape {} does not match relation logits {}".format(
                        tuple(scores.shape), tuple(logits.shape)
                    )
                )
            fused.append(
                logits
                + self.score_weight
                * scores.to(device=logits.device, dtype=logits.dtype)
            )
        return fused


class MTMPlugin(nn.Module):
    def __init__(
        self,
        config,
        clip_model,
        object_names,
        predicate_spec,
        output_dir,
        device,
    ):
        super().__init__()
        self.config = config
        self.predicate_spec = predicate_spec
        self.device = device
        object.__setattr__(self, "clip_model", clip_model)
        self.teacher = TripletTextTeacher(
            clip_model,
            object_names,
            predicate_spec.relation_names[1:],
            predicate_spec.text_filter,
            config.TEXT_TEACHER,
            device,
        )
        self.projector = MTMProjector(config.PROJECTOR)
        self.generator = (
            SHIPGenerator(
                clip_model, config.PROJECTOR.EMBED_DIM, config.SHIP
            )
            if config.SHIP.ENABLED
            else None
        )
        self.sampler = PredicateSampler(
            predicate_spec.relation_names,
            predicate_spec.base_indices,
            predicate_spec.novel_indices,
            config.SHIP,
        )
        self.loss_computer = MTMLossComputer(config.LOSS, self.teacher)
        self.debug = MTMDebugWriter(config.DEBUG, output_dir)
        self.clip_input_size = clip_model.visual.input_resolution
        self.image_normalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long))
        self.set_mode("base")

    def set_mode(self, mode):
        self.mode = mode
        self.active_relations, self.inference_filter = self.predicate_spec.resolve(mode)
        self.teacher.filtered_cache.clear()

    def ship_parameters(self):
        return tuple(self.generator.parameters()) if self.generator is not None else ()

    def _crop_union_features(self, image, proposal, pairs):
        boxes = proposal.bbox
        subjects = boxes.index_select(0, pairs[:, 0].to(boxes.device))
        objects = boxes.index_select(0, pairs[:, 1].to(boxes.device))
        union = torch.cat(
            [
                torch.min(subjects[:, :2], objects[:, :2]),
                torch.max(subjects[:, 2:], objects[:, 2:]),
            ],
            dim=-1,
        ).long()
        unique, inverse = torch.unique(
            union, dim=0, sorted=False, return_inverse=True
        )
        image = image[:, : int(proposal.size[1]), : int(proposal.size[0])]
        _, height, width = image.shape
        chunks = []
        batch_size = int(self.config.INFERENCE.UNION_CROP_BATCH_SIZE)
        with torch.no_grad():
            for offset in range(0, unique.size(0), batch_size):
                crops = []
                for box in unique[offset : offset + batch_size]:
                    left = int(box[0].clamp(0, max(width - 1, 0)).item())
                    top = int(box[1].clamp(0, max(height - 1, 0)).item())
                    right = int(box[2].clamp(left + 1, width).item())
                    bottom = int(box[3].clamp(top + 1, height).item())
                    crop = TF.resized_crop(
                        image,
                        top,
                        left,
                        bottom - top,
                        right - left,
                        (self.clip_input_size, self.clip_input_size),
                    )
                    crop = (crop * 255.0).to(torch.uint8).to(crop.dtype) / 255.0
                    crops.append(self.image_normalize(crop))
                encoded = self.clip_model.encode_image(
                    torch.stack(crops).to(self.device)
                )
                chunks.append(encoded[:, 0] if encoded.dim() == 3 else encoded)
        return (
            torch.cat(chunks)
            .index_select(0, inverse.to(self.device))
            .float()
        )

    def _empty_losses(self):
        return self.loss_computer.empty(self.projector.output_normalization.weight)

    def compute_losses(
        self,
        raw,
        subjects,
        objects,
        relations,
        subject_features=None,
        object_features=None,
    ):
        self.training_step.add_(1)
        step = int(self.training_step.item())
        fusion_enabled = bool(
            self.config.PROJECTOR.ENTITY_FUSION_ENABLED
        )
        max_pairs = int(self.config.MAX_TRAIN_PAIRS)
        if max_pairs > 0 and raw.size(0) > max_pairs:
            index = torch.linspace(
                0, raw.size(0) - 1, max_pairs, device=raw.device
            ).long()
            raw, subjects, objects, relations = [
                value.index_select(0, index)
                for value in (raw, subjects, objects, relations)
            ]
            if subject_features is not None:
                subject_features = subject_features.index_select(0, index)
                object_features = object_features.index_select(0, index)

        base_relation_features = self.projector.fuse_inputs(
            subject_features,
            raw,
            object_features,
            fusion_enabled,
        )

        base_texts, base_predicates, base_targets = self.teacher.build_targets(
            subjects, relations, objects, self.active_relations
        )
        base_targets = base_targets.to(raw.device)
        normalized_base_raw = F.normalize(raw.float(), dim=-1)
        zero = raw.sum() * 0.0
        reconstructed = None
        reconstruction_loss = zero
        kl_loss = zero
        ship = self.config.SHIP
        ship_ramp = linear_ramp(
            step, ship.WARMUP_STEPS, ship.RAMP_STEPS
        )

        replay_raw = replay_targets = replay_source_indices = None
        replay_subjects = replay_objects = None
        replay_predicates = []
        novel_raw = novel_source_indices = None
        if self.generator is not None:
            reconstructed, mean, logvar = self.generator.reconstruct(
                base_texts, normalized_base_raw
            )
            reconstruction_loss = (
                reconstructed - normalized_base_raw.detach()
            ).pow(2).sum(dim=-1).mean()
            kl_loss = (
                -0.5
                * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1)
            ).mean()
            if ship.BASE_REPLAY.ENABLED:
                replay_count = round(
                    raw.size(0)
                    * float(ship.BASE_REPLAY.PSEUDO_RATIO)
                    * ship_ramp
                )
                (
                    replay_texts,
                    replay_subjects,
                    replay_objects,
                    replay_predicates,
                    replay_source_indices,
                ) = self.sampler.sample_base(
                    subjects, objects, replay_count, self.teacher
                )
                if replay_texts:
                    replay_targets = self.teacher.encode(replay_texts).detach().to(
                        raw.device
                    )
                    replay_raw = self.generator.generate(replay_texts, raw.device)
                    replay_raw = replay_raw * raw.detach().float().norm(
                        dim=-1
                    ).mean().clamp(min=1e-6)
            novel_count = round(
                raw.size(0) * float(ship.PSEUDO_NOVEL_RATIO) * ship_ramp
            )
            (
                novel_texts,
                novel_subjects,
                novel_objects,
                novel_predicates,
                novel_source_indices,
            ) = self.sampler.sample_novel(
                subjects, objects, novel_count, self.teacher
            )
            if novel_texts:
                novel_targets = self.teacher.encode(novel_texts).detach().to(raw.device)
                novel_raw = self.generator.generate(novel_texts, raw.device)
                novel_raw = novel_raw * raw.detach().float().norm(
                    dim=-1
                ).mean().clamp(min=1e-6)
        if novel_raw is None:
            novel_predicates, novel_targets = [], None
            novel_subjects = novel_objects = None

        novel_relation_features = None
        if novel_raw is not None:
            novel_relation_features = self.projector.fuse_inputs(
                subject_features.index_select(0, novel_source_indices)
                if fusion_enabled
                else None,
                novel_raw,
                object_features.index_select(0, novel_source_indices)
                if fusion_enabled
                else None,
                fusion_enabled,
            )
        all_raw = (
            torch.cat([base_relation_features, novel_relation_features], dim=0)
            if novel_relation_features is not None
            else base_relation_features
        )
        all_targets = (
            torch.cat([base_targets, novel_targets], dim=0)
            if novel_targets is not None
            else base_targets
        )
        all_subjects = (
            torch.cat([subjects, novel_subjects])
            if novel_subjects is not None
            else subjects
        )
        all_objects = (
            torch.cat([objects, novel_objects])
            if novel_objects is not None
            else objects
        )
        all_predicates = base_predicates + novel_predicates

        visual, projected = self.projector(
            all_raw, self.config.PROJECTOR.VISUAL_ADAPTER_ENABLED
        )
        normalized_visual = F.normalize(visual.float(), dim=-1)
        normalized_projected = F.normalize(projected.float(), dim=-1)
        alignment, align_pos, align_neg = self.loss_computer.alignment(
            normalized_projected,
            all_targets,
            all_subjects,
            all_objects,
            all_predicates,
        )
        main_count = normalized_projected.size(0)
        if replay_raw is not None:
            replay_relation_features = self.projector.fuse_inputs(
                subject_features.index_select(0, replay_source_indices)
                if fusion_enabled
                else None,
                replay_raw,
                object_features.index_select(0, replay_source_indices)
                if fusion_enabled
                else None,
                fusion_enabled,
            )
            _, replay_projected = self.projector(
                replay_relation_features,
                self.config.PROJECTOR.VISUAL_ADAPTER_ENABLED,
            )
            replay_projected = F.normalize(replay_projected.float(), dim=-1)
            _, replay_align_pos, replay_align_neg = self.loss_computer.alignment(
                replay_projected,
                replay_targets,
                replay_subjects,
                replay_objects,
                replay_predicates,
            )
            replay_count = replay_projected.size(0)
            total_count = main_count + replay_count
            align_pos = (
                align_pos * main_count + replay_align_pos * replay_count
            ) / float(total_count)
            align_neg = (
                align_neg * main_count + replay_align_neg * replay_count
            ) / float(total_count)
            alignment = align_pos + float(
                self.config.LOSS.NEGATIVE_ALIGNMENT_WEIGHT
            ) * align_neg

        loss_config = self.config.LOSS
        structure_distance = loss_config.STRUCTURE_DISTANCE
        visual_structure, text_structure = structure_losses(
            normalized_visual,
            normalized_projected,
            all_targets,
            structure_distance,
        )
        structure_ramp = linear_ramp(
            step,
            loss_config.STRUCTURE_WARMUP_STEPS,
            loss_config.STRUCTURE_RAMP_STEPS,
        )

        base_count = raw.size(0)
        with torch.no_grad():
            base_errors = 1.0 - (
                normalized_projected[:base_count] * base_targets
            ).sum(dim=-1)
            self.sampler.update_base_difficulty(base_predicates, base_errors)
        base_visual_error, base_text_error = structure_losses(
            normalized_visual[:base_count],
            normalized_projected[:base_count],
            base_targets,
            structure_distance,
        )
        base_debug = {
            "raw": base_relation_features,
            "h": visual[:base_count],
            "q": normalized_projected[:base_count],
            "t": base_targets,
            "align_pos": align_pos.item(),
            "align_neg": align_neg.item(),
            "visual_error": base_visual_error.item(),
            "text_error": base_text_error.item(),
            "recon": (
                (reconstructed * normalized_base_raw).sum(-1).mean().item()
                if reconstructed is not None
                else float("nan")
            ),
            "kl": kl_loss.item(),
        }
        novel_debug = None
        if novel_raw is not None:
            novel_visual_error, novel_text_error = structure_losses(
                normalized_visual[base_count:],
                normalized_projected[base_count:],
                novel_targets,
                structure_distance,
            )
            novel_debug = {
                "raw": novel_relation_features,
                "h": visual[base_count:],
                "q": normalized_projected[base_count:],
                "t": novel_targets,
                "align": (
                    normalized_projected[base_count:] * novel_targets
                ).sum(-1).mean().item(),
                "visual_error": novel_visual_error.item(),
                "text_error": novel_text_error.item(),
                "ramp": ship_ramp,
            }
        with torch.no_grad():
            self.debug.write(
                step,
                base_debug,
                novel_debug,
                {
                    "base_names": self.sampler.base_names,
                    "base_counts": self.sampler.base_sample_counts,
                    "base_difficulty": self.sampler.base_difficulty_ema,
                    "novel_names": self.sampler.novel_names,
                    "novel_counts": self.sampler.novel_sample_counts,
                },
            )
        return self.loss_computer.weighted(
            alignment,
            visual_structure,
            text_structure,
            structure_ramp,
            reconstruction_loss,
            kl_loss,
            ship_ramp,
            ship,
        )

    def forward(
        self,
        images,
        proposals,
        pair_indices,
        relation_labels,
        object_labels,
        object_clip_features=None,
    ):
        if self.training and not self.config.TRAINING_ENABLED:
            return MTMOutput()
        if not self.training and (
            not self.config.INFERENCE.ENABLED
            or float(self.config.INFERENCE.SCORE_WEIGHT) == 0.0
        ):
            return MTMOutput()

        raw_features, subject_labels, object_label_batches, target_relations = (
            [],
            [],
            [],
            [],
        )
        subject_feature_batches, object_feature_batches = [], []
        inference_scores = []
        labels_per_image = (
            relation_labels
            if relation_labels is not None
            else [None] * len(pair_indices)
        )
        clip_features_per_image = (
            object_clip_features
            if object_clip_features is not None
            else [None] * len(pair_indices)
        )
        for image, proposal, pairs, predicted_objects, labels, clip_objects in zip(
            images,
            proposals,
            pair_indices,
            object_labels,
            labels_per_image,
            clip_features_per_image,
        ):
            pairs = pairs.long()
            if self.training:
                positive = torch.nonzero(
                    labels.to(pairs.device).view(-1) > 0, as_tuple=False
                ).view(-1)
                selected_pairs = pairs.index_select(0, positive)
            else:
                positive = None
                selected_pairs = pairs
            if selected_pairs.numel() == 0:
                if not self.training:
                    inference_scores.append(
                        image.new_zeros((pairs.size(0), len(self.active_relations)))
                    )
                continue
            raw = self._crop_union_features(image, proposal, selected_pairs)
            subject_visual = object_visual = None
            if self.config.PROJECTOR.ENTITY_FUSION_ENABLED:
                if clip_objects is None:
                    raise ValueError(
                        "MTM entity fusion requires precomputed object CLIP features"
                    )
                clip_cls = (
                    clip_objects[:, 0, :]
                    if clip_objects.dim() == 3
                    else clip_objects
                )
                selected_on_clip = selected_pairs.to(clip_cls.device)
                subject_visual = clip_cls.index_select(
                    0, selected_on_clip[:, 0]
                ).to(raw.device)
                object_visual = clip_cls.index_select(
                    0, selected_on_clip[:, 1]
                ).to(raw.device)
            subjects = predicted_objects.index_select(
                0, selected_pairs[:, 0]
            ).long().to(raw.device)
            objects = predicted_objects.index_select(
                0, selected_pairs[:, 1]
            ).long().to(raw.device)
            if self.training:
                if proposal.has_field("labels"):
                    ground_truth = proposal.get_field("labels").long().to(raw.device)
                    subjects = ground_truth.index_select(0, selected_pairs[:, 0])
                    objects = ground_truth.index_select(0, selected_pairs[:, 1])
                raw_features.append(raw)
                subject_labels.append(subjects)
                object_label_batches.append(objects)
                subject_feature_batches.append(subject_visual)
                object_feature_batches.append(object_visual)
                target_relations.append(
                    labels.to(raw.device).index_select(0, positive)
                )
            else:
                relation_features = self.projector.fuse_inputs(
                    subject_visual,
                    raw,
                    object_visual,
                    self.config.PROJECTOR.ENTITY_FUSION_ENABLED,
                )
                _, projected = self.projector(
                    relation_features,
                    self.config.PROJECTOR.VISUAL_ADAPTER_ENABLED,
                )
                projected = F.normalize(projected.float(), dim=-1)
                score_chunks = []
                batch_size = self.teacher.encode_batch_size
                for offset in range(0, projected.size(0), batch_size):
                    candidates = self.teacher.inference_candidates(
                        subjects[offset : offset + batch_size],
                        objects[offset : offset + batch_size],
                        self.active_relations,
                        self.inference_filter,
                    )
                    score_chunks.append(
                        (
                            projected[offset : offset + batch_size].unsqueeze(1)
                            * candidates
                        ).sum(dim=-1)
                    )
                scores = torch.cat(score_chunks, dim=0)
                if scores.size(1) != len(self.active_relations):
                    raise ValueError("MTM inference predicate columns are misaligned")
                inference_scores.append(scores)

        if not self.training:
            return MTMOutput(
                score_deltas=inference_scores,
                score_weight=float(self.config.INFERENCE.SCORE_WEIGHT),
            )
        if not raw_features:
            return MTMOutput(losses=self._empty_losses())
        return MTMOutput(
            losses=self.compute_losses(
                torch.cat(raw_features),
                torch.cat(subject_labels),
                torch.cat(object_label_batches),
                torch.cat(target_relations),
                (
                    torch.cat(subject_feature_batches)
                    if self.config.PROJECTOR.ENTITY_FUSION_ENABLED
                    else None
                ),
                (
                    torch.cat(object_feature_batches)
                    if self.config.PROJECTOR.ENTITY_FUSION_ENABLED
                    else None
                ),
            )
        )


def build_mtm_plugin(
    config, clip_model, object_names, predicate_spec, output_dir
):
    mtm_config = config.MODEL.ROI_RELATION_HEAD.MTM
    if not mtm_config.ENABLED:
        return None
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    device = config.MODEL.DEVICE
    return MTMPlugin(
        mtm_config,
        clip_model,
        object_names,
        predicate_spec,
        output_dir,
        device,
    ).to(device)
