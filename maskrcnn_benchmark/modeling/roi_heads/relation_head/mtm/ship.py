import torch
from torch import nn
from torch.nn import functional as F

from CLIP import clip


SHIP_INIT_STD = 0.02


def _init_ship_layer(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=SHIP_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ShipVisualEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, 2048), nn.ReLU(inplace=True))
        self.mean = nn.Linear(2048, dim)
        self.logvar = nn.Linear(2048, dim)
        self.apply(_init_ship_layer)

    def forward(self, features):
        hidden = self.body(features.float())
        return self.mean(hidden), self.logvar(hidden)


class ShipTextAdapter(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.normalization = nn.LayerNorm(dim)
        self.body = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.body.apply(_init_ship_layer)

    def forward(self, features):
        features = features.float()
        return F.normalize(
            features + self.body(self.normalization(features)), dim=-1
        )


class SHIPGenerator(nn.Module):
    def __init__(self, clip_model, embed_dim, config):
        super().__init__()
        object.__setattr__(self, "clip_model", clip_model)
        self.embed_dim = embed_dim
        self.context_length = int(config.CONTEXT_LENGTH)
        self.context = nn.Parameter(torch.empty(self.context_length, embed_dim))
        nn.init.normal_(self.context, std=SHIP_INIT_STD)
        self.encoder = ShipVisualEncoder(embed_dim)
        self.generator = nn.Sequential(
            nn.Linear(embed_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, embed_dim),
        )
        self.generator.apply(_init_ship_layer)
        self.text_adapter = (
            ShipTextAdapter(
                embed_dim,
                config.TEXT_ADAPTER.HIDDEN_DIM,
                config.TEXT_ADAPTER.DROPOUT,
            )
            if config.TEXT_ADAPTER.ENABLED
            else None
        )

    def _encode_prompts(self, texts, residuals):
        prefix = " ".join(["X"] * self.context_length)
        tokens = clip.tokenize([prefix + " " + text for text in texts]).to(
            residuals.device
        )
        with torch.no_grad():
            embeddings = self.clip_model.token_embedding(tokens).float()
        prompts = torch.cat(
            [
                embeddings[:, :1],
                self.context.unsqueeze(0).expand(residuals.size(0), -1, -1)
                + residuals.unsqueeze(1),
                embeddings[:, 1 + self.context_length :],
            ],
            dim=1,
        )
        dtype = self.clip_model.dtype
        encoded = prompts.to(dtype) + self.clip_model.positional_embedding.type(dtype)
        encoded = self.clip_model.transformer(encoded.permute(1, 0, 2)).permute(
            1, 0, 2
        )
        encoded = self.clip_model.ln_final(encoded).type(dtype)
        eos = tokens.argmax(dim=-1)
        encoded = (
            encoded[
                torch.arange(encoded.size(0), device=encoded.device), eos
            ]
            @ self.clip_model.text_projection
        ).float()
        encoded = F.normalize(encoded, dim=-1)
        return self.text_adapter(encoded) if self.text_adapter is not None else encoded

    def reconstruct(self, texts, raw_features):
        target = F.normalize(raw_features.detach().float(), dim=-1)
        mean, logvar = self.encoder(target)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return self._encode_prompts(texts, self.generator(latent)), mean, logvar

    def generate(self, texts, device):
        latent = torch.randn(len(texts), self.embed_dim, device=device)
        return self._encode_prompts(texts, self.generator(latent))


class PredicateSampler(nn.Module):
    def __init__(self, relation_names, base_indices, novel_indices, config):
        super().__init__()
        self.config = config
        self.base_entries = [
            (index, relation_names[index]) for index in base_indices if index != 0
        ]
        self.base_names = [name for _, name in self.base_entries]
        self.base_name_to_slot = {
            name: slot for slot, name in enumerate(self.base_names)
        }
        self.novel_entries = [
            (index, relation_names[index]) for index in novel_indices if index != 0
        ]
        self.novel_names = [name for _, name in self.novel_entries]
        self.novel_name_to_slot = {
            name: slot for slot, name in enumerate(self.novel_names)
        }
        self.base_candidates = {}
        self.novel_candidates = {}
        self.low_recall_names = frozenset(config.LOW_RECALL_RELATIONS)
        replay = config.BASE_REPLAY
        if not 0.0 <= float(replay.DIFFICULTY_MIX) <= 1.0:
            raise ValueError("SHIP.BASE_REPLAY.DIFFICULTY_MIX must be in [0, 1]")
        if not 0.0 <= float(replay.DIFFICULTY_EMA) < 1.0:
            raise ValueError("SHIP.BASE_REPLAY.DIFFICULTY_EMA must be in [0, 1)")
        if float(replay.DIFFICULTY_TEMPERATURE) <= 0.0:
            raise ValueError(
                "SHIP.BASE_REPLAY.DIFFICULTY_TEMPERATURE must be positive"
            )
        self.register_buffer(
            "base_difficulty_ema", torch.zeros(len(self.base_names))
        )
        self.register_buffer(
            "base_difficulty_updates",
            torch.zeros(len(self.base_names), dtype=torch.long),
        )
        self.register_buffer(
            "base_sample_counts", torch.zeros(len(self.base_names), dtype=torch.long)
        )
        self.register_buffer(
            "novel_sample_counts",
            torch.zeros(len(self.novel_names), dtype=torch.long),
        )

    @torch.no_grad()
    def update_base_difficulty(self, predicates, errors):
        if not self.base_names or errors.numel() == 0:
            return
        slots = torch.tensor(
            [self.base_name_to_slot.get(name, -1) for name in predicates],
            device=errors.device,
            dtype=torch.long,
        )
        valid = slots >= 0
        if not valid.any():
            return
        slots, errors = slots[valid], errors.detach().float()[valid]
        sums = errors.new_zeros(len(self.base_names))
        counts = errors.new_zeros(len(self.base_names))
        sums.scatter_add_(0, slots, errors)
        counts.scatter_add_(0, slots, torch.ones_like(errors))
        observed = counts > 0
        means = sums[observed] / counts[observed]
        previous = self.base_difficulty_ema[observed]
        first_update = self.base_difficulty_updates[observed] == 0
        decay = float(self.config.BASE_REPLAY.DIFFICULTY_EMA)
        updated = decay * previous + (1.0 - decay) * means
        self.base_difficulty_ema[observed] = torch.where(
            first_update, means, updated
        )
        self.base_difficulty_updates[observed] += counts[observed].long()

    def _base_probabilities(self):
        if not self.base_names:
            return self.base_difficulty_ema
        uniform = torch.full_like(
            self.base_difficulty_ema, 1.0 / float(len(self.base_names))
        )
        observed = self.base_difficulty_updates > 0
        if not observed.any():
            return uniform
        difficulty = self.base_difficulty_ema.clone()
        difficulty[~observed] = difficulty[observed].mean()
        replay = self.config.BASE_REPLAY
        difficulty = torch.softmax(
            difficulty / float(replay.DIFFICULTY_TEMPERATURE), dim=0
        )
        mix = float(replay.DIFFICULTY_MIX)
        return (1.0 - mix) * uniform + mix * difficulty

    def sample_base(self, subjects, objects, count, teacher):
        if count <= 0 or not self.base_names:
            return [], None, None, []
        pair_indices = torch.randint(subjects.numel(), (count,), device=subjects.device)
        sampled_subjects = subjects.index_select(0, pair_indices)
        sampled_objects = objects.index_select(0, pair_indices)
        probabilities = self._base_probabilities()
        texts, predicates, sampled_slots = [], [], []
        for subject, object_ in zip(
            sampled_subjects.detach().cpu().tolist(),
            sampled_objects.detach().cpu().tolist(),
        ):
            subject_name = teacher.object_names[int(subject)]
            object_name = teacher.object_names[int(object_)]
            compatible_slots = self.base_candidates.get(subject_name)
            if compatible_slots is None:
                compatible_slots = []
                if subject_name in teacher.text_filter.columns:
                    compatible_slots = [
                        slot
                        for slot, (index, name) in enumerate(self.base_entries)
                        if index < len(teacher.text_filter)
                        and str(teacher.text_filter.iloc[index][subject_name]) == name
                    ]
                compatible_slots = compatible_slots or list(range(len(self.base_names)))
                self.base_candidates[subject_name] = compatible_slots
            compatible = torch.tensor(
                compatible_slots, device=probabilities.device, dtype=torch.long
            )
            conditional = probabilities.index_select(0, compatible)
            conditional = conditional / conditional.sum().clamp(min=1e-12)
            slot = int(compatible[torch.multinomial(conditional, 1)].item())
            predicate = self.base_names[slot]
            sampled_slots.append(slot)
            predicates.append(predicate)
            texts.append(teacher.format_triplet(subject_name, predicate, object_name))
        sampled_slots = torch.tensor(
            sampled_slots, device=self.base_sample_counts.device, dtype=torch.long
        )
        self.base_sample_counts.add_(
            torch.bincount(sampled_slots, minlength=len(self.base_names))
        )
        return texts, sampled_subjects, sampled_objects, predicates

    def sample_novel(self, subjects, objects, count, teacher):
        if count <= 0 or not self.novel_names:
            return [], None, None, []
        indices = torch.randint(subjects.numel(), (count,), device=subjects.device)
        sampled_subjects = subjects.index_select(0, indices)
        sampled_objects = objects.index_select(0, indices)
        texts, predicates = [], []
        for subject, object_ in zip(
            sampled_subjects.cpu().tolist(), sampled_objects.cpu().tolist()
        ):
            subject_name = teacher.object_names[int(subject)]
            object_name = teacher.object_names[int(object_)]
            compatible = self.novel_candidates.get(subject_name)
            if compatible is None:
                compatible = []
                if subject_name in teacher.text_filter.columns:
                    compatible = [
                        name
                        for index, name in self.novel_entries
                        if index < len(teacher.text_filter)
                        and str(teacher.text_filter.iloc[index][subject_name]) == name
                    ]
                compatible = compatible or self.novel_names
                self.novel_candidates[subject_name] = compatible
            sampling_pool = compatible
            if self.config.LOW_RECALL_SAMPLING_ENABLED:
                low_recall = [
                    name for name in compatible if name in self.low_recall_names
                ]
                if low_recall:
                    sampling_pool = low_recall
            predicate = sampling_pool[
                int(torch.randint(len(sampling_pool), (1,)).item())
            ]
            predicates.append(predicate)
            texts.append(teacher.format_triplet(subject_name, predicate, object_name))
        sampled_slots = torch.tensor(
            [self.novel_name_to_slot[name] for name in predicates],
            device=self.novel_sample_counts.device,
            dtype=torch.long,
        )
        self.novel_sample_counts.add_(
            torch.bincount(sampled_slots, minlength=len(self.novel_names))
        )
        return texts, sampled_subjects, sampled_objects, predicates
