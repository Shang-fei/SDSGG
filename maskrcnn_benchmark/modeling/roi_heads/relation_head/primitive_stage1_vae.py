import json
import os

import torch
from torch import nn
from torch.nn import functional as F

from .CLIP import clip


def load_primitive_mapping(path):
    with open(path, "r") as f:
        return json.load(f)


def default_mapping_path():
    return os.path.join(os.path.dirname(__file__), "primitive_mapping.json")


class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.to(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot_pos = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eot_pos] @ self.text_projection


class RelationVAEEncoder(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=2048, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        hidden = self.net(x)
        return self.mean(hidden), self.log_var(hidden)


class PromptBiasGenerator(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=4096, prompt_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim),
        )
        self.apply(RelationVAEEncoder._init_weights)

    def forward(self, z):
        return self.net(z)


class PrimitivePromptLearner(nn.Module):
    def __init__(self, clip_model, num_slots=16, n_ctx=4, max_slots_per_predicate=4):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.num_slots = num_slots
        self.n_ctx = n_ctx
        self.max_slots_per_predicate = max_slots_per_predicate
        self.primitive_prompt_bank = nn.Parameter(torch.empty(num_slots, n_ctx, ctx_dim))
        nn.init.normal_(self.primitive_prompt_bank, std=0.02)

    @property
    def primitive_token_count(self):
        return self.n_ctx * self.max_slots_per_predicate

    def forward(self, slot_ids, bias):
        slot_ids = slot_ids.long()
        tokens = self.primitive_prompt_bank[slot_ids]
        tokens = tokens.reshape(slot_ids.shape[0], self.primitive_token_count, -1)
        if bias.shape[-1] != tokens.shape[-1]:
            raise RuntimeError(
                "Prompt bias dim {} must match CLIP token dim {}".format(
                    bias.shape[-1],
                    tokens.shape[-1],
                )
            )
        return tokens + bias.unsqueeze(1)

    def orthogonal_loss(self):
        slot_repr = F.normalize(self.primitive_prompt_bank.mean(dim=1), dim=-1)
        sim = slot_repr @ slot_repr.t()
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        return (sim - eye).pow(2).mean()

    def similarity_matrix(self):
        with torch.no_grad():
            slot_repr = F.normalize(self.primitive_prompt_bank.mean(dim=1), dim=-1)
            return slot_repr @ slot_repr.t()


class PrimitiveStage1VAE(nn.Module):
    def __init__(
        self,
        clip_model,
        num_slots=16,
        n_ctx=4,
        max_slots_per_predicate=4,
        feature_dim=None,
        hidden_dim=2048,
        latent_dim=512,
    ):
        super().__init__()
        clip_feature_dim = int(clip_model.text_projection.shape[1])
        token_dim = int(clip_model.ln_final.weight.shape[0])
        feature_dim = int(feature_dim or clip_feature_dim)
        self.clip_feature_dim = feature_dim
        self.clip_token_dim = token_dim
        self.clip_model = clip_model
        self.text_encoder = CLIPTextEncoder(clip_model)
        self.encoder = RelationVAEEncoder(feature_dim, hidden_dim, latent_dim)
        self.generator = PromptBiasGenerator(latent_dim, hidden_dim=4096, prompt_dim=token_dim)
        self.prompt_learner = PrimitivePromptLearner(
            clip_model,
            num_slots=num_slots,
            n_ctx=n_ctx,
            max_slots_per_predicate=max_slots_per_predicate,
        )
        self.latent_dim = latent_dim
        generator_out_dim = self.generator.net[-1].out_features
        if generator_out_dim != self.clip_token_dim:
            raise ValueError(
                "Generator output dim {} must match CLIP token dim {}".format(
                    generator_out_dim,
                    self.clip_token_dim,
                )
            )

        empty_tokens = clip.tokenize([""])
        self.sot_token = int(empty_tokens[0, 0])
        self.eot_token = int(empty_tokens[0, 1])

        for p in self.clip_model.parameters():
            p.requires_grad = False

    def encode_image(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        if features.dim() == 3:
            features = features[:, 0, :]
        return F.normalize(features.float(), dim=-1)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        return mean + torch.randn_like(std) * std

    def build_prompt(self, slot_ids, bias, triplet_texts):
        device = bias.device
        primitive_tokens = self.prompt_learner(slot_ids, bias)
        primitive_count = primitive_tokens.shape[1]
        context_length = 77
        suffix_len = context_length - 1 - primitive_count

        raw_tokens = clip.tokenize(triplet_texts, truncate=True).to(device)
        tokenized = torch.zeros(raw_tokens.shape[0], context_length, dtype=raw_tokens.dtype, device=device)
        tokenized[:, 0] = self.sot_token
        suffix = raw_tokens[:, 1:1 + suffix_len].clone()
        has_eot = suffix.eq(self.eot_token).any(dim=1)
        suffix[~has_eot, -1] = self.eot_token
        tokenized[:, 1 + primitive_count:] = suffix

        embeddings = self.clip_model.token_embedding(tokenized).float()
        embeddings[:, 1:1 + primitive_count] = primitive_tokens.float()
        return embeddings, tokenized

    def forward(self, target_features, slot_ids, triplet_texts, sample=True):
        if target_features.dim() == 3:
            target_features = target_features[:, 0, :]
        mean, log_var = self.encoder(target_features)
        z = self.reparameterize(mean, log_var) if sample else mean
        bias = self.generator(z)
        prompts, tokenized = self.build_prompt(slot_ids, bias, triplet_texts)
        recon = self.text_encoder(prompts, tokenized)
        return F.normalize(recon.float(), dim=-1), mean, log_var

    @torch.no_grad()
    def generate(self, slot_ids, triplet_texts, num_samples):
        device = self.primitive_prompt_bank_device
        if slot_ids.dim() == 1:
            slot_ids = slot_ids.unsqueeze(0).repeat(num_samples, 1)
        slot_ids = slot_ids.to(device)
        if len(triplet_texts) == 1 and num_samples > 1:
            triplet_texts = triplet_texts * num_samples
        z = torch.randn(num_samples, self.latent_dim, device=device)
        bias = self.generator(z)
        prompts, tokenized = self.build_prompt(slot_ids, bias, triplet_texts)
        features = self.text_encoder(prompts, tokenized)
        return F.normalize(features.float(), dim=-1)

    @property
    def primitive_prompt_bank_device(self):
        return self.prompt_learner.primitive_prompt_bank.device

    def trainable_state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "generator": self.generator.state_dict(),
            "prompt_learner": self.prompt_learner.state_dict(),
        }

    def load_trainable_state_dict(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.generator.load_state_dict(state["generator"])
        self.prompt_learner.load_state_dict(state["prompt_learner"])


def kl_loss(mean, log_var):
    return -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()


def reconstruction_loss(recon, target):
    return F.mse_loss(F.normalize(recon, dim=-1), F.normalize(target, dim=-1))


def build_clip_model(name, device):
    clip_model, preprocess = clip.load(name, device=device)
    clip_model = clip_model.float().eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model, preprocess
