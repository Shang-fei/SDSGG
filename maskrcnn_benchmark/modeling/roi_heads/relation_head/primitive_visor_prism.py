import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from .CLIP import clip


def slot_labels(slot_ids, num_slots):
    labels = torch.zeros(slot_ids.shape[0], num_slots, device=slot_ids.device, dtype=torch.float32)
    labels.scatter_(1, slot_ids.long().clamp(min=0, max=num_slots - 1), 1.0)
    return labels


def primitive_anchor_texts(mapping):
    anchors = []
    slots = mapping["slots"]
    for idx in range(int(mapping["num_slots"])):
        name = slots[str(idx)]["name"].replace("_", " ")
        anchors.append("a visual relation primitive of {}".format(name))
    return anchors


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def grad_reverse(x, weight=1.0):
    return GradientReverse.apply(x, weight)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class PrimitiveBank(nn.Module):
    def __init__(self, clip_model, mapping, residual_scale=0.1, random_anchors=False):
        super().__init__()
        self.num_slots = int(mapping["num_slots"])
        self.residual_scale = float(residual_scale)
        with torch.no_grad():
            if random_anchors:
                anchors = torch.randn(self.num_slots, int(clip_model.text_projection.shape[1]))
            else:
                tokenized = clip.tokenize(primitive_anchor_texts(mapping)).to(next(clip_model.parameters()).device)
                anchors = clip_model.encode_text(tokenized).float()
            anchors = F.normalize(anchors, dim=-1)
        self.register_buffer("anchors", anchors)
        self.delta = nn.Parameter(torch.zeros_like(anchors))

    def forward(self):
        return F.normalize(self.anchors + self.residual_scale * self.delta, dim=-1)

    def anchor_loss(self):
        return (1.0 - (self.forward() * self.anchors).sum(dim=-1)).mean()

    def diversity_loss(self, margin=0.85):
        primitives = self.forward()
        sim = primitives @ primitives.t()
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        return F.relu(sim[~eye] - margin).pow(2).mean()


class SparsePredicateComposer(nn.Module):
    def __init__(self, feature_dim=512, num_slots=16, top_m=4):
        super().__init__()
        self.top_m = int(top_m)
        self.logit_mlp = MLP(feature_dim, feature_dim, num_slots)
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, predicate_features, primitive_features):
        logits = self.logit_mlp(predicate_features)
        logits = logits + self.prior_scale * (predicate_features @ primitive_features.t())
        k = min(self.top_m, logits.shape[1])
        top_idx = logits.topk(k, dim=1).indices
        masked = logits.new_full(logits.shape, -1e4)
        masked.scatter_(1, top_idx, logits.gather(1, top_idx))
        weights = F.softmax(masked, dim=1)
        condition = F.normalize(weights @ primitive_features, dim=-1)
        return condition, weights, logits


class PrimitivePrototypeGenerator(nn.Module):
    def __init__(self, feature_dim=512, geom_dim=11, gate_rank=64, dropout=0.1):
        super().__init__()
        self.so_gate = MLP(feature_dim, feature_dim // 2, gate_rank)
        self.geom_proj = MLP(geom_dim, feature_dim // 2, gate_rank)
        self.decoder = MLP(feature_dim + gate_rank * 2, feature_dim * 2, feature_dim, dropout=dropout)

    def forward(self, primitive_condition, so_features, geometry, so_dropout=0.0, use_so=True):
        if self.training and so_dropout > 0:
            keep = torch.rand(so_features.shape[0], 1, device=so_features.device) > so_dropout
            so_features = so_features * keep.to(so_features.dtype)
        if not use_so:
            so_features = torch.zeros_like(so_features)
        gate = torch.tanh(self.so_gate(so_features))
        geom = torch.tanh(self.geom_proj(geometry))
        return F.normalize(self.decoder(torch.cat([primitive_condition, gate, geom], dim=-1)), dim=-1)


class PrimitiveActivationHead(nn.Module):
    def __init__(self, feature_dim=512, num_slots=16):
        super().__init__()
        self.head = MLP(feature_dim, feature_dim, num_slots)

    def forward(self, features):
        return self.head(features)


class ObjectPairAdversary(nn.Module):
    def __init__(self, feature_dim=512, num_obj_classes=151):
        super().__init__()
        self.subject = MLP(feature_dim, feature_dim, num_obj_classes)
        self.object = MLP(feature_dim, feature_dim, num_obj_classes)

    def forward(self, features, grl_weight=1.0):
        reversed_features = grad_reverse(features, grl_weight)
        return self.subject(reversed_features), self.object(reversed_features)


class VISORPRISM(nn.Module):
    def __init__(
        self,
        clip_model,
        mapping,
        num_obj_classes,
        top_m=4,
        geom_dim=11,
        random_anchors=False,
    ):
        super().__init__()
        feature_dim = int(clip_model.text_projection.shape[1])
        self.feature_dim = feature_dim
        self.num_slots = int(mapping["num_slots"])
        self.clip_model = clip_model
        self.primitive_bank = PrimitiveBank(clip_model, mapping, random_anchors=random_anchors)
        self.composer = SparsePredicateComposer(feature_dim, self.num_slots, top_m=top_m)
        self.generator = PrimitivePrototypeGenerator(feature_dim, geom_dim=geom_dim)
        self.activation_head = PrimitiveActivationHead(feature_dim, self.num_slots)
        self.so_only = MLP(feature_dim + geom_dim, feature_dim * 2, feature_dim)
        self.adversary = ObjectPairAdversary(feature_dim, num_obj_classes)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        for p in self.clip_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_image(self, images):
        features = self.clip_model.encode_image(images)
        if features.dim() == 3:
            features = features[:, 0, :]
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts, device):
        tokenized = clip.tokenize(texts, truncate=True).to(device)
        return F.normalize(self.clip_model.encode_text(tokenized).float(), dim=-1)

    def compose(self, predicate_features, primitive_override=None):
        primitive_features = self.primitive_bank() if primitive_override is None else primitive_override
        return self.composer(predicate_features, primitive_features)

    def generate(self, predicate_features, so_features, geometry, so_dropout=0.0, use_so=True, primitive_override=None):
        condition, weights, logits = self.compose(predicate_features, primitive_override=primitive_override)
        prototypes = self.generator(condition, so_features, geometry, so_dropout=so_dropout, use_so=use_so)
        return prototypes, weights, logits

    def generate_so_only(self, so_features, geometry):
        return F.normalize(self.so_only(torch.cat([so_features, geometry], dim=-1)), dim=-1)

    def trainable_state_dict(self):
        return {
            "primitive_bank": self.primitive_bank.state_dict(),
            "composer": self.composer.state_dict(),
            "generator": self.generator.state_dict(),
            "activation_head": self.activation_head.state_dict(),
            "so_only": self.so_only.state_dict(),
            "adversary": self.adversary.state_dict(),
            "logit_scale": self.logit_scale.detach().cpu(),
        }

    def load_trainable_state_dict(self, state):
        self.primitive_bank.load_state_dict(state["primitive_bank"])
        self.composer.load_state_dict(state["composer"])
        self.generator.load_state_dict(state["generator"])
        self.activation_head.load_state_dict(state["activation_head"])
        self.so_only.load_state_dict(state["so_only"])
        self.adversary.load_state_dict(state["adversary"])
        if "logit_scale" in state:
            self.logit_scale.data.copy_(state["logit_scale"].to(self.logit_scale.device))
