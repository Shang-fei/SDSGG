# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import math
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize, ToPILImage
from torchvision.transforms import functional as FF
from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from CLIP import clip
import time
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import os, sys
try:
    import yaml
except ImportError:
    yaml = None
curpath=os.path.dirname(__file__)
PRDCS_BASE = ['has', 'hanging from', 'from', 'in', 'walking on', 'watching', 'with', 'behind', 'for', 'to', 'belonging to', 'at', 'wearing', 'standing on', 'holding', 'riding', 'near', 'looking at', 'sitting on', 'on', 'over', 'in front of', 'between', 'made of', 'of', 'carrying', 'parked on', 'against', 'attached to', 'playing', 'covering', 'covered in', 'wears', 'above', 'under']
PRDCS_NOVEL = ['mounted on', 'says', 'part of', 'across', 'flying in', 'using', 'on back of', 'lying on', 'growing on', 'walking in', 'laying on', 'along', 'eating', 'and', 'painted on']
SEMAN = ["attached to", "carrying", "covered in", "covering", "eating", "flying in", "growing on",
                               "hanging from", "holding", "laying on", "looking at", "lying on", "mounted on",
                               "painted on", "parked on", "playing", "riding", "says", "sitting on", "standing on",
                               "using", "walking in", "walking on", "watching"]
# from utils_clip import *
def crop_and_resize(image, posi1, posi2):
    posi = torch.cat((torch.min(posi1[0:2], posi2[0:2]),
                      torch.max(posi1[2:], posi2[2:])), dim=0).int()


    cropped_image = FF.crop(image, posi[1], posi[0], posi[3] - posi[1], posi[2] - posi[0])  # top,left,height,width

    resized_image = FF.resize(cropped_image, (224, 224))

    return resized_image

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim2)

        nn.init.xavier_normal_(self.proj_q1.weight)
        nn.init.xavier_normal_(self.proj_k2.weight)
        nn.init.xavier_normal_(self.proj_v2.weight)
        nn.init.xavier_normal_(self.proj_o.weight)


        self.layer_norm = nn.LayerNorm(in_dim2)

    def forward(self, x1, x2, mask=None):  # q:x1-text k,v:x2-img

        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        x_cls = x2[:,0].clone()

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)

        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / (self.k_dim ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)

        output = self.proj_o(output)

        norm_output = self.layer_norm(output+x_cls.unsqueeze(1))

        norm_output = torch.mean(norm_output, dim=-2)

        return norm_output

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc1=nn.Linear(c_in, c_in // reduction, bias=False)
        self.relu1=nn.ReLU(inplace=True)
        self.cross_att=CrossAttention(in_dim1=512, in_dim2=c_in // reduction, k_dim=512, v_dim=c_in // reduction, num_heads=8)
        self.fc2=nn.Linear(c_in // reduction, c_in, bias=False)
        self.relu2=nn.ReLU(inplace=True)

    def forward(self, x_sub,x_obj):
        x_sub = self.relu1(self.fc1(x_sub))
        x_sub=self.cross_att(x_obj,x_sub)
        x_sub=self.relu2(self.fc2(x_sub))
        return x_sub

class MVA(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(512).half()
        self.adapter = Adapter(512, 4).half()
        self.linear=nn.Linear(1024, 512, bias=False).half()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, sub_features, obj_features, text_fea=None):
        adapted_features = self.adapter(sub_features, obj_features)
        if text_fea is not None:
            if text_fea.dim() == 1:
                text_features = text_fea.unsqueeze(0).expand(adapted_features.size(0), -1)
            elif text_fea.size(0) == 1 and adapted_features.size(0) != 1:
                text_features = text_fea.expand(adapted_features.size(0), -1)
            else:
                text_features = text_fea
            text_features = text_features.to(device=adapted_features.device, dtype=adapted_features.dtype)
            adapted_features = torch.cat([adapted_features, text_features], dim=-1)
            adapted_features = self.relu(self.linear(adapted_features))
        ratio = 0.5
        return ratio * adapted_features + (1 - ratio) * sub_features[:, 0, :]


class PrimitiveQueryActivationHead(nn.Module):
    def __init__(self, primitive_basis, num_heads=8, dropout=0.0):
        super(PrimitiveQueryActivationHead, self).__init__()
        hidden_dim = primitive_basis.shape[-1]
        self.primitive_queries = nn.Parameter(primitive_basis.float().clone())
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.activation_score = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.activation_score.weight, std=0.01)
        nn.init.constant_(self.activation_score.bias, 0.0)

    def forward(self, relation_tokens, tau):
        batch_size = relation_tokens.shape[0]
        primitive_queries = self.primitive_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended_features, _ = self.cross_attention(
            primitive_queries,
            relation_tokens.float(),
            relation_tokens.float(),
            need_weights=False,
        )
        primitive_features = self.query_norm(primitive_queries + attended_features)
        primitive_features = self.output_norm(primitive_features + self.ffn(primitive_features))
        primitive_logits = self.activation_score(primitive_features).squeeze(-1)
        primitive_activation = torch.sigmoid(primitive_logits / max(tau, 1e-6))
        return primitive_activation, primitive_logits


class RelationFeatureVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RelationFeatureVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def encode(self, x):
        hidden = self.encoder(x.float())
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        target = x.float()
        mu, logvar = self.encode(target)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, x):
        target = x.float()
        recon, mu, logvar = self.forward(target)
        recon_cos = 1.0 - F.cosine_similarity(
            F.normalize(recon, dim=-1),
            F.normalize(target, dim=-1),
            dim=-1,
        ).mean()
        recon_mse = F.mse_loss(recon, target)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return recon_cos + recon_mse, kl, mu, logvar, recon_cos.detach(), recon_mse.detach()


@registry.ROI_RELATION_PREDICTOR.register("RelationFeatureVAEPredictor")
class RelationFeatureVAEPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(RelationFeatureVAEPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.device = config.MODEL.DEVICE

        statistics = get_dataset_statistics(config)
        obj_classes = statistics["obj_classes"]
        rel_classes = statistics["rel_classes"]
        self.predicate_names = list(rel_classes)
        self.foreground_predicate_names = self.predicate_names[1:self.num_rel_cls]

        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        probe_cfg = config.MODEL.ROI_RELATION_HEAD.VAE_PROBE
        self.relation_dim = int(probe_cfg.RELATION_DIM)
        self.recon_loss_weight = float(probe_cfg.RECON_LOSS_WEIGHT)
        self.kl_loss_weight = float(probe_cfg.KL_LOSS_WEIGHT)
        self.foreground_only = bool(probe_cfg.FOREGROUND_ONLY)
        self.debug_interval = int(probe_cfg.DEBUG_INTERVAL)
        self.stats_csv = str(probe_cfg.STATS_CSV)
        latent_dim = int(probe_cfg.LATENT_DIM)
        geometry_dim = int(probe_cfg.GEOMETRY_DIM)

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for clip_parameter in self.clip_model.parameters():
            clip_parameter.requires_grad = False
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim, int(probe_cfg.HIDDEN_DIM)),
            nn.ReLU(inplace=True),
            nn.Linear(int(probe_cfg.HIDDEN_DIM), int(probe_cfg.HIDDEN_DIM)),
            nn.ReLU(inplace=True),
        )

        self.pair_projector = nn.Sequential(
            nn.Linear(512 * 3 + int(probe_cfg.HIDDEN_DIM), int(probe_cfg.HIDDEN_DIM)),
            nn.ReLU(inplace=True),
            nn.Linear(int(probe_cfg.HIDDEN_DIM), self.relation_dim),
        )
        self.vae = RelationFeatureVAE(
            input_dim=self.relation_dim,
            hidden_dim=int(probe_cfg.HIDDEN_DIM),
            latent_dim=latent_dim,
        )

        self.register_buffer("debug_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("predicate_count", torch.zeros(self.num_rel_cls - 1, dtype=torch.float32))
        self.register_buffer("mu_sum", torch.zeros(self.num_rel_cls - 1, latent_dim, dtype=torch.float32))
        self.register_buffer("logvar_sum", torch.zeros(self.num_rel_cls - 1, latent_dim, dtype=torch.float32))
        self.register_buffer("mu_second_sum", torch.zeros(self.num_rel_cls - 1, latent_dim, dtype=torch.float32))
        print(
            "RelationFeatureVAEPredictor: CLIP+geometry x_rel, relation_dim={}, latent_dim={}, "
            "geometry_dim={}, recon_weight={}, kl_weight={}, foreground_only={}".format(
                self.relation_dim,
                latent_dim,
                geometry_dim,
                self.recon_loss_weight,
                self.kl_loss_weight,
                self.foreground_only,
            )
        )

    def updata(self, mode):
        return

    def _crop_to_pil(self, image, box_a, box_b):
        crop = crop_and_resize(image, box_a, box_b)
        crop_array = crop[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        return Image.fromarray(np.uint8(crop_array))

    def _encode_clip_image_batch(self, pil_images, batch_size=128):
        if len(pil_images) == 0:
            return None
        features = []
        with torch.no_grad():
            for start_idx in range(0, len(pil_images), batch_size):
                image_tensor = torch.cat([
                    self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    for image in pil_images[start_idx:start_idx + batch_size]
                ], dim=0)
                image_features = self.clip_model.encode_image(image_tensor)
                if image_features.dim() == 3:
                    image_features = image_features[:, 0, :]
                features.append(image_features.float())
        return F.normalize(torch.cat(features, dim=0), dim=-1)

    def _geometry_features(self, proposal, pair_idx):
        boxes = proposal.bbox.float()
        subject_boxes = boxes[pair_idx[:, 0]]
        object_boxes = boxes[pair_idx[:, 1]]
        image_width, image_height = proposal.size
        eps = 1e-6
        width = float(image_width)
        height = float(image_height)
        image_area = max(width * height, eps)

        sx1, sy1, sx2, sy2 = subject_boxes.unbind(dim=-1)
        ox1, oy1, ox2, oy2 = object_boxes.unbind(dim=-1)
        sw = (sx2 - sx1).clamp_min(eps)
        sh = (sy2 - sy1).clamp_min(eps)
        ow = (ox2 - ox1).clamp_min(eps)
        oh = (oy2 - oy1).clamp_min(eps)
        scx = (sx1 + sx2) * 0.5
        scy = (sy1 + sy2) * 0.5
        ocx = (ox1 + ox2) * 0.5
        ocy = (oy1 + oy2) * 0.5

        inter_x1 = torch.max(sx1, ox1)
        inter_y1 = torch.max(sy1, oy1)
        inter_x2 = torch.min(sx2, ox2)
        inter_y2 = torch.min(sy2, oy2)
        inter_area = (inter_x2 - inter_x1).clamp_min(0.0) * (inter_y2 - inter_y1).clamp_min(0.0)
        subject_area = sw * sh
        object_area = ow * oh
        union_area = subject_area + object_area - inter_area

        return torch.stack([
            sx1 / width, sy1 / height, sx2 / width, sy2 / height,
            ox1 / width, oy1 / height, ox2 / width, oy2 / height,
            (scx - ocx) / width, (scy - ocy) / height,
            torch.log(sw / ow), torch.log(sh / oh),
            inter_area / union_area.clamp_min(eps),
            subject_area / image_area,
            object_area / image_area,
            union_area / image_area,
            subject_area / object_area.clamp_min(eps),
            (scy < ocy).float(),
            (scy > ocy).float(),
            (scx < ocx).float(),
            (scx > ocx).float(),
        ], dim=-1)

    def _build_relation_features(self, proposals, rel_pair_idxs, img):
        relation_inputs = []
        for image_idx, pair_idx in enumerate(rel_pair_idxs):
            num_rel = pair_idx.shape[0]
            if num_rel == 0:
                continue
            image = img[image_idx].unsqueeze(0)
            object_crops = [
                self._crop_to_pil(image, proposals[image_idx].bbox[box_idx], proposals[image_idx].bbox[box_idx])
                for box_idx in range(len(proposals[image_idx].bbox))
            ]
            union_crops = [
                self._crop_to_pil(image, proposals[image_idx].bbox[int(pair[0])], proposals[image_idx].bbox[int(pair[1])])
                for pair in pair_idx.detach().cpu().tolist()
            ]
            object_features = self._encode_clip_image_batch(object_crops)
            union_features = self._encode_clip_image_batch(union_crops)
            subject_features = object_features[pair_idx[:, 0]]
            object_pair_features = object_features[pair_idx[:, 1]]
            geometry_features = self._geometry_features(proposals[image_idx], pair_idx).to(
                device=subject_features.device,
                dtype=subject_features.dtype,
            )
            geometry_features = self.geometry_encoder(geometry_features.float())
            relation_inputs.append(torch.cat([
                subject_features.float(),
                object_pair_features.float(),
                union_features.float(),
                geometry_features.float(),
            ], dim=-1))
        if not relation_inputs:
            return self.mu_sum.new_zeros((0, self.relation_dim))
        relation_inputs = torch.cat(relation_inputs, dim=0)
        return F.normalize(self.pair_projector(relation_inputs.float()), dim=-1)

    def _update_latent_statistics(self, mu, logvar, labels):
        with torch.no_grad():
            foreground_mask = labels > 0
            if not foreground_mask.any():
                return
            foreground_mu = mu.detach()[foreground_mask].float()
            foreground_logvar = logvar.detach()[foreground_mask].float()
            foreground_labels = labels[foreground_mask].long() - 1
            for predicate_idx in foreground_labels.unique():
                class_mask = foreground_labels == predicate_idx
                class_mu = foreground_mu[class_mask]
                class_logvar = foreground_logvar[class_mask]
                idx = int(predicate_idx.item())
                self.mu_sum[idx] += class_mu.sum(dim=0)
                self.logvar_sum[idx] += class_logvar.sum(dim=0)
                self.mu_second_sum[idx] += class_mu.pow(2).sum(dim=0)
                self.predicate_count[idx] += float(class_mu.shape[0])

    def _format_latent_statistics(self, topk=8):
        valid_mask = self.predicate_count > 0
        if not valid_mask.any():
            return "no foreground predicate stats yet"
        mu_mean, logvar_mean, mu_var = self._compute_latent_statistics()
        predicate_mu_norm = mu_mean.norm(dim=-1)
        predicate_mu_var = mu_var.mean(dim=-1)
        masked_var = predicate_mu_var.masked_fill(~valid_mask, -1.0)
        top_values, top_indices = masked_var.topk(min(topk, int(valid_mask.sum().item())), dim=0)
        fragments = []
        for value, idx in zip(top_values, top_indices):
            pred_idx = int(idx.item())
            fragments.append(
                "{}:mu_norm={:.4f},mu_var={:.4f},logvar={:.4f},n={:.0f}".format(
                    self.foreground_predicate_names[pred_idx],
                    float(predicate_mu_norm[pred_idx].detach().cpu()),
                    float(value.detach().cpu()),
                    float(logvar_mean[pred_idx].mean().detach().cpu()),
                    float(self.predicate_count[pred_idx].detach().cpu()),
                )
            )
        return "; ".join(fragments)

    def _compute_latent_statistics(self):
        count = self.predicate_count.clamp_min(1.0).unsqueeze(-1)
        mu_mean = self.mu_sum / count
        logvar_mean = self.logvar_sum / count
        mu_second = self.mu_second_sum / count
        mu_var = (mu_second - mu_mean.pow(2)).clamp_min(0.0)
        return mu_mean, logvar_mean, mu_var

    def _save_latent_statistics(self):
        if not self.stats_csv:
            return
        valid_mask = self.predicate_count > 0
        if not valid_mask.any():
            return
        mu_mean, logvar_mean, mu_var = self._compute_latent_statistics()
        rows = []
        for predicate_idx, predicate_name in enumerate(self.foreground_predicate_names):
            if not bool(valid_mask[predicate_idx].item()):
                continue
            rows.append({
                "predicate": predicate_name,
                "count": float(self.predicate_count[predicate_idx].detach().cpu()),
                "mu_norm": float(mu_mean[predicate_idx].norm().detach().cpu()),
                "mu_var_mean": float(mu_var[predicate_idx].mean().detach().cpu()),
                "logvar_mean": float(logvar_mean[predicate_idx].mean().detach().cpu()),
            })
        stats_dir = os.path.dirname(self.stats_csv)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(self.stats_csv, index=False)

    def _maybe_print_debug(self, rec_cos, rec_mse, kl):
        if self.debug_interval <= 0:
            return
        self.debug_step += 1
        if int(self.debug_step.item()) % self.debug_interval != 0:
            return
        self._save_latent_statistics()
        print(
            "RelationFeatureVAEProbe step {}: rec_cos={:.4f}, rec_mse={:.4f}, kl={:.4f}, stats=[{}]".format(
                int(self.debug_step.item()),
                float(rec_cos.detach().cpu()),
                float(rec_mse.detach().cpu()),
                float(kl.detach().cpu()),
                self._format_latent_statistics(),
            )
        )

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None):
        if img is None:
            raise ValueError("RelationFeatureVAEPredictor requires images to build CLIP relation features.")
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        relation_features = self._build_relation_features(proposals, rel_pair_idxs, img)
        add_losses = {}

        if self.training and relation_features.shape[0] > 0:
            labels = cat(rel_labels, dim=0).to(relation_features.device).long()
            if self.foreground_only:
                train_mask = labels > 0
            else:
                train_mask = torch.ones_like(labels, dtype=torch.bool)
            if train_mask.any():
                rec_loss, kl_loss, mu, logvar, rec_cos, rec_mse = self.vae.loss(relation_features[train_mask])
                add_losses["loss_vae_rec"] = self.recon_loss_weight * rec_loss
                add_losses["loss_vae_kl"] = self.kl_loss_weight * kl_loss
                self._update_latent_statistics(mu, logvar, labels[train_mask])
                self._maybe_print_debug(rec_cos, rec_mse, kl_loss.detach())
            else:
                zero = relation_features.sum() * 0.0
                add_losses["loss_vae_rec"] = zero
                add_losses["loss_vae_kl"] = zero

        relation_logits = []
        start = 0
        for num_rel in num_rels:
            if num_rel == 0:
                relation_logits.append(roi_features.new_zeros((0, self.num_rel_cls)))
                continue
            placeholder = relation_features[start:start + num_rel].sum(dim=-1, keepdim=True) * 0.0
            relation_logits.append(placeholder.expand(-1, self.num_rel_cls))
            start += num_rel

        obj_dists = obj_dists.detach().split(num_objs, dim=0)
        return obj_dists, tuple(relation_logits), add_losses

@registry.ROI_RELATION_PREDICTOR.register("GQAClipPredictor")
class GQAClipPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(GQAClipPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        self.device=config.MODEL.DEVICE
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.adaper_clip1 = MVA()
        self.adaper_clip2 = MVA()

        self.obj_names = obj_classes

        a=time.time()
        self.texts1=[]
        self.texts2=[]
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        self.description_relation=pd.read_csv(curpath+"/description_relation_loss.csv")

        all_rel1=['Two or more objects partially overlap each other',
                'Interaction between objects',
                'A picture on another object',
                'Vertical positional relationship',
                'On a road',
                'May have contact behavior',
                'On a flat plane, it should appear balanced with no visible tilting',
                'Specialized structures resembling legs',
                'With hand, for grasping, support, or locomotion',
                'In a closed environment',
                'Connected or attached to a larger structure',
                'Have specialized equipment',
                'Belong to animal or human behavior',
                'Direct sensory organs toward the object of interest',
                'Posture indicating concentration',
                'Be a soft material',
                'Might have flat teeth or sharp teeth',
                'Development of leaves, stem, flowers',
                'Object is in motion',
                'Generally used for decoration',
                'Have a curvy body',
                'It has a positive directionality']
        all_rel2=['Each object is completely separate with clear space between them.',
                'Objects are isolated and have no interaction with each other.',
                'with the picture not being on or part of the object.',
                'Objects are positioned horizontally in relation to each other.',
                'Located off-road, in a non-road environment.',
                'They cannot touch or interact.',
                'Positioned on an uneven surface, appearing imbalanced or visibly tilted.',
                'Lacks structures resembling legs',
                'Lacks hands or similar structures',
                'Situated in an open, unrestricted environment.',
                'Completely detached and independent from any larger structure.',
                'Lacks any form of specialized equipment, entirely unaided or basic in form.',
                'possibly inanimate or mechanical.',
                'Sensory organs are directed away from or are indifferent to the object of interest.',
                'Posture indicative of distraction, disinterest, or relaxation.',
                'Composed of hard, rigid material.',
                'Possesses no teeth or structures resembling teeth.',
                'Lacks any botanical features.',
                'Object remains stationary',
                'not intended for decorative purposes.',
                'Have a straight body',
                'It has a negative directionality']

        self.base=[0,31, 48, 30, 29, 22, 8, 23, 21, 1, 50, 40, 43, 38, 41, 11, 46, 6, 13, 35, 47, 12]
        self.novel=[0,26, 14, 25, 15, 19, 44, 45, 18, 28, 7]

        mode="base"

        if mode=="base":
            self.description_relation = self.description_relation.iloc[self.base, 1:]
        elif mode=="novel":
            self.description_relation = self.description_relation.iloc[self.novel, 1:]

        self.description_relation=self.description_relation.applymap(lambda x: [int(s) for s in x.split(',')])
        self.description_relation=np.array(self.description_relation)
        self.description_relation = np.array([[np.array(item) for item in inner_list] for inner_list in self.description_relation])
        self.description_relation=torch.Tensor(self.description_relation).to(self.device)

        with torch.no_grad():

            text1=clip.tokenize( ["a photo of "+rel for rel in all_rel1]).to(self.device)
            text_features1 = self.clip_model.encode_text(text1)
            self.text_features1=text_features1

            text2=clip.tokenize( ["a photo of "+rel for rel in all_rel2]).to(self.device)
            text_features2 = self.clip_model.encode_text(text2)
            self.text_features2=text_features2

            text3=clip.tokenize(["a photo of subject " for x in self.obj_names]).to(self.device)
            text_features3 = self.clip_model.encode_text(text3)
            self.text_features3=text_features3

            text4=clip.tokenize(["a photo of object "  for x in self.obj_names]).to(self.device)
            text_features4 = self.clip_model.encode_text(text4)
            self.text_features4=text_features4

            self.texts5=[]

            for obj in self.obj_names:
                text5 = clip.tokenize(["a photo of " + tex for tex in list(self.prompt)]).to(
                    self.device)
                text_features5 = self.clip_model.encode_text(text5)
                text_features5 = text_features5
                self.texts5.append(text_features5.detach().cpu().numpy())

        b=time.time()
        print('init complete : '+str(b-a))
        self.linear1=nn.Linear(1024,512, bias=False).to(self.device).half()

    def updata(self,mode):
        print("now is "+mode)
        self.description_relation = pd.read_csv(
            curpath+"/description_relation_loss.csv")
        if mode=="base":
            self.description_relation = self.description_relation.iloc[self.base, 1:]
        elif mode=="novel":
            self.description_relation = self.description_relation.iloc[self.novel, 1:]

        self.description_relation=self.description_relation.applymap(lambda x: [int(s) for s in x.split(',')])
        self.description_relation=np.array(self.description_relation)
        self.description_relation = np.array([[np.array(item) for item in inner_list] for inner_list in self.description_relation])
        self.description_relation=torch.Tensor(self.description_relation).to(self.device)

        with torch.no_grad():
            self.texts5=[]

            for obj in self.obj_names:
                text5 = clip.tokenize(["a photo of " + tex for tex in list(self.prompt)]).to(
                    self.device)
                text_features5 = self.clip_model.encode_text(text5)
                text_features5 = text_features5
                self.texts5.append(text_features5.detach().cpu().numpy())

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,img=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        ##这里要思考如何使用联合box，boxlist_union操作注意print(list(self.obj_names)[rel_pair_idxs[0][0][0]])

        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)

        rel_dists=[]
        for i in range(len(num_rels)):
            rel_dist_per_batch=[]
            union_imges=[]
            image_features=[]
            with torch.no_grad():
                for j in range(len(proposals[i].bbox)):
                    union_img = crop_and_resize(img[i].unsqueeze(0), proposals[i].bbox[j], proposals[i].bbox[j])
                    iimg = union_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    iimg = Image.fromarray(np.uint8(iimg))
                    union_img = self.clip_preprocess(iimg).unsqueeze(0).to(self.device)
                    image_features.append(union_img)
                image_features = torch.cat(image_features)
                image_features = self.clip_model.encode_image(image_features)

            for la_count,rel_index in enumerate(rel_pair_idxs[i]):

                obj_n1,obj_n2=obj_preds[i][rel_index[0]],obj_preds[i][rel_index[1]]

                text_features1=self.text_features1
                text_features2=self.text_features2

                text_sub=self.text_features3[obj_n1]
                text_obj=self.text_features4[obj_n2]
                cross_output1=self.adaper_clip1(image_features[rel_index[0]].unsqueeze(0),image_features[rel_index[1]].unsqueeze(0),text_sub)
                cross_output2=self.adaper_clip2(image_features[rel_index[1]].unsqueeze(0),image_features[rel_index[0]].unsqueeze(0),text_obj)

                cross_output=(cross_output1+cross_output2)/2

                similarity1 = ((cross_output/ cross_output.norm(dim=-1, keepdim=True)) @ (text_features1/text_features1.norm(dim=-1, keepdim=True)).T)

                similarity2 = ((cross_output/ cross_output.norm(dim=-1, keepdim=True)) @ (text_features2/text_features2.norm(dim=-1, keepdim=True)).T)


                if self.adaper_clip1.training:

                    probs=(similarity1-similarity2)/0.05
                    image_features_clip=(image_features[rel_index[0]][0].unsqueeze(0)+image_features[rel_index[1]][0].unsqueeze(0))/2
                    similarit_origin_1=((image_features_clip/image_features_clip.norm(dim=-1, keepdim=True)) @
                                 (text_features1/text_features1.norm(dim=-1, keepdim=True)).T)

                    similarit_origin_2 = ((image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)) @
                                   (text_features2 / text_features2.norm(dim=-1, keepdim=True)).T)
                    similarit_origin=(similarit_origin_1-similarit_origin_2)/0.05

                    probs=torch.cat([probs,similarit_origin]).unsqueeze(0)


                else:
                    similarity_delta=(similarity1-similarity2)/0.05

                    probs=self.description_relation[:,2]*(similarity_delta)

                    probs = (probs.sum(-1) ).unsqueeze(0)

                    text_features5 = torch.Tensor(self.texts5[2]).to(self.device).half()
                    similarity31 = ((image_features[rel_index[0]][0].unsqueeze(0)/image_features[rel_index[0]][0].unsqueeze(0).norm(dim=-1, keepdim=True)) @ (text_features5/text_features5.norm(dim=-1, keepdim=True)).T/0.05)
                    similarity32 = ((image_features[rel_index[1]][0].unsqueeze(0)/image_features[rel_index[1]][0].unsqueeze(0).norm(dim=-1, keepdim=True)) @ (text_features5/text_features5.norm(dim=-1, keepdim=True)).T/0.05)
                    similarity3=(similarity31+similarity32)/2

                    probs=probs*0.2+similarity3*0.8


                rel_dist_per_batch.append(probs)

            rel_dist_per_batch=torch.cat(rel_dist_per_batch)

            rel_dists.append(rel_dist_per_batch)



        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}


        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("ClipPredictor")
class ClipPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(ClipPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        self.device=config.MODEL.DEVICE
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.adaper_clip1 = MVA()
        self.adaper_clip2 = MVA()
        self.obj_names = obj_classes
        a=time.time()
        self.texts1=[]
        self.texts2=[]
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        self.description_relation=pd.read_csv(curpath+"/description_relation.csv")

        all_rel1=['Two or more objects partially overlap each other',
                'Interaction between objects',
                'A picture on another object',
                'Vertical positional relationship',
                'On a road',
                'May have contact behavior',
                'On a flat plane, it should appear balanced with no visible tilting',
                'Specialized structures resembling legs',
                'With hand, for grasping, support, or locomotion',
                'In a closed environment',
                'Connected or attached to a larger structure',
                'Have specialized equipment',
                'Belong to animal or human behavior',
                'Direct sensory organs toward the object of interest',
                'Posture indicating concentration',
                'Be a soft material',
                'Might have flat teeth or sharp teeth',
                'Development of leaves, stem, flowers',
                'Object is in motion',
                'Generally used for decoration',
                'Have a curvy body',
                'It has a positive directionality']
        all_rel2=['Each object is completely separate with clear space between them.',
                'Objects are isolated and have no interaction with each other.',
                'with the picture not being on or part of the object.',
                'Objects are positioned horizontally in relation to each other.',
                'Located off-road, in a non-road environment.',
                'They cannot touch or interact.',
                'Positioned on an uneven surface, appearing imbalanced or visibly tilted.',
                'Lacks structures resembling legs',
                'Lacks hands or similar structures',
                'Situated in an open, unrestricted environment.',
                'Completely detached and independent from any larger structure.',
                'Lacks any form of specialized equipment, entirely unaided or basic in form.',
                'possibly inanimate or mechanical.',
                'Sensory organs are directed away from or are indifferent to the object of interest.',
                'Posture indicative of distraction, disinterest, or relaxation.',
                'Composed of hard, rigid material.',
                'Possesses no teeth or structures resemblaing teeth.',
                'Lacks any botanical features.',
                'Object remains stationary',
                'not intended for decorative purposes.',
                'Have a straight body',
                'It has a negative directionality']
        self.id_dict={'__background__': 0, 'above': 1, 'across': 2, 'against': 3, 'along': 4, 'and': 5, 'at': 6, 'attached to': 7, 'behind': 8, 'belonging to': 9, 'between': 10, 'carrying': 11, 'covered in': 12, 'covering': 13, 'eating': 14, 'flying in': 15, 'for': 16, 'from': 17, 'growing on': 18, 'hanging from': 19, 'has': 20, 'holding': 21, 'in': 22, 'in front of': 23, 'laying on': 24, 'looking at': 25, 'lying on': 26, 'made of': 27, 'mounted on': 28, 'near': 29, 'of': 30, 'on': 31, 'on back of': 32, 'over': 33, 'painted on': 34, 'parked on': 35, 'part of': 36, 'playing': 37, 'riding': 38, 'says': 39, 'sitting on': 40, 'standing on': 41, 'to': 42, 'under': 43, 'using': 44, 'walking in': 45, 'walking on': 46, 'watching': 47, 'wearing': 48, 'wears': 49, 'with': 50}

        self.base=[0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.PRDCS_BASE)]
        self.novel=[0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.PRDCS_NOVEL)]

        self.semantic = [0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.SEMAN)]
        mode="base"

        if mode=="base":

            self.description_relation = self.description_relation.iloc[self.base, 1:]

            self.sub_filter_novel = pd.read_csv(
            curpath+"/filter_total.csv").iloc[self.base, 1:]
        elif mode=="novel":
            self.description_relation = self.description_relation.iloc[self.novel, 1:]
            self.sub_filter_novel = pd.read_csv(
            curpath+"/filter_total.csv").iloc[self.novel, 1:]
        elif mode=="total":
            self.description_relation = self.description_relation.iloc[:, 1:]
            self.sub_filter_novel = pd.read_csv(
            curpath+"/filter_total.csv").iloc[:, 1:]
        elif mode=="semantic":
            self.description_relation = self.description_relation.iloc[self.semantic, 1:]
            self.sub_filter_novel = pd.read_csv(
            curpath+"/filter_total.csv").iloc[self.semantic, 1:]

        self.description_relation=self.description_relation.applymap(lambda x: [int(s) for s in x.split(',')])
        self.description_relation=np.array(self.description_relation)
        self.description_relation = np.array([[np.array(item) for item in inner_list] for inner_list in self.description_relation])
        self.description_relation=torch.Tensor(self.description_relation).to(self.device)

        with torch.no_grad():

            text1=clip.tokenize( ["a photo of "+rel for rel in all_rel1]).to(self.device)
            text_features1 = self.clip_model.encode_text(text1)
            self.text_features1=text_features1

            text2=clip.tokenize( ["a photo of "+rel for rel in all_rel2]).to(self.device)
            text_features2 = self.clip_model.encode_text(text2)
            self.text_features2=text_features2

            text3=clip.tokenize(["a photo of subject " for x in self.obj_names]).to(self.device)
            text_features3 = self.clip_model.encode_text(text3)
            self.text_features3=text_features3

            text4=clip.tokenize(["a photo of object "  for x in self.obj_names]).to(self.device)
            text_features4 = self.clip_model.encode_text(text4)
            self.text_features4=text_features4

            self.texts5=[]

            for obj in self.obj_names:
                text5 = clip.tokenize(["a photo of " + tex for tex in list(self.sub_filter_novel[obj])]).to(
                    self.device)
                text_features5 = self.clip_model.encode_text(text5)
                text_features5 = text_features5
                self.texts5.append(text_features5.detach().cpu().numpy())

        b=time.time()
        print('init complete : '+str(b-a))

        self.zhangliang=[]
        self.count=0

        self.linear1=nn.Linear(1024,512, bias=False).to(self.device).half()

    def updata(self,mode):
        print("now is "+mode)
        self.description_relation = pd.read_csv(
            curpath+"/description_relation.csv")
        if mode == "base":

            self.description_relation = self.description_relation.iloc[self.base, 1:]

            self.sub_filter_novel = pd.read_csv(
                curpath+"/filter_total.csv").iloc[
                                    self.base, 1:]
        elif mode == "novel":
            self.description_relation = self.description_relation.iloc[self.novel, 1:]
            self.sub_filter_novel = pd.read_csv(
                curpath+"/filter_total.csv").iloc[
                                    self.novel, 1:]
        elif mode == "total":
            self.description_relation = self.description_relation.iloc[:, 1:]
            self.sub_filter_novel = pd.read_csv(
                curpath+"/filter_total.csv").iloc[
                                    :, 1:]
        elif mode == "semantic":
            self.description_relation = self.description_relation.iloc[self.semantic, 1:]
            self.sub_filter_novel = pd.read_csv(
                curpath+"/filter_total.csv").iloc[
                                    self.semantic, 1:]

        print(self.description_relation)

        self.description_relation=self.description_relation.applymap(lambda x: [int(s) for s in x.split(',')])
        self.description_relation=np.array(self.description_relation)
        self.description_relation = np.array([[np.array(item) for item in inner_list] for inner_list in self.description_relation])
        self.description_relation=torch.Tensor(self.description_relation).to(self.device)

        with torch.no_grad():
            self.texts5=[]

            for obj in self.obj_names:
                text5 = clip.tokenize(["a photo of " + tex for tex in list(self.sub_filter_novel[obj])]).to(
                    self.device)

                timing = []

                a = time.time()

                text_features5 = self.clip_model.encode_text(text5)
                text_features5 = text_features5
                self.texts5.append(text_features5.detach().cpu().numpy())

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,img=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        ##这里要思考如何使用联合box，boxlist_union操作注意print(list(self.obj_names)[rel_pair_idxs[0][0][0]])

        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)


        rel_dists=[]
        for i in range(len(num_rels)):
            rel_dist_per_batch=[]
            union_imges=[]
            image_tensor=[]
            with torch.no_grad():
                for j in range(len(proposals[i].bbox)):
                    union_img = crop_and_resize(img[i].unsqueeze(0), proposals[i].bbox[j], proposals[i].bbox[j])
                    iimg = union_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    iimg = Image.fromarray(np.uint8(iimg))
                    union_img = self.clip_preprocess(iimg).unsqueeze(0).to(self.device)
                    image_tensor.append(union_img)
                image_tensor = torch.cat(image_tensor)

                image_features = self.clip_model.encode_image(image_tensor)

            for la_count,rel_index in enumerate(rel_pair_idxs[i]):

                obj_n1,obj_n2=obj_preds[i][rel_index[0]],obj_preds[i][rel_index[1]]#two object names

                text_features1=self.text_features1
                text_features2=self.text_features2

                text_sub=self.text_features3[obj_n1]
                text_obj=self.text_features4[obj_n2]



                cross_output1=self.adaper_clip1(image_features[rel_index[0]].unsqueeze(0),image_features[rel_index[1]].unsqueeze(0),text_sub)

                cross_output2=self.adaper_clip2(image_features[rel_index[1]].unsqueeze(0),image_features[rel_index[0]].unsqueeze(0),text_obj)

                cross_output=(cross_output1+cross_output2)/2

                similarity1 = ((cross_output/ cross_output.norm(dim=-1, keepdim=True)) @ (text_features1/text_features1.norm(dim=-1, keepdim=True)).T)

                similarity2 = ((cross_output/ cross_output.norm(dim=-1, keepdim=True)) @ (text_features2/text_features2.norm(dim=-1, keepdim=True)).T)


                if self.adaper_clip1.training:

                    probs=(similarity1-similarity2)/0.05
                    image_features_clip=(image_features[rel_index[0]][0].unsqueeze(0)+image_features[rel_index[1]][0].unsqueeze(0))/2
                    similarit_origin_1=((image_features_clip/image_features_clip.norm(dim=-1, keepdim=True)) @
                                 (text_features1/text_features1.norm(dim=-1, keepdim=True)).T)

                    similarit_origin_2 = ((image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)) @
                                   (text_features2 / text_features2.norm(dim=-1, keepdim=True)).T)
                    similarit_origin=(similarit_origin_1-similarit_origin_2)/0.05

                    probs=torch.cat([probs,similarit_origin]).unsqueeze(0)


                else:
                    similarity_delta=(similarity1-similarity2)/0.05

                    probs=self.description_relation[:,obj_n1]*(similarity_delta)

                    probs = (probs.sum(-1) ).unsqueeze(0)

                    text_features5 = torch.Tensor(self.texts5[obj_n1]).to(self.device).half()
                    similarity31 = ((image_features[rel_index[0]][0].unsqueeze(0)/image_features[rel_index[0]][0].unsqueeze(0).norm(dim=-1, keepdim=True)) @ (text_features5/text_features5.norm(dim=-1, keepdim=True)).T/0.05)
                    similarity32 = ((image_features[rel_index[1]][0].unsqueeze(0)/image_features[rel_index[1]][0].unsqueeze(0).norm(dim=-1, keepdim=True)) @ (text_features5/text_features5.norm(dim=-1, keepdim=True)).T/0.05)
                    similarity3=(similarity31+similarity32)/2

                    probs=probs*0.2+similarity3*0.8
                rel_dist_per_batch.append(probs)

            rel_dist_per_batch=torch.cat(rel_dist_per_batch)

            rel_dists.append(rel_dist_per_batch)



        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("SemanticBankGaussianPredictor")
class SemanticBankGaussianPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(SemanticBankGaussianPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None

        self.device = config.MODEL.DEVICE

        primitive_cfg = config.MODEL.ROI_RELATION_HEAD.PRIMITIVE_BANK
        primitive_config = self._load_primitive_config(primitive_cfg.CONFIG_PATH)
        primitive_settings = primitive_config.get("settings", {})

        self.w_source = str(self._cfg_value(primitive_cfg, primitive_settings, "W_SOURCE", "w_source", "text_similarity"))
        self.text_similarity_tau = float(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_SIMILARITY_TAU", "text_similarity_tau", 0.20
        ))
        self.text_similarity_topk = int(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_SIMILARITY_TOPK", "text_similarity_topk", 5
        ))
        self.text_similarity_threshold = float(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_SIMILARITY_THRESHOLD", "text_similarity_threshold", -1.0
        ))
        self.visual_activation_tau = float(self._cfg_value(
            primitive_cfg, primitive_settings, "VISUAL_ACTIVATION_TAU", "visual_activation_tau", 0.20
        ))
        self.composition_logit_scale = float(self._cfg_value(
            primitive_cfg, primitive_settings, "COMPOSITION_LOGIT_SCALE", "composition_logit_scale", 10.0
        ))
        self.foreground_only_loss = bool(self._cfg_value(
            primitive_cfg, primitive_settings, "FOREGROUND_ONLY_LOSS", "foreground_only_loss", True
        ))
        self.debug_interval = int(self._cfg_value(
            primitive_cfg, primitive_settings, "DEBUG_INTERVAL", "debug_interval", 100
        ))

        statistics = get_dataset_statistics(config)
        obj_classes = statistics["obj_classes"]
        rel_classes = statistics["rel_classes"]
        self.obj_names = list(obj_classes)
        self.predicate_names = list(rel_classes)
        self.num_active_predicates = self.num_rel_cls
        self.foreground_predicate_names = self.predicate_names[1:self.num_active_predicates]
        if len(self.foreground_predicate_names) != self.num_active_predicates - 1:
            raise ValueError(
                "Relation class mismatch: NUM_CLASSES={} but dataset provides {} relation names.".format(
                    self.num_active_predicates,
                    len(self.predicate_names),
                )
            )

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        for clip_parameter in self.clip_model.parameters():
            clip_parameter.requires_grad = False
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        self.subject_adapter = MVA()
        self.object_adapter = MVA()
        self.background_classifier = nn.Linear(512, 1).to(self.device)
        self.active_mode = "base"

        resources = primitive_config.get("resources", {})
        primitive_names, primitive_prompts, predicate_descriptions = self._load_primitive_resources(resources)
        self.primitive_names = primitive_names
        primitive_basis_init = self._encode_prompt_groups(primitive_prompts)
        predicate_description_features = self._encode_prompt_groups(predicate_descriptions)
        predicate_weights, primitive_basis, text_losses = self._build_text_anchor(
            predicate_description_features,
            primitive_basis_init,
        )

        self.register_buffer("primitive_basis", primitive_basis.clone())

        self.num_primitives = primitive_basis.shape[0]
        predicate_entropy = self._compute_weight_entropy(predicate_weights)
        self.register_buffer("predicate_primitive_weights", predicate_weights.float())
        self.register_buffer("predicate_entropy", predicate_entropy.float())
        self.register_buffer("visual_mean_stat", torch.zeros_like(predicate_weights.float()))
        self.register_buffer("visual_second_moment_stat", torch.zeros_like(predicate_weights.float()))
        self.register_buffer("predicate_count", torch.zeros(self.num_active_predicates - 1, dtype=torch.float32))
        self.register_buffer("debug_step", torch.zeros((), dtype=torch.long))

        self.primitive_activation_head = PrimitiveQueryActivationHead(primitive_basis).to(self.device)

        subject_role_prompts = ["a photo of subject {}".format(obj_name) for obj_name in self.obj_names]
        object_role_prompts = ["a photo of object {}".format(obj_name) for obj_name in self.obj_names]
        self.register_buffer("subject_role_text_features", self._encode_text_features(subject_role_prompts).float())
        self.register_buffer("object_role_text_features", self._encode_text_features(object_role_prompts).float())
        print(
            "SemanticBankGaussianPredictor primitive-query mode: {} primitives, {} predicates, "
            "w_source={}, text_tau={}, topk={}, visual_tau={}, scale={}, foreground_only_loss={}".format(
                self.num_primitives,
                self.num_active_predicates,
                self.w_source,
                self.text_similarity_tau,
                self.text_similarity_topk,
                self.visual_activation_tau,
                self.composition_logit_scale,
                self.foreground_only_loss,
            )
        )

    def _cfg_value(self, cfg_node, settings, cfg_key, yaml_key, default):
        if hasattr(cfg_node, cfg_key):
            return getattr(cfg_node, cfg_key)
        return settings.get(yaml_key, default)

    def _load_primitive_config(self, config_path):
        if yaml is None:
            raise ImportError("PyYAML is required to load primitive predictor YAML config.")
        if os.path.isabs(config_path):
            resolved_path = config_path
        else:
            repo_root = os.path.abspath(os.path.join(curpath, "../../../../"))
            resolved_path = os.path.join(repo_root, config_path)
        with open(resolved_path, "r") as config_file:
            primitive_config = yaml.safe_load(config_file)
        if not primitive_config or "resources" not in primitive_config:
            raise ValueError("primitive config must define resources")
        return primitive_config

    def _resolve_resource_path(self, resource_path):
        if os.path.isabs(resource_path):
            return resource_path
        repo_root = os.path.abspath(os.path.join(curpath, "../../../../"))
        return os.path.join(repo_root, resource_path)

    def _encode_text_features(self, prompts, batch_size=256):
        features = []
        with torch.no_grad():
            for start_idx in range(0, len(prompts), batch_size):
                prompt_batch = prompts[start_idx:start_idx + batch_size]
                text_tokens = clip.tokenize(prompt_batch).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = F.normalize(text_features.float(), dim=-1)
                features.append(text_features)
        return torch.cat(features, dim=0)

    def _encode_prompt_groups(self, prompt_groups):
        group_features = []
        for prompts in prompt_groups:
            prompt_features = self._encode_text_features(prompts)
            group_features.append(F.normalize(prompt_features.mean(dim=0, keepdim=True), dim=-1))
        return torch.cat(group_features, dim=0)

    def _load_primitive_resources(self, resources):
        primitive_csv = pd.read_csv(self._resolve_resource_path(resources["primitive_prototypes_csv"])).fillna("")
        predicate_desc_csv = pd.read_csv(self._resolve_resource_path(resources["predicate_descriptions_csv"])).fillna("")
        primitive_names = primitive_csv["primitive"].tolist()
        primitive_prompts = []
        for _, row in primitive_csv.iterrows():
            prompts = [str(row[col]).strip() for col in primitive_csv.columns if col != "primitive" and str(row[col]).strip()]
            primitive_prompts.append(prompts)

        predicate_description_map = {
            str(row["predicate"]).strip(): [
                str(row[col]).strip() for col in predicate_desc_csv.columns if col != "predicate" and str(row[col]).strip()
            ]
            for _, row in predicate_desc_csv.iterrows()
        }

        predicate_descriptions = []
        for predicate_name in self.foreground_predicate_names:
            if predicate_name not in predicate_description_map:
                raise ValueError("Missing predicate description for {}".format(predicate_name))
            predicate_descriptions.append(predicate_description_map[predicate_name])

        return primitive_names, primitive_prompts, predicate_descriptions

    def _build_text_anchor(self, predicate_features, primitive_basis_init):
        if self.w_source == "text_similarity":
            similarity = predicate_features.float() @ primitive_basis_init.float().t()
            weights = self._compute_sparse_similarity_weights(similarity)
            losses = self._compute_text_anchor_losses(predicate_features, primitive_basis_init, weights)
            self._print_text_similarity_summary(losses, weights)
            return weights.detach(), primitive_basis_init.detach(), losses
        raise ValueError("Primitive-query minimal predictor only supports W_SOURCE=text_similarity.")

    def _compute_sparse_similarity_weights(self, similarity):
        mask = torch.ones_like(similarity, dtype=torch.bool)
        if self.text_similarity_topk > 0 and self.text_similarity_topk < similarity.shape[-1]:
            _, topk_indices = torch.topk(similarity, self.text_similarity_topk, dim=-1)
            mask = torch.zeros_like(similarity, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
        if self.text_similarity_threshold > -1.0:
            mask = mask & (similarity >= self.text_similarity_threshold)
        empty_mask = ~mask.any(dim=-1)
        if empty_mask.any():
            fallback_indices = similarity[empty_mask].argmax(dim=-1, keepdim=True)
            mask[empty_mask] = False
            mask[empty_mask].scatter_(1, fallback_indices, True)
        masked_similarity = similarity.masked_fill(~mask, -1e4)
        return F.softmax(masked_similarity / max(self.text_similarity_tau, 1e-6), dim=-1)

    def _print_text_similarity_summary(self, losses, weights):
        with torch.no_grad():
            entropy = self._compute_weight_entropy(weights.float())
            max_weight = weights.float().max(dim=-1)[0]
            active_count = (weights.float() > 1e-6).float().sum(dim=-1)
            semantic_recovery = 1.0 - losses["recon"]
            print(
                "TextSemantic final: recovery {:.4f}, recovery_min {:.4f}, recovery_max {:.4f}, "
                "recovery_mean {:.4f}, recovery_std {:.4f}, W_min {:.4f}, W_max_global {:.4f}, "
                "W_mean {:.4f}, W_active {:.2f}/{:.2f}/{:.2f}, W_entropy {:.4f}/{:.4f}, "
                "W_max {:.4f}/{:.4f}, orth {:.4f}".format(
                    float(semantic_recovery.mean().detach().cpu()) if semantic_recovery.dim() > 0 else float(semantic_recovery.detach().cpu()),
                    float(losses["recovery_per_predicate"].min().detach().cpu()),
                    float(losses["recovery_per_predicate"].max().detach().cpu()),
                    float(losses["recovery_per_predicate"].mean().detach().cpu()),
                    float(losses["recovery_per_predicate"].std(unbiased=False).detach().cpu()),
                    float(weights.float().min().detach().cpu()),
                    float(weights.float().max().detach().cpu()),
                    float(weights.float().mean().detach().cpu()),
                    float(active_count.mean().detach().cpu()),
                    float(active_count.min().detach().cpu()),
                    float(active_count.max().detach().cpu()),
                    float(entropy.mean().detach().cpu()),
                    float(entropy.std(unbiased=False).detach().cpu()),
                    float(max_weight.mean().detach().cpu()),
                    float(max_weight.max().detach().cpu()),
                    float(losses["orth"].detach().cpu()),
                )
            )

    def _compute_text_anchor_losses(self, predicate_features, primitive_basis, weights):
        normalized_basis = F.normalize(primitive_basis.float(), dim=-1)
        reconstructed = F.normalize(weights.float() @ normalized_basis, dim=-1)
        recovery_per_predicate = (reconstructed * predicate_features.float()).sum(dim=-1)
        recon_loss = (1.0 - recovery_per_predicate).mean()
        sparse_loss = self._compute_weight_entropy(weights.float()).mean()
        gram = normalized_basis @ normalized_basis.t()
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        orth_loss = (gram - identity).pow(2).mean()
        return {
            "recon": recon_loss,
            "sparse": sparse_loss,
            "orth": orth_loss,
            "recovery_per_predicate": recovery_per_predicate.detach(),
        }

    def _compute_weight_entropy(self, weights):
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return entropy / math.log(float(weights.shape[-1]))

    def _compute_visual_variance(self):
        count = self.predicate_count.clamp_min(1.0).unsqueeze(-1)
        visual_mean = self.visual_mean_stat / count
        visual_second = self.visual_second_moment_stat / count
        visual_variance = (visual_second - visual_mean.pow(2)).clamp(min=0.0)
        return visual_mean, visual_variance

    def _compute_composition_logits(self, primitive_activation):
        return self.composition_logit_scale * (
            primitive_activation.float() @ self.predicate_primitive_weights.float().t()
        )

    def _update_visual_statistics(self, primitive_activation, relation_labels):
        if relation_labels is None:
            return
        with torch.no_grad():
            foreground_mask = relation_labels > 0
            if not foreground_mask.any():
                return
            foreground_activation = primitive_activation.detach()[foreground_mask].float()
            foreground_labels = relation_labels[foreground_mask].long() - 1
            for predicate_idx in foreground_labels.unique():
                class_mask = foreground_labels == predicate_idx
                class_activation = foreground_activation[class_mask]
                idx = int(predicate_idx.item())
                self.visual_mean_stat[idx] += class_activation.sum(dim=0)
                self.visual_second_moment_stat[idx] += class_activation.pow(2).sum(dim=0)
                self.predicate_count[idx] += float(class_activation.shape[0])

    def _encode_clip_image_batch(self, pil_images, batch_size=128):
        features = []
        with torch.no_grad():
            for start_idx in range(0, len(pil_images), batch_size):
                image_tensor = torch.cat([
                    self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    for image in pil_images[start_idx:start_idx + batch_size]
                ], dim=0)
                features.append(self.clip_model.encode_image(image_tensor))
        return torch.cat(features, dim=0)

    def _crop_to_pil(self, image, box_a, box_b):
        crop = crop_and_resize(image, box_a, box_b)
        crop_array = crop[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        return Image.fromarray(np.uint8(crop_array))

    def _pool_clip_features(self, clip_features):
        if clip_features.dim() == 3:
            return clip_features[:, 0, :]
        return clip_features

    def _maybe_print_debug(self, primitive_activation, primitive_logits, relation_scores):
        if self.debug_interval <= 0:
            return
        self.debug_step += 1
        if int(self.debug_step.item()) % self.debug_interval != 0:
            return
        _, visual_variance = self._compute_visual_variance()
        top_activation_values, top_activation_indices = primitive_activation.float().mean(dim=0).topk(
            min(5, self.num_primitives),
            dim=0,
        )
        top_activation_text = ", ".join([
            "{}:{:.3f}".format(self.primitive_names[int(idx)], float(value.detach().cpu()))
            for value, idx in zip(top_activation_values, top_activation_indices)
        ])
        top_predicate_text = self._format_predicate_top_primitives()
        print(
            "PrimitiveQuery debug step {}: W_entropy {:.4f}/{:.4f}, a_vis {:.4f}/{:.4f}/{:.4f}/{:.4f}, "
            "logit {:.4f}/{:.4f}/{:.4f}/{:.4f}, score {:.4f}/{:.4f}/{:.4f}/{:.4f}, "
            "visual_var {:.4f}/{:.4f}, count {:.2f}/{:.2f}, "
            "tau {:.3f}, scale {:.2f}, top_a [{}], top_W [{}]".format(
                int(self.debug_step.item()),
                float(self.predicate_entropy.float().mean().detach().cpu()),
                float(self.predicate_entropy.float().std(unbiased=False).detach().cpu()),
                float(primitive_activation.float().min().detach().cpu()),
                float(primitive_activation.float().mean().detach().cpu()),
                float(primitive_activation.float().max().detach().cpu()),
                float(primitive_activation.float().std(unbiased=False).detach().cpu()),
                float(primitive_logits.float().min().detach().cpu()),
                float(primitive_logits.float().mean().detach().cpu()),
                float(primitive_logits.float().max().detach().cpu()),
                float(primitive_logits.float().std(unbiased=False).detach().cpu()),
                float(relation_scores.float().min().detach().cpu()),
                float(relation_scores.float().mean().detach().cpu()),
                float(relation_scores.float().max().detach().cpu()),
                float(relation_scores.float().std(unbiased=False).detach().cpu()),
                float(visual_variance.float().mean().detach().cpu()),
                float(visual_variance.float().max().detach().cpu()),
                float(self.predicate_count.float().mean().detach().cpu()),
                float(self.predicate_count.float().min().detach().cpu()),
                self.visual_activation_tau,
                self.composition_logit_scale,
                top_activation_text,
                top_predicate_text,
            )
        )

    def _format_predicate_top_primitives(self, max_predicates=5, topk=3):
        predicate_count = min(max_predicates, len(self.foreground_predicate_names))
        fragments = []
        weights = self.predicate_primitive_weights.float()
        for predicate_idx in range(predicate_count):
            top_values, top_indices = weights[predicate_idx].topk(min(topk, self.num_primitives), dim=0)
            primitive_text = "/".join([
                "{}:{:.2f}".format(self.primitive_names[int(idx)], float(value.detach().cpu()))
                for value, idx in zip(top_values, top_indices)
            ])
            fragments.append("{}=>{}".format(self.foreground_predicate_names[predicate_idx], primitive_text))
        return "; ".join(fragments)

    def updata(self, mode):
        if mode != self.active_mode:
            print("SemanticBankGaussianPredictor primitive-query mode is initialized for {} predicates; requested mode {} is ignored.".format(
                self.active_mode, mode
            ))

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None):
        if img is None:
            raise ValueError("SemanticBankGaussianPredictor requires input images for CLIP visual features.")

        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)

        rel_dists = []
        debug_primitive_activations = []
        debug_primitive_logits = []
        debug_relation_scores = []

        for image_idx in range(len(num_rels)):
            image = img[image_idx].unsqueeze(0)
            object_crops = [
                self._crop_to_pil(image, proposals[image_idx].bbox[box_idx], proposals[image_idx].bbox[box_idx])
                for box_idx in range(len(proposals[image_idx].bbox))
            ]
            image_features = self._encode_clip_image_batch(object_crops)

            image_features = F.normalize(image_features, dim=-1)
            relation_indexes = rel_pair_idxs[image_idx]
            if relation_indexes.numel() == 0:
                rel_dists.append(torch.empty(0, self.num_active_predicates, device=self.device))
            else:
                subject_box_indexes = relation_indexes[:, 0]
                object_box_indexes = relation_indexes[:, 1]
                subject_classes = obj_preds[image_idx][subject_box_indexes].long()
                object_classes = obj_preds[image_idx][object_box_indexes].long()
                subject_raw_features = image_features[subject_box_indexes]
                object_raw_features = image_features[object_box_indexes]

                subject_text_features = self.subject_role_text_features[subject_classes].to(image_features.dtype)
                object_text_features = self.object_role_text_features[object_classes].to(image_features.dtype)
                subject_relation_features = self.subject_adapter(
                    subject_raw_features,
                    object_raw_features,
                    subject_text_features
                )
                object_relation_features = self.object_adapter(
                    object_raw_features,
                    subject_raw_features,
                    object_text_features
                )
                relation_features = F.normalize(
                    (subject_relation_features + object_relation_features) / 2.0, dim=-1
                )
                pooled_subject_features = self._pool_clip_features(subject_raw_features)
                pooled_object_features = self._pool_clip_features(object_raw_features)
                raw_relation_features = F.normalize((pooled_subject_features + pooled_object_features) / 2.0, dim=-1)
                relation_tokens = torch.stack(
                    [
                        subject_relation_features.float(),
                        object_relation_features.float(),
                        relation_features.float(),
                        raw_relation_features.float(),
                        subject_text_features.float(),
                        object_text_features.float(),
                    ],
                    dim=1,
                )
                primitive_activation, primitive_logits = self.primitive_activation_head(
                    relation_tokens,
                    tau=self.visual_activation_tau,
                )
                foreground_logits = self._compute_composition_logits(primitive_activation)
                debug_primitive_activations.append(primitive_activation.detach())
                debug_primitive_logits.append(primitive_logits.detach())
                debug_relation_scores.append(foreground_logits.detach())
                background_logit = self.background_classifier(
                    relation_features.float()
                ).float()
                relation_logits = torch.cat([background_logit, foreground_logits], dim=-1)
                rel_dists.append(relation_logits)

                if self.training:
                    relation_labels = rel_labels[image_idx].to(self.device)
                    self._update_visual_statistics(primitive_activation, relation_labels)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        if self.training and debug_primitive_activations:
            self._maybe_print_debug(
                torch.cat(debug_primitive_activations, dim=0),
                torch.cat(debug_primitive_logits, dim=0),
                torch.cat(debug_relation_scores, dim=0),
            )

        add_losses = {}
        return obj_dists, rel_dists, add_losses



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
