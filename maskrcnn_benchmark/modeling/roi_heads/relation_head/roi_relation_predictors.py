# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
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
import csv
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

    def forward(self, sub_features,obj_features,text_fea=None):
        x = self.adapter(sub_features,obj_features)
        if text_fea is not None:
            xx=[]
            for i in range(x.shape[0]):
                xx.append(torch.cat([x[i],text_fea]).unsqueeze(0))
            x=torch.cat(xx)
            x=self.linear(x)
            x=self.relu(x)
        ratio = 0.5
        sub_features= ratio * x + (1 - ratio) * sub_features[:,0,:]

        return sub_features


class PrimitiveCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super(PrimitiveCrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        query = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(query)
        return self.norm2(query + self.dropout(ffn_out))


class PrimitiveGuidedRelationAdapter(nn.Module):
    """
    Builds primitive-conditioned relation evidence for subject-object pairs.

    The output h_prim keeps the primitive axis. The visual memory follows the
    MVA design: one relation token for subject->object and one for object->subject.
    """

    def __init__(self, visual_dim=512, text_dim=512, hidden_dim=512, num_heads=8, dropout=0.1):
        super(PrimitiveGuidedRelationAdapter, self).__init__()
        self.hidden_dim = hidden_dim
        self.primitive_text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.subject_object_mva = MVA().float()
        self.object_subject_mva = MVA().float()
        self.memory_type = nn.Parameter(torch.zeros(2, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.pair_attn = PrimitiveCrossAttentionBlock(hidden_dim, num_heads, dropout)

        if visual_dim != hidden_dim:
            self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        else:
            self.visual_proj = nn.Identity()

    def _split_by_image(self, features, proposals):
        if isinstance(features, (list, tuple)):
            return list(features)
        num_objs = [len(proposal) for proposal in proposals]
        return list(features.split(num_objs, dim=0))

    def _collect_pair_visual_tokens(self, obj_visual_tokens, proposals, rel_pair_idxs):
        obj_tokens_per_image = self._split_by_image(obj_visual_tokens, proposals)
        sub_tokens = []
        obj_tokens = []

        for per_image_tokens, pair_idx in zip(obj_tokens_per_image, rel_pair_idxs):
            if pair_idx.numel() == 0:
                continue
            sub_tokens.append(per_image_tokens[pair_idx[:, 0]])
            obj_tokens.append(per_image_tokens[pair_idx[:, 1]])

        if len(sub_tokens) == 0:
            return None, None

        return torch.cat(sub_tokens, dim=0), torch.cat(obj_tokens, dim=0)

    def _project_visual_tokens(self, features):
        if features.dim() != 3:
            raise ValueError("obj_visual_tokens must be [N, T, D] CLIP visual tokens")
        return self.visual_proj(features)

    def _build_visual_memory(self, sub_tokens, obj_tokens):
        # 沿用 MVA 的双向交互思想，但一次性并行处理所有 pair，避免 ClipPredictor 的逐 pair 循环。
        subject_object = self.subject_object_mva(sub_tokens, obj_tokens)
        object_subject = self.object_subject_mva(obj_tokens, sub_tokens)
        visual_memory = torch.stack((subject_object, object_subject), dim=1)
        return visual_memory + self.memory_type.view(1, 2, -1)

    def forward(self, obj_visual_tokens, obj_text_features, proposals, rel_pair_idxs, primitive_text_features):
        """
        Args:
            obj_visual_tokens: Tensor [sum_obj, T, Dv] or list of per-image tensors.
                T is expected to be CLIP visual tokens, e.g. CLS + 196 patches.
            obj_text_features: Tensor [sum_obj, Dt] or list of per-image tensors.
            proposals: list[BoxList], aligned with rel_pair_idxs.
            rel_pair_idxs: list[Tensor], each [num_rel_i, 2] with local object indices.
            primitive_text_features: Tensor [K, Dt].

        Returns:
            h_prim: Tensor [sum_rel, K, hidden_dim].
        """
        dtype = self.primitive_text_proj[0].weight.dtype
        device = self.primitive_text_proj[0].weight.device
        obj_visual_tokens = obj_visual_tokens.to(device=device, dtype=dtype) \
            if torch.is_tensor(obj_visual_tokens) else [x.to(device=device, dtype=dtype) for x in obj_visual_tokens]
        obj_text_features = obj_text_features.to(device=device, dtype=dtype) \
            if torch.is_tensor(obj_text_features) else [x.to(device=device, dtype=dtype) for x in obj_text_features]
        primitive_text_features = primitive_text_features.to(device=device, dtype=dtype)

        sub_tokens, obj_tokens = self._collect_pair_visual_tokens(
            obj_visual_tokens, proposals, rel_pair_idxs
        )
        num_primitives = primitive_text_features.size(0)
        if sub_tokens is None:
            return primitive_text_features.new_zeros((0, num_primitives, self.hidden_dim))

        sub_tokens = self._project_visual_tokens(sub_tokens)
        obj_tokens = self._project_visual_tokens(obj_tokens)
        # query 只保留原语文本锚点，避免共享 prompt 或 pair 条件把 32 个原语拉到同一方向。
        primitive_query = self.primitive_text_proj(primitive_text_features)
        q_prim = self.query_norm(primitive_query).unsqueeze(0)
        q_prim = q_prim.expand(sub_tokens.size(0), -1, -1)
        visual_memory = self._build_visual_memory(sub_tokens, obj_tokens)
        return self.pair_attn(q_prim, visual_memory)


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

@registry.ROI_RELATION_PREDICTOR.register("SFClipPredictor")
class SFClipPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(SFClipPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        self.obj_names = obj_classes
        self.rel_names = rel_classes

        self.device=config.MODEL.DEVICE
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        a=time.time()
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        self.id_dict={'__background__': 0, 'above': 1, 'across': 2, 'against': 3, 'along': 4, 'and': 5, 'at': 6, 'attached to': 7, 'behind': 8, 'belonging to': 9, 'between': 10, 'carrying': 11, 'covered in': 12, 'covering': 13, 'eating': 14, 'flying in': 15, 'for': 16, 'from': 17, 'growing on': 18, 'hanging from': 19, 'has': 20, 'holding': 21, 'in': 22, 'in front of': 23, 'laying on': 24, 'looking at': 25, 'lying on': 26, 'made of': 27, 'mounted on': 28, 'near': 29, 'of': 30, 'on': 31, 'on back of': 32, 'over': 33, 'painted on': 34, 'parked on': 35, 'part of': 36, 'playing': 37, 'riding': 38, 'says': 39, 'sitting on': 40, 'standing on': 41, 'to': 42, 'under': 43, 'using': 44, 'walking in': 45, 'walking on': 46, 'watching': 47, 'wearing': 48, 'wears': 49, 'with': 50}

        self.base=[0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.PRDCS_BASE)]
        self.novel=[0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.PRDCS_NOVEL)]

        self.semantic = [0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.SEMAN)]
        primitive_rows = self._load_primitive_descriptions()
        self.primitive_names = [row[1] for row in primitive_rows]
        self.primitive_descriptions = [row[2] for row in primitive_rows]
        self.num_primitives = len(self.primitive_descriptions)

        with torch.no_grad():
            primitive_tokens = clip.tokenize([
                "a photo of a relation where " + desc for desc in self.primitive_descriptions
            ]).to(self.device)
            primitive_text_features = self.clip_model.encode_text(primitive_tokens).float()
            object_tokens = clip.tokenize(["a photo of " + obj for obj in self.obj_names]).to(self.device)
            object_text_features = self.clip_model.encode_text(object_tokens).float()
        self.register_buffer("primitive_text_features", primitive_text_features)
        self.register_buffer("object_text_features", object_text_features)

        predicate_prior = self._load_predicate_primitive_prior().to(self.device)
        self.register_buffer("predicate_primitive_prior", predicate_prior)

        self.primitive_adapter = PrimitiveGuidedRelationAdapter(
            visual_dim=primitive_text_features.size(-1),
            text_dim=primitive_text_features.size(-1),
            hidden_dim=512,
            num_heads=8,
            dropout=0.1,
        )
        self.primitive_temperature = 0.05
        self.clip_object_batch_size = getattr(config.MODEL.ROI_RELATION_HEAD, "CLIP_OBJECT_BATCH_SIZE", 32)
        self.sf_debug = os.environ.get("SFC_DEBUG", "0") == "1"
        self.sf_debug_interval = max(int(os.environ.get("SFC_DEBUG_INTERVAL", "200")), 1)
        self.sf_debug_step = 0
        self.updata(getattr(config.OV_SETTING, "TRAIN_PART", "base"))

        b=time.time()
        print('init complete : '+str(b-a))

    def _load_primitive_descriptions(self):
        primitive_path = os.path.join(curpath, "primitive_descriptions_32.csv")
        rows = []
        with open(primitive_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((int(row["index"]), row["name"], row["description"]))
        rows.sort(key=lambda x: x[0])
        return rows

    def _load_predicate_primitive_prior(self):
        prior_path = os.path.join(curpath, "description_relation_32.csv")
        rows = []
        with open(prior_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                primitive_vector = [float(x) for x in row["primitive_vector"].split(",")]
                rows.append((int(row["predicate_id"]), primitive_vector))
        rows.sort(key=lambda x: x[0])
        return torch.Tensor([row[1] for row in rows]).float()

    def updata(self,mode):
        print("now is "+mode)
        if mode == "base":
            active_ids = self.base
        elif mode == "novel":
            active_ids = self.novel
        elif mode == "total":
            active_ids = list(range(self.predicate_primitive_prior.size(0)))
        elif mode == "semantic":
            active_ids = self.semantic
        else:
            raise ValueError("Unsupported SFClipPredictor mode: {}".format(mode))
        self.mode = mode
        self.active_predicate_ids = active_ids
        self.active_predicate_ids_tensor = torch.LongTensor(active_ids).to(self.device)

    def _split_rel_tensor(self, rel_tensor, num_rels):
        if len(num_rels) == 0:
            return []
        return list(rel_tensor.split(num_rels, dim=0))

    def _extract_clip_object_tokens(self, proposals, img):
        if img is None:
            raise ValueError("SFClipPredictor requires img to extract CLIP object tokens")

        image_tensors = []
        for image, proposal in zip(img, proposals):
            for box in proposal.bbox:
                crop = crop_and_resize(image.unsqueeze(0), box, box)
                crop = crop[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                crop = Image.fromarray(np.uint8(crop))
                image_tensors.append(self.clip_preprocess(crop).unsqueeze(0))

        if len(image_tensors) == 0:
            return self.primitive_text_features.new_zeros((0, 197, self.primitive_text_features.size(-1)))

        image_tensors = torch.cat(image_tensors, dim=0).to(self.device)
        batch_size = max(int(self.clip_object_batch_size), 1)
        clip_features = []
        with torch.no_grad():
            for start in range(0, image_tensors.size(0), batch_size):
                clip_features.append(
                    self.clip_model.encode_image(image_tensors[start:start + batch_size]).float()
                )
        return torch.cat(clip_features, dim=0)

    def _compute_primitive_logits(self, h_prim):
        dtype = h_prim.dtype
        device = h_prim.device
        primitive_text_features = self.primitive_text_features.to(device=device, dtype=dtype)
        primitive_proto = self.primitive_adapter.primitive_text_proj(primitive_text_features)
        h_prim = F.normalize(h_prim, dim=-1)
        primitive_proto = F.normalize(primitive_proto, dim=-1)
        return (h_prim * primitive_proto.unsqueeze(0)).sum(-1) / self.primitive_temperature

    def _rel_name(self, rel_id):
        rel_id = int(rel_id)
        if 0 <= rel_id < len(self.rel_names):
            return self.rel_names[rel_id]
        return "rel_{}".format(rel_id)

    def _collect_pair_gt_labels(self, proposals, rel_pair_idxs, device):
        gt_labels = []
        has_gt = False
        for proposal, pair_idx in zip(proposals, rel_pair_idxs):
            if proposal.has_field("relation"):
                rel_matrix = proposal.get_field("relation").to(device=device)
                gt_labels.append(rel_matrix[pair_idx[:, 0], pair_idx[:, 1]].long())
                has_gt = True
            else:
                gt_labels.append(torch.zeros(pair_idx.size(0), dtype=torch.long, device=device))
        if not has_gt:
            return None
        return gt_labels

    def _format_rel_hist(self, labels, counts):
        order = counts.argsort(descending=True)[:5]
        return ", ".join([
            "{}:{}".format(self._rel_name(labels[i]), int(counts[i]))
            for i in order
        ])

    def _format_primitive_top(self, values):
        top_vals, top_ids = values.topk(min(5, values.numel()))
        return ", ".join([
            "{}:{:.3f}".format(self.primitive_names[int(idx)], val.item())
            for val, idx in zip(top_vals, top_ids)
        ])

    def _debug_sf_state(self, primitive_logits, h_prim=None, rel_scores=None, rel_labels=None):
        if not self.sf_debug:
            return
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return
        self.sf_debug_step += 1
        if self.sf_debug_step % self.sf_debug_interval != 0:
            return

        with torch.no_grad():
            logits = primitive_logits.detach().float()
            msg = [
                "[SFClipDebug step={} mode={} train={}]".format(
                    self.sf_debug_step, self.mode, self.training
                ),
                "primitive_logits shape={} mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
                    tuple(logits.shape),
                    logits.mean().item(),
                    logits.std(unbiased=False).item(),
                    logits.min().item(),
                    logits.max().item(),
                ),
            ]
            primitive_mean = logits.mean(dim=0)
            msg.append("pred primitive mean={}".format(self._format_primitive_top(primitive_mean)))

            if rel_labels is not None:
                all_labels = cat(rel_labels, dim=0).view(-1).long().to(logits.device)
                positive_mask = all_labels > 0
                positive_labels = all_labels[positive_mask]
                if positive_labels.numel() > 0:
                    uniq, counts = positive_labels.unique(return_counts=True)
                    gt_predicate_hist = self._format_rel_hist(uniq, counts)
                    gt_prior = self.predicate_primitive_prior.to(
                        device=logits.device, dtype=logits.dtype
                    )[positive_labels]
                    positive_logits = logits[positive_mask]
                    decisive = gt_prior != 0
                    if decisive.any():
                        gt_target = gt_prior * 2.0
                        decisive_mae = (positive_logits[decisive] - gt_target[decisive]).abs().mean().item()
                        signed_hit = (
                            torch.sign(positive_logits[decisive]) == torch.sign(gt_prior[decisive])
                        ).float().mean().item()
                        gt_primitive_mean = gt_prior.float().mean(dim=0).abs()
                        msg.append("gt primitive abs mean={}".format(self._format_primitive_top(gt_primitive_mean)))
                        msg.append("decisive mae={:.4f} sign hit={:.4f}".format(decisive_mae, signed_hit))

                    if h_prim is not None:
                        # 诊断 32 个原语 token 是否学成相似表示；越接近 1，原语轴越可能塌缩。
                        positive_h_prim = h_prim.detach().float()[positive_mask]
                        if positive_h_prim.size(0) > 0 and positive_h_prim.size(1) > 1:
                            norm_h_prim = F.normalize(positive_h_prim, dim=-1)
                            primitive_sim = torch.bmm(norm_h_prim, norm_h_prim.transpose(1, 2))
                            off_diag = ~torch.eye(
                                primitive_sim.size(1),
                                dtype=torch.bool,
                                device=primitive_sim.device,
                            ).unsqueeze(0)
                            primitive_sim = primitive_sim[off_diag.expand_as(primitive_sim)]
                            msg.append("primitive collapse cos={:.4f} abs={:.4f}".format(
                                primitive_sim.mean().item(),
                                primitive_sim.abs().mean().item(),
                            ))

                    if rel_scores is None:
                        active_prior = self.predicate_primitive_prior.to(
                            device=logits.device, dtype=logits.dtype
                        )[self.active_predicate_ids_tensor.to(logits.device)]
                        active_prior = F.normalize(active_prior, dim=-1)
                        train_scores = torch.matmul(logits, active_prior.t())
                        train_top_cols = train_scores[:, 1:].max(dim=1)[1]
                        train_pred_ids = self.active_predicate_ids_tensor.to(logits.device)[train_top_cols + 1]
                        gt_pair_pred_ids = train_pred_ids[positive_mask]
                        pred_uniq, pred_counts = gt_pair_pred_ids.unique(return_counts=True)
                        hit = (gt_pair_pred_ids == positive_labels).float().mean().item()
                        msg.append("gt predicate top={} | train pred top(gt pairs)={} | pred@gt hit={:.4f}".format(
                            gt_predicate_hist,
                            self._format_rel_hist(pred_uniq, pred_counts),
                            hit,
                        ))

            if rel_scores is not None and rel_scores.numel() > 0:
                if h_prim is not None:
                    # eval 阶段没有采样标签时，也统计全部候选 pair 的原语相似度。
                    eval_h_prim = h_prim.detach().float()
                    if eval_h_prim.size(0) > 0 and eval_h_prim.size(1) > 1:
                        norm_h_prim = F.normalize(eval_h_prim, dim=-1)
                        primitive_sim = torch.bmm(norm_h_prim, norm_h_prim.transpose(1, 2))
                        off_diag = ~torch.eye(
                            primitive_sim.size(1),
                            dtype=torch.bool,
                            device=primitive_sim.device,
                        ).unsqueeze(0)
                        primitive_sim = primitive_sim[off_diag.expand_as(primitive_sim)]
                        msg.append("primitive collapse cos={:.4f} abs={:.4f}".format(
                            primitive_sim.mean().item(),
                            primitive_sim.abs().mean().item(),
                        ))
                scores = rel_scores.detach().float()
                top_scores, top_cols = scores[:, 1:].max(dim=1)
                pred_ids = self.active_predicate_ids_tensor.to(top_cols.device)[top_cols + 1]
                if rel_labels is not None:
                    gt_labels = cat(rel_labels, dim=0).view(-1).long().to(pred_ids.device)
                    valid_gt = gt_labels > 0
                    if valid_gt.any():
                        hit = (pred_ids[valid_gt] == gt_labels[valid_gt]).float().mean().item()
                        gt_uniq, gt_counts = gt_labels[valid_gt].unique(return_counts=True)
                        eval_gt_hist = self._format_rel_hist(gt_uniq, gt_counts)
                uniq, counts = pred_ids.unique(return_counts=True)
                eval_pred_hist = self._format_rel_hist(uniq, counts)
                msg.append("rel_scores shape={} mean={:.4f} std={:.4f} pred top={} score mean={:.4f}".format(
                    tuple(scores.shape),
                    scores.mean().item(),
                    scores.std(unbiased=False).item(),
                    eval_pred_hist,
                    top_scores.mean().item(),
                ))
                if rel_labels is not None and 'eval_gt_hist' in locals():
                    msg.append("gt predicate top={} | eval pred top={} | pred@gt hit={:.4f}".format(
                        eval_gt_hist,
                        eval_pred_hist,
                        hit,
                    ))
            print(" | ".join(msg))

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,img=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        if sum(num_rels) == 0:
            obj_dists = obj_dists.split(num_objs, dim=0)
            if self.training:
                rel_dists = tuple(
                    roi_features.new_zeros((0, self.num_primitives)) for _ in num_rels
                )
            else:
                rel_dists = tuple(
                    roi_features.new_zeros((0, len(self.active_predicate_ids))) for _ in num_rels
                )
            return obj_dists, rel_dists, {}

        obj_text_features = self.object_text_features.to(device=roi_features.device)[obj_preds.long()]
        obj_visual_tokens = self._extract_clip_object_tokens(proposals, img).to(device=roi_features.device)
        h_prim = self.primitive_adapter(
            obj_visual_tokens,
            obj_text_features,
            proposals,
            rel_pair_idxs,
            self.primitive_text_features,
        )
        primitive_logits = self._compute_primitive_logits(h_prim)

        obj_dists = obj_dists.split(num_objs, dim=0)
        if self.training:
            self._debug_sf_state(primitive_logits, h_prim=h_prim, rel_labels=rel_labels)
            rel_dists = tuple(self._split_rel_tensor(primitive_logits, num_rels))
        else:
            active_prior = self.predicate_primitive_prior.to(
                device=primitive_logits.device, dtype=primitive_logits.dtype
            )[
                self.active_predicate_ids_tensor.to(primitive_logits.device)
            ]
            active_prior = F.normalize(active_prior, dim=-1)
            rel_scores = torch.matmul(primitive_logits, active_prior.t())
            debug_gt_labels = self._collect_pair_gt_labels(proposals, rel_pair_idxs, primitive_logits.device)
            self._debug_sf_state(primitive_logits, h_prim=h_prim, rel_scores=rel_scores, rel_labels=debug_gt_labels)
            rel_dists = tuple(self._split_rel_tensor(rel_scores, num_rels))

        return obj_dists, rel_dists, {}

def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
