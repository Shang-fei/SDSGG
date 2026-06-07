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
        return torch.sigmoid(primitive_logits / max(tau, 1e-6))


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
        self.text_recon_steps = int(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_RECON_STEPS", "text_recon_steps", 300
        ))
        self.text_recon_lr = float(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_RECON_LR", "text_recon_lr", 0.05
        ))
        self.text_recon_tau = float(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_RECON_TAU", "text_recon_tau", 0.20
        ))
        self.text_recon_debug_interval = int(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_RECON_DEBUG_INTERVAL", "text_recon_debug_interval", 50
        ))
        self.text_recon_weight = float(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_RECON_WEIGHT", "text_recon_weight", 1.0
        ))
        self.text_sparse_weight = float(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_SPARSE_WEIGHT", "text_sparse_weight", 0.02
        ))
        self.text_orth_weight = float(self._cfg_value(
            primitive_cfg, primitive_settings, "TEXT_ORTH_WEIGHT", "text_orth_weight", 0.05
        ))
        self.b_trainable_text_stage = bool(self._cfg_value(
            primitive_cfg, primitive_settings, "B_TRAINABLE_TEXT_STAGE", "b_trainable_text_stage", True
        ))
        self.b_trainable_visual_stage = bool(self._cfg_value(
            primitive_cfg, primitive_settings, "B_TRAINABLE_VISUAL_STAGE", "b_trainable_visual_stage", False
        ))
        self.sigma_min = float(self._cfg_value(primitive_cfg, primitive_settings, "SIGMA_MIN", "sigma_min", 0.05))
        self.sigma_max = float(self._cfg_value(primitive_cfg, primitive_settings, "SIGMA_MAX", "sigma_max", 0.30))
        self.visual_align_weight = float(self._cfg_value(
            primitive_cfg, primitive_settings, "VISUAL_ALIGN_WEIGHT", "visual_align_weight", 0.10
        ))
        self.origin_reg_weight = float(self._cfg_value(
            primitive_cfg, primitive_settings, "ORIGIN_REG_WEIGHT", "origin_reg_weight", 0.02
        ))
        self.visual_sparse_weight = float(self._cfg_value(
            primitive_cfg, primitive_settings, "VISUAL_SPARSE_WEIGHT", "visual_sparse_weight", 0.001
        ))
        self.visual_activation_tau = float(self._cfg_value(
            primitive_cfg, primitive_settings, "VISUAL_ACTIVATION_TAU", "visual_activation_tau", 0.20
        ))
        self.origin_activation_tau = float(self._cfg_value(
            primitive_cfg, primitive_settings, "ORIGIN_ACTIVATION_TAU", "origin_activation_tau", 0.20
        ))
        self.object_filter_weight = float(self._cfg_value(
            primitive_cfg, primitive_settings, "OBJECT_FILTER_WEIGHT", "object_filter_weight", 0.05
        ))
        self.use_object_filter = bool(self._cfg_value(
            primitive_cfg, primitive_settings, "USE_OBJECT_FILTER", "use_object_filter", True
        ))
        self.use_log_variance = bool(self._cfg_value(
            primitive_cfg, primitive_settings, "USE_LOG_VARIANCE", "use_log_variance", False
        ))
        self.use_composition_logits = bool(self._cfg_value(
            primitive_cfg, primitive_settings, "USE_COMPOSITION_LOGITS", "use_composition_logits", False
        ))
        self.use_visual_variance_in_logits = bool(self._cfg_value(
            primitive_cfg, primitive_settings, "USE_VISUAL_VARIANCE_IN_LOGITS", "use_visual_variance_in_logits", False
        ))
        self.use_novel_variance_transfer = bool(self._cfg_value(
            primitive_cfg, primitive_settings, "USE_NOVEL_VARIANCE_TRANSFER", "use_novel_variance_transfer", False
        ))
        self.novel_topk = int(self._cfg_value(primitive_cfg, primitive_settings, "NOVEL_TOPK", "novel_topk", 3))
        self.novel_tau = float(self._cfg_value(primitive_cfg, primitive_settings, "NOVEL_TAU", "novel_tau", 0.2))
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

        if self.b_trainable_visual_stage:
            self.primitive_basis = nn.Parameter(primitive_basis.clone())
        else:
            self.register_buffer("primitive_basis", primitive_basis.clone())

        self.num_primitives = primitive_basis.shape[0]
        predicate_entropy = self._compute_weight_entropy(predicate_weights)
        predicate_variance = self.sigma_min + (self.sigma_max - self.sigma_min) * predicate_entropy.unsqueeze(-1)
        predicate_variance = predicate_variance.expand_as(predicate_weights).contiguous()
        self.register_buffer("predicate_primitive_weights", predicate_weights.float())
        self.register_buffer("predicate_variance", predicate_variance.float())
        self.register_buffer("predicate_entropy", predicate_entropy.float())
        self.register_buffer("predicate_description_features", predicate_description_features.float())
        self.register_buffer("text_recon_loss", text_losses["recon"].detach().float())
        self.register_buffer("text_sparse_loss", text_losses["sparse"].detach().float())
        self.register_buffer("text_orth_loss", text_losses["orth"].detach().float())
        self.register_buffer("visual_mean_stat", torch.zeros_like(predicate_weights.float()))
        self.register_buffer("visual_second_moment_stat", torch.zeros_like(predicate_weights.float()))
        self.register_buffer("predicate_count", torch.zeros(self.num_active_predicates - 1, dtype=torch.float32))
        self.register_buffer("debug_step", torch.zeros((), dtype=torch.long))

        self.primitive_activation_head = PrimitiveQueryActivationHead(primitive_basis).to(self.device)

        subject_role_prompts = ["a photo of subject {}".format(obj_name) for obj_name in self.obj_names]
        object_role_prompts = ["a photo of object {}".format(obj_name) for obj_name in self.obj_names]
        self.register_buffer("subject_role_text_features", self._encode_text_features(subject_role_prompts).float())
        self.register_buffer("object_role_text_features", self._encode_text_features(object_role_prompts).float())
        self.register_buffer(
            "base_predicate_mask",
            torch.tensor([predicate_name in set(config.OV_SETTING.PRDCS_BASE) for predicate_name in self.foreground_predicate_names],
                         dtype=torch.bool),
        )
        self.register_buffer(
            "novel_predicate_mask",
            torch.tensor([predicate_name in set(config.OV_SETTING.PRDCS_NOVEL) for predicate_name in self.foreground_predicate_names],
                         dtype=torch.bool),
        )
        if self.use_object_filter:
            object_filter_text_features = self._build_object_filter_text_features(resources)
        else:
            object_filter_text_features = torch.empty(
                0,
                len(self.foreground_predicate_names),
                primitive_basis.shape[-1],
                device=self.device,
                dtype=torch.float32,
            )
        self.register_buffer("object_filter_text_features", object_filter_text_features.float())
        print(
            "SemanticBankGaussianPredictor frozen W0 mode: {} primitives, {} predicates, w_source={}, "
            "text semantic recovery/sparse/orth={:.4f}/{:.4f}/{:.4f}".format(
                self.num_primitives,
                self.num_active_predicates,
                self.w_source,
                float(self.text_recon_loss.cpu()),
                float(self.text_sparse_loss.cpu()),
                float(self.text_orth_loss.cpu()),
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
        if self.w_source == "reconstruction":
            return self._reconstruct_predicate_weights(predicate_features, primitive_basis_init)
        if self.w_source == "text_similarity":
            similarity = predicate_features.float() @ primitive_basis_init.float().t()
            weights = F.softmax(similarity / max(self.text_recon_tau, 1e-6), dim=-1)
            losses = self._compute_text_anchor_losses(predicate_features, primitive_basis_init, weights)
            self._print_text_recon_summary(losses, weights)
            return weights.detach(), primitive_basis_init.detach(), losses
        else:
            raise ValueError("Unsupported W_SOURCE: {}".format(self.w_source))

    def _reconstruct_predicate_weights(self, predicate_features, primitive_basis_init):
        predicate_features = predicate_features.detach().float()
        primitive_basis_init = primitive_basis_init.detach().float()
        init_similarity = predicate_features @ primitive_basis_init.t()
        weight_logits = nn.Parameter(init_similarity.clone())
        if self.b_trainable_text_stage:
            primitive_basis_param = nn.Parameter(primitive_basis_init.clone())
            optim_params = [weight_logits, primitive_basis_param]
        else:
            primitive_basis_param = primitive_basis_init
            optim_params = [weight_logits]

        optimizer = torch.optim.Adam(optim_params, lr=self.text_recon_lr)
        steps = max(self.text_recon_steps, 0)
        losses = None
        for step_idx in range(steps):
            weights = F.softmax(weight_logits / max(self.text_recon_tau, 1e-6), dim=-1)
            losses = self._compute_text_anchor_losses(predicate_features, primitive_basis_param, weights)
            loss = (
                self.text_recon_weight * losses["recon"]
                + self.text_sparse_weight * losses["sparse"]
                + self.text_orth_weight * losses["orth"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._maybe_print_text_recon_debug(step_idx + 1, steps, loss, losses, weights)

        weights = F.softmax(weight_logits / max(self.text_recon_tau, 1e-6), dim=-1).detach()
        primitive_basis = F.normalize(primitive_basis_param.detach().float(), dim=-1)
        losses = self._compute_text_anchor_losses(predicate_features, primitive_basis, weights)
        self._print_text_recon_summary(losses, weights)
        return weights.float(), primitive_basis.float(), losses

    def _maybe_print_text_recon_debug(self, step_idx, total_steps, loss, losses, weights):
        if self.text_recon_debug_interval <= 0:
            return
        if step_idx != 1 and step_idx != total_steps and step_idx % self.text_recon_debug_interval != 0:
            return
        with torch.no_grad():
            entropy = self._compute_weight_entropy(weights.float())
            max_weight = weights.float().max(dim=-1)[0]
            print(
                "TextRecon step {}/{}: loss {:.4f}, recon {:.4f}, sparse {:.4f}, orth {:.4f}, "
                "W_entropy {:.4f}/{:.4f}, W_max {:.4f}/{:.4f}".format(
                    step_idx,
                    total_steps,
                    float(loss.detach().cpu()),
                    float(losses["recon"].detach().cpu()),
                    float(losses["sparse"].detach().cpu()),
                    float(losses["orth"].detach().cpu()),
                    float(entropy.mean().detach().cpu()),
                    float(entropy.std(unbiased=False).detach().cpu()),
                    float(max_weight.mean().detach().cpu()),
                    float(max_weight.max().detach().cpu()),
                )
            )

    def _print_text_recon_summary(self, losses, weights):
        with torch.no_grad():
            entropy = self._compute_weight_entropy(weights.float())
            max_weight = weights.float().max(dim=-1)[0]
            semantic_recovery = 1.0 - losses["recon"]
            print(
                "TextSemantic final: recovery {:.4f}, recovery_min {:.4f}, recovery_max {:.4f}, "
                "recovery_mean {:.4f}, recovery_std {:.4f}, W_min {:.4f}, W_max_global {:.4f}, "
                "W_mean {:.4f}, W_entropy {:.4f}/{:.4f}, W_max {:.4f}/{:.4f}, orth {:.4f}".format(
                    float(semantic_recovery.mean().detach().cpu()) if semantic_recovery.dim() > 0 else float(semantic_recovery.detach().cpu()),
                    float(losses["recovery_per_predicate"].min().detach().cpu()),
                    float(losses["recovery_per_predicate"].max().detach().cpu()),
                    float(losses["recovery_per_predicate"].mean().detach().cpu()),
                    float(losses["recovery_per_predicate"].std(unbiased=False).detach().cpu()),
                    float(weights.float().min().detach().cpu()),
                    float(weights.float().max().detach().cpu()),
                    float(weights.float().mean().detach().cpu()),
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

    def _get_predicate_distribution(self):
        predicate_mean = self.predicate_primitive_weights.float()
        predicate_variance = self.predicate_variance.float()
        if self.use_novel_variance_transfer:
            predicate_variance = self._transfer_novel_variance(predicate_variance)
        return predicate_mean, predicate_variance.clamp(min=self.sigma_min, max=self.sigma_max)

    def _transfer_novel_variance(self, predicate_variance):
        base_indices = torch.nonzero(self.base_predicate_mask, as_tuple=False).squeeze(1)
        novel_indices = torch.nonzero(self.novel_predicate_mask, as_tuple=False).squeeze(1)
        if base_indices.numel() == 0 or novel_indices.numel() == 0:
            return predicate_variance
        transferred_variance = predicate_variance.clone()
        normalized_weights = F.normalize(self.predicate_primitive_weights.float(), dim=-1)
        base_weights = normalized_weights[base_indices]
        for novel_idx in novel_indices.tolist():
            similarity = normalized_weights[novel_idx].unsqueeze(0) @ base_weights.t()
            topk = min(self.novel_topk, base_indices.numel())
            topk_similarity, topk_pos = torch.topk(similarity.squeeze(0), topk, dim=0)
            transfer_indices = base_indices[topk_pos]
            transfer_weights = F.softmax(topk_similarity / max(self.novel_tau, 1e-6), dim=0)
            transferred_variance[novel_idx] = torch.sum(
                transfer_weights.unsqueeze(-1) * predicate_variance[transfer_indices],
                dim=0,
            )
        return transferred_variance

    def _compute_composition_logits(self, primitive_activation):
        return primitive_activation.float() @ self.predicate_primitive_weights.float().t()

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

    def _compute_origin_activation(self, raw_relation_features):
        raw_relation_features = self._pool_clip_features(raw_relation_features)
        primitive_basis = F.normalize(self.primitive_basis.float(), dim=-1)
        origin_similarity = raw_relation_features.float() @ primitive_basis.t()
        return F.softmax(origin_similarity / max(self.origin_activation_tau, 1e-6), dim=-1)

    def _build_object_filter_text_features(self, resources):
        if "object_filter_csv" not in resources:
            return torch.zeros(
                len(self.obj_names), len(self.foreground_predicate_names), self.primitive_basis.shape[-1],
                device=self.device,
                dtype=torch.float32,
            )
        filter_csv = pd.read_csv(self._resolve_resource_path(resources["object_filter_csv"])).fillna("__background__")
        object_columns = list(filter_csv.columns[1:])
        predicate_to_row = {}
        for row_idx in range(len(filter_csv)):
            if row_idx < len(self.predicate_names):
                predicate_to_row[self.predicate_names[row_idx]] = row_idx
        id_dict = {
            "__background__": 0, "above": 1, "across": 2, "against": 3, "along": 4, "and": 5, "at": 6,
            "attached to": 7, "behind": 8, "belonging to": 9, "between": 10, "carrying": 11,
            "covered in": 12, "covering": 13, "eating": 14, "flying in": 15, "for": 16, "from": 17,
            "growing on": 18, "hanging from": 19, "has": 20, "holding": 21, "in": 22, "in front of": 23,
            "laying on": 24, "looking at": 25, "lying on": 26, "made of": 27, "mounted on": 28, "near": 29,
            "of": 30, "on": 31, "on back of": 32, "over": 33, "painted on": 34, "parked on": 35,
            "part of": 36, "playing": 37, "riding": 38, "says": 39, "sitting on": 40, "standing on": 41,
            "to": 42, "under": 43, "using": 44, "walking in": 45, "walking on": 46, "watching": 47,
            "wearing": 48, "wears": 49, "with": 50,
        }
        prompts = []
        for obj_name in self.obj_names:
            column_name = obj_name if obj_name in object_columns else "__background__"
            for predicate_name in self.foreground_predicate_names:
                row_idx = id_dict.get(predicate_name, predicate_to_row.get(predicate_name, 0))
                filter_text = str(filter_csv.iloc[row_idx][column_name]).strip()
                prompts.append("a photo of {}".format(filter_text if filter_text else "__background__"))
        object_filter_features = self._encode_text_features(prompts, batch_size=256)
        return object_filter_features.view(
            len(self.obj_names),
            len(self.foreground_predicate_names),
            -1,
        )

    def _compute_object_filter_logits(self, subject_features, object_features, object_classes):
        if not self.use_object_filter or self.object_filter_weight == 0.0:
            return None
        subject_features = self._pool_clip_features(subject_features)
        object_features = self._pool_clip_features(object_features)
        text_features = self.object_filter_text_features[object_classes].to(subject_features.dtype)
        if text_features.shape[-1] != subject_features.shape[-1]:
            raise ValueError(
                "Object filter feature dim mismatch: text dim {} vs image dim {}.".format(
                    text_features.shape[-1],
                    subject_features.shape[-1],
                )
            )
        subject_similarity = torch.bmm(text_features, subject_features.unsqueeze(-1)).squeeze(-1)
        object_similarity = torch.bmm(text_features, object_features.unsqueeze(-1)).squeeze(-1)
        return (subject_similarity + object_similarity).float() * 0.5

    def _maybe_print_debug(self, primitive_activation, relation_scores):
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
            "score {:.4f}/{:.4f}/{:.4f}/{:.4f}, visual_var {:.4f}/{:.4f}, count {:.2f}/{:.2f}, "
            "tau {:.3f}, top_a [{}], top_W [{}]".format(
                int(self.debug_step.item()),
                float(self.predicate_entropy.float().mean().detach().cpu()),
                float(self.predicate_entropy.float().std(unbiased=False).detach().cpu()),
                float(primitive_activation.float().min().detach().cpu()),
                float(primitive_activation.float().mean().detach().cpu()),
                float(primitive_activation.float().max().detach().cpu()),
                float(primitive_activation.float().std(unbiased=False).detach().cpu()),
                float(relation_scores.float().min().detach().cpu()),
                float(relation_scores.float().mean().detach().cpu()),
                float(relation_scores.float().max().detach().cpu()),
                float(relation_scores.float().std(unbiased=False).detach().cpu()),
                float(visual_variance.float().mean().detach().cpu()),
                float(visual_variance.float().max().detach().cpu()),
                float(self.predicate_count.float().mean().detach().cpu()),
                float(self.predicate_count.float().min().detach().cpu()),
                self.visual_activation_tau,
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
            print("SemanticBankGaussianPredictor frozen W0 mode is initialized for {} predicates; requested mode {} is ignored.".format(
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
                primitive_activation = self.primitive_activation_head(
                    relation_tokens,
                    tau=self.visual_activation_tau,
                )
                foreground_logits = self._compute_composition_logits(primitive_activation)
                debug_primitive_activations.append(primitive_activation.detach())
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
                torch.cat(debug_relation_scores, dim=0),
            )

        add_losses = {}
        return obj_dists, rel_dists, add_losses



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
