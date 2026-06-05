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

        semantic_bank_cfg = config.MODEL.ROI_RELATION_HEAD.SEMANTIC_BANK
        bank_config = self._load_semantic_bank_config(semantic_bank_cfg.CONFIG_PATH)
        bank_settings = bank_config.get("settings", {})
        self.tau_bank = float(getattr(semantic_bank_cfg, "TAU_BANK", bank_settings.get("tau_bank", 0.07)))
        self.tau_cls = float(getattr(semantic_bank_cfg, "TAU_CLS", bank_settings.get("tau_cls", 1.0)))
        self.sigma_min = float(getattr(semantic_bank_cfg, "SIGMA_MIN", bank_settings.get("sigma_min", 0.06)))
        self.sigma_max = float(getattr(semantic_bank_cfg, "SIGMA_MAX", bank_settings.get("sigma_max", 0.35)))
        self.object_filter_weight = float(getattr(semantic_bank_cfg, "OBJECT_FILTER_WEIGHT", bank_settings.get("object_filter_weight", 0.05)))
        self.ema_momentum = float(getattr(semantic_bank_cfg, "EMA_MOMENTUM", bank_settings.get("ema_momentum", 0.95)))
        self.ema_min_count = int(getattr(semantic_bank_cfg, "EMA_MIN_COUNT", bank_settings.get("ema_min_count", 8)))
        self.loss_clip_teacher_weight = float(getattr(
            semantic_bank_cfg, "LOSS_CLIP_TEACHER_WEIGHT", bank_settings.get("loss_clip_teacher_weight", 0.02)
        ))
        self.union_teacher_weight = float(getattr(
            semantic_bank_cfg, "UNION_TEACHER_WEIGHT", bank_settings.get("union_teacher_weight", 1.0)
        ))
        self.subobj_teacher_weight = float(getattr(
            semantic_bank_cfg, "SUBOBJ_TEACHER_WEIGHT", bank_settings.get("subobj_teacher_weight", 0.5)
        ))
        self.debug_interval = int(getattr(semantic_bank_cfg, "DEBUG_INTERVAL", bank_settings.get("debug_interval", 100)))

        statistics = get_dataset_statistics(config)
        obj_classes = statistics["obj_classes"]
        rel_classes = statistics["rel_classes"]
        self.obj_names = list(obj_classes)
        self.predicate_names = list(rel_classes)
        self.num_active_predicates = len(self.predicate_names)

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        self.subject_adapter = MVA()
        self.object_adapter = MVA()
        self.background_classifier = nn.Linear(512, 1).to(self.device)

        self.predicate_id_by_name = {
            "__background__": 0, "above": 1, "across": 2, "against": 3, "along": 4, "and": 5, "at": 6,
            "attached to": 7, "behind": 8, "belonging to": 9, "between": 10, "carrying": 11,
            "covered in": 12, "covering": 13, "eating": 14, "flying in": 15, "for": 16, "from": 17,
            "growing on": 18, "hanging from": 19, "has": 20, "holding": 21, "in": 22, "in front of": 23,
            "laying on": 24, "looking at": 25, "lying on": 26, "made of": 27, "mounted on": 28, "near": 29,
            "of": 30, "on": 31, "on back of": 32, "over": 33, "painted on": 34, "parked on": 35,
            "part of": 36, "playing": 37, "riding": 38, "says": 39, "sitting on": 40, "standing on": 41,
            "to": 42, "under": 43, "using": 44, "walking in": 45, "walking on": 46, "watching": 47,
            "wearing": 48, "wears": 49, "with": 50
        }
        self.active_mode = "base"

        semantic_bank_axes = bank_config["semantic_bank_axes"]
        semantic_bank_features = self._encode_semantic_bank_features(semantic_bank_axes)
        self.num_bank_axes = semantic_bank_features.shape[0]
        self.register_buffer("semantic_bank_features", semantic_bank_features.float())

        predicate_descriptions = bank_config.get("predicate_descriptions", {})
        predicate_text_mean, predicate_text_variance = self._initialize_text_predicate_distribution(predicate_descriptions)
        self.register_buffer("predicate_text_mean", predicate_text_mean.float())
        self.register_buffer("predicate_text_variance", predicate_text_variance.float())
        self.register_buffer("predicate_mean_ema", predicate_text_mean.float().clone())
        self.register_buffer("predicate_variance_ema", predicate_text_variance.float().clone())
        self.register_buffer("predicate_count", torch.zeros(self.num_active_predicates - 1, dtype=torch.float32))
        self.register_buffer("debug_step", torch.zeros((), dtype=torch.long))

        subject_role_prompts = ["a photo of subject {}".format(obj_name) for obj_name in self.obj_names]
        object_role_prompts = ["a photo of object {}".format(obj_name) for obj_name in self.obj_names]
        self.register_buffer("subject_role_text_features", self._encode_text_features(subject_role_prompts).float())
        self.register_buffer("object_role_text_features", self._encode_text_features(object_role_prompts).float())
        self.register_buffer("object_filter_text_features", self._encode_object_filter_features().float())
        print("SemanticBankGaussianPredictor V2: {} bank axes, {} predicates, EMA shape {}, teacher=subject_object+union".format(
            self.num_bank_axes, self.num_active_predicates, tuple(self.predicate_mean_ema.shape)
        ))

    def _load_semantic_bank_config(self, config_path):
        if yaml is None:
            raise ImportError("PyYAML is required to load SemanticBankGaussianPredictor YAML config.")
        if os.path.isabs(config_path):
            resolved_path = config_path
        else:
            repo_root = os.path.abspath(os.path.join(curpath, "../../../../"))
            resolved_path = os.path.join(repo_root, config_path)
        with open(resolved_path, "r") as config_file:
            bank_config = yaml.safe_load(config_file)
        if not bank_config or "semantic_bank_axes" not in bank_config:
            raise ValueError("semantic bank config must define semantic_bank_axes")
        return bank_config

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

    def _encode_semantic_bank_features(self, semantic_bank_axes):
        axis_features = []
        for axis in semantic_bank_axes:
            prompts = axis.get("prompts", None)
            if prompts is None:
                prompts = [axis.get("positive", axis["name"])]
            prompt_features = self._encode_text_features(prompts)
            axis_features.append(F.normalize(prompt_features.mean(dim=0, keepdim=True), dim=-1))
        return torch.cat(axis_features, dim=0)

    def _compute_bank_activation(self, feature):
        normalized_feature = F.normalize(feature.float(), dim=-1)
        return torch.sigmoid((normalized_feature @ self.semantic_bank_features.t()) / self.tau_bank)

    def _initialize_text_predicate_distribution(self, predicate_descriptions):
        predicate_mean = []
        predicate_variance = []

        for predicate_name in self.predicate_names[1:]:
            descriptions = predicate_descriptions.get(predicate_name, None)
            if not descriptions:
                descriptions = [
                    "a photo where the subject is {} the object".format(predicate_name),
                    "a visual relationship where the subject is {} the object".format(predicate_name),
                ]
            description_features = self._encode_text_features(descriptions)
            description_activations = self._compute_bank_activation(description_features)

            mean = description_activations.mean(dim=0)
            variance = description_activations.var(dim=0, unbiased=False)
            variance = variance.clamp(min=self.sigma_min ** 2, max=self.sigma_max ** 2)
            predicate_mean.append(mean)
            predicate_variance.append(variance)

        return torch.stack(predicate_mean, dim=0), torch.stack(predicate_variance, dim=0)

    def _encode_object_filter_features(self):
        filter_table = pd.read_csv(os.path.join(curpath, "filter_total.csv")).iloc[:, 1:]
        feature_tensor = torch.zeros(
            len(self.obj_names), self.num_active_predicates, 512, device=self.device, dtype=torch.float32
        )
        prompts = []
        locations = []
        for object_idx, object_name in enumerate(self.obj_names):
            if object_name not in filter_table.columns:
                continue
            for predicate_idx, predicate_name in enumerate(self.predicate_names):
                if predicate_idx == 0:
                    continue
                predicate_row = self.predicate_id_by_name.get(predicate_name, None)
                if predicate_row is None or predicate_row >= len(filter_table):
                    continue
                prompt = str(filter_table[object_name].iloc[predicate_row]).strip()
                if not prompt or prompt == "__background__" or prompt.lower() == "nan":
                    continue
                prompts.append(prompt)
                locations.append((object_idx, predicate_idx))

        if prompts:
            encoded_features = self._encode_text_features(prompts)
            for feature, (object_idx, predicate_idx) in zip(encoded_features, locations):
                feature_tensor[object_idx, predicate_idx] = feature
        return feature_tensor

    def _get_predicate_distribution(self):
        count_mask = (self.predicate_count >= self.ema_min_count).float().unsqueeze(-1)
        predicate_mean = count_mask * self.predicate_mean_ema + (1.0 - count_mask) * self.predicate_text_mean
        predicate_variance = count_mask * self.predicate_variance_ema + (1.0 - count_mask) * self.predicate_text_variance
        predicate_mean = predicate_mean.float().clamp(min=0.0, max=1.0)
        predicate_variance = predicate_variance.float().clamp(min=self.sigma_min ** 2, max=self.sigma_max ** 2)
        return predicate_mean, predicate_variance

    def _compute_gaussian_logits(self, visual_bank_activation):
        predicate_mean, predicate_variance = self._get_predicate_distribution()
        bank_delta = visual_bank_activation.float().unsqueeze(1) - predicate_mean.unsqueeze(0)
        mahalanobis_distance = bank_delta.pow(2) / predicate_variance.unsqueeze(0)
        return -0.5 * mahalanobis_distance.mean(dim=-1) / self.tau_cls

    def _compute_object_filter_logits(self, image_features, relation_indexes, subject_classes, object_classes):
        subject_features = F.normalize(image_features[relation_indexes[:, 0], 0, :].float(), dim=-1)
        object_features = F.normalize(image_features[relation_indexes[:, 1], 0, :].float(), dim=-1)
        subject_filter_features = self.object_filter_text_features[subject_classes]
        object_filter_features = self.object_filter_text_features[object_classes]
        subject_logits = torch.bmm(subject_filter_features, subject_features.unsqueeze(-1)).squeeze(-1)
        object_logits = torch.bmm(object_filter_features, object_features.unsqueeze(-1)).squeeze(-1)
        object_filter_logits = (subject_logits + object_logits) / (2.0 * self.tau_bank)
        object_filter_logits[:, 0] = 0.0
        return object_filter_logits

    def _update_predicate_ema(self, visual_bank_activation, relation_labels):
        if relation_labels is None:
            return
        with torch.no_grad():
            foreground_mask = relation_labels > 0
            if not foreground_mask.any():
                return
            foreground_activation = visual_bank_activation.detach()[foreground_mask].float()
            foreground_labels = relation_labels[foreground_mask].long() - 1
            for predicate_idx in foreground_labels.unique():
                class_mask = foreground_labels == predicate_idx
                class_activation = foreground_activation[class_mask]
                batch_mean = class_activation.mean(dim=0)
                batch_variance = class_activation.var(dim=0, unbiased=False).clamp(
                    min=self.sigma_min ** 2, max=self.sigma_max ** 2
                )
                idx = int(predicate_idx.item())
                if self.predicate_count[idx] <= 0:
                    self.predicate_mean_ema[idx].copy_(batch_mean)
                    self.predicate_variance_ema[idx].copy_(batch_variance)
                else:
                    self.predicate_mean_ema[idx].mul_(self.ema_momentum).add_(batch_mean, alpha=1.0 - self.ema_momentum)
                    self.predicate_variance_ema[idx].mul_(self.ema_momentum).add_(
                        batch_variance, alpha=1.0 - self.ema_momentum
                    )
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

    def _maybe_print_debug(self, student_bank_activation, teacher_bank_activation, gaussian_logits):
        if self.debug_interval <= 0:
            return
        self.debug_step += 1
        if int(self.debug_step.item()) % self.debug_interval != 0:
            return
        _, predicate_variance = self._get_predicate_distribution()
        print(
            "SemanticBank debug step {}: student {:.4f}/{:.4f}, teacher {:.4f}/{:.4f}, "
            "gaussian {:.4f}/{:.4f}, count {:.2f}/{:.2f}, var {:.4f}/{:.4f}".format(
                int(self.debug_step.item()),
                float(student_bank_activation.float().mean().detach().cpu()),
                float(student_bank_activation.float().std(unbiased=False).detach().cpu()),
                float(teacher_bank_activation.float().mean().detach().cpu()),
                float(teacher_bank_activation.float().std(unbiased=False).detach().cpu()),
                float(gaussian_logits.float().mean().detach().cpu()),
                float(gaussian_logits.float().std(unbiased=False).detach().cpu()),
                float(self.predicate_count.float().mean().detach().cpu()),
                float(self.predicate_count.float().min().detach().cpu()),
                float(predicate_variance.float().mean().detach().cpu()),
                float(predicate_variance.float().min().detach().cpu()),
            )
        )

    def updata(self, mode):
        if mode != self.active_mode:
            print("SemanticBankGaussianPredictor is initialized for {} predicates; requested mode {} is ignored.".format(
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
        clip_teacher_losses = []

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

                subject_text_features = self.subject_role_text_features[subject_classes].to(image_features.dtype)
                object_text_features = self.object_role_text_features[object_classes].to(image_features.dtype)
                subject_relation_features = self.subject_adapter(
                    image_features[subject_box_indexes],
                    image_features[object_box_indexes],
                    subject_text_features
                )
                object_relation_features = self.object_adapter(
                    image_features[object_box_indexes],
                    image_features[subject_box_indexes],
                    object_text_features
                )
                student_relation_features = F.normalize(
                    (subject_relation_features + object_relation_features) / 2.0, dim=-1
                )
                student_bank_activation = self._compute_bank_activation(student_relation_features)
                foreground_logits = self._compute_gaussian_logits(student_bank_activation)
                background_logit = self.background_classifier(
                    student_relation_features.to(self.background_classifier.weight.dtype)
                ).float()
                relation_logits = torch.cat([background_logit, foreground_logits], dim=-1)

                object_filter_logits = self._compute_object_filter_logits(
                    image_features, relation_indexes, subject_classes, object_classes
                )
                relation_logits = relation_logits + self.object_filter_weight * object_filter_logits
                rel_dists.append(relation_logits)

                if self.training:
                    union_crops = [
                        self._crop_to_pil(
                            image,
                            proposals[image_idx].bbox[int(subject_idx.item())],
                            proposals[image_idx].bbox[int(object_idx.item())]
                        )
                        for subject_idx, object_idx in relation_indexes
                    ]
                    union_features = F.normalize(self._encode_clip_image_batch(union_crops), dim=-1)
                    subobj_teacher_features = F.normalize(
                        (image_features[subject_box_indexes, 0, :] + image_features[object_box_indexes, 0, :]).float() / 2.0,
                        dim=-1
                    )
                    union_teacher_features = F.normalize(union_features[:, 0, :].float(), dim=-1)
                    teacher_relation_features = F.normalize(
                        self.union_teacher_weight * union_teacher_features
                        + self.subobj_teacher_weight * subobj_teacher_features,
                        dim=-1
                    )
                    teacher_bank_activation = self._compute_bank_activation(teacher_relation_features)
                    clip_teacher_loss = F.mse_loss(
                        student_bank_activation.float(),
                        teacher_bank_activation.detach().float()
                    )
                    clip_teacher_losses.append(clip_teacher_loss)
                    self._update_predicate_ema(student_bank_activation, rel_labels[image_idx].to(self.device))
                    self._maybe_print_debug(
                        student_bank_activation,
                        teacher_bank_activation,
                        foreground_logits
                    )

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        if self.training:
            if clip_teacher_losses:
                add_losses["loss_clip_teacher"] = self.loss_clip_teacher_weight * torch.stack(clip_teacher_losses).mean()
        return obj_dists, rel_dists, add_losses



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
