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
try:
    import yaml
except ImportError:
    yaml = None
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import os, sys
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

class EdgePrivilegedAdapter(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, init_scale=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim).half()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False).half()
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False).half()
        self.scale = nn.Parameter(torch.tensor(init_scale).half())

    def forward(self, image_features):
        residual = self.fc2(self.relu(self.fc1(self.layer_norm(image_features))))
        image_features = image_features + self.scale.to(image_features.dtype) * residual
        return image_features / image_features.norm(dim=-1, keepdim=True)

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
        self.use_edge_map = config.MODEL.ROI_RELATION_HEAD.USE_EDGE_MAP
        self.edge_map_beta = config.MODEL.ROI_RELATION_HEAD.EDGE_MAP_BETA

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

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None, edge_maps=None):
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
        self.use_edge_map = config.MODEL.ROI_RELATION_HEAD.USE_EDGE_MAP
        self.edge_map_beta = config.MODEL.ROI_RELATION_HEAD.EDGE_MAP_BETA if self.use_edge_map else 0.0
        self.edge_map_train_only = config.MODEL.ROI_RELATION_HEAD.EDGE_MAP_TRAIN_ONLY if self.use_edge_map else True
        self.use_edge_adapter = config.MODEL.ROI_RELATION_HEAD.USE_EDGE_ADAPTER if self.use_edge_map else False
        self.edge_distill_weight = config.MODEL.ROI_RELATION_HEAD.EDGE_DISTILL_WEIGHT if self.use_edge_map else 0.0

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        self.device=config.MODEL.DEVICE
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.adaper_clip1 = MVA()
        self.adaper_clip2 = MVA()
        self.edge_adapter = EdgePrivilegedAdapter() if self.use_edge_adapter else None
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

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None, edge_maps=None):
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
        add_losses = {}
        edge_distill_losses = []
        for i in range(len(num_rels)):
            rel_dist_per_batch=[]
            union_imges=[]
            image_tensor=[]
            edge_tensor=[]
            with torch.no_grad():
                for j in range(len(proposals[i].bbox)):
                    union_img = crop_and_resize(img[i].unsqueeze(0), proposals[i].bbox[j], proposals[i].bbox[j])
                    iimg = union_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    iimg = Image.fromarray(np.uint8(iimg))
                    union_img = self.clip_preprocess(iimg).unsqueeze(0).to(self.device)
                    image_tensor.append(union_img)
                    if self.use_edge_map and edge_maps is not None:
                        edge_img = crop_and_resize(edge_maps[i].unsqueeze(0), proposals[i].bbox[j], proposals[i].bbox[j])
                        if edge_img.shape[1] == 1:
                            edge_img = edge_img.repeat(1, 3, 1, 1)
                        eimg = edge_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                        eimg = Image.fromarray(np.uint8(eimg))
                        edge_img = self.clip_preprocess(eimg).unsqueeze(0).to(self.device)
                        edge_tensor.append(edge_img)
                image_tensor = torch.cat(image_tensor)

                image_features = self.clip_model.encode_image(image_tensor)
                if self.use_edge_map and edge_maps is not None and len(edge_tensor) > 0:
                    edge_tensor = torch.cat(edge_tensor)
                    edge_features = self.clip_model.encode_image(edge_tensor)
                else:
                    edge_features = None

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if self.use_edge_adapter:
                student_features = self.edge_adapter(image_features)
            else:
                student_features = image_features

            if self.training and self.use_edge_map and edge_features is not None and self.edge_distill_weight > 0:
                edge_features = edge_features / edge_features.norm(dim=-1, keepdim=True)
                teacher_features = image_features + self.edge_map_beta * edge_features
                teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                teacher_features = teacher_features.detach()
                edge_distill_loss = 1 - F.cosine_similarity(student_features, teacher_features, dim=-1).mean()
                edge_distill_losses.append(edge_distill_loss)

            image_features = student_features

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

        if len(edge_distill_losses) > 0:
            edge_distill_loss_raw = torch.stack(edge_distill_losses).mean()
            add_losses["loss_edge_distill"] = self.edge_distill_weight * edge_distill_loss_raw
            add_losses["loss_edge_distill_raw"] = edge_distill_loss_raw.detach()
            add_losses["edge_teacher_student_cos"] = (1 - edge_distill_loss_raw).detach()
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
        self.use_edge_map = config.MODEL.ROI_RELATION_HEAD.USE_EDGE_MAP
        self.edge_map_beta = config.MODEL.ROI_RELATION_HEAD.EDGE_MAP_BETA if self.use_edge_map else 0.0
        self.use_edge_adapter = config.MODEL.ROI_RELATION_HEAD.USE_EDGE_ADAPTER if self.use_edge_map else False
        self.edge_distill_weight = config.MODEL.ROI_RELATION_HEAD.EDGE_DISTILL_WEIGHT if self.use_edge_map else 0.0

        semantic_bank_cfg = config.MODEL.ROI_RELATION_HEAD.SEMANTIC_BANK
        bank_config = self._load_semantic_bank_config(semantic_bank_cfg.CONFIG_PATH)
        bank_settings = bank_config.get("settings", {})
        self.tau_bank = float(getattr(semantic_bank_cfg, "TAU_BANK", bank_settings.get("tau_bank", 0.07)))
        self.tau_cls = float(getattr(semantic_bank_cfg, "TAU_CLS", bank_settings.get("tau_cls", 1.0)))
        self.sigma_min = float(getattr(semantic_bank_cfg, "SIGMA_MIN", bank_settings.get("sigma_min", 0.06)))
        self.sigma_max = float(getattr(semantic_bank_cfg, "SIGMA_MAX", bank_settings.get("sigma_max", 0.35)))
        self.loss_nll_weight = float(getattr(semantic_bank_cfg, "LOSS_NLL_WEIGHT", bank_settings.get("loss_nll_weight", 0.2)))
        self.loss_prior_weight = float(getattr(semantic_bank_cfg, "LOSS_PRIOR_WEIGHT", bank_settings.get("loss_prior_weight", 0.05)))
        self.loss_clip_raw_weight = float(getattr(semantic_bank_cfg, "LOSS_CLIP_RAW_WEIGHT", bank_settings.get("loss_clip_raw_weight", 0.05)))
        self.object_filter_weight = float(getattr(semantic_bank_cfg, "OBJECT_FILTER_WEIGHT", bank_settings.get("object_filter_weight", 0.2)))
        self.entropy_weight = float(bank_settings.get("entropy_weight", 0.5))
        self.uncertainty_weight = float(bank_settings.get("uncertainty_weight", 0.3))
        self.text_variance_weight = float(bank_settings.get("text_variance_weight", 0.2))

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
        self.edge_adapter = EdgePrivilegedAdapter() if self.use_edge_adapter else None
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
        positive_axis_prompts = [axis["positive"] for axis in semantic_bank_axes]
        negative_axis_prompts = [axis["negative"] for axis in semantic_bank_axes]
        semantic_bank_positive_features = self._encode_text_features(positive_axis_prompts)
        semantic_bank_negative_features = self._encode_text_features(negative_axis_prompts)
        self.register_buffer("semantic_bank_positive_features", semantic_bank_positive_features.float())
        self.register_buffer("semantic_bank_negative_features", semantic_bank_negative_features.float())

        predicate_descriptions = bank_config.get("predicate_descriptions", {})
        predicate_mean_init, predicate_log_variance_init = self._initialize_predicate_distribution(predicate_descriptions)
        self.register_buffer("predicate_mean_init", predicate_mean_init.float())
        self.register_buffer("predicate_log_variance_init", predicate_log_variance_init.float())
        self.predicate_mean = nn.Parameter(predicate_mean_init.float().clone())
        self.predicate_log_variance = nn.Parameter(predicate_log_variance_init.float().clone())

        subject_role_prompts = ["a photo of subject {}".format(obj_name) for obj_name in self.obj_names]
        object_role_prompts = ["a photo of object {}".format(obj_name) for obj_name in self.obj_names]
        self.register_buffer("subject_role_text_features", self._encode_text_features(subject_role_prompts).float())
        self.register_buffer("object_role_text_features", self._encode_text_features(object_role_prompts).float())
        self.register_buffer("object_filter_text_features", self._encode_object_filter_features().float())

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
                prompt_batch = ["a photo of {}".format(prompt) for prompt in prompts[start_idx:start_idx + batch_size]]
                text_tokens = clip.tokenize(prompt_batch).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = F.normalize(text_features.float(), dim=-1)
                features.append(text_features)
        return torch.cat(features, dim=0)

    def _compute_bank_activation(self, feature):
        normalized_feature = F.normalize(feature.float(), dim=-1)
        positive_similarity = normalized_feature @ self.semantic_bank_positive_features.t()
        negative_similarity = normalized_feature @ self.semantic_bank_negative_features.t()
        return torch.sigmoid((positive_similarity - negative_similarity) / self.tau_bank)

    def _initialize_predicate_distribution(self, predicate_descriptions):
        predicate_mean_init = []
        predicate_log_variance_init = []
        log_bank_count = math.log(len(self.semantic_bank_positive_features))

        for predicate_name in self.predicate_names[1:]:
            descriptions = predicate_descriptions.get(predicate_name, None)
            if not descriptions:
                descriptions = ["subject is {} object".format(predicate_name)]
            description_features = self._encode_text_features(descriptions)
            description_activations = self._compute_bank_activation(description_features)

            predicate_mean = description_activations.mean(dim=0)
            description_variance = description_activations.var(dim=0, unbiased=False)
            normalized_mean = predicate_mean / predicate_mean.sum().clamp(min=1e-6)
            predicate_entropy = -(normalized_mean * torch.log(normalized_mean.clamp(min=1e-6))).sum() / log_bank_count
            local_uncertainty = 4.0 * predicate_mean * (1.0 - predicate_mean)
            variance_ratio = (
                self.entropy_weight * predicate_entropy
                + self.uncertainty_weight * local_uncertainty
                + self.text_variance_weight * description_variance
            ).clamp(min=0.0, max=1.0)
            predicate_variance = self.sigma_min ** 2 + (self.sigma_max ** 2 - self.sigma_min ** 2) * variance_ratio

            predicate_mean_init.append(predicate_mean)
            predicate_log_variance_init.append(torch.log(predicate_variance.clamp(min=1e-6)))

        return torch.stack(predicate_mean_init, dim=0), torch.stack(predicate_log_variance_init, dim=0)

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

    def _compute_gaussian_logits(self, visual_bank_activation, apply_temperature=True):
        log_variance_min = math.log(self.sigma_min ** 2)
        log_variance_max = math.log(self.sigma_max ** 2)
        predicate_mean = self.predicate_mean.float()
        predicate_log_variance = self.predicate_log_variance.float().clamp(log_variance_min, log_variance_max)
        predicate_variance = torch.exp(predicate_log_variance)
        bank_delta = visual_bank_activation.float().unsqueeze(1) - predicate_mean.unsqueeze(0)
        gaussian_logits = -0.5 * ((bank_delta.pow(2) / predicate_variance.unsqueeze(0)) + predicate_log_variance.unsqueeze(0)).sum(dim=-1)
        if apply_temperature:
            gaussian_logits = gaussian_logits / self.tau_cls
        return gaussian_logits

    def _compute_object_filter_logits(self, image_features, relation_index, subject_class, object_class):
        subject_feature = F.normalize(image_features[relation_index[0], 0, :].float().unsqueeze(0), dim=-1)
        object_feature = F.normalize(image_features[relation_index[1], 0, :].float().unsqueeze(0), dim=-1)
        subject_filter_features = self.object_filter_text_features[subject_class]
        object_filter_features = self.object_filter_text_features[object_class]
        subject_logits = subject_feature @ subject_filter_features.t()
        object_logits = object_feature @ object_filter_features.t()
        object_filter_logits = (subject_logits + object_logits) / (2.0 * self.tau_bank)
        object_filter_logits[:, 0] = 0.0
        return object_filter_logits

    def _compute_auxiliary_losses(self, visual_bank_activations, raw_bank_activations, relation_labels):
        add_losses = {}
        if not visual_bank_activations:
            return add_losses

        visual_bank_activation = torch.cat(visual_bank_activations, dim=0).float()
        raw_bank_activation = torch.cat(raw_bank_activations, dim=0).float()
        relation_label = torch.cat(relation_labels, dim=0).long()
        foreground_mask = relation_label > 0

        if foreground_mask.any():
            foreground_visual_bank_activation = visual_bank_activation[foreground_mask]
            foreground_label = relation_label[foreground_mask] - 1
            foreground_logit = self._compute_gaussian_logits(foreground_visual_bank_activation, apply_temperature=False)
            normalizing_constant = 0.5 * foreground_visual_bank_activation.shape[1] * math.log(2.0 * math.pi)
            foreground_nll = (normalizing_constant - foreground_logit.gather(1, foreground_label.view(-1, 1))).mean()
            add_losses["loss_gaussian_nll"] = self.loss_nll_weight * foreground_nll

        log_variance_min = math.log(self.sigma_min ** 2)
        log_variance_max = math.log(self.sigma_max ** 2)
        current_log_variance = self.predicate_log_variance.float().clamp(log_variance_min, log_variance_max)
        gaussian_prior = (
            F.mse_loss(self.predicate_mean.float(), self.predicate_mean_init.float())
            + F.mse_loss(current_log_variance, self.predicate_log_variance_init.float())
        )
        clip_raw_regularizer = F.mse_loss(visual_bank_activation, raw_bank_activation)
        add_losses["loss_gaussian_prior"] = self.loss_prior_weight * gaussian_prior
        add_losses["loss_clip_raw_reg"] = self.loss_clip_raw_weight * clip_raw_regularizer
        return add_losses

    def updata(self, mode):
        if mode != self.active_mode:
            print("SemanticBankGaussianPredictor is initialized for {} predicates; requested mode {} is ignored.".format(
                self.active_mode, mode
            ))

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None, edge_maps=None):
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
        edge_distill_losses = []
        visual_bank_activations = []
        raw_bank_activations = []
        relation_labels = []

        for image_idx in range(len(num_rels)):
            image_tensor = []
            edge_tensor = []
            with torch.no_grad():
                for box_idx in range(len(proposals[image_idx].bbox)):
                    object_crop = crop_and_resize(img[image_idx].unsqueeze(0), proposals[image_idx].bbox[box_idx], proposals[image_idx].bbox[box_idx])
                    object_array = object_crop[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    object_image = Image.fromarray(np.uint8(object_array))
                    image_tensor.append(self.clip_preprocess(object_image).unsqueeze(0).to(self.device))

                    if self.use_edge_map and edge_maps is not None:
                        edge_crop = crop_and_resize(edge_maps[image_idx].unsqueeze(0), proposals[image_idx].bbox[box_idx], proposals[image_idx].bbox[box_idx])
                        if edge_crop.shape[1] == 1:
                            edge_crop = edge_crop.repeat(1, 3, 1, 1)
                        edge_array = edge_crop[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                        edge_image = Image.fromarray(np.uint8(edge_array))
                        edge_tensor.append(self.clip_preprocess(edge_image).unsqueeze(0).to(self.device))

                image_tensor = torch.cat(image_tensor, dim=0)
                image_features = self.clip_model.encode_image(image_tensor)
                if self.use_edge_map and edge_maps is not None and edge_tensor:
                    edge_tensor = torch.cat(edge_tensor, dim=0)
                    edge_features = self.clip_model.encode_image(edge_tensor)
                else:
                    edge_features = None

            image_features = F.normalize(image_features, dim=-1)
            if self.use_edge_adapter:
                student_features = self.edge_adapter(image_features)
            else:
                student_features = image_features

            if self.training and self.use_edge_map and edge_features is not None and self.edge_distill_weight > 0:
                edge_features = F.normalize(edge_features, dim=-1)
                teacher_features = F.normalize(image_features + self.edge_map_beta * edge_features, dim=-1).detach()
                edge_distill_loss = 1 - F.cosine_similarity(student_features, teacher_features, dim=-1).mean()
                edge_distill_losses.append(edge_distill_loss)

            image_features = student_features
            relation_logits_per_image = []

            for relation_index in rel_pair_idxs[image_idx]:
                subject_box_idx = relation_index[0]
                object_box_idx = relation_index[1]
                subject_class = int(obj_preds[image_idx][subject_box_idx].item())
                object_class = int(obj_preds[image_idx][object_box_idx].item())

                subject_text_feature = self.subject_role_text_features[subject_class].to(image_features.dtype)
                object_text_feature = self.object_role_text_features[object_class].to(image_features.dtype)
                subject_relation_feature = self.subject_adapter(
                    image_features[subject_box_idx].unsqueeze(0),
                    image_features[object_box_idx].unsqueeze(0),
                    subject_text_feature
                )
                object_relation_feature = self.object_adapter(
                    image_features[object_box_idx].unsqueeze(0),
                    image_features[subject_box_idx].unsqueeze(0),
                    object_text_feature
                )
                relation_clip_feature = F.normalize((subject_relation_feature + object_relation_feature) / 2.0, dim=-1)
                visual_bank_activation = self._compute_bank_activation(relation_clip_feature)
                foreground_logits = self._compute_gaussian_logits(visual_bank_activation)
                background_logit = self.background_classifier(
                    relation_clip_feature.to(self.background_classifier.weight.dtype)
                ).float()
                relation_logits = torch.cat([background_logit, foreground_logits], dim=-1)

                object_filter_logits = self._compute_object_filter_logits(
                    image_features, relation_index, subject_class, object_class
                )
                relation_logits = relation_logits + self.object_filter_weight * object_filter_logits
                relation_logits_per_image.append(relation_logits)

                if self.training:
                    raw_relation_feature = F.normalize(
                        (image_features[subject_box_idx, 0, :] + image_features[object_box_idx, 0, :]).float().unsqueeze(0) / 2.0,
                        dim=-1
                    )
                    raw_bank_activation = self._compute_bank_activation(raw_relation_feature)
                    visual_bank_activations.append(visual_bank_activation)
                    raw_bank_activations.append(raw_bank_activation)

            if relation_logits_per_image:
                rel_dists.append(torch.cat(relation_logits_per_image, dim=0))
            else:
                rel_dists.append(torch.empty(0, self.num_active_predicates, device=self.device))

            if self.training and rel_labels is not None:
                relation_labels.append(rel_labels[image_idx].to(self.device))

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        if self.training:
            add_losses.update(self._compute_auxiliary_losses(visual_bank_activations, raw_bank_activations, relation_labels))
        if edge_distill_losses:
            edge_distill_loss_raw = torch.stack(edge_distill_losses).mean()
            add_losses["loss_edge_distill"] = self.edge_distill_weight * edge_distill_loss_raw
            add_losses["loss_edge_distill_raw"] = edge_distill_loss_raw.detach()
            add_losses["edge_teacher_student_cos"] = (1 - edge_distill_loss_raw).detach()
        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
