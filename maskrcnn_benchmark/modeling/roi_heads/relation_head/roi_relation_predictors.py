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
from .utils_motifs import obj_edge_vectors
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
            if text_fea.dim() == 1:
                text_fea = text_fea.unsqueeze(0).expand(x.shape[0], -1)
            elif text_fea.shape[0] == 1 and x.shape[0] != 1:
                text_fea = text_fea.expand(x.shape[0], -1)
            x=torch.cat([x, text_fea.to(device=x.device, dtype=x.dtype)], dim=-1)
            x=self.linear(x)
            x=self.relu(x)
        ratio = 0.5
        sub_features= ratio * x + (1 - ratio) * sub_features[:,0,:]

        return sub_features

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


class RIPHead(nn.Module):
    def __init__(self, num_obj_cls, axis_dim, hidden_dim, obj_embed_dim, max_edit):
        super(RIPHead, self).__init__()
        self.max_edit = max_edit
        self.obj_embed = nn.Embedding(num_obj_cls, obj_embed_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(axis_dim + obj_embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.program_encoder = nn.Sequential(
            nn.Linear(axis_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edit = nn.Linear(hidden_dim, axis_dim)

    def forward(self, evidence, programs, subj_labels, obj_labels):
        evidence = evidence.float()
        programs = programs.float()
        subj_embed = self.obj_embed(subj_labels.long()).float()
        obj_embed = self.obj_embed(obj_labels.long()).float()
        state = self.state_encoder(torch.cat([evidence, subj_embed, obj_embed], dim=-1))
        program_state = self.program_encoder(programs)
        joint = torch.tanh(program_state + state.unsqueeze(1))
        delta = self.max_edit * torch.tanh(self.edit(joint))
        edited_evidence = evidence.unsqueeze(1) + delta
        energy = (edited_evidence - programs).abs().sum(dim=-1)
        return -energy, delta, energy


@registry.ROI_RELATION_PREDICTOR.register("ClipRIPPredictor")
class ClipRIPPredictor(ClipPredictor):
    def __init__(self, config, in_channels):
        super(ClipRIPPredictor, self).__init__(config, in_channels)

        rip_cfg = config.MODEL.ROI_RELATION_HEAD.RIP
        self.rip_logit_weight = rip_cfg.LOGIT_WEIGHT
        self.rip_margin = rip_cfg.MARGIN
        self.rip_hard_neg_topk = rip_cfg.HARD_NEG_TOPK
        self.rip_detach_evidence = rip_cfg.DETACH_EVIDENCE
        self.rip_energy_weight = rip_cfg.ENERGY_WEIGHT
        self.rip_rank_weight = rip_cfg.RANK_WEIGHT
        self.rip_cls_weight = rip_cfg.CLS_WEIGHT
        self.rip_cls_gamma = rip_cfg.CLS_GAMMA
        self.rip_sparse_weight = rip_cfg.SPARSE_WEIGHT
        self.rip_anchor_weight = rip_cfg.ANCHOR_WEIGHT
        self.rip_direction_weight = rip_cfg.DIRECTION_WEIGHT
        self.rip_distill_weight = rip_cfg.DISTILL_WEIGHT
        credo_cfg = config.MODEL.ROI_RELATION_HEAD.CREDO
        self.credoDebug = credo_cfg.DEBUG
        self.credoDebugPeriod = max(1, int(credo_cfg.DEBUG_PERIOD))
        self.credoIter = 0

        self.rip_head = RIPHead(
            self.num_obj_cls,
            22,
            rip_cfg.HIDDEN_DIM,
            rip_cfg.OBJ_EMBED_DIM,
            rip_cfg.MAX_EDIT,
        ).to(self.device)

        rel_prop = torch.tensor([0.5] + list(config.MODEL.ROI_RELATION_HEAD.REL_PROP),
                                device=self.device, dtype=torch.float32)
        tail_weights = rel_prop.clamp(min=1e-6).pow(-rip_cfg.TAIL_WEIGHT_POWER)
        tail_weights = tail_weights / tail_weights[1:].mean().clamp(min=1e-6)
        tail_weights[0] = 0.0
        self.register_buffer("rip_tail_weights", tail_weights)
        self._set_active_pred_ids("base")

    def _set_active_pred_ids(self, mode):
        if mode == "base":
            active_ids = self.base
        elif mode == "novel":
            active_ids = self.novel
        elif mode == "semantic":
            active_ids = self.semantic
        else:
            active_ids = list(range(self.num_rel_cls))
        self.active_pred_ids = torch.tensor(active_ids, device=self.device, dtype=torch.long)
        label_to_col = torch.full((self.num_rel_cls,), -1, device=self.device, dtype=torch.long)
        label_to_col[self.active_pred_ids] = torch.arange(
            len(active_ids), device=self.device, dtype=torch.long
        )
        self.active_label_to_col = label_to_col

    def updata(self, mode):
        super(ClipRIPPredictor, self).updata(mode)
        self._set_active_pred_ids(mode)

    def _clip_cls_features(self, image_features):
        if image_features.dim() == 3:
            return image_features[:, 0, :]
        return image_features

    def _active_programs(self, subj_labels):
        programs = self.description_relation.index_select(1, subj_labels.long())
        return programs.permute(1, 0, 2).contiguous()

    def _center_scores(self, scores):
        scores = scores.float()
        return scores - scores.mean(dim=-1, keepdim=True)

    def _program_score(self, programs, evidence):
        return torch.einsum("rca,ra->rc", programs.float(), evidence.float())

    def _filter_similarity(self, cls_features, sub_idx, obj_idx, subj_labels, num_preds):
        out = cls_features.new_zeros((sub_idx.numel(), num_preds))
        for label in subj_labels.unique():
            mask = subj_labels == label
            rel_pos = torch.nonzero(mask, as_tuple=False).view(-1)
            text_features5 = torch.Tensor(self.texts5[int(label.item())]).to(self.device).half()
            text_features5 = text_features5.to(dtype=cls_features.dtype)
            sub_feat = cls_features.index_select(0, sub_idx.index_select(0, rel_pos))
            obj_feat = cls_features.index_select(0, obj_idx.index_select(0, rel_pos))
            text_norm = text_features5 / text_features5.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            sub_norm = sub_feat / sub_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            obj_norm = obj_feat / obj_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            sim = ((sub_norm @ text_norm.t()) + (obj_norm @ text_norm.t())) / 2 / 0.05
            out.index_copy_(0, rel_pos, sim[:, :num_preds])
        return out

    def _rip_losses(self, evidence, programs, base_scores, rip_scores, delta, energy, rel_labels):
        zero = rip_scores.sum() * 0.0
        labels = rel_labels.view(-1).long()
        mapped = self.active_label_to_col.index_select(0, labels.clamp(min=0, max=self.num_rel_cls - 1))
        pos = (labels > 0) & (mapped >= 0)
        if pos.sum() == 0:
            return {
                "loss_rip_energy": zero,
                "loss_rip_rank": zero,
                "loss_rip_cls": zero,
                "loss_rip_sparse": zero,
                "loss_rip_anchor": zero,
                "loss_rip_direction": zero,
                "loss_rip_distill": zero,
            }

        pos_cols = mapped[pos]
        pos_energy = energy[pos].gather(1, pos_cols.view(-1, 1)).squeeze(1)
        pos_scores = rip_scores[pos].gather(1, pos_cols.view(-1, 1)).squeeze(1)

        neg_scores = rip_scores[pos].clone()
        neg_scores[:, 0] = -1e4
        neg_scores.scatter_(1, pos_cols.view(-1, 1), -1e4)
        topk = min(self.rip_hard_neg_topk, max(1, neg_scores.size(1) - 1))
        hard_neg = neg_scores.topk(topk, dim=1)[0]
        rank = F.relu(self.rip_margin + hard_neg - pos_scores.unsqueeze(1)).mean(dim=1)

        sample_weights = self.rip_tail_weights.index_select(
            0, labels[pos].clamp(min=0, max=self.rip_tail_weights.numel() - 1)
        ).to(rank.device)
        sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-6)

        log_probs = F.log_softmax(rip_scores[pos].float(), dim=-1)
        log_pt = log_probs.gather(1, pos_cols.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        cls_loss = -torch.pow(1.0 - pt, self.rip_cls_gamma) * log_pt

        pos_delta = delta[pos].gather(
            1, pos_cols.view(-1, 1, 1).expand(-1, 1, delta.size(-1))
        ).squeeze(1)
        pos_program = programs[pos].gather(
            1, pos_cols.view(-1, 1, 1).expand(-1, 1, programs.size(-1))
        ).squeeze(1)
        direction = pos_program - evidence[pos].float()
        direction_loss = F.relu(-(pos_delta * direction).sum(dim=-1)).mean()
        centered_scores = self._center_scores(rip_scores)
        distill_loss = F.smooth_l1_loss(
            self._center_scores(rip_scores),
            self._center_scores(base_scores.detach().float()),
        )
        return {
            "loss_rip_energy": self.rip_energy_weight * (pos_energy * sample_weights).mean(),
            "loss_rip_rank": self.rip_rank_weight * (rank * sample_weights).mean(),
            "loss_rip_cls": self.rip_cls_weight * (cls_loss * sample_weights).mean(),
            "loss_rip_sparse": self.rip_sparse_weight * pos_delta.abs().mean(),
            "loss_rip_anchor": self.rip_anchor_weight * centered_scores.pow(2).mean(),
            "loss_rip_direction": self.rip_direction_weight * direction_loss,
            "loss_rip_distill": self.rip_distill_weight * distill_loss,
        }

    def _merge_losses(self, losses):
        if len(losses) == 0:
            zero = self.rip_head.edit.weight.sum() * 0.0
            return {
                "loss_rip_energy": zero,
                "loss_rip_rank": zero,
                "loss_rip_cls": zero,
                "loss_rip_sparse": zero,
                "loss_rip_anchor": zero,
                "loss_rip_direction": zero,
                "loss_rip_distill": zero,
            }
        return {
            key: torch.stack([loss[key] for loss in losses]).mean()
            for key in losses[0].keys()
        }

    def collectCredoDebug(self, baseScores, finalScores, energy, delta, relLabels):
        labels = relLabels.view(-1).long()
        mapped = self.active_label_to_col.index_select(0, labels.clamp(min=0, max=self.num_rel_cls - 1))
        pos = (labels > 0) & (mapped >= 0)
        stats = {
            "num_rel": int(labels.numel()),
            "active_predicates": int(baseScores.size(1)),
            "base_mean": baseScores.detach().float().mean().item(),
            "base_std": baseScores.detach().float().std().item() if baseScores.numel() > 1 else 0.0,
            "final_mean": finalScores.detach().float().mean().item(),
            "final_std": finalScores.detach().float().std().item() if finalScores.numel() > 1 else 0.0,
            "verify_pos": 0.0,
            "verify_neg": 0.0,
            "verify_margin": 0.0,
            "trans_pos": 0.0,
            "trans_neg": 0.0,
            "trans_margin": 0.0,
            "gate_pos": 0.0,
            "gate_neg": 0.0,
            "gate_active_ratio": 0.0,
            "edit_l1": delta.detach().float().abs().mean().item(),
            "distill_l1": (finalScores.detach().float() - baseScores.detach().float()).abs().mean().item(),
            "max_mem_mb": 0.0,
            "nan_or_inf": False,
        }
        if torch.cuda.is_available():
            stats["max_mem_mb"] = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        tensors = [baseScores, finalScores, energy, delta]
        stats["nan_or_inf"] = any(
            not torch.isfinite(t.detach()).all().item()
            for t in tensors
        )

        gateProb = torch.sigmoid(finalScores.detach().float() - baseScores.detach().float())
        stats["gate_active_ratio"] = (gateProb > 0.5).float().mean().item()

        if pos.sum() == 0:
            return stats

        posCols = mapped[pos]
        posEnergy = energy[pos].gather(1, posCols.view(-1, 1)).squeeze(1).detach().float()
        negEnergy = energy[pos].detach().float().clone()
        negEnergy[:, 0] = 1e4
        negEnergy.scatter_(1, posCols.view(-1, 1), 1e4)
        negEnergy = negEnergy.min(dim=1)[0]
        stats["verify_pos"] = posEnergy.mean().item()
        stats["verify_neg"] = negEnergy.mean().item()
        stats["verify_margin"] = (negEnergy - posEnergy).mean().item()

        transEnergy = (energy.detach().float() + delta.detach().float().abs().sum(dim=-1))
        posTrans = transEnergy[pos].gather(1, posCols.view(-1, 1)).squeeze(1)
        negTrans = transEnergy[pos].clone()
        negTrans[:, 0] = 1e4
        negTrans.scatter_(1, posCols.view(-1, 1), 1e4)
        negTrans = negTrans.min(dim=1)[0]
        stats["trans_pos"] = posTrans.mean().item()
        stats["trans_neg"] = negTrans.mean().item()
        stats["trans_margin"] = (negTrans - posTrans).mean().item()

        posGate = gateProb[pos].gather(1, posCols.view(-1, 1)).squeeze(1)
        negGate = gateProb[pos].clone()
        negGate[:, 0] = 0.0
        negGate.scatter_(1, posCols.view(-1, 1), 0.0)
        negGate = negGate.max(dim=1)[0]
        stats["gate_pos"] = posGate.mean().item()
        stats["gate_neg"] = negGate.mean().item()
        return stats

    def logCredoDebug(self, logger, debugStats):
        if logger is None or len(debugStats) == 0:
            return
        keys = debugStats[0].keys()
        merged = {}
        for key in keys:
            values = [stats[key] for stats in debugStats]
            if key == "num_rel":
                merged[key] = int(sum(values))
            elif key == "active_predicates":
                merged[key] = int(max(values))
            elif key == "nan_or_inf":
                merged[key] = any(values)
            elif key == "max_mem_mb":
                merged[key] = max(values)
            else:
                merged[key] = sum(values) / max(1, len(values))

        message = (
            "[CREDO DEBUG] "
            "iter={iter} num_rel={num_rel} active_predicates={active_predicates} "
            "base_mean={base_mean:.4f} base_std={base_std:.4f} "
            "final_mean={final_mean:.4f} final_std={final_std:.4f} "
            "verify_pos={verify_pos:.4f} verify_neg={verify_neg:.4f} verify_margin={verify_margin:.4f} "
            "trans_pos={trans_pos:.4f} trans_neg={trans_neg:.4f} trans_margin={trans_margin:.4f} "
            "gate_pos={gate_pos:.4f} gate_neg={gate_neg:.4f} gate_active_ratio={gate_active_ratio:.4f} "
            "edit_l1={edit_l1:.4f} distill_l1={distill_l1:.4f} max_mem_mb={max_mem_mb:.1f}"
        ).format(iter=self.credoIter, **merged)
        logger.info(message)
        if merged["nan_or_inf"]:
            logger.info("[CREDO WARNING] nan_or_inf=True")

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,img=None):
        if self.training:
            self.credoIter += 1

        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)

        text_features1 = self.text_features1
        text_features2 = self.text_features2
        text1_norm = text_features1 / text_features1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        text2_norm = text_features2 / text_features2.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        rel_dists = []
        rip_losses = []
        credoDebugStats = []
        for i in range(len(num_rels)):
            image_tensor = []
            with torch.no_grad():
                for j in range(len(proposals[i].bbox)):
                    union_img = crop_and_resize(img[i].unsqueeze(0), proposals[i].bbox[j], proposals[i].bbox[j])
                    iimg = union_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    iimg = Image.fromarray(np.uint8(iimg))
                    union_img = self.clip_preprocess(iimg).unsqueeze(0).to(self.device)
                    image_tensor.append(union_img)
                image_tensor = torch.cat(image_tensor)
                image_features = self.clip_model.encode_image(image_tensor)

            if rel_pair_idxs[i].numel() == 0:
                if self.training:
                    rel_dists.append(image_features.new_zeros((0, 2, 22)))
                else:
                    rel_dists.append(image_features.new_zeros((0, self.description_relation.size(0))))
                continue

            pair_idx = rel_pair_idxs[i].long()
            sub_idx = pair_idx[:, 0]
            obj_idx = pair_idx[:, 1]
            subj_labels = obj_preds[i].index_select(0, sub_idx).long()
            obj_labels = obj_preds[i].index_select(0, obj_idx).long()

            sub_features = image_features.index_select(0, sub_idx)
            obj_features = image_features.index_select(0, obj_idx)
            text_sub = self.text_features3.index_select(0, subj_labels)
            text_obj = self.text_features4.index_select(0, obj_labels)

            cross_output1 = self.adaper_clip1(sub_features, obj_features, text_sub)
            cross_output2 = self.adaper_clip2(obj_features, sub_features, text_obj)
            cross_output = (cross_output1 + cross_output2) / 2

            cross_norm = cross_output / cross_output.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            similarity1 = cross_norm @ text1_norm.t()
            similarity2 = cross_norm @ text2_norm.t()
            evidence = (similarity1 - similarity2) / 0.05

            cls_features = self._clip_cls_features(image_features)
            cls_pair = (cls_features.index_select(0, sub_idx) + cls_features.index_select(0, obj_idx)) / 2
            cls_pair_norm = cls_pair / cls_pair.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            origin_1 = cls_pair_norm @ text1_norm.t()
            origin_2 = cls_pair_norm @ text2_norm.t()
            origin_evidence = (origin_1 - origin_2) / 0.05

            if self.adaper_clip1.training:
                probs = torch.stack([evidence, origin_evidence], dim=1)
            else:
                programs = self._active_programs(subj_labels)
                program_score = self._program_score(programs, evidence)
                similarity3 = self._filter_similarity(
                    cls_features, sub_idx, obj_idx, subj_labels, programs.size(1)
                )
                probs = program_score * 0.2 + similarity3.float() * 0.8
                rip_evidence = evidence.detach() if self.rip_detach_evidence else evidence
                rip_scores, delta, energy = self.rip_head(
                    rip_evidence, programs, subj_labels, obj_labels
                )
                probs = probs + self.rip_logit_weight * self._center_scores(rip_scores)
                probs = probs.to(dtype=evidence.dtype)

            if self.training:
                programs = self._active_programs(subj_labels)
                base_scores = self._program_score(programs, evidence)
                rip_evidence = evidence.detach() if self.rip_detach_evidence else evidence
                rip_scores, delta, energy = self.rip_head(
                    rip_evidence, programs, subj_labels, obj_labels
                )
                final_scores = base_scores + self.rip_logit_weight * self._center_scores(rip_scores)
                rip_losses.append(
                    self._rip_losses(rip_evidence, programs, base_scores, rip_scores, delta, energy, rel_labels[i])
                )
                if self.credoDebug and self.credoIter % self.credoDebugPeriod == 0:
                    credoDebugStats.append(
                        self.collectCredoDebug(base_scores, final_scores, energy, delta, rel_labels[i])
                    )

            rel_dists.append(probs)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)
        add_losses = self._merge_losses(rip_losses) if self.training else {}
        if self.training and self.credoDebug and self.credoIter % self.credoDebugPeriod == 0:
            self.logCredoDebug(logger, credoDebugStats)
        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("ClipCREDOPredictor")
class ClipCREDOPredictor(ClipRIPPredictor):
    pass


@registry.ROI_RELATION_PREDICTOR.register("ClipReSAPredictor")
class ClipReSAPredictor(ClipPredictor):
    def __init__(self, config, in_channels):
        super(ClipReSAPredictor, self).__init__(config, in_channels)

        resa_cfg = config.MODEL.ROI_RELATION_HEAD.RESA
        self.atom_dim = resa_cfg.ATOM_DIM
        self.atom_weight = resa_cfg.ATOM_WEIGHT
        self.recon_weight = resa_cfg.RECON_WEIGHT
        self.sparse_weight = resa_cfg.SPARSE_WEIGHT
        self.diversity_weight = resa_cfg.DIVERSITY_WEIGHT
        self.consist_weight = resa_cfg.CONSIST_WEIGHT
        self.contrast_weight = resa_cfg.CONTRAST_WEIGHT

        self.atom_encoder = ReSASparseAtomEncoder(
            512,
            self.atom_dim,
            config.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE,
        ).to(self.device).half()
        self.atom_dictionary = nn.Parameter(torch.empty(self.atom_dim, 512, device=self.device, dtype=torch.float16))
        self.pred_atom_logits = nn.Parameter(torch.empty(self.num_rel_cls, self.atom_dim, device=self.device, dtype=torch.float16))
        self.atom_logit_scale = nn.Parameter(torch.tensor(10.0, device=self.device, dtype=torch.float16))
        nn.init.xavier_normal_(self.atom_dictionary)
        nn.init.uniform_(self.pred_atom_logits, 0.0, 0.1)

        if resa_cfg.INIT_PRED_ATOMS_FROM_TEXT:
            rel_embed_vecs = obj_edge_vectors(self._all_rel_classes(), wv_dir=config.GLOVE_DIR, wv_dim=config.MODEL.ROI_RELATION_HEAD.EMBED_DIM)
            self._init_pred_atoms_from_text(rel_embed_vecs)

        self._confusable_groups = self._build_confusable_groups(self._all_rel_classes())
        self._set_active_pred_ids("base")

    def _all_rel_classes(self):
        rel_classes = [None] * len(self.id_dict)
        for name, idx in self.id_dict.items():
            rel_classes[idx] = name
        return rel_classes

    def _set_active_pred_ids(self, mode):
        if mode == "base":
            active_ids = self.base
        elif mode == "novel":
            active_ids = self.novel
        elif mode == "semantic":
            active_ids = self.semantic
        else:
            active_ids = list(range(self.num_rel_cls))
        self.active_pred_ids = torch.tensor(active_ids, device=self.device, dtype=torch.long)

    def updata(self, mode):
        super(ClipReSAPredictor, self).updata(mode)
        self._set_active_pred_ids(mode)

    def _init_pred_atoms_from_text(self, text_vectors):
        if text_vectors.size(0) != self.num_rel_cls:
            return
        with torch.no_grad():
            init = self.pred_atom_logits.new_zeros((self.num_rel_cls, self.atom_dim))
            copy_dim = min(text_vectors.size(1), self.atom_dim)
            init[:, :copy_dim] = text_vectors[:, :copy_dim].to(init.device, dtype=init.dtype)
            init = init - init.min(dim=-1, keepdim=True)[0]
            init = init / init.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            init[0].zero_()
            self.pred_atom_logits.copy_(init + 1e-3)

    def _build_confusable_groups(self, rel_classes):
        name_to_idx = {name.lower(): idx for idx, name in enumerate(rel_classes) if name is not None}
        groups = [
            ["on", "standing on", "sitting on", "laying on", "lying on", "mounted on", "parked on", "on back of"],
            ["holding", "carrying", "wearing", "wears", "using", "attached to"],
            ["above", "under", "over", "behind", "in front of", "near", "against", "between"],
            ["in", "covered in", "covering", "part of", "of", "has"],
            ["looking at", "watching", "playing", "eating", "riding", "walking on", "walking in"],
        ]
        confusable = {}
        for group in groups:
            ids = [name_to_idx[name] for name in group if name in name_to_idx]
            for idx in ids:
                confusable[idx] = [other for other in ids if other != idx]
        return confusable

    def _pred_atom_code(self):
        pred_act = F.relu(self.pred_atom_logits.float())
        pred_code = pred_act / pred_act.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pred_code = pred_code.clone()
        pred_code[0].zero_()
        return pred_code

    def _atom_logits(self, atom_code):
        pred_code = self._pred_atom_code()
        atom_query = F.normalize(atom_code.float(), dim=-1)
        atom_proto = F.normalize(pred_code, dim=-1)
        return self.atom_logit_scale.float().clamp(1.0, 30.0) * torch.matmul(atom_query, atom_proto.t())

    def _active_atom_logits(self, relation_feature, out_dtype):
        atom_raw_logits, atom_act, atom_code = self.atom_encoder(relation_feature.half())
        atom_logits = self._atom_logits(atom_code)
        atom_logits = atom_logits.index_select(1, self.active_pred_ids)
        return atom_logits.to(dtype=out_dtype)

    def _diversity_loss(self):
        dictionary = F.normalize(self.atom_dictionary.float(), dim=-1)
        gram = torch.matmul(dictionary, dictionary.t())
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - eye).pow(2).mean()

    def _sparse_loss(self, atom_code):
        atom_code = atom_code.float()
        entropy = -(atom_code * atom_code.clamp(min=1e-6).log()).sum(dim=-1)
        return entropy.mean()

    def _consistency_loss(self, atom_code, rel_labels):
        if rel_labels is None:
            return atom_code.float().sum() * 0.0
        labels = torch.cat(rel_labels, dim=0).view(-1)
        pos = labels > 0
        if pos.sum() == 0:
            return atom_code.float().sum() * 0.0
        pred_code = self._pred_atom_code()
        target_code = pred_code[labels[pos].long()]
        return (1.0 - F.cosine_similarity(atom_code[pos].float(), target_code, dim=-1)).mean()

    def _contrast_loss(self, atom_logits, rel_labels):
        if rel_labels is None:
            return atom_logits.sum() * 0.0
        labels = torch.cat(rel_labels, dim=0).view(-1).long()
        pos = labels > 0
        if pos.sum() == 0:
            return atom_logits.sum() * 0.0
        losses = []
        for label in labels[pos].unique():
            label_int = int(label.item())
            negatives = self._confusable_groups.get(label_int, None)
            if not negatives:
                continue
            sample_mask = pos & (labels == label)
            neg_ids = torch.tensor(negatives, device=atom_logits.device, dtype=torch.long)
            pos_score = atom_logits[sample_mask, label_int].unsqueeze(1)
            neg_score = atom_logits[sample_mask].index_select(1, neg_ids)
            losses.append(F.relu(0.2 + neg_score - pos_score).mean())
        if len(losses) == 0:
            return atom_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _resa_losses(self, relation_features, rel_labels):
        if len(relation_features) == 0:
            zero = self.pred_atom_logits.float().sum() * 0.0
            return {
                "loss_resa_recon": zero,
                "loss_resa_sparse": zero,
                "loss_resa_diversity": zero,
                "loss_resa_consist": zero,
                "loss_resa_contrast": zero,
            }
        relation_features = torch.cat(relation_features, dim=0)
        atom_raw_logits, atom_act, atom_code = self.atom_encoder(relation_features.half())
        atom_logits = self._atom_logits(atom_code)
        recon = torch.matmul(atom_code.float(), self.atom_dictionary.float())
        return {
            "loss_resa_recon": self.recon_weight * F.smooth_l1_loss(recon, relation_features.detach().float()),
            "loss_resa_sparse": self.sparse_weight * self._sparse_loss(atom_code),
            "loss_resa_diversity": self.diversity_weight * self._diversity_loss(),
            "loss_resa_consist": self.consist_weight * self._consistency_loss(atom_code, rel_labels),
            "loss_resa_contrast": self.contrast_weight * self._contrast_loss(atom_logits, rel_labels),
        }

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,img=None):
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)

        rel_dists = []
        relation_features_for_loss = []
        for i in range(len(num_rels)):
            rel_dist_per_batch = []
            image_tensor = []
            with torch.no_grad():
                for j in range(len(proposals[i].bbox)):
                    union_img = crop_and_resize(img[i].unsqueeze(0), proposals[i].bbox[j], proposals[i].bbox[j])
                    iimg = union_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    iimg = Image.fromarray(np.uint8(iimg))
                    union_img = self.clip_preprocess(iimg).unsqueeze(0).to(self.device)
                    image_tensor.append(union_img)
                image_tensor = torch.cat(image_tensor)
                image_features = self.clip_model.encode_image(image_tensor)

            for la_count, rel_index in enumerate(rel_pair_idxs[i]):
                obj_n1, obj_n2 = obj_preds[i][rel_index[0]], obj_preds[i][rel_index[1]]

                text_features1 = self.text_features1
                text_features2 = self.text_features2
                text_sub = self.text_features3[obj_n1]
                text_obj = self.text_features4[obj_n2]

                cross_output1 = self.adaper_clip1(image_features[rel_index[0]].unsqueeze(0), image_features[rel_index[1]].unsqueeze(0), text_sub)
                cross_output2 = self.adaper_clip2(image_features[rel_index[1]].unsqueeze(0), image_features[rel_index[0]].unsqueeze(0), text_obj)
                cross_output = (cross_output1 + cross_output2) / 2
                relation_features_for_loss.append(cross_output)

                similarity1 = ((cross_output / cross_output.norm(dim=-1, keepdim=True)) @ (text_features1 / text_features1.norm(dim=-1, keepdim=True)).T)
                similarity2 = ((cross_output / cross_output.norm(dim=-1, keepdim=True)) @ (text_features2 / text_features2.norm(dim=-1, keepdim=True)).T)

                if self.adaper_clip1.training:
                    probs = (similarity1 - similarity2) / 0.05
                    image_features_clip = (image_features[rel_index[0]][0].unsqueeze(0) + image_features[rel_index[1]][0].unsqueeze(0)) / 2
                    similarit_origin_1 = ((image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)) @
                                 (text_features1 / text_features1.norm(dim=-1, keepdim=True)).T)
                    similarit_origin_2 = ((image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)) @
                                   (text_features2 / text_features2.norm(dim=-1, keepdim=True)).T)
                    similarit_origin = (similarit_origin_1 - similarit_origin_2) / 0.05
                    probs = torch.cat([probs, similarit_origin]).unsqueeze(0)
                else:
                    similarity_delta = (similarity1 - similarity2) / 0.05
                    probs = self.description_relation[:, obj_n1] * similarity_delta
                    probs = probs.sum(-1).unsqueeze(0)

                    text_features5 = torch.Tensor(self.texts5[obj_n1]).to(self.device).half()
                    similarity31 = ((image_features[rel_index[0]][0].unsqueeze(0) / image_features[rel_index[0]][0].unsqueeze(0).norm(dim=-1, keepdim=True)) @ (text_features5 / text_features5.norm(dim=-1, keepdim=True)).T / 0.05)
                    similarity32 = ((image_features[rel_index[1]][0].unsqueeze(0) / image_features[rel_index[1]][0].unsqueeze(0).norm(dim=-1, keepdim=True)) @ (text_features5 / text_features5.norm(dim=-1, keepdim=True)).T / 0.05)
                    similarity3 = (similarity31 + similarity32) / 2
                    probs = probs * 0.2 + similarity3 * 0.8
                    probs = probs + self.atom_weight * self._active_atom_logits(cross_output, probs.dtype)
                rel_dist_per_batch.append(probs)

            rel_dist_per_batch = torch.cat(rel_dist_per_batch)
            rel_dists.append(rel_dist_per_batch)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        if self.training:
            add_losses = self._resa_losses(relation_features_for_loss, rel_labels)
        return obj_dists, rel_dists, add_losses



class ELRGFAxisHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_axes, dropout):
        super(ELRGFAxisHead, self).__init__()
        self.axis_logits = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_axes),
        )
        self.axis_feat = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, pair_rep):
        return self.axis_logits(pair_rep), self.axis_feat(pair_rep)


class ELRGFMetricScorer(nn.Module):
    def __init__(self, pair_dim, num_rel_cls, metric_dim, axis_dim, low_rank, dropout):
        super(ELRGFMetricScorer, self).__init__()
        self.num_rel_cls = num_rel_cls
        self.metric_dim = metric_dim
        self.low_rank = low_rank

        self.visual_proj = nn.Sequential(
            nn.Linear(pair_dim, metric_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(metric_dim, metric_dim),
        )
        self.pred_prototypes = nn.Parameter(torch.empty(num_rel_cls, metric_dim))
        self.diag_head = nn.Linear(axis_dim, metric_dim)
        self.lowrank_head = nn.Linear(axis_dim, metric_dim * low_rank)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.pred_prototypes, std=0.01)

    def init_from_classifier(self, classifier):
        if classifier.weight.size(0) != self.num_rel_cls:
            return
        if classifier.weight.size(1) != self.metric_dim:
            return
        with torch.no_grad():
            self.pred_prototypes.copy_(F.normalize(classifier.weight.detach(), dim=-1))

    def init_from_text_vectors(self, text_vectors):
        if text_vectors.size(0) != self.num_rel_cls:
            return
        with torch.no_grad():
            proto = self.pred_prototypes.new_zeros((self.num_rel_cls, self.metric_dim))
            copy_dim = min(text_vectors.size(1), self.metric_dim)
            proto[:, :copy_dim] = text_vectors[:, :copy_dim].to(proto.device, dtype=proto.dtype)
            self.pred_prototypes.copy_(F.normalize(proto, dim=-1))

    def forward(self, pair_rep, axis_feat):
        q = F.normalize(self.visual_proj(pair_rep), dim=-1)
        proto = F.normalize(self.pred_prototypes, dim=-1)

        diag = F.softplus(self.diag_head(axis_feat)) + 1e-4
        low_rank = torch.tanh(self.lowrank_head(axis_feat))
        low_rank = low_rank.view(pair_rep.size(0), self.metric_dim, self.low_rank)

        diff = q[:, None, :] - proto[None, :, :]
        diag_dist = (diff.pow(2) * diag[:, None, :]).sum(-1)
        low_rank_dist = torch.einsum("ncd,ndr->ncr", diff, low_rank).pow(2).sum(-1)
        dist = diag_dist + low_rank_dist

        logits = -self.logit_scale.clamp(0.1, 20.0) * dist
        metric_reg = (diag - 1.0).pow(2).mean() + 0.01 * low_rank.pow(2).mean()
        return logits, metric_reg


class ReSASparseAtomEncoder(nn.Module):
    def __init__(self, pair_dim, atom_dim, dropout):
        super(ReSASparseAtomEncoder, self).__init__()
        self.atom_logits = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(pair_dim, atom_dim),
        )
        nn.init.constant_(self.atom_logits[-1].bias, 0.1)

    def forward(self, pair_rep):
        atom_logits = self.atom_logits(pair_rep)
        atom_act = F.relu(atom_logits)
        atom_code = atom_act / atom_act.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return atom_logits, atom_act, atom_code


@registry.ROI_RELATION_PREDICTOR.register("ReSAPredictor")
class ReSAPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(ReSAPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        self.rel_classes = rel_classes
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        resa_cfg = config.MODEL.ROI_RELATION_HEAD.RESA
        self.atom_dim = resa_cfg.ATOM_DIM
        self.pair_dim = resa_cfg.PAIR_DIM
        self.base_weight = resa_cfg.BASE_WEIGHT
        self.atom_weight = resa_cfg.ATOM_WEIGHT
        self.freq_weight = resa_cfg.FREQ_WEIGHT
        self.recon_weight = resa_cfg.RECON_WEIGHT
        self.sparse_weight = resa_cfg.SPARSE_WEIGHT
        self.diversity_weight = resa_cfg.DIVERSITY_WEIGHT
        self.consist_weight = resa_cfg.CONSIST_WEIGHT
        self.contrast_weight = resa_cfg.CONTRAST_WEIGHT
        self.use_union_features = resa_cfg.USE_UNION_FEATURES and self.use_vision

        hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        pair_input_dim = hidden_dim * 2 + 128
        if self.use_union_features:
            self.union_proj = nn.Sequential(
                nn.Linear(pooling_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            pair_input_dim += hidden_dim
        else:
            self.union_proj = None

        self.geom_embed = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_input_dim, self.pair_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE),
            nn.Linear(self.pair_dim, self.pair_dim),
            nn.ReLU(inplace=True),
        )
        self.rel_compress = nn.Linear(self.pair_dim, self.num_rel_cls)
        layer_init(self.rel_compress, xavier=True)

        self.atom_encoder = ReSASparseAtomEncoder(
            self.pair_dim,
            self.atom_dim,
            config.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE,
        )
        self.atom_dictionary = nn.Parameter(torch.empty(self.atom_dim, self.pair_dim))
        self.pred_atom_logits = nn.Parameter(torch.empty(self.num_rel_cls, self.atom_dim))
        self.atom_logit_scale = nn.Parameter(torch.tensor(10.0))
        nn.init.xavier_normal_(self.atom_dictionary)
        nn.init.uniform_(self.pred_atom_logits, 0.0, 0.1)

        if resa_cfg.INIT_PRED_ATOMS_FROM_TEXT:
            rel_embed_vecs = obj_edge_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=config.MODEL.ROI_RELATION_HEAD.EMBED_DIM)
            self._init_pred_atoms_from_text(rel_embed_vecs)

        self.freq_bias = FrequencyBias(config, statistics) if self.use_bias else None
        self._confusable_groups = self._build_confusable_groups(rel_classes)

    def updata(self, mode):
        return

    def _init_pred_atoms_from_text(self, text_vectors):
        if text_vectors.size(0) != self.num_rel_cls:
            return
        with torch.no_grad():
            init = self.pred_atom_logits.new_zeros((self.num_rel_cls, self.atom_dim))
            copy_dim = min(text_vectors.size(1), self.atom_dim)
            init[:, :copy_dim] = text_vectors[:, :copy_dim].to(init.device, dtype=init.dtype)
            init = init - init.min(dim=-1, keepdim=True)[0]
            init = init / init.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            init[0].zero_()
            self.pred_atom_logits.copy_(init + 1e-3)

    def _build_confusable_groups(self, rel_classes):
        name_to_idx = {name.lower(): idx for idx, name in enumerate(rel_classes)}
        groups = [
            ["on", "standing on", "sitting on", "laying on", "lying on", "mounted on", "parked on", "on back of"],
            ["holding", "touching", "carrying", "wearing", "wears", "using", "attached to"],
            ["above", "under", "over", "behind", "in front of", "near", "against", "between"],
            ["in", "inside", "covered in", "covering", "part of", "of", "has"],
            ["looking at", "watching", "playing", "eating", "riding", "walking on", "walking in"],
        ]
        confusable = {}
        for group in groups:
            ids = [name_to_idx[name] for name in group if name in name_to_idx]
            for idx in ids:
                confusable[idx] = [other for other in ids if other != idx]
        return confusable

    def _pair_geometry(self, proposal, pair_idx):
        box_info = get_box_info(proposal.bbox, proposal=proposal)
        sub_info = box_info[pair_idx[:, 0]]
        obj_info = box_info[pair_idx[:, 1]]
        return get_box_pair_info(sub_info, obj_info)

    def _build_pair_rep(self, edge_ctx, proposals, rel_pair_idxs, union_features):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        edge_ctx = edge_ctx.split(num_objs, dim=0)
        if self.use_union_features and union_features is not None:
            union_splits = union_features.split(num_rels, dim=0)
        else:
            union_splits = [None] * len(num_rels)

        pair_reps = []
        for img_idx, pair_idx in enumerate(rel_pair_idxs):
            sub_ctx = edge_ctx[img_idx][pair_idx[:, 0]]
            obj_ctx = edge_ctx[img_idx][pair_idx[:, 1]]
            geom_embed = self.geom_embed(self._pair_geometry(proposals[img_idx], pair_idx))
            features = [sub_ctx, obj_ctx, geom_embed]
            if self.use_union_features and union_splits[img_idx] is not None:
                features.append(self.union_proj(union_splits[img_idx]))
            pair_reps.append(torch.cat(features, dim=-1))
        pair_reps = torch.cat(pair_reps, dim=0)
        return self.pair_proj(pair_reps)

    def _pred_atom_code(self):
        pred_act = F.relu(self.pred_atom_logits)
        pred_code = pred_act / pred_act.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pred_code = pred_code.clone()
        pred_code[0].zero_()
        return pred_code

    def _atom_logits(self, atom_code, base_logits):
        pred_code = self._pred_atom_code()
        atom_query = F.normalize(atom_code, dim=-1)
        atom_proto = F.normalize(pred_code, dim=-1)
        atom_logits = self.atom_logit_scale.clamp(1.0, 30.0) * torch.matmul(atom_query, atom_proto.t())
        atom_logits = atom_logits.clone()
        atom_logits[:, 0] = 0.0
        return atom_logits

    def _diversity_loss(self):
        dictionary = F.normalize(self.atom_dictionary, dim=-1)
        gram = torch.matmul(dictionary, dictionary.t())
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - eye).pow(2).mean()

    def _sparse_loss(self, atom_code):
        entropy = -(atom_code * atom_code.clamp(min=1e-6).log()).sum(dim=-1)
        return entropy.mean()

    def _consistency_loss(self, atom_code, rel_labels):
        if rel_labels is None:
            return atom_code.sum() * 0.0
        labels = torch.cat(rel_labels, dim=0).view(-1)
        pos = labels > 0
        if pos.sum() == 0:
            return atom_code.sum() * 0.0
        pred_code = self._pred_atom_code()
        target_code = pred_code[labels[pos].long()]
        return (1.0 - F.cosine_similarity(atom_code[pos], target_code, dim=-1)).mean()

    def _contrast_loss(self, atom_logits, rel_labels):
        if rel_labels is None:
            return atom_logits.sum() * 0.0
        labels = torch.cat(rel_labels, dim=0).view(-1).long()
        pos = labels > 0
        if pos.sum() == 0:
            return atom_logits.sum() * 0.0
        losses = []
        for label in labels[pos].unique():
            label_int = int(label.item())
            negatives = self._confusable_groups.get(label_int, None)
            if not negatives:
                continue
            sample_mask = pos & (labels == label)
            neg_ids = torch.tensor(negatives, device=atom_logits.device, dtype=torch.long)
            pos_score = atom_logits[sample_mask, label_int].unsqueeze(1)
            neg_score = atom_logits[sample_mask].index_select(1, neg_ids)
            losses.append(F.relu(0.2 + neg_score - pos_score).mean())
        if len(losses) == 0:
            return atom_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None):
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        pair_rep = self._build_pair_rep(edge_ctx, proposals, rel_pair_idxs, union_features)
        base_logits = self.rel_compress(pair_rep)
        atom_raw_logits, atom_act, atom_code = self.atom_encoder(pair_rep)
        atom_logits = self._atom_logits(atom_code, base_logits)
        rel_logits = self.base_weight * base_logits + self.atom_weight * atom_logits
        rel_logits[:, 0] = base_logits[:, 0]

        if self.freq_bias is not None and self.freq_weight != 0:
            obj_preds_split = obj_preds.split(num_objs, dim=0)
            pair_obj_preds = []
            for img_idx, pair_idx in enumerate(rel_pair_idxs):
                pair_obj_preds.append(torch.stack((obj_preds_split[img_idx][pair_idx[:, 0]], obj_preds_split[img_idx][pair_idx[:, 1]]), dim=1))
            pair_obj_preds = torch.cat(pair_obj_preds, dim=0)
            rel_logits = rel_logits + self.freq_weight * self.freq_bias.index_with_labels(pair_obj_preds.long())
            rel_logits[:, 0] = base_logits[:, 0]

        rel_dists = tuple(rel_logits.split(num_rels, dim=0))
        obj_dists = obj_dists.split(num_objs, dim=0)

        add_losses = {}
        if self.training:
            recon = torch.matmul(atom_code, self.atom_dictionary)
            add_losses["loss_resa_recon"] = self.recon_weight * F.smooth_l1_loss(recon.float(), pair_rep.detach().float())
            add_losses["loss_resa_sparse"] = self.sparse_weight * self._sparse_loss(atom_code)
            add_losses["loss_resa_diversity"] = self.diversity_weight * self._diversity_loss()
            add_losses["loss_resa_consist"] = self.consist_weight * self._consistency_loss(atom_code, rel_labels)
            add_losses["loss_resa_contrast"] = self.contrast_weight * self._contrast_loss(atom_logits, rel_labels)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("ELRGFPredictor")
class ELRGFPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(ELRGFPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        self.rel_classes = rel_classes
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        elrgf_cfg = config.MODEL.ROI_RELATION_HEAD.ELRGF
        self.metric_dim = elrgf_cfg.METRIC_DIM
        self.axis_dim = elrgf_cfg.AXIS_DIM
        self.num_axes = elrgf_cfg.NUM_AXES
        self.base_logit_weight = elrgf_cfg.BASE_LOGIT_WEIGHT
        self.metric_weight = elrgf_cfg.METRIC_WEIGHT
        self.freq_weight = elrgf_cfg.FREQ_WEIGHT
        self.axis_loss_weight = elrgf_cfg.AXIS_LOSS_WEIGHT
        self.metric_reg_weight = elrgf_cfg.METRIC_REG_WEIGHT
        self.axiom_loss_weight = elrgf_cfg.AXIOM_LOSS_WEIGHT
        self.axis_dropout = elrgf_cfg.AXIS_DROPOUT
        self.detach_axis = elrgf_cfg.DETACH_AXIS
        self.use_axis_pseudo_labels = elrgf_cfg.USE_AXIS_PSEUDO_LABELS
        self.center_metric_logits = elrgf_cfg.CENTER_METRIC_LOGITS
        self.disable_bg_metric = elrgf_cfg.DISABLE_BG_METRIC
        self.use_union_features = elrgf_cfg.USE_UNION_FEATURES and self.use_vision

        hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        pair_input_dim = hidden_dim * 2 + 128
        if self.use_union_features:
            self.union_proj = nn.Sequential(
                nn.Linear(pooling_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            pair_input_dim += hidden_dim
        else:
            self.union_proj = None

        self.geom_embed = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_input_dim, self.metric_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE),
            nn.Linear(self.metric_dim, self.metric_dim),
            nn.ReLU(inplace=True),
        )
        self.rel_compress = nn.Linear(self.metric_dim, self.num_rel_cls)
        layer_init(self.rel_compress, xavier=True)

        self.axis_head = ELRGFAxisHead(self.metric_dim, self.axis_dim, self.num_axes, self.axis_dropout)
        self.metric_scorer = ELRGFMetricScorer(
            self.metric_dim,
            self.num_rel_cls,
            self.metric_dim,
            self.axis_dim,
            elrgf_cfg.LOW_RANK,
            self.axis_dropout,
        )
        if elrgf_cfg.INIT_PROTOTYPES_FROM_CLASSIFIER:
            self.metric_scorer.init_from_classifier(self.rel_compress)
        if elrgf_cfg.INIT_PROTOTYPES_FROM_TEXT:
            rel_embed_vecs = obj_edge_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=config.MODEL.ROI_RELATION_HEAD.EMBED_DIM)
            self.metric_scorer.init_from_text_vectors(rel_embed_vecs)

        self.freq_bias = FrequencyBias(config, statistics) if self.use_bias else None
        self._axis_predicate_map = self._build_axis_predicate_map(rel_classes)

    def updata(self, mode):
        return

    def _build_axis_predicate_map(self, rel_classes):
        axis_sets = [
            {"on", "standing on", "sitting on", "lying on", "laying on", "mounted on", "parked on"},
            {"on", "touching", "holding", "wearing", "wears", "carrying", "attached to", "covering", "covered in"},
            {"in", "inside", "within", "covered in"},
            {"behind", "in front of", "covered in", "covering", "under"},
            {"above", "under", "behind", "in front of", "over", "between", "near", "against"},
            {"riding", "sitting on", "standing on", "holding", "carrying", "wearing", "watching", "looking at", "playing", "using"},
        ]
        mapping = []
        for axis_set in axis_sets:
            ids = []
            for idx, name in enumerate(rel_classes):
                if name.lower() in axis_set:
                    ids.append(idx)
            mapping.append(ids)
        return mapping

    def _pair_geometry(self, proposal, pair_idx):
        box_info = get_box_info(proposal.bbox, proposal=proposal)
        sub_info = box_info[pair_idx[:, 0]]
        obj_info = box_info[pair_idx[:, 1]]
        return get_box_pair_info(sub_info, obj_info)

    def _axis_targets_from_geometry(self, proposal, pair_idx, rel_labels_per_img):
        num_pairs = pair_idx.size(0)
        targets = proposal.bbox.new_full((num_pairs, self.num_axes), -1.0)
        boxes = proposal.bbox
        sub = boxes[pair_idx[:, 0]]
        obj = boxes[pair_idx[:, 1]]

        sub_w = (sub[:, 2] - sub[:, 0] + 1.0).clamp(min=1.0)
        sub_h = (sub[:, 3] - sub[:, 1] + 1.0).clamp(min=1.0)
        obj_w = (obj[:, 2] - obj[:, 0] + 1.0).clamp(min=1.0)
        obj_h = (obj[:, 3] - obj[:, 1] + 1.0).clamp(min=1.0)
        sub_area = sub_w * sub_h
        obj_area = obj_w * obj_h

        inter_x1 = torch.max(sub[:, 0], obj[:, 0])
        inter_y1 = torch.max(sub[:, 1], obj[:, 1])
        inter_x2 = torch.min(sub[:, 2], obj[:, 2])
        inter_y2 = torch.min(sub[:, 3], obj[:, 3])
        inter_w = (inter_x2 - inter_x1 + 1.0).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1 + 1.0).clamp(min=0.0)
        inter_area = inter_w * inter_h
        x_overlap = inter_w / torch.min(sub_w, obj_w)
        subj_inside_obj = inter_area / sub_area.clamp(min=1.0)

        sub_cx = (sub[:, 0] + sub[:, 2]) * 0.5
        sub_cy = (sub[:, 1] + sub[:, 3]) * 0.5
        obj_cx = (obj[:, 0] + obj[:, 2]) * 0.5
        obj_cy = (obj[:, 1] + obj[:, 3]) * 0.5

        img_scale = float(max(max(proposal.size[0], proposal.size[1]), 100))
        horizontal_gap = torch.max(torch.max(obj[:, 0] - sub[:, 2], sub[:, 0] - obj[:, 2]), sub[:, 0].new_zeros(num_pairs))
        vertical_gap = torch.max(torch.max(obj[:, 1] - sub[:, 3], sub[:, 1] - obj[:, 3]), sub[:, 0].new_zeros(num_pairs))
        edge_distance = torch.sqrt(horizontal_gap.pow(2) + vertical_gap.pow(2)) / img_scale

        support_pos = (x_overlap > 0.25) & ((sub[:, 3] - obj[:, 1]).abs() / obj_h < 0.35) & (sub_cy < obj_cy)
        support_neg = (x_overlap < 0.05) | (sub_cy > obj_cy)
        targets[support_pos, 0] = 1.0
        targets[support_neg, 0] = 0.0

        contact_pos = (inter_area > 0) | (edge_distance < 0.03)
        contact_neg = edge_distance > 0.20
        targets[contact_pos, 1] = 1.0
        targets[contact_neg, 1] = 0.0

        containment_pos = subj_inside_obj > 0.70
        containment_neg = subj_inside_obj < 0.05
        targets[containment_pos, 2] = 1.0
        targets[containment_neg, 2] = 0.0

        occlusion_pos = (inter_area / torch.min(sub_area, obj_area).clamp(min=1.0)) > 0.35
        occlusion_neg = inter_area == 0
        targets[occlusion_pos, 3] = 1.0
        targets[occlusion_neg, 3] = 0.0

        relative_order_pos = ((sub_cx - obj_cx).abs() / obj_w > 0.25) | ((sub_cy - obj_cy).abs() / obj_h > 0.25)
        targets[relative_order_pos, 4] = 1.0

        if rel_labels_per_img is not None:
            labels = rel_labels_per_img.to(proposal.bbox.device)
            for axis_id, rel_ids in enumerate(self._axis_predicate_map):
                if len(rel_ids) == 0:
                    continue
                rel_ids_tensor = torch.tensor(rel_ids, device=labels.device, dtype=labels.dtype)
                matched = (labels[:, None] == rel_ids_tensor[None, :]).any(dim=1)
                targets[matched, axis_id] = 1.0

        return targets

    def _axis_loss(self, axis_logits, axis_targets):
        valid = axis_targets >= 0
        if valid.sum() == 0:
            return axis_logits.sum() * 0.0
        loss = F.binary_cross_entropy_with_logits(axis_logits[valid], axis_targets[valid], reduction="none")
        pos = axis_targets[valid] > 0.5
        if pos.any() and (~pos).any():
            pos_weight = torch.sqrt((~pos).float().sum() / pos.float().sum().clamp(min=1.0)).clamp(max=10.0)
            loss = torch.where(pos, loss * pos_weight, loss)
        return loss.mean()

    def _axiom_loss(self, axis_logits):
        axis_prob = torch.sigmoid(axis_logits)
        support = axis_prob[:, 0]
        contact = axis_prob[:, 1]
        containment = axis_prob[:, 2]
        return 0.5 * (F.relu(support - contact).mean() + F.relu(containment - contact).mean())

    def _build_pair_rep(self, edge_ctx, proposals, rel_pair_idxs, union_features):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        edge_ctx = edge_ctx.split(num_objs, dim=0)
        if self.use_union_features and union_features is not None:
            union_splits = union_features.split(num_rels, dim=0)
        else:
            union_splits = [None] * len(num_rels)

        pair_reps = []
        for img_idx, pair_idx in enumerate(rel_pair_idxs):
            sub_ctx = edge_ctx[img_idx][pair_idx[:, 0]]
            obj_ctx = edge_ctx[img_idx][pair_idx[:, 1]]
            geom_embed = self.geom_embed(self._pair_geometry(proposals[img_idx], pair_idx))
            features = [sub_ctx, obj_ctx, geom_embed]
            if self.use_union_features and union_splits[img_idx] is not None:
                features.append(self.union_proj(union_splits[img_idx]))
            pair_reps.append(torch.cat(features, dim=-1))
        pair_reps = torch.cat(pair_reps, dim=0)
        return self.pair_proj(pair_reps)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None):
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        pair_rep = self._build_pair_rep(edge_ctx, proposals, rel_pair_idxs, union_features)
        base_logits = self.rel_compress(pair_rep)
        axis_logits, axis_feat = self.axis_head(pair_rep)

        metric_axis_feat = F.dropout(axis_feat, p=self.axis_dropout, training=self.training)
        if self.detach_axis and self.training:
            metric_axis_feat = metric_axis_feat.detach()
        metric_logits, metric_reg = self.metric_scorer(pair_rep, metric_axis_feat)
        if self.disable_bg_metric:
            metric_logits[:, 0] = 0.0
        if self.center_metric_logits:
            metric_logits = metric_logits - metric_logits.mean(dim=-1, keepdim=True)
        if self.disable_bg_metric:
            metric_logits[:, 0] = 0.0

        rel_logits = self.base_logit_weight * base_logits + self.metric_weight * metric_logits

        if self.freq_bias is not None and self.freq_weight != 0:
            obj_preds_split = obj_preds.split(num_objs, dim=0)
            pair_obj_preds = []
            for img_idx, pair_idx in enumerate(rel_pair_idxs):
                pair_obj_preds.append(torch.stack((obj_preds_split[img_idx][pair_idx[:, 0]], obj_preds_split[img_idx][pair_idx[:, 1]]), dim=1))
            pair_obj_preds = torch.cat(pair_obj_preds, dim=0)
            rel_logits = rel_logits + self.freq_weight * self.freq_bias.index_with_labels(pair_obj_preds.long())

        rel_dists = tuple(rel_logits.split(num_rels, dim=0))
        obj_dists = obj_dists.split(num_objs, dim=0)

        add_losses = {}
        if self.training:
            add_losses["loss_elrgf_metric_reg"] = self.metric_reg_weight * metric_reg
            if self.axiom_loss_weight > 0:
                add_losses["loss_elrgf_axiom"] = self.axiom_loss_weight * self._axiom_loss(axis_logits)
            if self.use_axis_pseudo_labels and rel_labels is not None and self.axis_loss_weight > 0:
                axis_targets = []
                for img_idx, pair_idx in enumerate(rel_pair_idxs):
                    axis_targets.append(self._axis_targets_from_geometry(proposals[img_idx], pair_idx, rel_labels[img_idx]))
                axis_targets = torch.cat(axis_targets, dim=0)
                add_losses["loss_elrgf_axis"] = self.axis_loss_weight * self._axis_loss(axis_logits, axis_targets)

        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
