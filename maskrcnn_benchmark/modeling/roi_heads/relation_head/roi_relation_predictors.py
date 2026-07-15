# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize
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
from .mtm_ship import MTMShipBranch
from maskrcnn_benchmark.data import get_dataset_statistics
from CLIP import clip
import time
from PIL import Image
import pandas as pd
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


class RelationnessHead(nn.Module):
    def __init__(self, relationDim=512, spatialDim=32, hiddenDim=256, dropout=0.1):
        super(RelationnessHead, self).__init__()
        self.spatialProj = nn.Sequential(
            nn.Linear(spatialDim, hiddenDim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(relationDim + hiddenDim, hiddenDim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hiddenDim, 1),
        )

    def forward(self, relationFeatures, spatialFeatures):
        spatialFeatures = self.spatialProj(spatialFeatures.float())
        features = torch.cat([relationFeatures.float(), spatialFeatures], dim=-1)
        return self.classifier(features).squeeze(-1)


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

        mtmConfig = config.MODEL.ROI_RELATION_HEAD.MTM
        self.mtmEnabled = mtmConfig.ENABLED
        self.mtmTrainEnabled = mtmConfig.ENABLED and mtmConfig.TRAIN_ENABLED
        self.mtmInferenceEnabled = mtmConfig.ENABLED and mtmConfig.INFERENCE_ENABLED
        self.mtmInferenceWeight = mtmConfig.INFERENCE_WEIGHT
        self.mtm_branch = None
        if self.mtmEnabled:
            self.mtmRelNames = [
                '__background__', 'above', 'across', 'against', 'along', 'and', 'at',
                'attached to', 'behind', 'belonging to', 'between', 'carrying',
                'covered in', 'covering', 'eating', 'flying in', 'for', 'from',
                'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of',
                'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near',
                'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of',
                'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under',
                'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with',
            ]
            self.mtmTextFilter = pd.read_csv(curpath + "/filter_total.csv").iloc[:, 1:]
            self.mtmActiveIndices = list(self.base)
            self.mtmActiveRelNames = [self.mtmRelNames[index] for index in self.mtmActiveIndices]
            self.mtmInferenceFilter = self.mtmTextFilter.iloc[self.mtmActiveIndices]
            self._validate_mtm_relation_order()
            for param in self.clip_model.parameters():
                param.requires_grad_(False)
            self.mtm_branch = MTMShipBranch(
                mtmConfig,
                self.clip_model,
                self.obj_names,
                self.mtmRelNames,
                self.mtmActiveRelNames,
                self.base,
                self.novel,
                self.mtmTextFilter,
                self.mtmInferenceFilter,
                self.device,
                config.OUTPUT_DIR,
            ).to(self.device)

    def _validate_mtm_relation_order(self):
        expected = [self.mtmRelNames[index] for index in self.mtmActiveIndices]
        assert self.mtmActiveRelNames == expected
        assert self.mtmInferenceFilter.index.tolist() == self.mtmActiveIndices
        assert len(self.mtmActiveRelNames) == self.description_relation.size(0)

    def updata(self,mode):
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

        if self.mtm_branch is not None:
            if mode == "base":
                active_indices = self.base
            elif mode == "novel":
                active_indices = self.novel
            else:
                raise ValueError("Unsupported GQA predicate evaluation mode: {}".format(mode))
            self.mtmActiveIndices = list(active_indices)
            self.mtmActiveRelNames = [self.mtmRelNames[index] for index in self.mtmActiveIndices]
            self.mtmInferenceFilter = self.mtmTextFilter.iloc[self.mtmActiveIndices]
            self._validate_mtm_relation_order()
            self.mtm_branch.set_mode(self.mtmActiveRelNames, self.mtmInferenceFilter)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,img=None):
        # Returns:
        #     obj_dists (list[Tensor]): logits of object label distribution
        #     rel_dists (list[Tensor])
        #     rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
        #     union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        ##这里要思考如何使用联合box，boxlist_union操作注意print(list(self.obj_names)[rel_pair_idxs[0][0][0]])

        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)

        mtm_output = {"scores": [None] * len(num_rels), "losses": {}}
        run_mtm = (
            self.mtm_branch is not None
            and ((self.training and self.mtmTrainEnabled)
                 or (not self.training and self.mtmInferenceEnabled and self.mtmInferenceWeight != 0))
        )
        if run_mtm:
            mtm_output = self.mtm_branch(
                img,
                proposals,
                rel_pair_idxs,
                rel_labels,
                obj_preds,
            )

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

            if not self.training and self.mtmInferenceEnabled and self.mtmInferenceWeight != 0:
                mtm_scores = mtm_output["scores"][i]
                if mtm_scores is not None:
                    assert mtm_scores.size(1) == rel_dist_per_batch.size(1)
                    assert mtm_scores.size(1) == len(self.mtmActiveRelNames)
                    rel_dist_per_batch = rel_dist_per_batch + self.mtmInferenceWeight * mtm_scores.to(
                        device=rel_dist_per_batch.device,
                        dtype=rel_dist_per_batch.dtype,
                    )

            rel_dists.append(rel_dist_per_batch)



        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}

        if self.training and self.mtmTrainEnabled:
            add_losses.update(mtm_output["losses"])


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
        self.relNames = [None] * len(self.id_dict)
        for rel_name, rel_id in self.id_dict.items():
            self.relNames[rel_id] = rel_name

        self.base=[0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.PRDCS_BASE)]
        self.novel=[0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.PRDCS_NOVEL)]

        self.semantic = [0]+[self.id_dict[x] for x in sorted(config.OV_SETTING.SEMAN)]
        self.mtmTextSvdFilter = pd.read_csv(curpath+"/filter_total.csv").iloc[:, 1:]
        mode="base"

        if mode=="base":

            self.description_relation = self.description_relation.iloc[self.base, 1:]
            activeRelNames = [self.relNames[idx] for idx in self.base]

            self.sub_filter_novel = pd.read_csv(
            curpath+"/filter_total.csv").iloc[self.base, 1:]
        elif mode=="novel":
            self.description_relation = self.description_relation.iloc[self.novel, 1:]
            activeRelNames = [self.relNames[idx] for idx in self.novel]
            self.sub_filter_novel = pd.read_csv(
            curpath+"/filter_total.csv").iloc[self.novel, 1:]
        elif mode=="total":
            self.description_relation = self.description_relation.iloc[:, 1:]
            activeRelNames = self.relNames
            self.sub_filter_novel = pd.read_csv(
            curpath+"/filter_total.csv").iloc[:, 1:]
        elif mode=="semantic":
            self.description_relation = self.description_relation.iloc[self.semantic, 1:]
            activeRelNames = [self.relNames[idx] for idx in self.semantic]
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

            self.texts5 = self.encodeSubjectFilterTextFeatures(self.sub_filter_novel)
            self.texts5Tensor = F.normalize(torch.stack(self.texts5, dim=0), dim=-1)

        b=time.time()
        print('init complete : '+str(b-a))

        self.zhangliang=[]
        self.count=0

        self.linear1=nn.Linear(1024,512, bias=False).to(self.device).half()
        mtmConfig = config.MODEL.ROI_RELATION_HEAD.MTM
        relationnessConfig = config.MODEL.ROI_RELATION_HEAD.RELATIONNESS
        self.mtmEnabled = mtmConfig.ENABLED
        self.mtmTrainEnabled = mtmConfig.ENABLED and mtmConfig.TRAIN_ENABLED
        self.mtmInferenceEnabled = mtmConfig.ENABLED and mtmConfig.INFERENCE_ENABLED
        self.mtmInferenceWeight = mtmConfig.INFERENCE_WEIGHT
        self.relationnessEnabled = relationnessConfig.ENABLED
        self.relationnessLossEnabled = relationnessConfig.ENABLED and relationnessConfig.LOSS_ENABLED
        self.relationnessLossWeight = relationnessConfig.LOSS_WEIGHT
        self.useRelationnessInference = relationnessConfig.ENABLED and relationnessConfig.USE_INFERENCE
        self.clipInputSize = self.clip_model.visual.input_resolution
        self.clipImageNormalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        self.relationnessHead = RelationnessHead(
            relationDim=mtmConfig.EMBED_DIM,
            hiddenDim=mtmConfig.EMBED_DIM // 2,
            dropout=mtmConfig.DROPOUT,
        ).to(self.device)
        self.activeRelNames = activeRelNames
        self.mtm_branch = MTMShipBranch(
            mtmConfig,
            self.clip_model,
            self.obj_names,
            self.relNames,
            self.activeRelNames,
            self.base,
            self.novel,
            self.mtmTextSvdFilter,
            self.sub_filter_novel,
            self.device,
            config.OUTPUT_DIR,
        ).to(self.device) if self.mtmEnabled else None
        self.textEncodeChunkSize = max(int(mtmConfig.TEXT_ENCODE_CHUNK_SIZE), 1)
        self.mtmModeStateCache = {
            mode: {
                "description_relation": self.description_relation,
                "sub_filter_novel": self.sub_filter_novel,
                "activeRelNames": self.activeRelNames,
                "texts5": self.texts5,
                "texts5Tensor": self.texts5Tensor,
            }
        }

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Keep learned MTM/SHIP weights loadable after moving them into mtm_branch.
        keyMappings = (
            ("relationMtm.adapter.inputProj", "mtm_branch.projector.visual_adapter.input_proj"),
            ("relationMtm.adapter.inputNorm", "mtm_branch.projector.visual_adapter.norm"),
            ("relationMtm.adapter.downProj", "mtm_branch.projector.visual_adapter.down"),
            ("relationMtm.adapter.upProj", "mtm_branch.projector.visual_adapter.up"),
            ("relationMtm.adapter.gate", "mtm_branch.projector.visual_adapter.gate"),
            ("relationMtm.fc", "mtm_branch.projector.fc"),
            ("relationMtm.selfAttention", "mtm_branch.projector.self_attention"),
            ("relationMtm.norm", "mtm_branch.projector.norm"),
            ("shipFeatureGenerator.ctx", "mtm_branch.generator.ctx"),
            ("shipFeatureGenerator.visualEncoder.featureEncoder", "mtm_branch.generator.encoder.body"),
            ("shipFeatureGenerator.visualEncoder.mean", "mtm_branch.generator.encoder.mean"),
            ("shipFeatureGenerator.visualEncoder.logvar", "mtm_branch.generator.encoder.logvar"),
            ("shipFeatureGenerator.latentGenerator.net", "mtm_branch.generator.generator"),
            ("shipFeatureGenerator.textVisualAdapter.norm", "mtm_branch.generator.text_adapter.norm"),
            ("shipFeatureGenerator.textVisualAdapter.net", "mtm_branch.generator.text_adapter.body"),
            ("shipTrainingStep", "mtm_branch.training_step"),
        )
        for oldName, newName in keyMappings:
            oldPrefix = prefix + oldName
            newPrefix = prefix + newName
            for key in [key for key in state_dict if key == oldPrefix or key.startswith(oldPrefix + ".")]:
                mappedKey = newPrefix + key[len(oldPrefix):]
                if mappedKey not in state_dict:
                    state_dict[mappedKey] = state_dict[key]
                del state_dict[key]
        state_dict.pop(prefix + "mtmDebugIteration", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def encodeSubjectFilterTextFeatures(self, subjectFilter):
        allTexts = []
        featureCounts = []
        for objName in self.obj_names:
            objectTexts = ["a photo of " + text for text in list(subjectFilter[objName])]
            allTexts.extend(objectTexts)
            featureCounts.append(len(objectTexts))

        encodedChunks = []
        chunkSize = getattr(self, "textEncodeChunkSize", 1024)
        with torch.no_grad():
            for offset in range(0, len(allTexts), chunkSize):
                tokens = clip.tokenize(allTexts[offset : offset + chunkSize]).to(self.device)
                encodedChunks.append(self.clip_model.encode_text(tokens).detach())
        encodedFeatures = torch.cat(encodedChunks, dim=0)
        return list(encodedFeatures.split(featureCounts, dim=0))

    def cropImagePadding(self, image, proposal):
        imageWidth, imageHeight = proposal.size
        return image[:, : int(imageHeight), : int(imageWidth)]

    def encodeClipBoxCrops(self, image, boxes, returnTokens=False):
        boxes = boxes.long()
        _, imageHeight, imageWidth = image.shape
        if boxes.size(0) == 0:
            featureShape = (0, 50, 512) if returnTokens else (0, 512)
            return image.new_zeros(featureShape).float()

        encodedChunks = []
        chunkSize = max(int(self.mtm_branch.cfg.UNION_CROP_CHUNK_SIZE), 1) if self.mtm_branch else 128
        for offset in range(0, boxes.size(0), chunkSize):
            cropChunk = []
            for box in boxes[offset : offset + chunkSize]:
                left = int(box[0].clamp(min=0, max=max(imageWidth - 1, 0)).item())
                top = int(box[1].clamp(min=0, max=max(imageHeight - 1, 0)).item())
                right = int(box[2].clamp(min=left + 1, max=imageWidth).item())
                bottom = int(box[3].clamp(min=top + 1, max=imageHeight).item())
                crop = FF.resized_crop(
                    image, top, left, bottom - top, right - left,
                    (self.clipInputSize, self.clipInputSize),
                )
                crop = (crop * 255.0).to(torch.uint8).to(crop.dtype) / 255.0
                cropChunk.append(self.clipImageNormalize(crop))
            encoded = self.clip_model.encode_image(torch.stack(cropChunk).to(self.device))
            if encoded.dim() == 3 and not returnTokens:
                encoded = encoded[:, 0, :]
            encodedChunks.append(encoded)
        return torch.cat(encodedChunks, dim=0)

    def buildPairSpatialFeatures(self, proposal, pairIdx):
        boxInfo = get_box_info(proposal.bbox, need_norm=True, proposal=proposal)
        subjBoxInfo = boxInfo.index_select(0, pairIdx[:, 0])
        objBoxInfo = boxInfo.index_select(0, pairIdx[:, 1])
        return get_box_pair_info(subjBoxInfo, objBoxInfo)

    def computeRelationnessLoss(self, relationnessLogits, relationLabels):
        if len(relationnessLogits) == 0 or relationLabels is None:
            zero = self.relationnessHead.classifier[-1].weight.sum() * 0.0
            return {"loss_mtm_relationness": zero}
        relationnessLogits = torch.cat(relationnessLogits, dim=0).float()
        relationLabels = torch.cat(relationLabels, dim=0).view(-1).to(relationnessLogits.device)
        targets = (relationLabels > 0).float()
        probabilities = torch.sigmoid(relationnessLogits).clamp(min=1e-6, max=1.0 - 1e-6)
        gamma = 2.0
        positiveLoss = -targets * torch.log(probabilities) * (1.0 - probabilities).pow(gamma)
        negativeLoss = -(1.0 - targets) * torch.log(1.0 - probabilities) * probabilities.pow(gamma)
        normalizer = targets.sum().clamp(min=1.0)
        loss = (positiveLoss + negativeLoss).sum() / normalizer
        return {"loss_mtm_relationness": self.relationnessLossWeight * loss}

    def computePairFilterScores(self, subFeatures, objFeatures, subjLabels, numRelations):
        textFeatureBank = self.texts5Tensor.to(
            device=subFeatures.device,
            dtype=subFeatures.dtype,
        )
        scores = subFeatures.new_empty((subFeatures.size(0), numRelations))
        chunkSize = self.textEncodeChunkSize
        for offset in range(0, subFeatures.size(0), chunkSize):
            end = min(offset + chunkSize, subFeatures.size(0))
            relationTextFeatures = textFeatureBank.index_select(
                0,
                subjLabels[offset:end],
            )[:, :numRelations]
            subScores = torch.bmm(
                relationTextFeatures,
                subFeatures[offset:end].unsqueeze(-1),
            ).squeeze(-1) / 0.05
            objScores = torch.bmm(
                relationTextFeatures,
                objFeatures[offset:end].unsqueeze(-1),
            ).squeeze(-1) / 0.05
            scores[offset:end] = (subScores + objScores) / 2
        return scores

    def updata(self,mode):
        if mode in self.mtmModeStateCache:
            cachedState = self.mtmModeStateCache[mode]
            self.description_relation = cachedState["description_relation"]
            self.sub_filter_novel = cachedState["sub_filter_novel"]
            self.activeRelNames = cachedState["activeRelNames"]
            self.texts5 = cachedState["texts5"]
            self.texts5Tensor = cachedState["texts5Tensor"]
            if self.mtm_branch is not None:
                self.mtm_branch.set_mode(self.activeRelNames, self.sub_filter_novel)
            return

        self.description_relation = pd.read_csv(
            curpath+"/description_relation.csv")
        if mode == "base":

            self.description_relation = self.description_relation.iloc[self.base, 1:]
            activeRelNames = [self.relNames[idx] for idx in self.base]

            self.sub_filter_novel = pd.read_csv(
                curpath+"/filter_total.csv").iloc[
                                    self.base, 1:]
        elif mode == "novel":
            self.description_relation = self.description_relation.iloc[self.novel, 1:]
            activeRelNames = [self.relNames[idx] for idx in self.novel]
            self.sub_filter_novel = pd.read_csv(
                curpath+"/filter_total.csv").iloc[
                                    self.novel, 1:]
        elif mode == "total":
            self.description_relation = self.description_relation.iloc[:, 1:]
            activeRelNames = self.relNames
            self.sub_filter_novel = pd.read_csv(
                curpath+"/filter_total.csv").iloc[
                                    :, 1:]
        elif mode == "semantic":
            self.description_relation = self.description_relation.iloc[self.semantic, 1:]
            activeRelNames = [self.relNames[idx] for idx in self.semantic]
            self.sub_filter_novel = pd.read_csv(
                curpath+"/filter_total.csv").iloc[
                                    self.semantic, 1:]
        else:
            raise ValueError("Unsupported predicate evaluation mode: {}".format(mode))

        self.description_relation=self.description_relation.applymap(lambda x: [int(s) for s in x.split(',')])
        self.description_relation=np.array(self.description_relation)
        self.description_relation = np.array([[np.array(item) for item in inner_list] for inner_list in self.description_relation])
        self.description_relation=torch.Tensor(self.description_relation).to(self.device)
        self.activeRelNames = activeRelNames
        if self.mtm_branch is not None:
            self.mtm_branch.set_mode(self.activeRelNames, self.sub_filter_novel)

        self.texts5 = self.encodeSubjectFilterTextFeatures(self.sub_filter_novel)
        self.texts5Tensor = F.normalize(torch.stack(self.texts5, dim=0), dim=-1)

        self.mtmModeStateCache[mode] = {
            "description_relation": self.description_relation,
            "sub_filter_novel": self.sub_filter_novel,
            "activeRelNames": self.activeRelNames,
            "texts5": self.texts5,
            "texts5Tensor": self.texts5Tensor,
        }

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,img=None):
        # Returns:
        #     obj_dists (list[Tensor]): logits of object label distribution
        #     rel_dists (list[Tensor])
        #     rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
        #     union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        ##这里要思考如何使用联合box，boxlist_union操作注意print(list(self.obj_names)[rel_pair_idxs[0][0][0]])

        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)

        mtm_output = {"scores": [None] * len(num_rels), "losses": {}}
        run_mtm = (
            self.mtm_branch is not None
            and ((self.training and self.mtmTrainEnabled)
                 or (not self.training and self.mtmInferenceEnabled and self.mtmInferenceWeight != 0))
        )
        if run_mtm:
            mtm_output = self.mtm_branch(
                img,
                proposals,
                rel_pair_idxs,
                rel_labels,
                obj_preds,
            )

        rel_dists=[]
        relationnessLogitsForLoss = []
        relationnessLabelsForLoss = []
        for i in range(len(num_rels)):
            with torch.no_grad():
                clipImage = self.cropImagePadding(img[i], proposals[i])
                image_features = self.encodeClipBoxCrops(
                    clipImage,
                    proposals[i].bbox,
                    returnTokens=True,
                )

            pair_idx = rel_pair_idxs[i].long()
            if pair_idx.numel() == 0:
                if self.adaper_clip1.training:
                    rel_dists.append(image_features.new_zeros((0, 2, self.text_features1.size(0))))
                else:
                    rel_dists.append(image_features.new_zeros((0, self.description_relation.size(0))))
                continue

            sub_idx = pair_idx[:, 0]
            obj_idx = pair_idx[:, 1]
            obj_n1 = obj_preds[i].index_select(0, sub_idx).long()
            obj_n2 = obj_preds[i].index_select(0, obj_idx).long()

            text_features1 = self.text_features1
            text_features2 = self.text_features2
            text1_norm = text_features1 / text_features1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            text2_norm = text_features2 / text_features2.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            sub_features = image_features.index_select(0, sub_idx)
            obj_features = image_features.index_select(0, obj_idx)
            text_sub = self.text_features3.index_select(0, obj_n1)
            text_obj = self.text_features4.index_select(0, obj_n2)

            cross_output1 = self.adaper_clip1(sub_features, obj_features, text_sub)
            cross_output2 = self.adaper_clip2(obj_features, sub_features, text_obj)
            cross_output = (cross_output1 + cross_output2) / 2
            pairSpatialFeatures = self.buildPairSpatialFeatures(proposals[i], pair_idx).to(
                device=cross_output.device,
                dtype=cross_output.dtype,
            )
            if self.relationnessEnabled:
                relationnessLogits = self.relationnessHead(cross_output.detach(), pairSpatialFeatures)
                relationnessScores = torch.sigmoid(relationnessLogits).to(
                    dtype=cross_output.dtype
                ).unsqueeze(-1)
            else:
                relationnessLogits = None
                relationnessScores = cross_output.new_ones((cross_output.size(0), 1))
            if self.training and self.relationnessLossEnabled and rel_labels is not None:
                relationnessLogitsForLoss.append(relationnessLogits)
                relationnessLabelsForLoss.append(rel_labels[i].to(relationnessLogits.device))
            cross_norm = cross_output / cross_output.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            similarity1 = cross_norm @ text1_norm.t()
            similarity2 = cross_norm @ text2_norm.t()

            if self.adaper_clip1.training:
                probs = (similarity1 - similarity2) / 0.05
                image_features_clip = (image_features.index_select(0, sub_idx)[:, 0, :] + image_features.index_select(0, obj_idx)[:, 0, :]) / 2
                image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                similarit_origin_1 = image_features_clip @ text1_norm.t()
                similarit_origin_2 = image_features_clip @ text2_norm.t()
                similarit_origin = (similarit_origin_1 - similarit_origin_2) / 0.05
                rel_dist_per_batch = torch.stack([probs, similarit_origin], dim=1)
            else:
                similarity_delta = (similarity1 - similarity2) / 0.05
                description_relation = self.description_relation.index_select(1, obj_n1)
                description_scores = (description_relation.permute(1, 0, 2) * similarity_delta.unsqueeze(1)).sum(-1)

                cls_features = image_features[:, 0, :]
                cls_norm = cls_features / cls_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                sub_norm = cls_norm.index_select(0, sub_idx)
                obj_norm = cls_norm.index_select(0, obj_idx)
                filter_scores = self.computePairFilterScores(
                    sub_norm,
                    obj_norm,
                    obj_n1,
                    description_scores.size(1),
                ).to(dtype=description_scores.dtype)
                rel_dist_per_batch = description_scores * 0.2 + filter_scores * 0.8
                if self.mtmInferenceEnabled and self.mtmInferenceWeight != 0:
                    mtm_scores = mtm_output["scores"][i]
                    if mtm_scores is not None:
                        rel_dist_per_batch = rel_dist_per_batch + self.mtmInferenceWeight * mtm_scores.to(
                            device=rel_dist_per_batch.device,
                            dtype=rel_dist_per_batch.dtype,
                        )
                if self.useRelationnessInference:
                    proposals[i].add_field(
                        "relationness_scores",
                        relationnessScores.squeeze(-1).detach(),
                    )
            rel_dists.append(rel_dist_per_batch)



        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        if self.training and self.mtmTrainEnabled:
            add_losses.update(mtm_output["losses"])
        if self.training and self.relationnessLossEnabled:
            add_losses.update(
                self.computeRelationnessLoss(
                    relationnessLogitsForLoss,
                    relationnessLabelsForLoss,
                )
            )
        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
