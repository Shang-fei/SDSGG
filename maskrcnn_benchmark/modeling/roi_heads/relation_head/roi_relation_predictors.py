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
from .mtm_debug import MTMDebugger
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


class MTMRelationAdapter(nn.Module):
    def __init__(self, inputDim=512, embedDim=512, dropout=0.1, reduction=4):
        super(MTMRelationAdapter, self).__init__()
        bottleneckDim = max(embedDim // reduction, 1)
        self.inputProj = nn.Linear(inputDim, embedDim)
        self.inputNorm = nn.LayerNorm(embedDim)
        self.downProj = nn.Linear(embedDim, bottleneckDim)
        self.upProj = nn.Linear(bottleneckDim, embedDim)
        self.gate = nn.Linear(embedDim, embedDim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, relationFeatures):
        x = self.inputProj(relationFeatures.float())
        residual = x
        x = self.inputNorm(x)
        delta = self.downProj(x)
        delta = F.gelu(delta)
        delta = self.dropout(delta)
        delta = self.upProj(delta)
        gate = torch.sigmoid(self.gate(x))
        return residual + gate * delta


class RelationModalityTransfer(nn.Module):
    def __init__(self, inputDim=512, embedDim=512, fcLayers=3, attentionLayers=3, numHeads=8, dropout=0.1):
        super(RelationModalityTransfer, self).__init__()
        self.adapter = MTMRelationAdapter(inputDim, embedDim, dropout)
        layers = []
        currentDim = embedDim
        for _ in range(max(fcLayers - 1, 0)):
            layers.append(nn.Linear(currentDim, embedDim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            currentDim = embedDim
        self.fc = nn.Sequential(*layers)
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=embedDim,
            nhead=numHeads,
            dim_feedforward=embedDim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.selfAttention = nn.TransformerEncoder(encoderLayer, num_layers=attentionLayers)
        self.norm = nn.LayerNorm(embedDim)

    def encode_visual(self, relationFeatures):
        return self.adapter(relationFeatures)

    def encode_text_space(self, visualFeatures):
        x = self.fc(visualFeatures).unsqueeze(0)
        x = self.selfAttention(x).squeeze(0)
        return self.norm(x)

    def forward(self, relationFeatures):
        visualFeatures = self.encode_visual(relationFeatures)
        return self.encode_text_space(visualFeatures)


class PromptDistributionLearner(nn.Module):
    def __init__(self, clipModel, promptNum=4, ctxLen=4, initStd=0.02, device="cuda"):
        super(PromptDistributionLearner, self).__init__()
        self.clipModel = clipModel
        self.promptNum = int(promptNum)
        self.ctxLen = int(ctxLen)
        width = clipModel.token_embedding.embedding_dim
        if self.ctxLen > 0:
            self.context = nn.Parameter(torch.empty(self.promptNum, self.ctxLen, width))
            nn.init.normal_(self.context, std=float(initStd))
        else:
            self.context = nn.Parameter(torch.empty(self.promptNum, 0, width))
        self.device = device

    def _soft_texts(self, labels):
        prefix = " ".join(["X"] * self.ctxLen)
        if prefix:
            return [prefix + " " + str(label) for label in labels]
        return [str(label) for label in labels]

    def _encode_tokens_with_context(self, tokens):
        dtype = self.clipModel.dtype
        tokenEmbeddings = self.clipModel.token_embedding(tokens).type(dtype)
        batchSize = tokenEmbeddings.size(0)
        repeated = tokenEmbeddings.unsqueeze(0).repeat(self.promptNum, 1, 1, 1).clone()
        if self.ctxLen > 0:
            context = self.context[:, None, :, :].expand(-1, batchSize, -1, -1).to(dtype=dtype, device=tokens.device)
            repeated[:, :, 1: 1 + self.ctxLen, :] = context
        flat = repeated.view(self.promptNum * batchSize, repeated.size(2), repeated.size(3))
        flatTokens = tokens.unsqueeze(0).repeat(self.promptNum, 1, 1).view(self.promptNum * batchSize, -1)

        x = flat + self.clipModel.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)
        x = self.clipModel.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clipModel.ln_final(x).type(dtype)
        x = x[torch.arange(x.shape[0], device=x.device), flatTokens.argmax(dim=-1)] @ self.clipModel.text_projection
        x = F.normalize(x.float(), dim=-1)
        return x.view(self.promptNum, batchSize, -1).permute(1, 0, 2)

    def forward(self, labels):
        if len(labels) == 0:
            empty = self.context.new_zeros((0, self.clipModel.text_projection.size(1)))
            return empty, empty
        tokens = clip.tokenize(self._soft_texts(labels)).to(self.device)
        samples = self._encode_tokens_with_context(tokens)
        mean = F.normalize(samples.mean(dim=1), dim=-1)
        if samples.size(1) <= 1:
            sigma = samples.new_zeros(mean.shape)
        else:
            sigma = samples.std(dim=1, unbiased=False)
        return mean, sigma


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
        self.trainRelNames = list(rel_classes)
        self.trainPart = config.OV_SETTING.TRAIN_PART
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)

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
        self.trainRelNames = list(rel_classes)
        self.trainPart = config.OV_SETTING.TRAIN_PART
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)

        self.adaper_clip1 = MVA()
        self.adaper_clip2 = MVA()
        self.mtm_adaper_clip1 = MVA()
        self.mtm_adaper_clip2 = MVA()
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
        mtmConfig = config.MODEL.ROI_RELATION_HEAD.MTM
        relationnessConfig = config.MODEL.ROI_RELATION_HEAD.RELATIONNESS
        self.mtmEnabled = mtmConfig.ENABLED
        self.mtmLossEnabled = mtmConfig.ENABLED and mtmConfig.LOSS_ENABLED
        self.mtmLossWeight = mtmConfig.LOSS_WEIGHT
        self.mtmAlignWeight = mtmConfig.ALIGN_WEIGHT
        self.mtmStructureWeight = mtmConfig.STRUCTURE_WEIGHT
        self.mtmStructureLossType = str(mtmConfig.STRUCTURE_LOSS_TYPE).lower()
        self.mtmStructureTemperature = mtmConfig.STRUCTURE_TEMPERATURE
        self.mtmEntityAlignWeight = mtmConfig.ENTITY_ALIGN_WEIGHT
        self.mtmRelAlignWeight = mtmConfig.REL_ALIGN_WEIGHT
        self.mtmEntityStructureWeight = mtmConfig.ENTITY_STRUCTURE_WEIGHT
        self.mtmRelStructureWeight = mtmConfig.REL_STRUCTURE_WEIGHT
        self.mtmPromptAnchorWeight = mtmConfig.PROMPT_ANCHOR_WEIGHT
        self.mtmTripletSubjectWeight = mtmConfig.TRIPLET_SUBJECT_WEIGHT
        self.mtmTripletPredicateWeight = mtmConfig.TRIPLET_PREDICATE_WEIGHT
        self.mtmTripletObjectWeight = mtmConfig.TRIPLET_OBJECT_WEIGHT
        self.mtmMaxPairs = mtmConfig.MAX_PAIRS
        self.mtmUseInference = mtmConfig.ENABLED and mtmConfig.USE_INFERENCE
        self.mtmInferenceWeight = mtmConfig.INFERENCE_WEIGHT
        self.mtmDebugger = MTMDebugger(config.OUTPUT_DIR) if mtmConfig.DEBUG else None
        self.mtmDebugTripletInterval = mtmConfig.DEBUG_TRIPLET_INTERVAL
        self.mtmDebugTripletMaxSamples = mtmConfig.DEBUG_TRIPLET_MAX_SAMPLES
        self.mtmDebugStep = 0
        self.relationnessEnabled = relationnessConfig.ENABLED
        self.relationnessLossEnabled = relationnessConfig.ENABLED and relationnessConfig.LOSS_ENABLED
        self.relationnessLossWeight = relationnessConfig.LOSS_WEIGHT
        self.useRelationnessInference = relationnessConfig.ENABLED and relationnessConfig.USE_INFERENCE
        self.relationMtm = RelationModalityTransfer(
            mtmConfig.INPUT_DIM,
            mtmConfig.EMBED_DIM,
            mtmConfig.FC_LAYERS,
            mtmConfig.ATTENTION_LAYERS,
            mtmConfig.NUM_HEADS,
            mtmConfig.DROPOUT,
        ).to(self.device)
        self.entityMtm = RelationModalityTransfer(
            mtmConfig.INPUT_DIM,
            mtmConfig.EMBED_DIM,
            mtmConfig.FC_LAYERS,
            mtmConfig.ATTENTION_LAYERS,
            mtmConfig.NUM_HEADS,
            mtmConfig.DROPOUT,
        ).to(self.device)
        self.entityPromptLearner = PromptDistributionLearner(
            self.clip_model,
            mtmConfig.SOFT_PROMPT_NUM,
            mtmConfig.SOFT_PROMPT_CTX_LEN,
            mtmConfig.SOFT_PROMPT_INIT_STD,
            self.device,
        ).to(self.device)
        self.predicatePromptLearner = PromptDistributionLearner(
            self.clip_model,
            mtmConfig.SOFT_PROMPT_NUM,
            mtmConfig.SOFT_PROMPT_CTX_LEN,
            mtmConfig.SOFT_PROMPT_INIT_STD,
            self.device,
        ).to(self.device)
        self.relationnessHead = RelationnessHead(
            relationDim=mtmConfig.EMBED_DIM,
            hiddenDim=mtmConfig.EMBED_DIM // 2,
            dropout=mtmConfig.DROPOUT,
        ).to(self.device)
        self.activeRelNames = self.trainRelNames
        self.filteredTripletEmbeddingCache = {}

    def encodeTripletTexts(self, tripletTexts):
        with torch.no_grad():
            tripletTokens = clip.tokenize(tripletTexts).to(self.device)
            tripletTextFeatures = self.clip_model.encode_text(tripletTokens).float()
            return F.normalize(tripletTextFeatures, dim=-1)

    def encodePredicateTexts(self, predicateNames):
        texts = ["a photo of relation " + predicateName for predicateName in predicateNames]
        return self.encodeTripletTexts(texts)

    def clipFeatureToVector(self, features):
        if features.dim() == 3:
            return features[:, 0, :]
        return features

    def labelsToObjectNames(self, labels):
        return [self.obj_names[int(label)] for label in labels.detach().cpu().tolist()]

    def labelsToPredicateNames(self, labels):
        names = []
        for label in labels.detach().cpu().tolist():
            relIndex = int(label)
            if relIndex >= len(self.activeRelNames):
                raise ValueError(
                    "MTM received relation label {} but active relation names only has {} entries.".format(
                        relIndex, len(self.activeRelNames)
                    )
                )
            names.append(self.activeRelNames[relIndex])
        return names

    def encodeHardEntityTexts(self, names):
        return self.encodeTripletTexts(["a photo of " + name for name in names])

    def encodeHardPredicateTexts(self, names):
        return self.encodeTripletTexts(["a photo of relation " + name for name in names])

    def getEntityTextMuSigma(self, labels):
        names = self.labelsToObjectNames(labels)
        mu, sigma = self.entityPromptLearner(names)
        return mu, sigma

    def getPredicateTextMuSigma(self, labels):
        names = self.labelsToPredicateNames(labels)
        mu, sigma = self.predicatePromptLearner(names)
        return mu, sigma

    def getEntityTextDistribution(self, labels):
        names = self.labelsToObjectNames(labels)
        mu, sigma = self.entityPromptLearner(names)
        with torch.no_grad():
            anchor = self.encodeHardEntityTexts(names).to(mu.device)
        return mu, sigma, anchor

    def getPredicateTextDistribution(self, labels):
        names = self.labelsToPredicateNames(labels)
        mu, sigma = self.predicatePromptLearner(names)
        with torch.no_grad():
            anchor = self.encodeHardPredicateTexts(names).to(mu.device)
        return mu, sigma, anchor

    def composeTripletTeacher(self, subjMu, predMu, objMu):
        teacher = (
            float(self.mtmTripletSubjectWeight) * subjMu
            + float(self.mtmTripletPredicateWeight) * predMu
            + float(self.mtmTripletObjectWeight) * objMu
        )
        return F.normalize(teacher, dim=-1)

    def computeVisualStructureLoss(self, projectedNorm, visualNorm, referenceLoss):
        if projectedNorm.size(0) < 2:
            return referenceLoss * 0.0

        projectedSimilarity = torch.matmul(projectedNorm, projectedNorm.t())
        visualSimilarity = torch.matmul(visualNorm, visualNorm.t())
        offDiagonal = ~torch.eye(
            projectedSimilarity.size(0),
            dtype=torch.bool,
            device=projectedSimilarity.device,
        )

        temperature = max(float(self.mtmStructureTemperature), 1e-6)
        if self.mtmStructureLossType == "l1":
            projectedSimilarity = projectedSimilarity / temperature
            visualSimilarity = visualSimilarity / temperature
            return (visualSimilarity - projectedSimilarity).abs()[offDiagonal].mean()

        if self.mtmStructureLossType == "kl":
            projectedLogProb = F.log_softmax(projectedSimilarity.masked_fill(~offDiagonal, -1e4) / temperature, dim=1)
            visualProb = F.softmax(visualSimilarity.masked_fill(~offDiagonal, -1e4) / temperature, dim=1)
            return F.kl_div(projectedLogProb, visualProb, reduction="batchmean")

        raise ValueError("Unsupported MTM structure loss type: " + str(self.mtmStructureLossType))

    def computeMtmStructureLosses(self, predicted_norm, adapted_visual_norm, target_norm, alignLoss):
        if predicted_norm.size(0) < 2:
            return alignLoss * 0.0, alignLoss * 0.0

        predictedSimilarity = torch.matmul(predicted_norm, predicted_norm.t())
        adaptedVisualSimilarity = torch.matmul(adapted_visual_norm, adapted_visual_norm.t())
        targetSimilarity = torch.matmul(target_norm, target_norm.t())
        offDiagonal = ~torch.eye(
            predictedSimilarity.size(0),
            dtype=torch.bool,
            device=predictedSimilarity.device,
        )

        temperature = max(float(self.mtmStructureTemperature), 1e-6)
        if self.mtmStructureLossType == "l1":
            predictedSimilarity = predictedSimilarity / temperature
            adaptedVisualSimilarity = adaptedVisualSimilarity / temperature
            targetSimilarity = targetSimilarity / temperature
            visualStructureLoss = (adaptedVisualSimilarity - predictedSimilarity).abs()[offDiagonal].mean()
            textStructureLoss = (predictedSimilarity - targetSimilarity).abs()[offDiagonal].mean()
            return visualStructureLoss, textStructureLoss

        if self.mtmStructureLossType == "kl":
            predictedLogProb = F.log_softmax(predictedSimilarity.masked_fill(~offDiagonal, -1e4) / temperature, dim=1)
            visualProb = F.softmax(
                adaptedVisualSimilarity.detach().masked_fill(~offDiagonal, -1e4) / temperature,
                dim=1,
            )
            textProb = F.softmax(
                targetSimilarity.detach().masked_fill(~offDiagonal, -1e4) / temperature,
                dim=1,
            )
            visualStructureLoss = F.kl_div(predictedLogProb, visualProb, reduction="batchmean")
            textStructureLoss = F.kl_div(predictedLogProb, textProb, reduction="batchmean")
            return visualStructureLoss, textStructureLoss

        raise ValueError("Unsupported MTM structure loss type: " + str(self.mtmStructureLossType))

    def buildTargetTripletTexts(self, subjLabels, relationLabels, objLabels):
        texts = []
        for subjLabel, relationLabel, objLabel in zip(
            subjLabels.detach().cpu().tolist(),
            relationLabels.detach().cpu().tolist(),
            objLabels.detach().cpu().tolist(),
        ):
            subjName = self.obj_names[int(subjLabel)]
            relIndex = int(relationLabel)
            if relIndex >= len(self.activeRelNames):
                raise ValueError(
                    "MTM relation label {} is outside active relation names length {}. "
                    "The dataset relation labels and MTM relation-name table are misaligned.".format(
                        relIndex, len(self.activeRelNames)
                    )
                )
            relationName = self.activeRelNames[relIndex]
            objName = self.obj_names[int(objLabel)]
            texts.append("a photo of a " + subjName + " " + relationName + " a " + objName)
        return texts

    def buildTripletPromptVariants(self, subjName, relationName, objName):
        return [
            "a photo of a " + subjName + " " + relationName + " a " + objName,
            "a scene where a " + subjName + " is " + relationName + " a " + objName,
            "a visual relation of " + subjName + " " + relationName + " " + objName,
            "a " + subjName + " and a " + objName + " with relation " + relationName,
        ]

    def buildTargetTripletRecords(self, subjLabels, relationLabels, objLabels):
        records = []
        for subjLabel, relationLabel, objLabel in zip(
            subjLabels.detach().cpu().tolist(),
            relationLabels.detach().cpu().tolist(),
            objLabels.detach().cpu().tolist(),
        ):
            subjName = self.obj_names[int(subjLabel)]
            relIndex = int(relationLabel)
            if relIndex >= len(self.activeRelNames):
                raise ValueError(
                    "MTM relation label {} is outside active relation names length {}. "
                    "The dataset relation labels and MTM relation-name table are misaligned.".format(
                        relIndex, len(self.activeRelNames)
                    )
                )
            relationName = self.activeRelNames[relIndex]
            objName = self.obj_names[int(objLabel)]
            variants = self.buildTripletPromptVariants(subjName, relationName, objName)
            records.append({
                "subject": subjName,
                "relation": relationName,
                "object": objName,
                "prompt": variants[0],
                "variants": variants,
            })
        return records

    def buildFilteredTripletTexts(self, subjLabel, objLabels):
        texts = []
        subjName = self.obj_names[int(subjLabel)]
        candidateRelations = list(self.sub_filter_novel[subjName])
        for objLabel in objLabels.detach().cpu().tolist():
            objName = self.obj_names[int(objLabel)]
            for relationName in candidateRelations:
                texts.append("a photo of a " + subjName + " " + relationName + " a " + objName)
        return texts, len(candidateRelations)

    def getFilteredTripletEmbeddings(self, subjLabel, objLabel):
        cacheKey = (int(subjLabel), int(objLabel))
        if cacheKey not in self.filteredTripletEmbeddingCache:
            objLabels = torch.tensor([cacheKey[1]], device=self.device)
            texts, numCandidateRelations = self.buildFilteredTripletTexts(cacheKey[0], objLabels)
            embeddings = self.encodeTripletTexts(texts).cpu()
            self.filteredTripletEmbeddingCache[cacheKey] = (embeddings, numCandidateRelations)
        embeddings, numCandidateRelations = self.filteredTripletEmbeddingCache[cacheKey]
        return embeddings.to(self.device), numCandidateRelations

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
        relationnessTargets = (relationLabels > 0).float()
        relationnessProb = torch.sigmoid(relationnessLogits).clamp(min=1e-6, max=1.0 - 1e-6)
        gamma = 2.0
        positiveLoss = -relationnessTargets * torch.log(relationnessProb) * (1.0 - relationnessProb).pow(gamma)
        negativeLoss = -(1.0 - relationnessTargets) * torch.log(1.0 - relationnessProb) * relationnessProb.pow(gamma)
        normalizer = relationnessTargets.sum().clamp(min=1.0)
        loss = (positiveLoss + negativeLoss).sum() / normalizer
        return {"loss_mtm_relationness": self.relationnessLossWeight * loss}

    def maybeRecordTripletPromptSimilarity(self, target_norm, subjLabels, relationLabels, objLabels):
        if self.mtmDebugger is None or self.mtmDebugTripletInterval <= 0:
            return
        self.mtmDebugStep += 1
        if self.mtmDebugStep % self.mtmDebugTripletInterval != 0:
            return
        if target_norm.size(0) < 2:
            return

        records = self.buildTargetTripletRecords(subjLabels, relationLabels, objLabels)
        maxSamples = max(int(self.mtmDebugTripletMaxSamples), 1)
        if target_norm.size(0) <= maxSamples:
            selectedIndices = list(range(target_norm.size(0)))
        else:
            selectedIndices = torch.linspace(
                0,
                target_norm.size(0) - 1,
                steps=maxSamples,
                device=target_norm.device,
            ).long().detach().cpu().tolist()

        variantTexts = []
        variantCounts = [len(record["variants"]) for record in records]
        for selectedIndex in selectedIndices:
            variantTexts.extend(records[int(selectedIndex)]["variants"])
        variantEmbeddings = self.encodeTripletTexts(variantTexts).to(target_norm.device)
        self.mtmDebugger.record_triplet_similarity(
            self.mtmDebugStep,
            records,
            target_norm,
            variantEmbeddings,
            variantCounts,
            selectedIndices,
        )

    def computeFactorizedMtmLosses(self, subFeatures, objFeatures, relFeatures, relationLabels, subjLabels, objLabels):
        if (
            not self.mtmLossEnabled
            or len(relFeatures) == 0
            or relationLabels is None
        ):
            zero = self.relationMtm.norm.weight.sum() * 0.0 + self.entityMtm.norm.weight.sum() * 0.0
            return {
                "loss_mtm_sub_align": zero,
                "loss_mtm_obj_align": zero,
                "loss_mtm_rel_align": zero,
                "loss_mtm_sub_visual_structure": zero,
                "loss_mtm_obj_visual_structure": zero,
                "loss_mtm_rel_visual_structure": zero,
                "loss_mtm_prompt_anchor": zero,
            }

        subFeatures = self.clipFeatureToVector(torch.cat(subFeatures, dim=0)).float()
        objFeatures = self.clipFeatureToVector(torch.cat(objFeatures, dim=0)).float()
        relFeatures = self.clipFeatureToVector(torch.cat(relFeatures, dim=0)).float()
        relationLabels = torch.cat(relationLabels, dim=0).view(-1).long().to(relFeatures.device)
        subjLabels = torch.cat(subjLabels, dim=0).view(-1).long().to(relFeatures.device)
        objLabels = torch.cat(objLabels, dim=0).view(-1).long().to(relFeatures.device)

        positive = relationLabels > 0
        if positive.any() and relationLabels[positive].max().item() >= len(self.activeRelNames):
            raise ValueError(
                "MTM received relation label {} but active relation names only has {} entries.".format(
                    relationLabels[positive].max().item(), len(self.activeRelNames)
                )
            )
        if positive.sum() == 0:
            zero = relFeatures.sum() * 0.0
            return {
                "loss_mtm_sub_align": zero,
                "loss_mtm_obj_align": zero,
                "loss_mtm_rel_align": zero,
                "loss_mtm_sub_visual_structure": zero,
                "loss_mtm_obj_visual_structure": zero,
                "loss_mtm_rel_visual_structure": zero,
                "loss_mtm_prompt_anchor": zero,
            }

        subFeatures = subFeatures[positive]
        objFeatures = objFeatures[positive]
        relFeatures = relFeatures[positive]
        relationLabels = relationLabels[positive]
        subjLabels = subjLabels[positive]
        objLabels = objLabels[positive]

        if self.mtmMaxPairs > 0 and relFeatures.size(0) > self.mtmMaxPairs:
            sampleIndex = torch.linspace(
                0,
                relFeatures.size(0) - 1,
                steps=self.mtmMaxPairs,
                device=relFeatures.device,
            ).long()
            subFeatures = subFeatures.index_select(0, sampleIndex)
            objFeatures = objFeatures.index_select(0, sampleIndex)
            relFeatures = relFeatures.index_select(0, sampleIndex)
            relationLabels = relationLabels.index_select(0, sampleIndex)
            subjLabels = subjLabels.index_select(0, sampleIndex)
            objLabels = objLabels.index_select(0, sampleIndex)

        subVisual = self.entityMtm.encode_visual(subFeatures)
        objVisual = self.entityMtm.encode_visual(objFeatures)
        relVisual = self.relationMtm.encode_visual(relFeatures)
        subPred = self.entityMtm.encode_text_space(subVisual)
        objPred = self.entityMtm.encode_text_space(objVisual)
        relPred = self.relationMtm.encode_text_space(relVisual)

        subMu, subSigma, subAnchor = self.getEntityTextDistribution(subjLabels)
        objMu, objSigma, objAnchor = self.getEntityTextDistribution(objLabels)
        predMu, predSigma, predAnchor = self.getPredicateTextDistribution(relationLabels)
        tripletMu = self.composeTripletTeacher(subMu, predMu, objMu)

        subPredNorm = F.normalize(subPred.float(), dim=-1)
        objPredNorm = F.normalize(objPred.float(), dim=-1)
        relPredNorm = F.normalize(relPred.float(), dim=-1)
        subVisualNorm = F.normalize(subVisual.float(), dim=-1)
        objVisualNorm = F.normalize(objVisual.float(), dim=-1)
        relVisualNorm = F.normalize(relVisual.float(), dim=-1)

        subAlign = (1.0 - (subPredNorm * subMu).sum(dim=-1)).mean()
        objAlign = (1.0 - (objPredNorm * objMu).sum(dim=-1)).mean()
        relAlign = (1.0 - (relPredNorm * tripletMu).sum(dim=-1)).mean()

        subStructure = self.computeVisualStructureLoss(subPredNorm, subVisualNorm, subAlign)
        objStructure = self.computeVisualStructureLoss(objPredNorm, objVisualNorm, objAlign)
        relStructure = self.computeVisualStructureLoss(relPredNorm, relVisualNorm, relAlign)

        entityAnchor = (
            (1.0 - (subMu * subAnchor).sum(dim=-1)).mean()
            + (1.0 - (objMu * objAnchor).sum(dim=-1)).mean()
        ) * 0.5
        predAnchorLoss = (1.0 - (predMu * predAnchor).sum(dim=-1)).mean()
        promptAnchor = 0.5 * entityAnchor + 0.5 * predAnchorLoss
        if self.mtmDebugger is not None:
            self.mtmDebugger.record_factorized_prompt_distribution(
                subMu,
                objMu,
                predMu,
                tripletMu,
                subSigma,
                objSigma,
                predSigma,
            )

        return {
            "loss_mtm_sub_align": (
                self.mtmLossWeight * self.mtmAlignWeight * self.mtmEntityAlignWeight * subAlign
            ),
            "loss_mtm_obj_align": (
                self.mtmLossWeight * self.mtmAlignWeight * self.mtmEntityAlignWeight * objAlign
            ),
            "loss_mtm_rel_align": (
                self.mtmLossWeight * self.mtmAlignWeight * self.mtmRelAlignWeight * relAlign
            ),
            "loss_mtm_sub_visual_structure": (
                self.mtmLossWeight * self.mtmStructureWeight * self.mtmEntityStructureWeight * subStructure
            ),
            "loss_mtm_obj_visual_structure": (
                self.mtmLossWeight * self.mtmStructureWeight * self.mtmEntityStructureWeight * objStructure
            ),
            "loss_mtm_rel_visual_structure": (
                self.mtmLossWeight * self.mtmStructureWeight * self.mtmRelStructureWeight * relStructure
            ),
            "loss_mtm_prompt_anchor": self.mtmLossWeight * self.mtmPromptAnchorWeight * promptAnchor,
        }

    def computeFactorizedMtmInferenceScores(self, relationFeatures, subjLabels, objLabels, outShape, outDtype):
        relationFeatures = self.clipFeatureToVector(relationFeatures).float()
        predicted = F.normalize(self.relationMtm(relationFeatures).float(), dim=-1)
        subjMu, _ = self.getEntityTextMuSigma(subjLabels.to(relationFeatures.device))
        objMu, _ = self.getEntityTextMuSigma(objLabels.to(relationFeatures.device))
        predicateNames = list(self.activeRelNames)[:outShape[1]]
        predMu, _ = self.predicatePromptLearner(predicateNames)
        predMu = predMu.to(predicted.device)
        tripletMu = (
            float(self.mtmTripletSubjectWeight) * subjMu.unsqueeze(1)
            + float(self.mtmTripletPredicateWeight) * predMu.unsqueeze(0)
            + float(self.mtmTripletObjectWeight) * objMu.unsqueeze(1)
        )
        tripletMu = F.normalize(tripletMu, dim=-1)
        scores = (predicted.unsqueeze(1) * tripletMu).sum(dim=-1)
        return scores[:, :outShape[1]].to(dtype=outDtype)

    def updata(self,mode):
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

        self.description_relation=self.description_relation.applymap(lambda x: [int(s) for s in x.split(',')])
        self.description_relation=np.array(self.description_relation)
        self.description_relation = np.array([[np.array(item) for item in inner_list] for inner_list in self.description_relation])
        self.description_relation=torch.Tensor(self.description_relation).to(self.device)
        self.activeRelNames = self.trainRelNames if mode == self.trainPart else activeRelNames
        self.filteredTripletEmbeddingCache = {}

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


        rel_dists=[]
        subFeaturesForMtm = []
        objFeaturesForMtm = []
        relFeaturesForMtm = []
        relationLabelsForMtm = []
        subjLabelsForMtm = []
        objLabelsForMtm = []
        relationnessLogitsForLoss = []
        relationnessLabelsForLoss = []
        for i in range(len(num_rels)):
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

            mtm_cross_output = None
            pair_spatial_features = self.buildPairSpatialFeatures(proposals[i], pair_idx).to(
                device=cross_output.device,
                dtype=cross_output.dtype,
            )
            if self.relationnessEnabled:
                relationness_logits = self.relationnessHead(cross_output.detach(), pair_spatial_features)
                relationness_scores = torch.sigmoid(relationness_logits).to(dtype=cross_output.dtype).unsqueeze(-1)
            else:
                relationness_logits = None
                relationness_scores = cross_output.new_ones((cross_output.size(0), 1))
            if self.training and self.relationnessLossEnabled and rel_labels is not None:
                relationnessLogitsForLoss.append(relationness_logits)
                relationnessLabelsForLoss.append(rel_labels[i].to(relationness_logits.device))

            if self.training and self.mtmLossEnabled and rel_labels is not None:
                mtm_feature_pos = torch.nonzero(
                    rel_labels[i].to(pair_idx.device).view(-1) > 0,
                    as_tuple=False,
                ).view(-1)
                if mtm_feature_pos.numel() > 0:
                    subFeatureForMtm = sub_features.index_select(0, mtm_feature_pos)
                    objFeatureForMtm = obj_features.index_select(0, mtm_feature_pos)
                    textSubForMtm = text_sub.index_select(0, mtm_feature_pos).detach()
                    textObjForMtm = text_obj.index_select(0, mtm_feature_pos).detach()
                    mtmCrossOutput1 = self.mtm_adaper_clip1(subFeatureForMtm, objFeatureForMtm, textSubForMtm)
                    mtmCrossOutput2 = self.mtm_adaper_clip2(objFeatureForMtm, subFeatureForMtm, textObjForMtm)
                    relFeatureForMtm = (mtmCrossOutput1 + mtmCrossOutput2) / 2
                    subFeaturesForMtm.append(subFeatureForMtm)
                    objFeaturesForMtm.append(objFeatureForMtm)
                    relFeaturesForMtm.append(relFeatureForMtm)
                    relationLabelsForMtm.append(rel_labels[i].to(mtm_feature_pos.device).index_select(0, mtm_feature_pos))
                    if proposals[i].has_field("labels"):
                        gtObjLabels = proposals[i].get_field("labels").long()
                        subjLabelsForMtm.append(gtObjLabels.index_select(0, sub_idx).index_select(0, mtm_feature_pos))
                        objLabelsForMtm.append(gtObjLabels.index_select(0, obj_idx).index_select(0, mtm_feature_pos))
                    else:
                        subjLabelsForMtm.append(obj_n1.index_select(0, mtm_feature_pos))
                        objLabelsForMtm.append(obj_n2.index_select(0, mtm_feature_pos))

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

                grouped_filter_scores = []
                cls_features = image_features[:, 0, :]
                cls_norm = cls_features / cls_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                sub_norm = cls_norm.index_select(0, sub_idx)
                obj_norm = cls_norm.index_select(0, obj_idx)
                for label in obj_n1.unique():
                    mask = obj_n1 == label
                    rel_pos = torch.nonzero(mask, as_tuple=False).view(-1)
                    text_features5 = torch.Tensor(self.texts5[int(label.item())]).to(self.device).to(dtype=cls_norm.dtype)
                    text_features5 = text_features5 / text_features5.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    similarity31 = sub_norm.index_select(0, rel_pos) @ text_features5.t() / 0.05
                    similarity32 = obj_norm.index_select(0, rel_pos) @ text_features5.t() / 0.05
                    label_filter_scores = (similarity31 + similarity32) / 2
                    grouped_filter_scores.append((rel_pos, label_filter_scores[:, :description_scores.size(1)]))
                filter_scores = description_scores.new_zeros(description_scores.shape)
                for rel_pos, scores in grouped_filter_scores:
                    filter_scores.index_copy_(0, rel_pos, scores.to(dtype=filter_scores.dtype))
                rel_dist_per_batch = description_scores * 0.2 + filter_scores * 0.8
                if self.mtmUseInference and self.mtmInferenceWeight != 0:
                    mtm_text_sub = text_sub.detach()
                    mtm_text_obj = text_obj.detach()
                    mtm_cross_output1 = self.mtm_adaper_clip1(sub_features, obj_features, mtm_text_sub)
                    mtm_cross_output2 = self.mtm_adaper_clip2(obj_features, sub_features, mtm_text_obj)
                    mtm_cross_output = (mtm_cross_output1 + mtm_cross_output2) / 2
                    rawMtmScores = self.computeFactorizedMtmInferenceScores(
                        mtm_cross_output,
                        obj_n1,
                        obj_n2,
                        rel_dist_per_batch.shape,
                        rel_dist_per_batch.dtype,
                    )
                    rel_dist_per_batch = rel_dist_per_batch + self.mtmInferenceWeight * rawMtmScores
                if self.useRelationnessInference:
                    proposals[i].add_field("relationness_scores", relationness_scores.squeeze(-1).detach())

            rel_dists.append(rel_dist_per_batch)



        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        if self.training and self.mtmLossEnabled:
            add_losses.update(
                self.computeFactorizedMtmLosses(
                    subFeaturesForMtm,
                    objFeaturesForMtm,
                    relFeaturesForMtm,
                    relationLabelsForMtm,
                    subjLabelsForMtm,
                    objLabelsForMtm,
                )
            )
        if self.training and self.relationnessLossEnabled:
            add_losses.update(self.computeRelationnessLoss(relationnessLogitsForLoss, relationnessLabelsForLoss))
        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
