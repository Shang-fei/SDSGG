# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize, ToPILImage
from torchvision.transforms import functional as FF
from torchvision.ops import roi_align
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


class MTMUnionGateFusion(nn.Module):
    def __init__(self, featureDim=512, spatialDim=32, dropout=0.1):
        super(MTMUnionGateFusion, self).__init__()
        priorDim = featureDim * 2 + spatialDim
        deltaDim = featureDim * 3 + spatialDim
        self.priorNet = nn.Sequential(
            nn.Linear(priorDim, featureDim),
            nn.LayerNorm(featureDim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.deltaNet = nn.Sequential(
            nn.Linear(deltaDim, featureDim),
            nn.LayerNorm(featureDim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(featureDim, featureDim),
        )
        self.gate = nn.Linear(featureDim, featureDim)
        self.outputNorm = nn.LayerNorm(featureDim)

    def forward(self, unionFeatures, subFeatures, objFeatures, spatialFeatures):
        unionFeatures = unionFeatures.float()
        subFeatures = subFeatures.float()
        objFeatures = objFeatures.float()
        spatialFeatures = spatialFeatures.float()
        prior = self.priorNet(torch.cat([subFeatures, objFeatures, spatialFeatures], dim=-1))
        delta = self.deltaNet(torch.cat([unionFeatures, subFeatures, objFeatures, spatialFeatures], dim=-1))
        gate = torch.sigmoid(self.gate(prior))
        return self.outputNorm(unionFeatures + gate * delta)


class BPLPosteriorNet(nn.Module):
    def __init__(self, inputDim=512, hiddenDim=256, outputDim=512):
        super(BPLPosteriorNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ELU(inplace=True),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ELU(inplace=True),
        )
        self.mean = nn.Linear(hiddenDim, outputDim)
        self.logvar = nn.Linear(hiddenDim, outputDim)

    def forward(self, imageFeatures):
        hidden = self.shared(imageFeatures.float())
        return self.mean(hidden), self.logvar(hidden).clamp(min=-10.0, max=10.0)


class BayesianTripletPromptLearner(nn.Module):
    def __init__(self, clipModel, ctxLen=4, embedDim=512, initStd=0.02):
        super(BayesianTripletPromptLearner, self).__init__()
        self.clipModel = clipModel
        self.ctxLen = ctxLen
        self.ctx = nn.Parameter(torch.empty(ctxLen, embedDim))
        nn.init.normal_(self.ctx, std=initStd)
        self.posterior = BPLPosteriorNet(embedDim, embedDim // 2, embedDim)

    def sampleResiduals(self, mu, logvar, sampleNum):
        eps = torch.randn(
            sampleNum,
            mu.size(0),
            mu.size(1),
            device=mu.device,
            dtype=mu.dtype,
        )
        return mu.unsqueeze(0) + eps * torch.exp(0.5 * logvar).unsqueeze(0)

    def encodePromptEmbeddings(self, prompts, tokenizedPrompts):
        dtype = self.clipModel.dtype
        x = prompts.to(dtype=dtype) + self.clipModel.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)
        x = self.clipModel.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clipModel.ln_final(x).type(dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokenizedPrompts.argmax(dim=-1)] @ self.clipModel.text_projection
        return x.float()

    def buildPromptSamples(self, tripletTexts, imageFeatures, sampleNum):
        promptPrefix = " ".join(["X"] * self.ctxLen)
        promptedTexts = [promptPrefix + " " + text for text in tripletTexts]
        tokenizedPrompts = clip.tokenize(promptedTexts).to(imageFeatures.device)
        with torch.no_grad():
            tokenEmbeddings = self.clipModel.token_embedding(tokenizedPrompts).float()
        mu, logvar = self.posterior(imageFeatures.detach().float())
        residuals = self.sampleResiduals(mu, logvar, sampleNum)
        prompts = []
        prefix = tokenEmbeddings[:, :1, :]
        suffix = tokenEmbeddings[:, 1 + self.ctxLen :, :]
        ctx = self.ctx.unsqueeze(0).expand(imageFeatures.size(0), -1, -1)
        for residual in residuals:
            shiftedCtx = ctx + residual.unsqueeze(1)
            prompts.append(torch.cat([prefix, shiftedCtx, suffix], dim=1))
        prompts = torch.cat(prompts, dim=0)
        repeatedTokens = tokenizedPrompts.repeat(sampleNum, 1)
        features = self.encodePromptEmbeddings(prompts, repeatedTokens)
        features = F.normalize(features, dim=-1)
        return features.view(sampleNum, imageFeatures.size(0), -1), mu, logvar

    def klLoss(self, mu, logvar):
        return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()


class RelationModalityTransfer(nn.Module):
    def __init__(self, inputDim=512, embedDim=512, fcLayers=3, attentionLayers=3, numHeads=8, dropout=0.1):
        super(RelationModalityTransfer, self).__init__()
        self.inputProj = nn.Sequential(
            nn.Linear(inputDim, embedDim),
            nn.LayerNorm(embedDim),
        )
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

    def forward(self, relationFeatures):
        x = self.inputProj(relationFeatures.float())
        x = self.fc(x).unsqueeze(0)
        x = self.selfAttention(x).squeeze(0)
        return self.norm(x)


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
        self.mtmVisualStructureWeight = mtmConfig.VISUAL_STRUCTURE_WEIGHT
        self.mtmMaxPairs = mtmConfig.MAX_PAIRS
        self.mtmUseInference = mtmConfig.ENABLED and mtmConfig.USE_INFERENCE
        self.mtmInferenceWeight = mtmConfig.INFERENCE_WEIGHT
        self.mtmTextSvdEnabled = mtmConfig.TEXT_SVD_ENABLED
        self.mtmTextSvdComponents = mtmConfig.TEXT_SVD_COMPONENTS
        self.mtmTextSvdChunkSize = mtmConfig.TEXT_SVD_CHUNK_SIZE
        self.mtmUnionGateFusionEnabled = mtmConfig.UNION_GATE_FUSION
        self.mtmBplEnabled = mtmConfig.BPL_ENABLED
        self.mtmBplSampleNum = mtmConfig.BPL_SAMPLE_NUM
        self.mtmBplTestSampleNum = mtmConfig.BPL_TEST_SAMPLE_NUM
        self.mtmBplKlWeight = mtmConfig.BPL_KL_WEIGHT
        self.mtmBplSvdAnchorWeight = mtmConfig.BPL_SVD_ANCHOR_WEIGHT
        self.clipInputSize = self.clip_model.visual.input_resolution
        self.clipImageNormalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        self.relationnessEnabled = relationnessConfig.ENABLED
        self.relationnessLossEnabled = relationnessConfig.ENABLED and relationnessConfig.LOSS_ENABLED
        self.relationnessLossWeight = relationnessConfig.LOSS_WEIGHT
        self.useRelationnessInference = relationnessConfig.ENABLED and relationnessConfig.USE_INFERENCE
        self.mtmUnionGateFusion = MTMUnionGateFusion(
            featureDim=mtmConfig.INPUT_DIM,
            spatialDim=32,
            dropout=mtmConfig.DROPOUT,
        ).to(self.device)
        self.bplPromptLearner = BayesianTripletPromptLearner(
            self.clip_model,
            ctxLen=mtmConfig.BPL_CTX_LEN,
            embedDim=mtmConfig.EMBED_DIM,
            initStd=mtmConfig.BPL_INIT_STD,
        ).to(self.device)
        self.relationMtm = RelationModalityTransfer(
            mtmConfig.INPUT_DIM,
            mtmConfig.EMBED_DIM,
            mtmConfig.FC_LAYERS,
            mtmConfig.ATTENTION_LAYERS,
            mtmConfig.NUM_HEADS,
            mtmConfig.DROPOUT,
        ).to(self.device)
        self.relationnessHead = RelationnessHead(
            relationDim=mtmConfig.EMBED_DIM,
            hiddenDim=mtmConfig.EMBED_DIM // 2,
            dropout=mtmConfig.DROPOUT,
        ).to(self.device)
        self.activeRelNames = activeRelNames
        self.filteredTripletEmbeddingCache = {}
        self.tripletTextSvdCache = {}
        self.tripletTextPrincipalComponents = None
        self.updateMtmTextSvdBasis()

    def encodeRawTripletTexts(self, tripletTexts):
        with torch.no_grad():
            tripletTokens = clip.tokenize(tripletTexts).to(self.device)
            tripletTextFeatures = self.clip_model.encode_text(tripletTokens).float()
            return F.normalize(tripletTextFeatures, dim=-1)

    def iterFilteredTripletTextChunks(self):
        chunk = []
        chunkSize = max(int(self.mtmTextSvdChunkSize), 1)
        objectNames = self.obj_names[1:] if len(self.obj_names) > 1 else self.obj_names
        for subjName in objectNames:
            candidateRelations = list(self.mtmTextSvdFilter[subjName])
            for relationName in candidateRelations:
                if relationName == "__background__":
                    continue
                for objName in objectNames:
                    chunk.append("a photo of a " + subjName + " " + relationName + " a " + objName)
                    if len(chunk) >= chunkSize:
                        yield chunk
                        chunk = []
        if len(chunk) > 0:
            yield chunk

    def updateMtmTextSvdBasis(self):
        if not self.mtmEnabled or not self.mtmTextSvdEnabled or self.mtmTextSvdComponents <= 0:
            self.tripletTextPrincipalComponents = None
            return
        cacheKey = "filter_total"
        if cacheKey in self.tripletTextSvdCache:
            cachedComponents = self.tripletTextSvdCache[cacheKey]
            if cachedComponents is None:
                self.tripletTextPrincipalComponents = None
            else:
                self.tripletTextPrincipalComponents = cachedComponents.to(self.device)
            return

        covariance = None
        numTriplets = 0
        with torch.no_grad():
            for tripletTexts in self.iterFilteredTripletTextChunks():
                tripletFeatures = self.encodeRawTripletTexts(tripletTexts).float()
                if covariance is None:
                    featureDim = tripletFeatures.size(-1)
                    covariance = tripletFeatures.new_zeros((featureDim, featureDim))
                covariance = covariance + torch.matmul(tripletFeatures.t(), tripletFeatures)
                numTriplets += tripletFeatures.size(0)

        if covariance is None or numTriplets < 2:
            self.tripletTextPrincipalComponents = None
            self.tripletTextSvdCache[cacheKey] = None
            return

        covariance = covariance / float(numTriplets)
        try:
            _, eigvecs = torch.linalg.eigh(covariance.float())
        except AttributeError:
            _, eigvecs = torch.symeig(covariance.float(), eigenvectors=True)
        numComponents = min(int(self.mtmTextSvdComponents), eigvecs.size(1))
        principalComponents = eigvecs[:, -numComponents:].contiguous()
        principalComponents = F.normalize(principalComponents, dim=0).detach().cpu()
        self.tripletTextSvdCache[cacheKey] = principalComponents
        self.tripletTextPrincipalComponents = principalComponents.to(self.device)

    def refineTripletTextFeatures(self, tripletTextFeatures):
        tripletTextFeatures = F.normalize(tripletTextFeatures.float(), dim=-1)
        if self.tripletTextPrincipalComponents is None:
            return tripletTextFeatures
        components = self.tripletTextPrincipalComponents.to(
            device=tripletTextFeatures.device,
            dtype=tripletTextFeatures.dtype,
        )
        projection = torch.matmul(torch.matmul(tripletTextFeatures, components), components.t())
        return F.normalize(tripletTextFeatures - projection, dim=-1)

    def encodeTripletTexts(self, tripletTexts):
        tripletTextFeatures = self.encodeRawTripletTexts(tripletTexts)
        return self.refineTripletTextFeatures(tripletTextFeatures)

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

    def encodeClipImageFeatureMap(self, image):
        imageInput = FF.resize(image, (self.clipInputSize, self.clipInputSize))
        imageInput = self.clipImageNormalize(imageInput).unsqueeze(0).to(self.device)
        tokens = self.clip_model.encode_image(imageInput)
        if tokens.dim() != 3:
            raise ValueError("MTM RoIAlign requires CLIP ViT token features, got shape {}".format(tuple(tokens.shape)))
        globalFeature = tokens[:, 0, :].float()
        patchTokens = tokens[:, 1:, :]
        gridSize = int(patchTokens.size(1) ** 0.5)
        if gridSize * gridSize != patchTokens.size(1):
            raise ValueError("CLIP patch token count {} is not a square grid".format(patchTokens.size(1)))
        featureMap = patchTokens.permute(0, 2, 1).contiguous().view(1, patchTokens.size(-1), gridSize, gridSize)
        return globalFeature, featureMap

    def scaleBoxesToClipInput(self, boxes, image):
        _, imageHeight, imageWidth = image.shape
        scaledBoxes = boxes.float().clone()
        scaledBoxes[:, 0::2] = scaledBoxes[:, 0::2] * (float(self.clipInputSize) / float(imageWidth))
        scaledBoxes[:, 1::2] = scaledBoxes[:, 1::2] * (float(self.clipInputSize) / float(imageHeight))
        return scaledBoxes

    def poolClipRoiFeatures(self, clipFeatureMap, boxes):
        if boxes.numel() == 0:
            return clipFeatureMap.new_zeros((0, clipFeatureMap.size(1)))
        rois = torch.cat(
            [
                boxes.new_zeros((boxes.size(0), 1)),
                boxes.to(device=clipFeatureMap.device, dtype=clipFeatureMap.dtype),
            ],
            dim=1,
        )
        spatialScale = float(clipFeatureMap.size(-1)) / float(self.clipInputSize)
        pooled = roi_align(
            clipFeatureMap,
            rois,
            output_size=(1, 1),
            spatial_scale=spatialScale,
            aligned=True,
        )
        return pooled.flatten(1)

    def buildUnionBoxes(self, boxes, pairIdx):
        subBoxes = boxes.index_select(0, pairIdx[:, 0].to(boxes.device))
        objBoxes = boxes.index_select(0, pairIdx[:, 1].to(boxes.device))
        topLeft = torch.min(subBoxes[:, :2], objBoxes[:, :2])
        bottomRight = torch.max(subBoxes[:, 2:], objBoxes[:, 2:])
        return torch.cat([topLeft, bottomRight], dim=-1)

    def buildMtmRoiRelationFeatures(self, clipFeatureMap, proposal, pairIdx, image):
        pairIdx = pairIdx.to(proposal.bbox.device)
        clipBoxes = self.scaleBoxesToClipInput(proposal.bbox, image)
        subBoxes = clipBoxes.index_select(0, pairIdx[:, 0])
        objBoxes = clipBoxes.index_select(0, pairIdx[:, 1])
        unionBoxes = self.buildUnionBoxes(clipBoxes, pairIdx)
        subFeatures = self.poolClipRoiFeatures(clipFeatureMap, subBoxes)
        objFeatures = self.poolClipRoiFeatures(clipFeatureMap, objBoxes)
        unionFeatures = self.poolClipRoiFeatures(clipFeatureMap, unionBoxes)
        spatialFeatures = self.buildPairSpatialFeatures(proposal, pairIdx).to(
            device=unionFeatures.device,
            dtype=unionFeatures.dtype,
        )
        if not self.mtmUnionGateFusionEnabled:
            return unionFeatures
        return self.mtmUnionGateFusion(unionFeatures, subFeatures, objFeatures, spatialFeatures)

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

    def computeMtmLosses(self, relationFeatures, relationLabels, subjLabels, objLabels, bplConditionFeatures=None):
        if not self.mtmLossEnabled or len(relationFeatures) == 0 or relationLabels is None:
            zero = self.relationMtm.norm.weight.sum() * 0.0
            return {
                "loss_mtm_align": zero,
                "loss_mtm_visual_structure": zero,
                "loss_mtm_text_structure": zero,
                "loss_mtm_bpl_kl": zero,
                "loss_mtm_bpl_svd_anchor": zero,
            }
        relationFeatures = torch.cat(relationFeatures, dim=0).float()
        relationLabels = torch.cat(relationLabels, dim=0).view(-1).long().to(relationFeatures.device)
        subjLabels = torch.cat(subjLabels, dim=0).view(-1).long().to(relationFeatures.device)
        objLabels = torch.cat(objLabels, dim=0).view(-1).long().to(relationFeatures.device)
        if bplConditionFeatures is not None and len(bplConditionFeatures) > 0:
            bplConditionFeatures = torch.cat(bplConditionFeatures, dim=0).float().to(relationFeatures.device)
        positive = relationLabels > 0
        if positive.any() and relationLabels[positive].max().item() >= len(self.activeRelNames):
            raise ValueError(
                "MTM received relation label {} but active relation names only has {} entries. "
                "Use dataset rel_classes for MTM targets or regenerate stale dataset-statistics cache.".format(
                    relationLabels[positive].max().item(), len(self.activeRelNames)
                )
            )
        valid = positive
        if valid.sum() == 0:
            zero = relationFeatures.sum() * 0.0
            return {
                "loss_mtm_align": zero,
                "loss_mtm_visual_structure": zero,
                "loss_mtm_text_structure": zero,
                "loss_mtm_bpl_kl": zero,
                "loss_mtm_bpl_svd_anchor": zero,
            }
        relationFeatures = relationFeatures[valid]
        relationLabels = relationLabels[valid]
        subjLabels = subjLabels[valid]
        objLabels = objLabels[valid]
        if bplConditionFeatures is not None:
            bplConditionFeatures = bplConditionFeatures[valid]
        if self.mtmMaxPairs > 0 and relationFeatures.size(0) > self.mtmMaxPairs:
            sample_index = torch.linspace(
                0,
                relationFeatures.size(0) - 1,
                steps=self.mtmMaxPairs,
                device=relationFeatures.device,
            ).long()
            relationFeatures = relationFeatures.index_select(0, sample_index)
            relationLabels = relationLabels.index_select(0, sample_index)
            subjLabels = subjLabels.index_select(0, sample_index)
            objLabels = objLabels.index_select(0, sample_index)
            if bplConditionFeatures is not None:
                bplConditionFeatures = bplConditionFeatures.index_select(0, sample_index)

        predicted_text_embeddings = self.relationMtm(relationFeatures)
        target_texts = self.buildTargetTripletTexts(subjLabels, relationLabels, objLabels)
        predicted_norm = F.normalize(predicted_text_embeddings.float(), dim=-1)
        input_visual_norm = F.normalize(relationFeatures.float(), dim=-1)
        bplKlLoss = predicted_norm.sum() * 0.0
        bplSvdAnchorLoss = predicted_norm.sum() * 0.0
        textStructureLoss = predicted_norm.sum() * 0.0

        if self.mtmBplEnabled:
            if bplConditionFeatures is None:
                raise ValueError("BPL MTM loss requires pair-level condition features when MTM.BPL_ENABLED is True")
            target_samples, bplMu, bplLogvar = self.bplPromptLearner.buildPromptSamples(
                target_texts,
                bplConditionFeatures.to(predicted_text_embeddings.device),
                int(self.mtmBplSampleNum),
            )
            target_norm = F.normalize(target_samples.float(), dim=-1)
            svd_anchor = self.encodeTripletTexts(target_texts).to(predicted_text_embeddings.device)
            svd_anchor = F.normalize(svd_anchor.float(), dim=-1)
            alignLoss = (1.0 - (predicted_norm.unsqueeze(0) * target_norm).sum(dim=-1)).mean()
            bplSvdAnchorLoss = (1.0 - (target_norm * svd_anchor.unsqueeze(0)).sum(dim=-1)).mean()
            bplKlLoss = self.bplPromptLearner.klLoss(bplMu, bplLogvar)
        else:
            target_embeddings = self.encodeTripletTexts(target_texts).to(predicted_text_embeddings.device)
            target_norm = F.normalize(target_embeddings.float(), dim=-1)
            alignLoss = (1.0 - (predicted_norm * target_norm).sum(dim=-1)).mean()

        if predicted_norm.size(0) < 2:
            visualStructureLoss = alignLoss * 0.0
        else:
            predictedSimilarity = torch.matmul(predicted_norm, predicted_norm.t())
            inputVisualSimilarity = torch.matmul(input_visual_norm, input_visual_norm.t())
            offDiagonal = ~torch.eye(
                predictedSimilarity.size(0),
                dtype=torch.bool,
                device=predictedSimilarity.device,
            )
            visualStructureLoss = (inputVisualSimilarity - predictedSimilarity).abs()[offDiagonal].mean()
        return {
            "loss_mtm_align": self.mtmLossWeight * self.mtmAlignWeight * alignLoss,
            "loss_mtm_visual_structure": self.mtmLossWeight * self.mtmStructureWeight * self.mtmVisualStructureWeight * visualStructureLoss,
            "loss_mtm_text_structure": textStructureLoss,
            "loss_mtm_bpl_kl": self.mtmBplKlWeight * bplKlLoss,
            "loss_mtm_bpl_svd_anchor": self.mtmLossWeight * self.mtmBplSvdAnchorWeight * bplSvdAnchorLoss,
        }

    def computeMtmInferenceScores(self, relationFeatures, subjLabels, objLabels, outShape, outDtype):
        predicted_embeddings = F.normalize(self.relationMtm(relationFeatures).float(), dim=-1)
        scores = relationFeatures.new_zeros(outShape, dtype=torch.float32)
        for label in subjLabels.unique():
            mask = subjLabels == label
            rel_pos = torch.nonzero(mask, as_tuple=False).view(-1)
            label_predicted_embeddings = predicted_embeddings.index_select(0, rel_pos)
            label_obj_labels = objLabels.index_select(0, rel_pos)
            label_condition_features = relationFeatures.index_select(0, rel_pos)
            candidate_features = []
            num_candidate_relations = None
            for pairOffset, objLabel in enumerate(label_obj_labels):
                if self.mtmBplEnabled:
                    candidateObjLabels = torch.tensor([int(objLabel.item())], device=self.device)
                    tripletTexts, candidate_count = self.buildFilteredTripletTexts(int(label.item()), candidateObjLabels)
                    conditionFeatures = label_condition_features[pairOffset : pairOffset + 1].to(
                        predicted_embeddings.device
                    ).expand(len(tripletTexts), -1)
                    samples, _, _ = self.bplPromptLearner.buildPromptSamples(
                        tripletTexts,
                        conditionFeatures,
                        int(self.mtmBplTestSampleNum),
                    )
                    samples = F.normalize(samples.float(), dim=-1)
                    candidate_features.append(samples)
                else:
                    embeddings, candidate_count = self.getFilteredTripletEmbeddings(label.item(), objLabel.item())
                    candidate_features.append(embeddings)
                num_candidate_relations = candidate_count
            if self.mtmBplEnabled:
                candidate_samples = torch.stack(candidate_features, dim=1).to(predicted_embeddings.device).float()
                candidate_samples = candidate_samples.view(
                    int(self.mtmBplTestSampleNum),
                    rel_pos.size(0),
                    num_candidate_relations,
                    -1,
                )
                label_scores = (label_predicted_embeddings.unsqueeze(0).unsqueeze(2) * candidate_samples).sum(-1).mean(dim=0)
            else:
                candidate_embeddings = torch.stack(candidate_features, dim=0)
                candidate_embeddings = candidate_embeddings.to(predicted_embeddings.device).float()
                candidate_embeddings = candidate_embeddings.view(rel_pos.size(0), num_candidate_relations, -1)
                label_scores = (label_predicted_embeddings.unsqueeze(1) * candidate_embeddings).sum(-1)
            scores.index_copy_(0, rel_pos, label_scores[:, :outShape[1]])
        return scores.to(dtype=outDtype)

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
        self.activeRelNames = activeRelNames
        self.filteredTripletEmbeddingCache = {}
        self.updateMtmTextSvdBasis()

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
        relationFeaturesForMtm = []
        relationLabelsForMtm = []
        subjLabelsForMtm = []
        objLabelsForMtm = []
        bplConditionFeatures = []
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

            mtm_relation_features = None
            need_mtm_features = (
                (self.training and self.mtmLossEnabled)
                or ((not self.training) and self.mtmUseInference and self.mtmInferenceWeight != 0)
            )
            if need_mtm_features:
                mtm_pair_idx = pair_idx
                mtm_feature_pos = None
                if self.training and rel_labels is not None:
                    mtm_feature_pos = torch.nonzero(
                        rel_labels[i].to(pair_idx.device).view(-1) > 0,
                        as_tuple=False,
                    ).view(-1)
                    mtm_pair_idx = pair_idx.index_select(0, mtm_feature_pos)
                if mtm_pair_idx.numel() > 0:
                    with torch.no_grad():
                        _, mtm_clip_feature_map = self.encodeClipImageFeatureMap(img[i])
                    mtm_relation_features = self.buildMtmRoiRelationFeatures(
                        mtm_clip_feature_map,
                        proposals[i],
                        mtm_pair_idx,
                        img[i],
                    )

            if self.training and self.mtmLossEnabled and mtm_relation_features is not None:
                relationFeaturesForMtm.append(mtm_relation_features)
                bplConditionFeatures.append(mtm_relation_features.detach())
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
                    mtmScores = self.computeMtmInferenceScores(
                        mtm_relation_features,
                        obj_n1,
                        obj_n2,
                        rel_dist_per_batch.shape,
                        rel_dist_per_batch.dtype,
                    )
                    rel_dist_per_batch = rel_dist_per_batch + self.mtmInferenceWeight * mtmScores
                if self.useRelationnessInference:
                    proposals[i].add_field("relationness_scores", relationness_scores.squeeze(-1).detach())

            rel_dists.append(rel_dist_per_batch)



        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        if self.training and self.mtmLossEnabled:
            add_losses.update(self.computeMtmLosses(relationFeaturesForMtm, relationLabelsForMtm, subjLabelsForMtm, objLabelsForMtm, bplConditionFeatures))
        if self.training and self.relationnessLossEnabled:
            add_losses.update(self.computeRelationnessLoss(relationnessLogitsForLoss, relationnessLabelsForLoss))
        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
