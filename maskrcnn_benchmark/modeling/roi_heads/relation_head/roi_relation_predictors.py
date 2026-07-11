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


class MTMVisualResidualAdapter(nn.Module):
    def __init__(self, featureDim=512, spatialDim=32, dropout=0.1, mode="union", maxScale=0.1):
        super(MTMVisualResidualAdapter, self).__init__()
        mode = str(mode).lower()
        if mode not in {"union", "context"}:
            raise ValueError("MTM.VISUAL_ADAPTER_MODE must be 'union' or 'context', got {}".format(mode))
        self.mode = mode
        inputDim = featureDim if mode == "union" else featureDim * 3 + spatialDim
        hiddenDim = max(featureDim // 4, 1)
        self.deltaNet = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenDim, featureDim),
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self.maxScale = float(maxScale)

    def residualScale(self):
        return self.maxScale * torch.tanh(self.alpha)

    def forward(self, unionFeatures, subFeatures, objFeatures, spatialFeatures):
        unionFeatures = unionFeatures.float()
        if self.mode == "context":
            adapterInputs = torch.cat(
                [
                    unionFeatures,
                    subFeatures.float(),
                    objFeatures.float(),
                    spatialFeatures.float(),
                ],
                dim=-1,
            )
        else:
            adapterInputs = unionFeatures
        delta = self.deltaNet(adapterInputs)
        return unionFeatures + self.residualScale() * delta


def initializeShipLayer(module, initStd=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=initStd)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ShipVisualEncoder(nn.Module):
    def __init__(self, visualDim=512, hiddenDim=2048, latentDim=512, initStd=0.02):
        super(ShipVisualEncoder, self).__init__()
        self.featureEncoder = nn.Sequential(
            nn.Linear(visualDim, hiddenDim),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(hiddenDim, latentDim)
        self.logvar = nn.Linear(hiddenDim, latentDim)
        self.apply(lambda module: initializeShipLayer(module, initStd))

    def forward(self, visualFeatures):
        hidden = self.featureEncoder(visualFeatures.float())
        return self.mean(hidden), self.logvar(hidden)


class ShipLatentGenerator(nn.Module):
    def __init__(self, latentDim=512, hiddenDim=4096, outputDim=512, initStd=0.02):
        super(ShipLatentGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latentDim, hiddenDim),
            nn.ReLU(inplace=True),
            nn.Linear(hiddenDim, outputDim),
        )
        self.apply(lambda module: initializeShipLayer(module, initStd))

    def forward(self, latentFeatures):
        return self.net(latentFeatures.float())


class ShipTripletFeatureGenerator(nn.Module):
    def __init__(self, clipModel, visualDim=512, embedDim=512, latentDim=512, ctxLen=4, initStd=0.02):
        super(ShipTripletFeatureGenerator, self).__init__()
        if visualDim != embedDim:
            raise ValueError("SHIP prompt reconstruction requires visualDim == embedDim")
        object.__setattr__(self, "clipModel", clipModel)
        self.latentDim = latentDim
        self.ctxLen = ctxLen
        self.ctx = nn.Parameter(torch.empty(ctxLen, embedDim))
        nn.init.normal_(self.ctx, mean=0.0, std=initStd)
        self.visualEncoder = ShipVisualEncoder(
            visualDim=visualDim,
            hiddenDim=2048,
            latentDim=latentDim,
            initStd=initStd,
        )
        self.latentGenerator = ShipLatentGenerator(
            latentDim=latentDim,
            hiddenDim=4096,
            outputDim=embedDim,
            initStd=initStd,
        )

    def reparameterize(self, mean, logvar):
        noise = torch.randn_like(mean)
        return mean + noise * torch.exp(0.5 * logvar)

    def encodePromptEmbeddings(self, prompts, tokenizedPrompts):
        dtype = self.clipModel.dtype
        x = prompts.to(dtype=dtype) + self.clipModel.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)
        x = self.clipModel.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clipModel.ln_final(x).type(dtype)
        eosPositions = tokenizedPrompts.argmax(dim=-1)
        x = x[torch.arange(x.size(0), device=x.device), eosPositions]
        return (x @ self.clipModel.text_projection).float()

    def encodeResidualPrompts(self, tripletTexts, residuals):
        promptPrefix = " ".join(["X"] * self.ctxLen)
        promptedTexts = [promptPrefix + " " + text for text in tripletTexts]
        tokenizedPrompts = clip.tokenize(promptedTexts).to(residuals.device)
        with torch.no_grad():
            tokenEmbeddings = self.clipModel.token_embedding(tokenizedPrompts).float()
        prefix = tokenEmbeddings[:, :1, :]
        suffix = tokenEmbeddings[:, 1 + self.ctxLen :, :]
        ctx = self.ctx.unsqueeze(0).expand(residuals.size(0), -1, -1)
        prompts = torch.cat([prefix, ctx + residuals.unsqueeze(1), suffix], dim=1)
        return F.normalize(self.encodePromptEmbeddings(prompts, tokenizedPrompts), dim=-1)

    def reconstruct(self, tripletTexts, visualFeatures):
        reconstructionTargets = F.normalize(visualFeatures.detach().float(), dim=-1)
        mean, logvar = self.visualEncoder(reconstructionTargets)
        latentFeatures = self.reparameterize(mean, logvar)
        residuals = self.latentGenerator(latentFeatures)
        reconstructedFeatures = self.encodeResidualPrompts(tripletTexts, residuals)
        return reconstructedFeatures, mean, logvar

    def generate(self, tripletTexts, device):
        latentFeatures = torch.randn(
            len(tripletTexts),
            self.latentDim,
            device=device,
            dtype=torch.float32,
        )
        residuals = self.latentGenerator(latentFeatures)
        return self.encodeResidualPrompts(tripletTexts, residuals)

    def klLoss(self, mean, logvar):
        return (-0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1)).mean()


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

            self.texts5 = self.encodeSubjectFilterTextFeatures(self.sub_filter_novel)
            self.texts5Tensor = F.normalize(torch.stack(self.texts5, dim=0), dim=-1)

        b=time.time()
        print('init complete : '+str(b-a))

        self.zhangliang=[]
        self.count=0

        self.linear1=nn.Linear(1024,512, bias=False).to(self.device).half()
        mtmConfig = config.MODEL.ROI_RELATION_HEAD.MTM
        self.mtmEnabled = mtmConfig.ENABLED
        self.mtmLossEnabled = mtmConfig.ENABLED and mtmConfig.LOSS_ENABLED
        self.mtmLossWeight = mtmConfig.LOSS_WEIGHT
        self.mtmAlignWeight = mtmConfig.ALIGN_WEIGHT
        self.mtmStructureWeight = mtmConfig.STRUCTURE_WEIGHT
        self.mtmVisualStructureWeight = mtmConfig.VISUAL_STRUCTURE_WEIGHT
        self.mtmStructureDebugEnabled = mtmConfig.STRUCTURE_DEBUG_ENABLED
        self.mtmStructureDebugStep = max(int(mtmConfig.STRUCTURE_DEBUG_STEP), 1)
        self.mtmMaxPairs = mtmConfig.MAX_PAIRS
        self.mtmUseInference = mtmConfig.ENABLED and mtmConfig.USE_INFERENCE
        self.mtmInferenceWeight = mtmConfig.INFERENCE_WEIGHT
        self.mtmTextSvdEnabled = mtmConfig.TEXT_SVD_ENABLED
        self.mtmTextSvdComponents = mtmConfig.TEXT_SVD_COMPONENTS
        self.mtmTextSvdChunkSize = mtmConfig.TEXT_SVD_CHUNK_SIZE
        self.mtmVisualAdapterEnabled = mtmConfig.VISUAL_ADAPTER_ENABLED
        self.mtmVisualAdapterMode = mtmConfig.VISUAL_ADAPTER_MODE
        self.mtmShipEnabled = mtmConfig.SHIP_ENABLED
        self.mtmShipPseudoRatio = mtmConfig.SHIP_PSEUDO_RATIO
        self.mtmShipReconWeight = mtmConfig.SHIP_RECON_WEIGHT
        self.mtmShipKlWeight = mtmConfig.SHIP_KL_WEIGHT
        self.mtmShipNovelAlignWeight = mtmConfig.SHIP_NOVEL_ALIGN_WEIGHT
        self.mtmShipVisualTransferEnabled = mtmConfig.SHIP_VISUAL_TRANSFER_ENABLED
        self.mtmShipVisualTransferWeight = mtmConfig.SHIP_VISUAL_TRANSFER_WEIGHT
        self.mtmShipTransferTemperature = max(float(mtmConfig.SHIP_TRANSFER_TEMPERATURE), 1e-6)
        self.mtmShipWarmupIters = max(int(mtmConfig.SHIP_WARMUP_ITERS), 0)
        self.mtmShipRampIters = max(int(mtmConfig.SHIP_RAMP_ITERS), 1)
        self.clipInputSize = self.clip_model.visual.input_resolution
        self.clipImageNormalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        self.mtmVisualAdapter = MTMVisualResidualAdapter(
            featureDim=mtmConfig.INPUT_DIM,
            spatialDim=32,
            dropout=mtmConfig.DROPOUT,
            mode=self.mtmVisualAdapterMode,
            maxScale=mtmConfig.VISUAL_ADAPTER_MAX_SCALE,
        ).to(self.device)
        self.shipFeatureGenerator = None
        if self.mtmShipEnabled:
            if self.mtmVisualAdapterEnabled and self.mtmVisualAdapterMode != "union":
                raise ValueError("SHIP pseudo union features require MTM.VISUAL_ADAPTER_MODE='union'")
            self.shipFeatureGenerator = ShipTripletFeatureGenerator(
                self.clip_model,
                visualDim=mtmConfig.INPUT_DIM,
                embedDim=mtmConfig.EMBED_DIM,
                latentDim=mtmConfig.EMBED_DIM,
                ctxLen=mtmConfig.SHIP_CTX_LEN,
                initStd=mtmConfig.SHIP_INIT_STD,
            ).to(self.device)
        self.relationMtm = RelationModalityTransfer(
            mtmConfig.INPUT_DIM,
            mtmConfig.EMBED_DIM,
            mtmConfig.FC_LAYERS,
            mtmConfig.ATTENTION_LAYERS,
            mtmConfig.NUM_HEADS,
            mtmConfig.DROPOUT,
        ).to(self.device)
        self.activeRelNames = activeRelNames
        self.filteredTripletEmbeddingCache = {}
        self.tripletTextSvdCache = {}
        self.tripletTextPrincipalComponents = None
        self.novelRelNames = [self.relNames[index] for index in self.novel if index != 0]
        self.novelRelationCandidates = {}
        self.register_buffer("shipTrainingStep", torch.zeros((), dtype=torch.long))
        self.register_buffer("mtmDebugIteration", torch.zeros((), dtype=torch.long))
        self.updateMtmTextSvdBasis()
        self.mtmModeStateCache = {
            mode: {
                "description_relation": self.description_relation,
                "sub_filter_novel": self.sub_filter_novel,
                "activeRelNames": self.activeRelNames,
                "texts5": self.texts5,
                "texts5Tensor": self.texts5Tensor,
            }
        }
        self.filteredTripletEmbeddingCaches = {mode: self.filteredTripletEmbeddingCache}

    def encodeRawTripletTexts(self, tripletTexts):
        with torch.no_grad():
            tripletTokens = clip.tokenize(tripletTexts).to(self.device)
            tripletTextFeatures = self.clip_model.encode_text(tripletTokens).float()
            return F.normalize(tripletTextFeatures, dim=-1)

    def encodeSubjectFilterTextFeatures(self, subjectFilter):
        allTexts = []
        featureCounts = []
        for objName in self.obj_names:
            objectTexts = ["a photo of " + text for text in list(subjectFilter[objName])]
            allTexts.extend(objectTexts)
            featureCounts.append(len(objectTexts))

        encodedChunks = []
        chunkSize = max(int(getattr(self, "mtmTextSvdChunkSize", 1024)), 1)
        with torch.no_grad():
            for offset in range(0, len(allTexts), chunkSize):
                tokens = clip.tokenize(allTexts[offset : offset + chunkSize]).to(self.device)
                encodedChunks.append(self.clip_model.encode_text(tokens).detach())
        encodedFeatures = torch.cat(encodedChunks, dim=0)
        return list(encodedFeatures.split(featureCounts, dim=0))

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

    def buildNovelTripletTexts(self, subjLabels, objLabels, pseudoCount):
        if pseudoCount <= 0 or len(self.novelRelNames) == 0:
            return [], None, None
        sampledPairIndices = torch.randint(
            subjLabels.numel(),
            (pseudoCount,),
            device=subjLabels.device,
        )
        sampledSubjLabels = subjLabels.index_select(0, sampledPairIndices).detach().cpu().tolist()
        sampledObjLabels = objLabels.index_select(0, sampledPairIndices).detach().cpu().tolist()
        texts = []
        for subjLabel, objLabel in zip(sampledSubjLabels, sampledObjLabels):
            subjName = self.obj_names[int(subjLabel)]
            objName = self.obj_names[int(objLabel)]
            if subjName not in self.novelRelationCandidates:
                compatibleRelations = []
                if subjName in self.mtmTextSvdFilter.columns:
                    for relationIndex in self.novel:
                        if relationIndex == 0 or relationIndex >= len(self.mtmTextSvdFilter):
                            continue
                        relationName = self.relNames[relationIndex]
                        if str(self.mtmTextSvdFilter.iloc[relationIndex][subjName]) == relationName:
                            compatibleRelations.append(relationName)
                if len(compatibleRelations) == 0:
                    compatibleRelations = self.novelRelNames
                self.novelRelationCandidates[subjName] = compatibleRelations
            compatibleRelations = self.novelRelationCandidates[subjName]
            sampledRelation = compatibleRelations[
                int(torch.randint(len(compatibleRelations), (1,)).item())
            ]
            texts.append("a photo of a " + subjName + " " + sampledRelation + " a " + objName)
        return (
            texts,
            subjLabels.index_select(0, sampledPairIndices),
            objLabels.index_select(0, sampledPairIndices),
        )

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

    def populateFilteredTripletEmbeddingCache(self, subjLabels, objLabels):
        labelPairs = torch.stack([subjLabels, objLabels], dim=-1).detach().cpu().tolist()
        missingKeys = []
        seenKeys = set()
        for subjLabel, objLabel in labelPairs:
            cacheKey = (int(subjLabel), int(objLabel))
            if cacheKey not in self.filteredTripletEmbeddingCache and cacheKey not in seenKeys:
                missingKeys.append(cacheKey)
                seenKeys.add(cacheKey)
        if len(missingKeys) == 0:
            return

        pendingTexts = []
        pendingEntries = []
        chunkSize = max(int(self.mtmTextSvdChunkSize), 1)

        def encodePendingTexts():
            if len(pendingTexts) == 0:
                return
            encoded = self.encodeTripletTexts(pendingTexts).detach().cpu()
            offset = 0
            for cacheKey, candidateCount in pendingEntries:
                self.filteredTripletEmbeddingCache[cacheKey] = (
                    encoded[offset : offset + candidateCount],
                    candidateCount,
                )
                offset += candidateCount
            del pendingTexts[:]
            del pendingEntries[:]

        for cacheKey in missingKeys:
            objLabelsForText = torch.tensor([cacheKey[1]], device=self.device)
            texts, candidateCount = self.buildFilteredTripletTexts(cacheKey[0], objLabelsForText)
            if len(pendingTexts) > 0 and len(pendingTexts) + len(texts) > chunkSize:
                encodePendingTexts()
            pendingTexts.extend(texts)
            pendingEntries.append((cacheKey, candidateCount))
        encodePendingTexts()

    def encodeClipImageFeatureMap(self, image):
        imageInput = FF.resize(image, (self.clipInputSize, self.clipInputSize))
        imageInput = self.clipImageNormalize(imageInput).unsqueeze(0).to(self.device)
        tokens = self.clip_model.encode_image(imageInput)
        if tokens.dim() != 3:
            raise ValueError("MTM RoIAlign requires CLIP ViT token features, got shape {}".format(tuple(tokens.shape)))
        globalFeature = tokens[:, 0, :].float()
        patchTokens = tokens[:, 1:, :].float()
        gridSize = int(patchTokens.size(1) ** 0.5)
        if gridSize * gridSize != patchTokens.size(1):
            raise ValueError("CLIP patch token count {} is not a square grid".format(patchTokens.size(1)))
        featureMap = patchTokens.permute(0, 2, 1).contiguous().view(1, patchTokens.size(-1), gridSize, gridSize)
        return globalFeature, featureMap

    def cropImagePadding(self, image, proposal):
        imageWidth, imageHeight = proposal.size
        return image[:, : int(imageHeight), : int(imageWidth)]

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
                torch.zeros((boxes.size(0), 1), device=clipFeatureMap.device, dtype=clipFeatureMap.dtype),
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

    def buildMtmRoiRelationFeatures(self, clipFeatureMap, proposal, pairIdx, image, returnRawUnion=False):
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
        if self.mtmVisualAdapterEnabled:
            relationFeatures = self.mtmVisualAdapter(
                unionFeatures,
                subFeatures,
                objFeatures,
                spatialFeatures,
            )
        else:
            relationFeatures = unionFeatures
        if returnRawUnion:
            return relationFeatures, unionFeatures.detach()
        return relationFeatures

    def buildPairSpatialFeatures(self, proposal, pairIdx):
        boxInfo = get_box_info(proposal.bbox, need_norm=True, proposal=proposal)
        subjBoxInfo = boxInfo.index_select(0, pairIdx[:, 0])
        objBoxInfo = boxInfo.index_select(0, pairIdx[:, 1])
        return get_box_pair_info(subjBoxInfo, objBoxInfo)

    def emptyMtmLosses(self, zero):
        return {
            "loss_mtm_align": zero,
            "loss_mtm_novel_align": zero,
            "loss_mtm_visual_structure": zero,
            "loss_mtm_ship_recon": zero,
            "loss_mtm_ship_kl": zero,
            "loss_mtm_ship_visual_transfer": zero,
        }

    def computeMtmLosses(
        self,
        relationFeatures,
        rawUnionFeatures,
        relationLabels,
        subjLabels,
        objLabels,
        logger=None,
    ):
        if not self.mtmLossEnabled or len(relationFeatures) == 0 or relationLabels is None:
            return self.emptyMtmLosses(self.relationMtm.norm.weight.sum() * 0.0)

        relationFeatures = torch.cat(relationFeatures, dim=0).float()
        rawUnionFeatures = torch.cat(rawUnionFeatures, dim=0).float().to(relationFeatures.device)
        relationLabels = torch.cat(relationLabels, dim=0).view(-1).long().to(relationFeatures.device)
        subjLabels = torch.cat(subjLabels, dim=0).view(-1).long().to(relationFeatures.device)
        objLabels = torch.cat(objLabels, dim=0).view(-1).long().to(relationFeatures.device)
        positive = relationLabels > 0
        if positive.any() and relationLabels[positive].max().item() >= len(self.activeRelNames):
            raise ValueError(
                "MTM received relation label {} but active relation names only has {} entries. "
                "Use dataset rel_classes for MTM targets or regenerate stale dataset-statistics cache.".format(
                    relationLabels[positive].max().item(), len(self.activeRelNames)
                )
            )
        if not positive.any():
            return self.emptyMtmLosses(relationFeatures.sum() * 0.0)

        relationFeatures = relationFeatures[positive]
        rawUnionFeatures = rawUnionFeatures[positive]
        relationLabels = relationLabels[positive]
        subjLabels = subjLabels[positive]
        objLabels = objLabels[positive]
        self.mtmDebugIteration.add_(1)
        if self.mtmMaxPairs > 0 and relationFeatures.size(0) > self.mtmMaxPairs:
            sampleIndex = torch.linspace(
                0,
                relationFeatures.size(0) - 1,
                steps=self.mtmMaxPairs,
                device=relationFeatures.device,
            ).long()
            relationFeatures = relationFeatures.index_select(0, sampleIndex)
            rawUnionFeatures = rawUnionFeatures.index_select(0, sampleIndex)
            relationLabels = relationLabels.index_select(0, sampleIndex)
            subjLabels = subjLabels.index_select(0, sampleIndex)
            objLabels = objLabels.index_select(0, sampleIndex)

        baseVisualFeatures = F.normalize(relationFeatures.float(), dim=-1)
        baseRawUnionFeatures = F.normalize(rawUnionFeatures.detach().float(), dim=-1)
        baseTexts = self.buildTargetTripletTexts(subjLabels, relationLabels, objLabels)
        baseTextTargets = F.normalize(
            self.encodeTripletTexts(baseTexts).to(baseVisualFeatures.device).float(),
            dim=-1,
        ).detach()
        zero = baseVisualFeatures.sum() * 0.0
        novelAlignLoss = zero
        reconstructionLoss = zero
        klLoss = zero
        visualTransferLoss = zero
        visualTransferCosine = None
        pseudoVisualFeatures = None
        pseudoRawUnionFeatures = None
        pseudoWeight = 0.0
        klWeight = 0.0

        if self.mtmShipEnabled:
            self.shipTrainingStep.add_(1)
            currentStep = int(self.shipTrainingStep.item())
            rampProgress = min(
                max(float(currentStep - self.mtmShipWarmupIters), 0.0) / float(self.mtmShipRampIters),
                1.0,
            )
            pseudoWeight = rampProgress
            klWeight = rampProgress
            reconstructedFeatures, shipMean, shipLogvar = self.shipFeatureGenerator.reconstruct(
                baseTexts,
                baseRawUnionFeatures,
            )
            reconstructionLoss = (
                reconstructedFeatures - baseRawUnionFeatures
            ).pow(2).sum(dim=-1).mean()
            klLoss = self.shipFeatureGenerator.klLoss(shipMean, shipLogvar)
            if currentStep > self.mtmShipWarmupIters and self.mtmShipPseudoRatio > 0:
                pseudoCount = int(round(baseVisualFeatures.size(0) * float(self.mtmShipPseudoRatio)))
                if pseudoCount == 0:
                    pseudoCount = 1
                novelTexts, novelSubjLabels, novelObjLabels = self.buildNovelTripletTexts(
                    subjLabels,
                    objLabels,
                    pseudoCount,
                )
            else:
                novelTexts = []
                novelSubjLabels = None
                novelObjLabels = None
            if len(novelTexts) > 0:
                novelTextTargets = F.normalize(
                    self.encodeTripletTexts(novelTexts).to(baseVisualFeatures.device).float(),
                    dim=-1,
                ).detach()
                pseudoRawUnionFeatures = self.shipFeatureGenerator.generate(
                    novelTexts,
                    baseVisualFeatures.device,
                )
                if self.mtmVisualAdapterEnabled:
                    pseudoVisualFeatures = self.mtmVisualAdapter(
                        pseudoRawUnionFeatures,
                        None,
                        None,
                        None,
                    )
                else:
                    pseudoVisualFeatures = pseudoRawUnionFeatures
                pseudoVisualFeatures = F.normalize(pseudoVisualFeatures.float(), dim=-1)
                allVisualFeatures = torch.cat([baseVisualFeatures, pseudoVisualFeatures], dim=0)
            else:
                novelTextTargets = None
                allVisualFeatures = baseVisualFeatures
        else:
            novelTexts = []
            novelTextTargets = None
            allVisualFeatures = baseVisualFeatures

        predictedFeatures = F.normalize(self.relationMtm(allVisualFeatures).float(), dim=-1)
        baseCount = baseVisualFeatures.size(0)
        basePredictedFeatures = predictedFeatures[:baseCount]
        baseAlignLoss = (1.0 - (basePredictedFeatures * baseTextTargets).sum(dim=-1)).mean()

        if pseudoVisualFeatures is not None:
            novelPredictedFeatures = predictedFeatures[baseCount:]
            novelAlignLoss = (
                1.0 - (novelPredictedFeatures * novelTextTargets).sum(dim=-1)
            ).mean()
            if self.mtmShipVisualTransferEnabled:
                transferTargets = []
                detachedBaseKeys = basePredictedFeatures.detach()
                detachedBaseValues = baseVisualFeatures.detach()
                for novelIndex in range(pseudoVisualFeatures.size(0)):
                    candidateMask = (
                        (subjLabels == novelSubjLabels[novelIndex])
                        & (objLabels == novelObjLabels[novelIndex])
                    )
                    candidateRelations = relationLabels[candidateMask]
                    candidateKeys = []
                    candidateValues = []
                    maskedKeys = detachedBaseKeys[candidateMask]
                    maskedValues = detachedBaseValues[candidateMask]
                    for baseRelation in candidateRelations.unique():
                        relationMask = candidateRelations == baseRelation
                        candidateKeys.append(maskedKeys[relationMask].mean(dim=0))
                        candidateValues.append(maskedValues[relationMask].mean(dim=0))
                    candidateKeys = F.normalize(torch.stack(candidateKeys, dim=0), dim=-1)
                    candidateValues = F.normalize(torch.stack(candidateValues, dim=0), dim=-1)
                    transferLogits = torch.matmul(
                        candidateKeys,
                        novelTextTargets[novelIndex],
                    ) / self.mtmShipTransferTemperature
                    transferWeights = F.softmax(transferLogits, dim=0)
                    transferTargets.append(
                        torch.sum(candidateValues * transferWeights.unsqueeze(-1), dim=0)
                    )
                transferTargets = F.normalize(torch.stack(transferTargets, dim=0), dim=-1)
                visualTransferCosine = (
                    pseudoVisualFeatures * transferTargets
                ).sum(dim=-1).mean()
                visualTransferLoss = 1.0 - visualTransferCosine

        if predictedFeatures.size(0) < 2:
            visualStructureLoss = zero
        else:
            inputVisualSimilarity = torch.matmul(allVisualFeatures, allVisualFeatures.t())
            predictedSimilarity = torch.matmul(predictedFeatures, predictedFeatures.t())
            structureError = (inputVisualSimilarity - predictedSimilarity).abs()
            offDiagonal = ~torch.eye(
                predictedSimilarity.size(0),
                dtype=torch.bool,
                device=predictedSimilarity.device,
            )
            structureWeights = torch.ones_like(structureError)
            if pseudoVisualFeatures is not None:
                novelPositions = torch.arange(
                    predictedFeatures.size(0),
                    device=predictedFeatures.device,
                ) >= baseCount
                involvesNovel = novelPositions.unsqueeze(0) | novelPositions.unsqueeze(1)
                structureWeights = torch.where(
                    involvesNovel,
                    structureWeights.new_full((), pseudoWeight),
                    structureWeights,
                )
            validWeights = structureWeights[offDiagonal]
            visualStructureLoss = (
                structureError[offDiagonal] * validWeights
            ).sum() / validWeights.sum().clamp(min=1.0)
            if (
                self.mtmStructureDebugEnabled
                and logger is not None
                and baseCount >= 2
                and int(self.mtmDebugIteration.item()) % self.mtmStructureDebugStep == 0
            ):
                with torch.no_grad():
                    def similarityStats(similarity, mask):
                        values = similarity[mask].float().clamp(min=-1.0, max=1.0)
                        return (
                            values.mean().item(),
                            values.max().item(),
                            values.min().item(),
                            values.std(unbiased=False).item(),
                        )

                    baseMask = ~torch.eye(
                        baseCount,
                        dtype=torch.bool,
                        device=predictedSimilarity.device,
                    )
                    baseRawSimilarity = torch.matmul(
                        baseRawUnionFeatures,
                        baseRawUnionFeatures.t(),
                    )
                    baseVisualSimilarity = inputVisualSimilarity[:baseCount, :baseCount]
                    baseMtmSimilarity = predictedSimilarity[:baseCount, :baseCount]
                    baseTextSimilarity = torch.matmul(baseTextTargets, baseTextTargets.t())
                    baseRawStats = similarityStats(baseRawSimilarity, baseMask)
                    baseTextStats = similarityStats(baseTextSimilarity, baseMask)
                    baseMtmStats = similarityStats(baseMtmSimilarity, baseMask)
                    baseVisualStats = similarityStats(baseVisualSimilarity, baseMask)
                    baseStructureError = (
                        baseVisualSimilarity - baseMtmSimilarity
                    ).abs()[baseMask].mean().item()
                    baseAlignCosine = (
                        basePredictedFeatures * baseTextTargets
                    ).sum(dim=-1).mean().item()
                    logger.info(
                        "MTM debug i=%d base "
                        "raw(mean/max/min/std)=%.4f/%.4f/%.4f/%.4f "
                        "xrel(mean/max/min/std)=%.4f/%.4f/%.4f/%.4f "
                        "qrel(mean/max/min/std)=%.4f/%.4f/%.4f/%.4f "
                        "tsvd(mean/max/min/std)=%.4f/%.4f/%.4f/%.4f "
                        "align=%.4f err=%.4f scale=%.4f",
                        int(self.mtmDebugIteration.item()),
                        *baseRawStats,
                        *baseVisualStats,
                        *baseMtmStats,
                        *baseTextStats,
                        baseAlignCosine,
                        baseStructureError,
                        self.mtmVisualAdapter.residualScale().item()
                        if self.mtmVisualAdapterEnabled else 0.0,
                    )
                    novelCount = predictedFeatures.size(0) - baseCount
                    if pseudoVisualFeatures is not None and novelCount >= 2:
                        novelMask = ~torch.eye(
                            novelCount,
                            dtype=torch.bool,
                            device=predictedSimilarity.device,
                        )
                        novelVisualSimilarity = inputVisualSimilarity[baseCount:, baseCount:]
                        novelMtmSimilarity = predictedSimilarity[baseCount:, baseCount:]
                        novelTextSimilarity = torch.matmul(novelTextTargets, novelTextTargets.t())
                        novelRawSimilarity = torch.matmul(
                            pseudoRawUnionFeatures,
                            pseudoRawUnionFeatures.t(),
                        )
                        novelRawStats = similarityStats(novelRawSimilarity, novelMask)
                        novelTextStats = similarityStats(novelTextSimilarity, novelMask)
                        novelMtmStats = similarityStats(novelMtmSimilarity, novelMask)
                        novelVisualStats = similarityStats(novelVisualSimilarity, novelMask)
                        novelStructureError = (
                            novelVisualSimilarity - novelMtmSimilarity
                        ).abs()[novelMask].mean().item()
                        novelAlignCosine = (
                            novelPredictedFeatures * novelTextTargets
                        ).sum(dim=-1).mean().item()
                        crossStructureError = (
                            inputVisualSimilarity[:baseCount, baseCount:]
                            - predictedSimilarity[:baseCount, baseCount:]
                        ).abs().mean().item()
                        logger.info(
                            "MTM debug i=%d novel "
                            "raw(mean/max/min/std)=%.4f/%.4f/%.4f/%.4f "
                            "xrel(mean/max/min/std)=%.4f/%.4f/%.4f/%.4f "
                            "qrel(mean/max/min/std)=%.4f/%.4f/%.4f/%.4f "
                            "tsvd(mean/max/min/std)=%.4f/%.4f/%.4f/%.4f "
                            "align=%.4f transfer=%.4f err=%.4f cross_err=%.4f ramp=%.4f",
                            int(self.mtmDebugIteration.item()),
                            *novelRawStats,
                            *novelVisualStats,
                            *novelMtmStats,
                            *novelTextStats,
                            novelAlignCosine,
                            visualTransferCosine.item()
                            if visualTransferCosine is not None else float("nan"),
                            novelStructureError,
                            crossStructureError,
                            pseudoWeight,
                        )

        return {
            "loss_mtm_align": self.mtmLossWeight * self.mtmAlignWeight * baseAlignLoss,
            "loss_mtm_novel_align": self.mtmLossWeight * self.mtmShipNovelAlignWeight * pseudoWeight * novelAlignLoss,
            "loss_mtm_visual_structure": self.mtmLossWeight * self.mtmStructureWeight * self.mtmVisualStructureWeight * visualStructureLoss,
            "loss_mtm_ship_recon": self.mtmLossWeight * self.mtmShipReconWeight * reconstructionLoss,
            "loss_mtm_ship_kl": self.mtmLossWeight * self.mtmShipKlWeight * klWeight * klLoss,
            "loss_mtm_ship_visual_transfer": self.mtmLossWeight * self.mtmShipVisualTransferWeight * pseudoWeight * visualTransferLoss,
        }

    def computeMtmInferenceScores(self, relationFeatures, subjLabels, objLabels, outShape, outDtype):
        normalizedRelationFeatures = F.normalize(relationFeatures.float(), dim=-1)
        predicted_embeddings = F.normalize(self.relationMtm(normalizedRelationFeatures).float(), dim=-1)
        scores = relationFeatures.new_zeros(outShape, dtype=torch.float32)
        self.populateFilteredTripletEmbeddingCache(subjLabels, objLabels)
        labelPairs = torch.stack([subjLabels, objLabels], dim=-1).detach().cpu().tolist()
        chunkSize = max(int(self.mtmTextSvdChunkSize), 1)
        for offset in range(0, len(labelPairs), chunkSize):
            chunkPairs = labelPairs[offset : offset + chunkSize]
            candidateFeatures = [
                self.filteredTripletEmbeddingCache[(int(subjLabel), int(objLabel))][0]
                for subjLabel, objLabel in chunkPairs
            ]
            candidateEmbeddings = torch.stack(candidateFeatures, dim=0).to(
                device=predicted_embeddings.device,
                dtype=torch.float32,
            )
            chunkPredicted = predicted_embeddings[offset : offset + len(chunkPairs)]
            chunkScores = (chunkPredicted.unsqueeze(1) * candidateEmbeddings).sum(dim=-1)
            scores[offset : offset + len(chunkPairs), :outShape[1]] = chunkScores[:, :outShape[1]]
        return scores.to(dtype=outDtype)

    def computePairFilterScores(self, subFeatures, objFeatures, subjLabels, numRelations):
        textFeatureBank = self.texts5Tensor.to(
            device=subFeatures.device,
            dtype=subFeatures.dtype,
        )
        scores = subFeatures.new_empty((subFeatures.size(0), numRelations))
        chunkSize = max(int(self.mtmTextSvdChunkSize), 1)
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
            self.filteredTripletEmbeddingCache = self.filteredTripletEmbeddingCaches[mode]
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
        self.filteredTripletEmbeddingCache = self.filteredTripletEmbeddingCaches.setdefault(mode, {})
        self.updateMtmTextSvdBasis()

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


        rel_dists=[]
        relationFeaturesForMtm = []
        rawUnionFeaturesForMtm = []
        relationLabelsForMtm = []
        subjLabelsForMtm = []
        objLabelsForMtm = []
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
                    mtm_image = self.cropImagePadding(img[i], proposals[i])
                    with torch.no_grad():
                        _, mtm_clip_feature_map = self.encodeClipImageFeatureMap(mtm_image)
                    mtmFeatureOutput = self.buildMtmRoiRelationFeatures(
                        mtm_clip_feature_map,
                        proposals[i],
                        mtm_pair_idx,
                        mtm_image,
                        returnRawUnion=(self.training and self.mtmLossEnabled),
                    )
                    if self.training and self.mtmLossEnabled:
                        mtm_relation_features, mtm_raw_union_features = mtmFeatureOutput
                    else:
                        mtm_relation_features = mtmFeatureOutput

            if self.training and self.mtmLossEnabled and mtm_relation_features is not None:
                relationFeaturesForMtm.append(mtm_relation_features)
                rawUnionFeaturesForMtm.append(mtm_raw_union_features)
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
                if self.mtmUseInference and self.mtmInferenceWeight != 0:
                    mtmScores = self.computeMtmInferenceScores(
                        mtm_relation_features,
                        obj_n1,
                        obj_n2,
                        rel_dist_per_batch.shape,
                        rel_dist_per_batch.dtype,
                    )
                    rel_dist_per_batch = rel_dist_per_batch + self.mtmInferenceWeight * mtmScores
            rel_dists.append(rel_dist_per_batch)



        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        if self.training and self.mtmLossEnabled:
            add_losses.update(
                self.computeMtmLosses(
                    relationFeaturesForMtm,
                    rawUnionFeaturesForMtm,
                    relationLabelsForMtm,
                    subjLabelsForMtm,
                    objLabelsForMtm,
                    logger,
                )
            )
        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
