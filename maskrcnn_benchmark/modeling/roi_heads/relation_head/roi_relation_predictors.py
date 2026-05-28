# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import json
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as FF
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .low_rank_text import (
    CoreRelationTextAdapter,
    load_relation_prompt_texts,
    build_predicate_splits,
    build_full_predicate_names,
)
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.utils.comm import is_main_process
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
                text_fea = text_fea.unsqueeze(0).expand(x.size(0), -1)
            x = torch.cat((x, text_fea.to(x.device, dtype=x.dtype)), dim=-1)
            x=self.linear(x)
            x=self.relu(x)
        ratio = 0.5
        sub_features= ratio * x + (1 - ratio) * sub_features[:,0,:]

        return sub_features


def load_relation_concepts(concept_json, fallback_concepts):
    if concept_json:
        with open(concept_json, "r") as f:
            concept_data = json.load(f)
        if isinstance(concept_data, dict):
            concept_data = concept_data.get("concepts", concept_data.get("relation_concepts"))
        if not isinstance(concept_data, list):
            raise ValueError("EZPC_TEXT.CONCEPT_JSON must contain a list or a 'concepts' field.")
        return [str(concept) for concept in concept_data]
    return [str(concept) for concept in fallback_concepts]


class EZPCRelationConceptProjector(nn.Module):
    """EZPC-style concept projection for relation predicates.

    A is initialized from CLIP text embeddings of human-readable relation
    concepts. Both pair visual features and predicate text features are
    projected through A, so each predicate logit is decomposable into concept
    contributions.
    """

    def __init__(
        self,
        concept_features,
        predicate_text_features,
        temperature=0.07,
        match_loss_weight=0.01,
        recon_loss_weight=0.1,
        freeze_a=False,
    ):
        super(EZPCRelationConceptProjector, self).__init__()
        concept_features = F.normalize(concept_features.float(), dim=-1)
        predicate_text_features = F.normalize(predicate_text_features.float(), dim=-1)
        concept_matrix = concept_features.t().contiguous()

        if freeze_a:
            self.register_buffer("A", concept_matrix)
        else:
            self.A = nn.Parameter(concept_matrix)
        self.register_buffer("concept_anchor", concept_matrix.detach().clone())
        self.register_buffer("predicate_text_features", predicate_text_features)

        self.temperature = float(temperature)
        self.match_loss_weight = float(match_loss_weight)
        self.recon_loss_weight = float(recon_loss_weight)

    def active_predicate_text(self, fg_ids):
        return self.predicate_text_features[fg_ids.to(self.predicate_text_features.device).long()].float()

    def project(self, features):
        return features.float() @ self.A.float()

    def logits(self, visual_features, fg_ids):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        text_features = self.active_predicate_text(fg_ids)
        text_features = F.normalize(text_features.float(), dim=-1)

        visual_concepts = self.project(visual_features)
        text_concepts = self.project(text_features)
        temperature = max(self.temperature, 1e-6)
        concept_logits = visual_concepts @ text_concepts.t() / temperature

        with torch.no_grad():
            teacher_logits = visual_features @ text_features.t() / temperature
        return concept_logits, teacher_logits, visual_concepts, text_concepts

    def concept_contributions(self, visual_features, fg_ids):
        visual_features = F.normalize(visual_features.float(), dim=-1)
        text_features = F.normalize(self.active_predicate_text(fg_ids), dim=-1)
        visual_concepts = self.project(visual_features)
        text_concepts = self.project(text_features)
        return visual_concepts.unsqueeze(1) * text_concepts.unsqueeze(0)

    def losses(self, concept_logits=None, teacher_logits=None):
        losses = {}
        if self.match_loss_weight > 0:
            losses["loss_ezpc_match"] = (
                F.mse_loss(self.A.float(), self.concept_anchor.float()) * self.match_loss_weight
            )
        if (
            self.recon_loss_weight > 0
            and concept_logits is not None
            and teacher_logits is not None
            and concept_logits.numel() > 0
        ):
            losses["loss_ezpc_recon"] = (
                F.kl_div(
                    F.log_softmax(concept_logits.float(), dim=-1),
                    F.softmax(teacher_logits.float(), dim=-1),
                    reduction="batchmean",
                )
                * self.recon_loss_weight
            )
        return losses


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


@registry.ROI_RELATION_PREDICTOR.register("EZPCClipPredictor")
class EZPCClipPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(EZPCClipPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.device = config.MODEL.DEVICE
        self.ezpc_cfg = config.MODEL.ROI_RELATION_HEAD.EZPC_TEXT

        statistics = get_dataset_statistics(config)
        obj_classes = statistics["obj_classes"]
        rel_classes = statistics["rel_classes"]

        self.obj_names = obj_classes
        self.train_predicate_names = rel_classes
        self.predicate_names = build_full_predicate_names(config)
        self.rel_splits = build_predicate_splits(config)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        self.mva_s2o = MVA()
        self.mva_o2s = MVA()
        self.update_split(config.OV_SETTING.TRAIN_PART)
        self._check_train_predicate_order()

        with torch.no_grad():
            self._encode_object_role_texts()
            concept_features = self._encode_relation_concepts()
            predicate_text_features = self._encode_predicate_texts(config)

        self.concept_projector = EZPCRelationConceptProjector(
            concept_features=concept_features,
            predicate_text_features=predicate_text_features,
            temperature=self.ezpc_cfg.CLASSIFIER_TEMPERATURE,
            match_loss_weight=self.ezpc_cfg.MATCH_LOSS_WEIGHT,
            recon_loss_weight=self.ezpc_cfg.RECON_LOSS_WEIGHT,
            freeze_a=self.ezpc_cfg.FREEZE_A,
        ).to(self.device)

    def _encode_texts(self, texts):
        text_tokens = clip.tokenize(texts).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        return F.normalize(text_features, dim=-1)

    def _encode_relation_concepts(self):
        concepts = load_relation_concepts(
            self.ezpc_cfg.CONCEPT_JSON,
            self.ezpc_cfg.CONCEPTS,
        )
        if not concepts:
            raise ValueError("EZPC_TEXT must provide at least one relation concept.")
        self.relation_concepts = concepts
        return self._encode_texts(["a photo showing " + concept for concept in concepts])

    def _encode_predicate_texts(self, config):
        prompt_json = config.MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.PROMPT_JSON
        prompt_field = config.MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.PROMPT_FIELD
        if prompt_json and os.path.exists(prompt_json):
            relation_texts = load_relation_prompt_texts(prompt_json, self.predicate_names, prompt_field)
        else:
            template = self.ezpc_cfg.PREDICATE_PROMPT
            relation_texts = [template.format(name) for name in self.predicate_names[1:]]
        return self._encode_texts(relation_texts)

    def _encode_object_role_texts(self):
        self.subject_role_text = self._encode_texts(["a photo of subject" for _ in self.obj_names])
        self.object_role_text = self._encode_texts(["a photo of object" for _ in self.obj_names])

    def update_split(self, mode):
        print("now is " + mode)
        if mode not in ("base", "novel", "semantic", "total"):
            raise ValueError("Unsupported OV relation split: {}".format(mode))
        self.mode = mode
        self.rel_ids = torch.as_tensor(self.rel_splits[mode], device=self.device, dtype=torch.long)
        self.fg_rel_ids = self.rel_ids[1:] - 1

    def updata(self, mode):
        self.update_split(mode)

    def _check_train_predicate_order(self):
        rel_ids = self.rel_ids.detach().cpu().tolist()
        active_names = [self.predicate_names[idx] for idx in rel_ids]
        if active_names != self.train_predicate_names:
            raise ValueError(
                "EZPC active predicate order does not match dataset labels. "
                "active={} dataset={}".format(active_names, self.train_predicate_names)
            )

    def _encode_object_crops(self, proposal, image):
        image_tensor = []
        for box in proposal.bbox:
            crop = crop_and_resize(image.unsqueeze(0), box, box)
            crop = crop[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            crop = Image.fromarray(np.uint8(crop))
            crop = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
            image_tensor.append(crop)
        image_tensor = torch.cat(image_tensor)
        return self.clip_model.encode_image(image_tensor)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None):
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)

        rel_dists = []
        concept_logit_list = []
        teacher_logit_list = []
        for i in range(len(num_rels)):
            if num_rels[i] == 0:
                rel_dists.append(roi_features.new_zeros((0, len(self.rel_ids))))
                continue

            with torch.no_grad():
                image_features = self._encode_object_crops(proposals[i], img[i])

            pair_idxs = rel_pair_idxs[i]
            subject_tokens = F.normalize(image_features[pair_idxs[:, 0]], dim=-1)
            object_tokens = F.normalize(image_features[pair_idxs[:, 1]], dim=-1)
            subj_labels = obj_preds[i][pair_idxs[:, 0]]
            obj_labels = obj_preds[i][pair_idxs[:, 1]]

            cross_output1 = self.mva_s2o(
                subject_tokens,
                object_tokens,
                self.subject_role_text[subj_labels],
            )
            cross_output2 = self.mva_o2s(
                object_tokens,
                subject_tokens,
                self.object_role_text[obj_labels],
            )
            pair_features = F.normalize((cross_output1.float() + cross_output2.float()) / 2, dim=-1)

            concept_logits, teacher_logits, _, _ = self.concept_projector.logits(
                pair_features,
                self.fg_rel_ids,
            )
            bg_logits = concept_logits.new_zeros((concept_logits.size(0), 1))
            logits = torch.cat((bg_logits, concept_logits), dim=1)

            rel_dists.append(logits)
            if self.training and rel_labels is not None:
                fg_mask = rel_labels[i].to(concept_logits.device).long() > 0
                if fg_mask.any():
                    concept_logit_list.append(concept_logits[fg_mask])
                    teacher_logit_list.append(teacher_logits[fg_mask])
            else:
                concept_logit_list.append(concept_logits)
                teacher_logit_list.append(teacher_logits)

        obj_dists = obj_dists.split(num_objs, dim=0)
        add_losses = {}
        if self.training:
            concept_logits = torch.cat(concept_logit_list, dim=0) if concept_logit_list else None
            teacher_logits = torch.cat(teacher_logit_list, dim=0) if teacher_logit_list else None
            add_losses.update(self.concept_projector.losses(concept_logits, teacher_logits))
        return obj_dists, tuple(rel_dists), add_losses


@registry.ROI_RELATION_PREDICTOR.register("LowRankClipPredictor")
class LowRankClipPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(LowRankClipPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.device = config.MODEL.DEVICE
        self.low_rank_cfg = config.MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT
        self.debug_step = 0

        statistics = get_dataset_statistics(config)
        obj_classes = statistics["obj_classes"]
        rel_classes = statistics["rel_classes"]

        self.obj_names = obj_classes
        self.train_predicate_names = rel_classes
        self.predicate_names = build_full_predicate_names(config)
        self.rel_splits = build_predicate_splits(config)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        self.mva_s2o = MVA()
        self.mva_o2s = MVA()
        self.update_split(config.OV_SETTING.TRAIN_PART)
        self._check_train_predicate_order()

        with torch.no_grad():
            relation_text_features = self._encode_relation_texts()
            self._encode_object_role_texts()

        self.relation_text_adapter = CoreRelationTextAdapter(
            relation_text_features,
            rank=self.low_rank_cfg.RANK,
            init_method=self.low_rank_cfg.INIT_METHOD,
            train_basis=self.low_rank_cfg.TRAIN_BASIS,
            train_mode=self.low_rank_cfg.TRAIN_MODE,
            logit_temperature=self.low_rank_cfg.CLASSIFIER_TEMPERATURE,
            recon_loss_weight=self.low_rank_cfg.RECON_LOSS_WEIGHT,
            sparsity_weight=self.low_rank_cfg.SPARSITY_WEIGHT,
            basis_decorr_weight=self.low_rank_cfg.BASIS_DECORR_WEIGHT,
            weight_decorr_weight=self.low_rank_cfg.WEIGHT_DECORR_WEIGHT,
            weight_delta_scale=self.low_rank_cfg.WEIGHT_DELTA_SCALE,
            weight_anchor_weight=self.low_rank_cfg.WEIGHT_ANCHOR_WEIGHT,
            w_active_threshold=self.low_rank_cfg.W_ACTIVE_THRESHOLD,
        ).to(self.device)

    def _encode_relation_texts(self):
        relation_texts = load_relation_prompt_texts(
            self.low_rank_cfg.PROMPT_JSON,
            self.predicate_names,
            self.low_rank_cfg.PROMPT_FIELD,
        )
        text_tokens = clip.tokenize(relation_texts).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def _encode_object_role_texts(self):
        text_sub = clip.tokenize(["a photo of subject " for _ in self.obj_names]).to(self.device)
        text_obj = clip.tokenize(["a photo of object " for _ in self.obj_names]).to(self.device)
        self.subject_role_text = self.clip_model.encode_text(text_sub)
        self.object_role_text = self.clip_model.encode_text(text_obj)
        sub_filter = pd.read_csv(curpath + "/filter_total.csv").iloc[:, 1:]
        self.object_relation_text = []
        for obj in self.obj_names:
            text5 = clip.tokenize(["a photo of " + tex for tex in list(sub_filter[obj])]).to(self.device)
            text_features5 = self.clip_model.encode_text(text5)
            self.object_relation_text.append(text_features5.detach().cpu().numpy())

    def update_split(self, mode):
        print("now is " + mode)
        if mode not in ("base", "novel", "semantic", "total"):
            raise ValueError("Unsupported OV relation split: {}".format(mode))
        self.mode = mode
        self.rel_ids = torch.as_tensor(self.rel_splits[mode], device=self.device, dtype=torch.long)
        self.fg_rel_ids = self.rel_ids[1:] - 1

    def updata(self, mode):
        self.update_split(mode)

    def _check_train_predicate_order(self):
        rel_ids = self.rel_ids.detach().cpu().tolist()
        active_names = [self.predicate_names[idx] for idx in rel_ids]
        if active_names != self.train_predicate_names:
            raise ValueError(
                "Low-rank active predicate order does not match dataset labels. "
                "active={} dataset={}".format(active_names, self.train_predicate_names)
            )

    def _encode_object_crops(self, proposal, image):
        image_tensor = []
        for box in proposal.bbox:
            crop = crop_and_resize(image.unsqueeze(0), box, box)
            crop = crop[0].permute(1, 2, 0).detach().cpu().numpy() * 255
            crop = Image.fromarray(np.uint8(crop))
            crop = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
            image_tensor.append(crop)
        image_tensor = torch.cat(image_tensor)
        return self.clip_model.encode_image(image_tensor)

    def _compress_visual_tokens(self, image_features):
        token_limit = int(self.low_rank_cfg.VISUAL_TOKEN_LIMIT)
        if image_features.dim() != 3 or token_limit <= 0 or image_features.size(1) <= token_limit:
            return image_features
        if token_limit == 1:
            return image_features[:, :1]

        patch_budget = token_limit - 1
        patch_tokens = image_features[:, 1:]
        if patch_tokens.size(1) <= patch_budget:
            return image_features

        sample_idx = torch.linspace(
            0,
            patch_tokens.size(1) - 1,
            steps=patch_budget,
            device=image_features.device,
        ).round().long()
        sampled_patches = patch_tokens.index_select(1, sample_idx)
        return torch.cat((image_features[:, :1], sampled_patches), dim=1)

    def _original_clip_similarity_logits(self, image_features, rel_index, subj_label, dtype):
        subj_label = int(subj_label.detach().cpu().item())
        text_features5 = torch.as_tensor(
            self.object_relation_text[subj_label],
            device=self.device,
            dtype=dtype,
        )
        subj_feature = image_features[rel_index[0]][0].unsqueeze(0)
        obj_feature = image_features[rel_index[1]][0].unsqueeze(0)
        text_features5 = F.normalize(text_features5.float(), dim=-1)
        similarity31 = F.normalize(subj_feature.float(), dim=-1) @ text_features5.t() / 0.05
        similarity32 = F.normalize(obj_feature.float(), dim=-1) @ text_features5.t() / 0.05
        similarity = (similarity31 + similarity32) / 2
        rel_ids = self.rel_ids.to(similarity.device)
        return similarity[:, rel_ids].to(dtype)

    def _format_predicate_hist(self, indices, counts, max_items=5):
        items = []
        for idx, count in zip(indices[:max_items].tolist(), counts[:max_items].tolist()):
            global_idx = self.rel_ids[idx].item()
            name = self.predicate_names[global_idx]
            items.append("{}:{}".format(name, int(count)))
        return ",".join(items)

    def _log_low_rank_debug(self, basis_logits, relation_logits, labels, logger):
        interval = int(self.low_rank_cfg.DEBUG_INTERVAL)
        if interval <= 0 or self.debug_step % interval != 0 or not is_main_process():
            return

        with torch.no_grad():
            stats = self.relation_text_adapter.debug_stats()
            weight_stats = self.relation_text_adapter.weight_usage_stats(
                self.fg_rel_ids,
                threshold=self.low_rank_cfg.W_ACTIVE_THRESHOLD,
            )
            basis_std = basis_logits.float().std()
            basis_abs = basis_logits.float().abs().mean()
            logit_std = relation_logits.float().std()
            logit_abs = relation_logits.float().abs().mean()

            parts = ["LowRankDebug step={}".format(self.debug_step)]
            parts.append("lowrank_abs={:.4f}".format(basis_abs.item()))
            parts.append("lowrank_std={:.4f}".format(basis_std.item()))
            parts.append("logit_abs={:.4f}".format(logit_abs.item()))
            parts.append("logit_std={:.4f}".format(logit_std.item()))
            parts.append("W_abs={:.4f}".format(stats["W_abs_mean"].item()))
            parts.append("W_delta={:.4f}".format(stats["W_delta_mean"].item()))
            parts.append("B_abs={:.4f}".format(stats["B_abs_mean"].item()))
            parts.append("W_active={:.2f}".format(weight_stats["W_active"].item()))
            parts.append("W_max_share={:.4f}".format(weight_stats["W_max_share"].item()))
            parts.append("recon_cos={:.4f}".format(stats["recon_cos"].item()))
            if labels is not None and labels.numel() > 0:
                pred_labels = relation_logits[:, 1:].argmax(dim=1) + 1
                valid_gt = labels.long() > 0
                bg_gt = ~valid_gt
                parts.append("fg_count={}".format(int(valid_gt.sum().item())))
                parts.append("bg_count={}".format(int(bg_gt.sum().item())))

                if valid_gt.any():
                    gt_labels = labels.long()[valid_gt]
                    fg_pred_labels = pred_labels[valid_gt]
                    fg_pred_counts = torch.bincount(
                        fg_pred_labels,
                        minlength=relation_logits.size(1),
                    )
                    fg_pred_counts_no_bg = fg_pred_counts[1:]
                    fg_pred_top_counts, fg_pred_top_idx = fg_pred_counts_no_bg.sort(descending=True)
                    fg_pred_top_idx = fg_pred_top_idx + 1
                    fg_pred_dist = fg_pred_counts.float() / fg_pred_counts.sum().clamp_min(1).float()
                    fg_pred_entropy = -(
                        fg_pred_dist[fg_pred_dist > 0] * fg_pred_dist[fg_pred_dist > 0].log()
                    ).sum()
                    parts.append(
                        "fg_pred_unique={}".format(int((fg_pred_counts_no_bg > 0).sum().item()))
                    )
                    parts.append("fg_pred_entropy={:.4f}".format(fg_pred_entropy.item()))
                    parts.append(
                        "fg_top_pred={}".format(
                            self._format_predicate_hist(fg_pred_top_idx, fg_pred_top_counts)
                        )
                    )

                    gt_counts = torch.bincount(gt_labels, minlength=relation_logits.size(1))
                    gt_counts_no_bg = gt_counts[1:]
                    gt_top_counts, gt_top_idx = gt_counts_no_bg.sort(descending=True)
                    gt_top_idx = gt_top_idx + 1
                    parts.append("gt_unique={}".format(int((gt_counts_no_bg > 0).sum().item())))
                    parts.append(
                        "top_gt={}".format(
                            self._format_predicate_hist(gt_top_idx, gt_top_counts)
                        )
                    )
                if bg_gt.any():
                    bg_pred_labels = pred_labels[bg_gt]
                    bg_pred_counts = torch.bincount(
                        bg_pred_labels,
                        minlength=relation_logits.size(1),
                    )
                    bg_pred_counts_no_bg = bg_pred_counts[1:]
                    bg_pred_top_counts, bg_pred_top_idx = bg_pred_counts_no_bg.sort(descending=True)
                    bg_pred_top_idx = bg_pred_top_idx + 1
                    parts.append(
                        "bg_forced_top_pred={}".format(
                            self._format_predicate_hist(bg_pred_top_idx, bg_pred_top_counts)
                        )
                    )

        message = " | ".join(parts)
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, img=None):
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_preds = obj_preds.split(num_objs, dim=0)

        rel_dists = []
        basis_logit_list = []
        class_logit_list = []
        for i in range(len(num_rels)):
            with torch.no_grad():
                image_features = self._encode_object_crops(proposals[i], img[i])
            relation_image_features = self._compress_visual_tokens(image_features)

            pair_idxs = rel_pair_idxs[i]
            subject_tokens = F.normalize(relation_image_features[pair_idxs[:, 0]], dim=-1)
            object_tokens = F.normalize(relation_image_features[pair_idxs[:, 1]], dim=-1)
            subj_labels = obj_preds[i][pair_idxs[:, 0]]
            obj_labels = obj_preds[i][pair_idxs[:, 1]]
            cross_output1 = self.mva_s2o(
                subject_tokens,
                object_tokens,
                self.subject_role_text[subj_labels],
            )
            cross_output2 = self.mva_o2s(
                object_tokens,
                subject_tokens,
                self.object_role_text[obj_labels],
            )
            pair_features = F.normalize((cross_output1 + cross_output2) / 2, dim=-1)
            subject_features = subject_tokens[:, 0]
            object_features = object_tokens[:, 0]
            raw_pair_features = F.normalize((subject_features + object_features) / 2, dim=-1)
            logits, basis_logits = self.relation_text_adapter.logits(
                pair_features,
                self.fg_rel_ids,
                raw_visual_features=raw_pair_features,
            )
            bg_logits = logits.new_zeros((logits.size(0), 1))
            logits = torch.cat((bg_logits, logits), dim=1)
            original_clip_eval_weight = min(max(float(self.low_rank_cfg.ORIGINAL_CLIP_EVAL_WEIGHT), 0.0), 1.0)
            if not self.training and original_clip_eval_weight > 0:
                original_clip_logits = []
                for rel_index in pair_idxs:
                    subj_label = obj_preds[i][rel_index[0]]
                    original_clip_logits.append(
                        self._original_clip_similarity_logits(
                            image_features,
                            rel_index,
                            subj_label,
                            logits.dtype,
                        )
                    )
                original_clip_logits = torch.cat(original_clip_logits, dim=0)
                logits = logits * (1 - original_clip_eval_weight) + original_clip_logits * original_clip_eval_weight

            class_logit_list.append(logits)
            basis_logit_list.append(basis_logits)
            rel_dists.append(logits)

        obj_dists = obj_dists.split(num_objs, dim=0)
        add_losses = {}
        if self.training:
            self.debug_step += 1
            basis_logits = torch.cat(basis_logit_list, dim=0) if basis_logit_list else None
            relation_logits = torch.cat(class_logit_list, dim=0) if class_logit_list else None
            labels = cat(rel_labels, dim=0) if rel_labels is not None else None
            if basis_logits is not None and relation_logits is not None:
                self._log_low_rank_debug(basis_logits, relation_logits, labels, logger)
            add_losses.update(self.relation_text_adapter.losses())
        return obj_dists, tuple(rel_dists), add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
