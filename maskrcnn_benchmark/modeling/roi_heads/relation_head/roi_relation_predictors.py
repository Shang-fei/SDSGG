# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as FF
from .model_transformer import TransformerContext
from .low_rank_text import (
    build_predicate_splits,
    build_full_predicate_names,
)
from .primitive_low_rank_text import (
    PrimitiveLowRankTextAdapter,
    load_primitive_spec,
)
from maskrcnn_benchmark.data import get_dataset_statistics
from CLIP import clip
import time
from PIL import Image
import pandas as pd
curpath=os.path.dirname(__file__)
CLIP_MVA_CHUNK_SIZE = 8

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

def mva_in_chunks(mva, sub_features, obj_features, text_features, chunk_size):
    outputs = []
    for start in range(0, sub_features.size(0), chunk_size):
        end = start + chunk_size
        outputs.append(mva(
            sub_features[start:end],
            obj_features[start:end],
            text_features[start:end],
        ))
    return torch.cat(outputs, dim=0)

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

    def _texts5_tensor(self, obj_label):
        return torch.as_tensor(self.texts5[int(obj_label)], device=self.device, dtype=torch.float16)

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

    def update_split(self, mode):
        self.updata(mode)

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

            pair_idxs = rel_pair_idxs[i]
            subj_idx = pair_idxs[:, 0]
            obj_idx = pair_idxs[:, 1]
            obj_n1 = obj_preds[i][subj_idx]
            obj_n2 = obj_preds[i][obj_idx]

            text_features1 = self.text_features1
            text_features2 = self.text_features2
            text_features1_norm = F.normalize(text_features1, dim=-1)
            text_features2_norm = F.normalize(text_features2, dim=-1)

            cross_output1 = mva_in_chunks(
                self.adaper_clip1,
                image_features[subj_idx],
                image_features[obj_idx],
                self.text_features3[obj_n1],
                chunk_size=CLIP_MVA_CHUNK_SIZE,
            )
            cross_output2 = mva_in_chunks(
                self.adaper_clip2,
                image_features[obj_idx],
                image_features[subj_idx],
                self.text_features4[obj_n2],
                chunk_size=CLIP_MVA_CHUNK_SIZE,
            )
            cross_output = (cross_output1 + cross_output2) / 2
            cross_output = F.normalize(cross_output, dim=-1)

            similarity1 = cross_output @ text_features1_norm.t()
            similarity2 = cross_output @ text_features2_norm.t()
            similarity_delta = (similarity1 - similarity2) / 0.05

            if self.adaper_clip1.training:
                image_features_clip = (image_features[subj_idx, 0] + image_features[obj_idx, 0]) / 2
                image_features_clip = F.normalize(image_features_clip, dim=-1)
                similarit_origin_1 = image_features_clip @ text_features1_norm.t()
                similarit_origin_2 = image_features_clip @ text_features2_norm.t()
                similarit_origin = (similarit_origin_1 - similarit_origin_2) / 0.05
                probs = torch.stack([similarity_delta, similarit_origin], dim=1)
            else:
                desc = self.description_relation[:, obj_n1].permute(1, 0, 2)
                probs = (desc * similarity_delta.unsqueeze(1)).sum(-1)

                similarity3 = probs.new_zeros(probs.shape)
                for obj_label in torch.unique(obj_n1):
                    mask = obj_n1 == obj_label
                    text_features5 = F.normalize(self._texts5_tensor(obj_label.item()), dim=-1)
                    subj_clip = F.normalize(image_features[subj_idx[mask], 0], dim=-1)
                    obj_clip = F.normalize(image_features[obj_idx[mask], 0], dim=-1)
                    similarity31 = subj_clip @ text_features5.t() / 0.05
                    similarity32 = obj_clip @ text_features5.t() / 0.05
                    similarity3[mask] = (similarity31 + similarity32) / 2

                probs = probs * 0.2 + similarity3 * 0.8
            rel_dist_per_batch.append(probs)

            rel_dist_per_batch=torch.cat(rel_dist_per_batch)

            rel_dists.append(rel_dist_per_batch)



        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = tuple(rel_dists)

        add_losses = {}
        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("PrimitiveLowRankClipPredictor")
class PrimitiveLowRankClipPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(PrimitiveLowRankClipPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.device = config.MODEL.DEVICE
        self.primitive_cfg = config.MODEL.ROI_RELATION_HEAD.PRIMITIVE_TEXT
        self.debug_log_period = int(self.primitive_cfg.DEBUG_LOG_PERIOD)
        self.use_object_filter = bool(self.primitive_cfg.USE_OBJECT_FILTER)
        self.object_filter_weight = float(self.primitive_cfg.OBJECT_FILTER_WEIGHT)
        self.object_filter_temperature = float(self.primitive_cfg.OBJECT_FILTER_TEMPERATURE)

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
        self.object_filter_table = pd.read_csv(os.path.join(curpath, "filter_total.csv"))
        self.update_split(config.OV_SETTING.TRAIN_PART)
        self._check_train_predicate_order()
        self.register_buffer("_debug_forward_count", torch.zeros((), dtype=torch.long))

        primitive_json = self._resolve_primitive_json(self.primitive_cfg.JSON_PATH)
        (
            primitive_texts,
            predicate_texts,
            self.predicate_primitives,
            predicate_primitive_mask,
        ) = load_primitive_spec(primitive_json, self.predicate_names)
        with torch.no_grad():
            self._encode_object_role_texts()
            primitive_features = self._encode_texts(["a visual relation primitive: " + text for text in primitive_texts])
            predicate_features = self._encode_texts(predicate_texts)

        self.primitive_text_adapter = PrimitiveLowRankTextAdapter(
            primitive_features=primitive_features,
            predicate_features=predicate_features,
            predicate_primitive_mask=predicate_primitive_mask,
            train_basis=self.primitive_cfg.TRAIN_BASIS,
            train_weight=self.primitive_cfg.TRAIN_WEIGHT,
            logit_temperature=self.primitive_cfg.CLASSIFIER_TEMPERATURE,
            recon_loss_weight=self.primitive_cfg.RECON_LOSS_WEIGHT,
            cosine_recon_loss_weight=self.primitive_cfg.COSINE_RECON_LOSS_WEIGHT,
            sparsity_weight=self.primitive_cfg.SPARSITY_WEIGHT,
            mask_out_weight=self.primitive_cfg.MASK_OUT_WEIGHT,
            basis_decorr_weight=self.primitive_cfg.BASIS_DECORR_WEIGHT,
            weight_decorr_weight=self.primitive_cfg.WEIGHT_DECORR_WEIGHT,
            basis_anchor_weight=self.primitive_cfg.BASIS_ANCHOR_WEIGHT,
        ).to(self.device)
        print(
            "PrimitiveLowRankClipPredictor: predicates={} primitives={} train_basis={} "
            "train_weight={} temperature={} recon_cos_weight={} mask_out_weight={} "
            "object_filter={} object_filter_weight={} debug_log_period={}".format(
                len(predicate_texts),
                len(primitive_texts),
                self.primitive_cfg.TRAIN_BASIS,
                self.primitive_cfg.TRAIN_WEIGHT,
                self.primitive_cfg.CLASSIFIER_TEMPERATURE,
                self.primitive_cfg.COSINE_RECON_LOSS_WEIGHT,
                self.primitive_cfg.MASK_OUT_WEIGHT,
                self.use_object_filter,
                self.object_filter_weight,
                self.debug_log_period,
            )
        )

    def _resolve_primitive_json(self, json_path):
        if os.path.isabs(json_path):
            return json_path
        return os.path.join(curpath, json_path)

    def _encode_texts(self, texts):
        text_tokens = clip.tokenize(texts).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        return F.normalize(text_features, dim=-1)

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
        self._encode_object_filter_texts()

    def updata(self, mode):
        self.update_split(mode)

    def _encode_object_filter_texts(self):
        if not self.use_object_filter:
            self.object_filter_texts = None
            return
        active_rows = self.rel_ids.detach().cpu().tolist()[1:]
        filter_rows = self.object_filter_table.iloc[active_rows]
        object_filter_texts = []
        with torch.no_grad():
            for obj_name in self.obj_names:
                texts = ["a photo of " + str(text) for text in list(filter_rows[obj_name])]
                object_filter_texts.append(self._encode_texts(texts).detach())
        self.object_filter_texts = object_filter_texts

    def _check_train_predicate_order(self):
        rel_ids = self.rel_ids.detach().cpu().tolist()
        active_names = [self.predicate_names[idx] for idx in rel_ids]
        if active_names != self.train_predicate_names:
            raise ValueError(
                "Primitive active predicate order does not match dataset labels. "
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

    def _object_filter_logits(self, image_features, pair_idxs, subj_labels, obj_labels):
        if not self.use_object_filter or self.object_filter_weight <= 0:
            return None
        subject_features = image_features[pair_idxs[:, 0]].float()
        object_features = image_features[pair_idxs[:, 1]].float()
        if subject_features.dim() == 3:
            subject_features = subject_features[:, 0, :]
            object_features = object_features[:, 0, :]
        subject_features = F.normalize(subject_features, dim=-1)
        object_features = F.normalize(object_features, dim=-1)
        prior_logits = []
        for pair_idx in range(pair_idxs.size(0)):
            text_features = self.object_filter_texts[int(subj_labels[pair_idx].item())]
            text_features = text_features.to(subject_features.device, dtype=subject_features.dtype)
            subject_score = subject_features[pair_idx:pair_idx + 1] @ text_features.t()
            object_score = object_features[pair_idx:pair_idx + 1] @ text_features.t()
            prior_logits.append((subject_score + object_score) * 0.5)
        prior_logits = torch.cat(prior_logits, dim=0)
        return prior_logits / max(self.object_filter_temperature, 1e-6)

    def _log_debug_stats(self, rel_dists, primitive_logits, pair_features, logger):
        if not self.training or self.debug_log_period <= 0:
            return
        self._debug_forward_count.add_(1)
        if int(self._debug_forward_count.item()) % self.debug_log_period != 0:
            return

        valid_rel_dists = [x for x in rel_dists if x.numel() > 0]
        if not valid_rel_dists or not primitive_logits or not pair_features:
            return
        fg_logits = torch.cat([x[:, 1:].detach().float() for x in valid_rel_dists], dim=0)
        if fg_logits.numel() == 0:
            return
        primitive_logits = torch.cat(primitive_logits, dim=0).detach().float()
        pair_features = torch.cat(pair_features, dim=0).detach().float()
        stats = self.primitive_text_adapter.debug_stats()
        message = (
            "PrimitiveLowRank debug step={step} split={split} pairs={pairs} "
            "pred_logit_mean={pred_mean:.4f} pred_logit_std={pred_std:.4f} "
            "pred_logit_min={pred_min:.4f} pred_logit_max={pred_max:.4f} "
            "primitive_logit_mean={prim_mean:.4f} primitive_logit_std={prim_std:.4f} "
            "primitive_logit_min={prim_min:.4f} primitive_logit_max={prim_max:.4f} "
            "pair_norm={pair_norm:.4f} w_abs_mean={w_abs_mean:.4f} "
            "w_abs_max={w_abs_max:.4f} w_nonzero_005={w_nonzero_005:.4f} "
            "w_mask_in_abs={w_mask_in_abs:.4f} w_mask_out_abs={w_mask_out_abs:.4f} "
            "basis_norm_mean={basis_norm_mean:.4f} basis_rel_shift_mean={basis_rel_shift_mean:.4f} "
            "basis_rel_shift_max={basis_rel_shift_max:.4f} basis_corr_offdiag={basis_corr_offdiag:.4f} "
            "recon_cos_mean={recon_cos_mean:.4f} recon_cos_min={recon_cos_min:.4f} "
            "recon_cos_neg_frac={recon_cos_neg_frac:.4f}"
        ).format(
            step=int(self._debug_forward_count.item()),
            split=self.mode,
            pairs=fg_logits.size(0),
            pred_mean=fg_logits.mean().item(),
            pred_std=fg_logits.std(unbiased=False).item(),
            pred_min=fg_logits.min().item(),
            pred_max=fg_logits.max().item(),
            prim_mean=primitive_logits.mean().item(),
            prim_std=primitive_logits.std(unbiased=False).item(),
            prim_min=primitive_logits.min().item(),
            prim_max=primitive_logits.max().item(),
            pair_norm=pair_features.norm(dim=-1).mean().item(),
            **stats
        )
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
        primitive_logits_for_debug = []
        pair_features_for_debug = []
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
            logits, primitive_logits = self.primitive_text_adapter.logits(
                pair_features,
                self.fg_rel_ids,
            )
            object_filter_logits = self._object_filter_logits(
                image_features,
                pair_idxs,
                subj_labels,
                obj_labels,
            )
            if object_filter_logits is not None:
                filter_weight = min(max(self.object_filter_weight, 0.0), 1.0)
                logits = logits * (1.0 - filter_weight) + object_filter_logits * filter_weight
            if self.training and self.debug_log_period > 0:
                primitive_logits_for_debug.append(primitive_logits)
                pair_features_for_debug.append(pair_features)
            bg_logits = logits.new_zeros((logits.size(0), 1))
            rel_dists.append(torch.cat((bg_logits, logits), dim=1))

        obj_dists = obj_dists.split(num_objs, dim=0)
        self._log_debug_stats(rel_dists, primitive_logits_for_debug, pair_features_for_debug, logger)
        add_losses = self.primitive_text_adapter.losses() if self.training else {}
        return obj_dists, tuple(rel_dists), add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
