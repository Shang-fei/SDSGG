import torch
from torch import nn
from torch.nn import functional as F

from CLIP import clip


class TripletTextTeacher(nn.Module):
    """Frozen CLIP triplet teacher with one shared global SVD refinement."""

    def __init__(self, clip_model, object_names, relation_names, text_filter, config, device):
        super().__init__()
        object.__setattr__(self, "clip_model", clip_model)
        self.object_names = object_names
        self.relation_names = relation_names
        self.text_filter = text_filter
        self.device = device
        self.svd_enabled = config.SVD_ENABLED
        self.svd_components = int(config.NUM_SVD_COMPONENTS)
        self.encode_batch_size = max(int(config.ENCODE_BATCH_SIZE), 1)
        self.candidate_cache = {}
        self.filtered_cache = {}
        self.register_buffer("principal_components", None)
        self.update_svd_basis()

    @staticmethod
    def format_triplet(subject, predicate, object_name):
        return "a photo of a {} {} a {}".format(subject, predicate, object_name)

    def encode_raw(self, texts):
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(texts), self.encode_batch_size):
                tokens = clip.tokenize(texts[offset : offset + self.encode_batch_size]).to(self.device)
                chunks.append(self.clip_model.encode_text(tokens).float())
        return F.normalize(torch.cat(chunks, dim=0), dim=-1)

    def encode(self, texts):
        features = self.encode_raw(texts)
        if self.principal_components is None:
            return features
        components = self.principal_components.to(features)
        projection = (features @ components) @ components.t()
        return F.normalize(features - projection, dim=-1)

    def _svd_text_batches(self):
        batch = []
        object_names = self.object_names[1:] if len(self.object_names) > 1 else self.object_names
        for subject in object_names:
            predicates = (
                list(self.text_filter[subject])
                if subject in self.text_filter.columns
                else self.relation_names
            )
            for predicate in predicates:
                if predicate == "__background__":
                    continue
                for object_name in object_names:
                    batch.append(self.format_triplet(subject, predicate, object_name))
                    if len(batch) == self.encode_batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch

    def update_svd_basis(self):
        if not self.svd_enabled or self.svd_components <= 0:
            self.principal_components = None
            return
        covariance = None
        feature_count = 0
        with torch.no_grad():
            for texts in self._svd_text_batches():
                features = self.encode_raw(texts).float()
                if covariance is None:
                    covariance = features.new_zeros(features.size(1), features.size(1))
                covariance.add_(features.t() @ features)
                feature_count += features.size(0)
        if covariance is None or feature_count < 2:
            self.principal_components = None
            return
        _, eigenvectors = torch.linalg.eigh(covariance / float(feature_count))
        component_count = min(self.svd_components, eigenvectors.size(1))
        self.principal_components = F.normalize(
            eigenvectors[:, -component_count:], dim=0
        ).detach()

    def build_targets(self, subject_labels, relation_labels, object_labels, active_relations):
        texts, predicates = [], []
        for subject, predicate, object_label in zip(
            subject_labels.detach().cpu().tolist(),
            relation_labels.detach().cpu().tolist(),
            object_labels.detach().cpu().tolist(),
        ):
            predicate_name = active_relations[int(predicate)]
            predicates.append(predicate_name)
            texts.append(
                self.format_triplet(
                    self.object_names[int(subject)],
                    predicate_name,
                    self.object_names[int(object_label)],
                )
            )
        return texts, predicates, self.encode(texts).detach()

    def all_predicates(self, subject_label, object_label):
        key = (int(subject_label), int(object_label))
        if key not in self.candidate_cache:
            subject = self.object_names[key[0]]
            object_name = self.object_names[key[1]]
            texts = [
                self.format_triplet(subject, predicate, object_name)
                for predicate in self.relation_names
            ]
            self.candidate_cache[key] = self.encode(texts).detach().cpu()
        return self.candidate_cache[key].to(self.device)

    def alignment_candidates(self, subject_labels, object_labels, predicates, device):
        candidates = torch.stack(
            [
                self.all_predicates(subject, object_label)
                for subject, object_label in zip(
                    subject_labels.detach().cpu().tolist(),
                    object_labels.detach().cpu().tolist(),
                )
            ]
        ).to(device=device, dtype=torch.float32)
        predicate_to_index = {
            name: index for index, name in enumerate(self.relation_names)
        }
        targets = torch.tensor(
            [predicate_to_index[name] for name in predicates],
            device=device,
            dtype=torch.long,
        )
        return candidates, targets

    def inference_candidates(
        self, subject_labels, object_labels, active_relations, inference_filter
    ):
        features = []
        for subject, object_label in zip(
            subject_labels.detach().cpu().tolist(),
            object_labels.detach().cpu().tolist(),
        ):
            subject_name = self.object_names[int(subject)]
            candidate_relations = (
                tuple(inference_filter[subject_name])
                if subject_name in inference_filter.columns
                else tuple(active_relations)
            )
            key = (int(subject), int(object_label), candidate_relations)
            if key not in self.filtered_cache:
                object_name = self.object_names[key[1]]
                texts = [
                    self.format_triplet(subject_name, predicate, object_name)
                    for predicate in candidate_relations
                ]
                self.filtered_cache[key] = self.encode(texts).detach().cpu()
            features.append(self.filtered_cache[key])
        return torch.stack(features).to(self.device)
