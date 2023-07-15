import copy
import random
from collections import Counter

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from sklearn import metrics
from sklearn.preprocessing import label_binarize

from algorithms.client.client import Client
from util.result_util import set_fixed_seed


class clientFedNH(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,)
        self.FedNH_fix_scaling = False
        self.FedNH_head_init = 'orthogonal'
        self.dim = 2
        self.model = copy.deepcopy(args.model)
        self.FedNH_client_adv_prototype_agg = False
        self._initialize_model()

        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            self.data_name = kwargs['data_name']
            self.train_loader = kwargs['train_loader']
            self.test_loader = kwargs['test_loader']

        self.count_by_class = torch.zeros(self.num_classes)
        self.num_train_samples = len(self.train_loader.dataset)
        for x, y in self.train_loader:
            for yy in y:
                self.count_by_class[yy.item()] += 1
        self.label_dist = {i: self.count_by_class[i] / self.num_train_samples for i in range(self.num_classes)}
        temp = [self.count_by_class[cls] if self.count_by_class[cls] != 0 else 1e-12 for cls in range(self.num_classes)]
        self.count_by_class_full = torch.tensor(temp).to(self.device)

    def train(self):
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            trainloader = self.train_loader
        else:
            trainloader = self.load_train_data()

        start_time = time.time()

        self.model.train()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                yhat = self.model.forward(x)
                loss = self.loss(yhat, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def _initialize_model(self):
        try:
            self.model.prototype.requires_grad_(False)
            if self.FedNH_head_init == 'orthogonal':
                # method 1:
                # torch.nn.init.orthogonal_ has a bug when first called.
                # self.model.prototype = torch.nn.init.orthogonal_(self.model.prototype)
                # method 2: might be slow
                # m, n = self.model.prototype.shape
                # self.model.prototype.data = self._get_orthonormal_basis(m, n)
                # method 3:
                m, n = self.model.prototype.shape
                self.model.prototype.data = torch.nn.init.orthogonal_(torch.rand(m, n)).to(self.device)
            elif self.FedNH_head_init == 'uniform' and self.dim == 2:
                r = 1.0
                num_cls = self.num_classes
                W = torch.zeros(num_cls, 2)
                for i in range(num_cls):
                    theta = i * 2 * torch.pi / num_cls
                    W[i, :] = torch.tensor([r * math.cos(theta), r * math.sin(theta)])
                self.model.prototype.copy_(W)
            else:
                raise NotImplementedError(
                    f"{self.FedNH_head_init} + {self.num_class}d")
        except AttributeError:
            raise NotImplementedError("Only support linear layers now.")
        if self.FedNH_fix_scaling == True:
            # 30.0 is a common choice in the paper
            self.model.scaling.requires_grad_(False)
            self.model.scaling.data = torch.tensor(30.0).to(self.device)
            print('self.model.scaling.data:', self.model.scaling.data)

    def _estimate_prototype(self):
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            trainloader = self.train_loader
        else:
            trainloader = self.load_train_data()
        self.model.eval()
        self.model.return_embedding = True
        embedding_dim = self.model.prototype.shape[1]
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                feature_embedding, _ = self.model.forward(x)
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    mask = (y == cls)
                    feature_embedding_in_cls = torch.sum(feature_embedding[mask, :], dim=0)
                    prototype[cls] += feature_embedding_in_cls
        for cls in range(self.num_classes):
            # sample mean
            prototype[cls] /= self.count_by_class[cls]
            # normalization so that self.W.data is of the sampe scale as prototype_cls_norm
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

            # reweight it for aggregartion
            prototype[cls] *= self.count_by_class[cls]

        self.model.return_embedding = False

        to_share = {'scaled_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def upload(self):
        if self.FedNH_client_adv_prototype_agg:
            return self.model, self._estimate_prototype_adv()
        else:
            return self.model, self._estimate_prototype()

    def _estimate_prototype_adv(self):
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            trainloader = self.train_loader
        else:
            trainloader = self.load_train_data()
        self.model.eval()
        self.model.return_embedding = True
        embeddings = []
        labels = []
        weights = []
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                # use the latest prototype
                feature_embedding, logits = self.model.forward(x)
                prob_ = F.softmax(logits, dim=1)
                prob = torch.gather(prob_, dim=1, index=y.view(-1, 1))
                labels.append(y)
                weights.append(prob)
                embeddings.append(feature_embedding)
        self.model.return_embedding = False
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)
        for cls in range(self.num_classes):
            mask = (labels == cls)
            weights_in_cls = weights[mask, :]
            feature_embedding_in_cls = embeddings[mask, :]
            prototype[cls] = torch.sum(feature_embedding_in_cls * weights_in_cls, dim=0) / torch.sum(weights_in_cls)
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

        # calculate predictive power
        to_share = {'adv_agg_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share
