import copy

import math
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F

from sklearn.preprocessing import label_binarize
from sklearn import metrics

from algorithms.client.client import Client


class clientFedours16(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        self.loss = nn.CrossEntropyLoss()

        self.p_fea = copy.deepcopy(self.model.base)
        self.gate = DNN_gate(self.model.head.in_features * 2, self.model.head.in_features, device=args.device)
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters()},
                                          {'params': self.gate.parameters()},
                                          ], lr=self.learning_rate)
        self.opt_pfea = torch.optim.SGD(self.p_fea.parameters(), lr=self.learning_rate)
        # self.opt_head = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate, )
        self.opt_gate = torch.optim.SGD(self.gate.parameters(), lr=self.learning_rate)

        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            self.data_name = kwargs['data_name']
            self.train_loader = kwargs['train_loader']
            self.test_loader = kwargs['test_loader']
        self.g_proj = kwargs['g_proj']
        self.p_proj = kwargs['p_proj']

        self.trainepo = 0

    def train(self):
        for param in self.p_fea.parameters():
            param.grad = None

        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            trainloader = self.train_loader
        else:
            trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        self.p_fea.train()
        self.gate.train()
        max_local_steps = self.local_steps
        self.trainepo += 1
        trade_off = 1.0
        for step in range(max_local_steps):

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                g_proj = torch.Tensor(self.g_proj).to(self.device)
                p_proj = torch.Tensor(self.p_proj).to(self.device)
                g_fea = self.model.base(x)
                p_fea = self.p_fea(x)
                g_fea = torch.mm(g_fea, g_proj)
                p_fea = torch.mm(p_fea, p_proj)
                fea = p_fea
                self.opt_pfea.zero_grad()
                output = self.model.head(fea)
                loss = self.loss(output, y)
                loss.backward()
                self.opt_pfea.step()

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                g_proj = torch.Tensor(self.g_proj).to(self.device)
                p_proj = torch.Tensor(self.p_proj).to(self.device)
                g_fea = self.model.base(x)
                p_fea = self.p_fea(x)

                g_fea = torch.mm(g_fea, g_proj)
                p_fea = torch.mm(p_fea, p_proj)

                gate_fea = torch.cat([g_fea, p_fea], dim=1)
                # gate_fea = g_fea + p_fea
                gate_out = self.gate(gate_fea)
                fea = gate_out * g_fea + (1 - gate_out) * p_fea
                g_out = self.model.head(g_fea)
                p_out = self.model.head(p_fea)
                # fea = torch.cat([g_fea, p_fea], dim=1)
                self.optimizer.zero_grad()
                output = self.model.head(fea)
                loss = self.loss(g_out, y) + self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            train_loader = self.train_loader
        else:
            train_loader = self.load_train_data()
        self.model.eval()
        self.p_fea.eval()
        self.gate.eval()
        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in train_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            g_proj = torch.Tensor(self.g_proj).to(self.device)
            p_proj = torch.Tensor(self.p_proj).to(self.device)

            g_fea = self.model.base(x)
            p_fea = self.p_fea(x)
            g_fea = torch.mm(g_fea, g_proj)
            p_fea = torch.mm(p_fea, p_proj)
            gate_fea = torch.cat([g_fea, p_fea], dim=1)
            gate_out = self.gate(gate_fea)
            fea = gate_out * g_fea + (1 - gate_out) * p_fea
            output = self.model.head(fea)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]

        return loss, train_num, train_acc

    def test_metrics(self):
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            test_loader_full = self.test_loader
        else:
            test_loader_full = self.load_test_data()
        self.model.eval()
        self.p_fea.eval()
        self.gate.eval()
        test_acc = 0
        test_num = 0
        loss = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in test_loader_full:
                x = x.to(self.device)
                y = y.to(self.device)
                g_proj = torch.Tensor(self.g_proj).to(self.device)
                p_proj = torch.Tensor(self.p_proj).to(self.device)

                g_fea = self.model.base(x)
                p_fea = self.p_fea(x)
                g_fea = torch.mm(g_fea, g_proj)
                p_fea = torch.mm(p_fea, p_proj)
                gate_fea = torch.cat([g_fea, p_fea], dim=1)
                gate_out = self.gate(gate_fea)
                fea = gate_out * g_fea + (1 - gate_out) * p_fea

                if self.trainepo % 20 == 0:
                    print(self.data_name, "--", gate_out.mean(1))
                output = self.model.head(fea)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                loss += self.loss(output, y).item() * y.shape[0]
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        self.test_acc.append(test_acc / test_num)
        self.test_loss.append(loss / test_num)
        return test_acc, test_num, auc, loss

class DNN_gate(nn.Module):
    def __init__(self, in_features=256 * 6 * 6, out_features=2, device='cpu'):
        super(DNN_gate, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(in_features, out_features),
                # nn.LayerNorm(out_features),
                # nn.BatchNorm1d(out_features),
                # nn.ReLU(inplace=True),
        )

        self.to(device)
    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x
