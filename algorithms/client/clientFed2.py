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
from models.LeNet import gate_DNN
from util.loss_util import CorrelationAlignmentLoss, MMD_loss


class clientFedours2(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        self.loss = nn.CrossEntropyLoss()

        self.p_fea = copy.deepcopy(self.model.base)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,)
        self.opt_pfea = torch.optim.SGD(self.p_fea.parameters(), lr=self.learning_rate,)

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
        max_local_steps = self.local_steps
        self.trainepo += 1
        trade_off = 1.0
        # if self.trainepo <= 50:
        #     trade_off = 0.0
        # elif self.trainepo > 50 and self.trainepo <= 100:
        #     trade_off = 1.0
        # elif self.trainepo > 100 and self.trainepo <= 150:
        #     trade_off = 5.0
        # else:
        #     trade_off = 10.0
        for step in range(max_local_steps):


            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.p_fea.parameters():
                param.requires_grad = False

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
                fea = g_fea
                # fea = torch.cat([g_fea, p_fea], dim=1)
                self.optimizer.zero_grad()
                output = self.model.head(fea)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

            # test_acc, test_num, auc, loss = self.test_metrics()
            # print('name:', self.data_name, 'after_g_acc:', test_acc/test_num)
            # self.model.train()
            # self.p_fea.train()

            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.p_fea.parameters():
                param.requires_grad = True

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

                # print('cor_loss:', cor_loss)
                # fea = 0.5 * g_fea + 0.5 * p_fea
                fea = p_fea
                self.opt_pfea.zero_grad()
                output = self.model.head(fea)
                loss = self.loss(output, y)
                loss.backward()
                self.opt_pfea.step()

            # test_acc, test_num, auc, loss = self.test_metrics()
            # print('name:', self.data_name, 'acc:', test_acc / test_num)
            #
            # self.model.train()
            # self.p_fea.train()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            train_loader = self.train_loader
        else:
            train_loader = self.load_train_data()
        self.model.eval()
        self.p_fea.eval()
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
            fea = 0.5 * g_fea + 0.5 * p_fea
            # fea = g_fea + p_fea
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
                fea = 0.5 * g_fea + 0.5 * p_fea
                # fea = g_fea + p_fea
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
        # print(self.id, "_testAcc:", test_acc * 1.0 / test_num)
        self.test_acc.append(test_acc / test_num)
        self.test_loss.append(loss / test_num)
        return test_acc, test_num, auc, loss

    # def set_parameters(self, model):
    #     for key in model.state_dict().keys():
    #         if 'bn' not in key:
    #             self.model.state_dict()[key].data.copy_(model.state_dict()[key])

    def dis_loss(self, p, q):
        # 计算KL散度
        kl = F.kl_div(p.softmax(dim=-1).log(), q.softmax(dim=-1), reduction='sum')
        # print('kl:', kl)
        # 定义一个e为底的幂函数
        def exp(x):
            # return x
            return math.e ** x
        # 将KL散度作为e为底的幂函数的指数
        loss = exp(-1 * kl)
        return loss


