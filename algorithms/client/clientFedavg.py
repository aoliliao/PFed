import copy
import random

import torch
import torch.nn as nn
import numpy as np
import time

from sklearn import metrics
from sklearn.preprocessing import label_binarize

from algorithms.client.client import Client
from util.result_util import set_fixed_seed


class clientFedavg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.id = id
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,)

        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            self.data_name = kwargs['data_name']
            self.train_loader = kwargs['train_loader']
            self.test_loader = kwargs['test_loader']
            # print(self.data_name, len(self.train_loader), len(self.test_loader))

    def train(self):
        if self.dataset == "digits" or self.dataset == "office" or self.dataset == "domainnet":
            trainloader = self.train_loader
        else:
            trainloader = self.load_train_data()

        # print(self.data_name, len(self.train_loader), len(self.test_loader))
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # print(y.shape)
                self.optimizer.zero_grad()
                output = self.model(x)
                # print(output.shape)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()


        # print("++++++++++++++++++++++++++++++++++")
        # for name, param in self.model.fc.named_parameters():
        #     # if 'bn' in name:
        #     if param.grad is not None:
        #         print(f"Parameter name: {name}")
        #         print(f"Gradient shape: {param.grad.shape}")
        #         print(f"Gradient values: {param.grad}")
        #         print("++++++++++++++++++++++++++++++++++")
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


