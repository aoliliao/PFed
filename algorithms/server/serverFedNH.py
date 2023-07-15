import copy
import time
import random

import torch
import torch.nn as nn
import numpy as np
from threading import Thread

from algorithms.client.clientFedNH import clientFedNH
from algorithms.client.clientFedavg import clientFedavg
from algorithms.server.server import Server
import torch.nn.functional as F


class FedNH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.FedNH_server_adv_prototype_agg = False
        if args.dataset == 'digits' or args.dataset == 'office' or args.dataset == 'domainnet':
            self.set_clients_bn(args, clientObj=clientFedNH)
        else:
            self.set_clients(args, clientObj=clientFedNH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.args = args
        self.train_sets = None,
        self.test_sets = None

    def train(self):
        avg_acc, avg_train_loss, glo_acc, avg_test_loss = [], [], [], []
        avg_train_acc, mean_testaccs, mean_trainaccs = [], [], []

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                test_acc, test_num, auc = self.test_generic_metric(self.num_class, self.device, self.global_model, test_data=self.concat_test_dataloader)
                print("Global Test Accurancy: {:.4f}".format(test_acc / test_num))
                print("Global Test AUC: {:.4f}".format(auc))
                glo_acc.append(test_acc / test_num)

                train_loss, avg_test_acc, test_loss, train_acc, mean_testacc, mean_trainacc = self.evaluate()
                avg_train_loss.append(train_loss)
                avg_acc.append(avg_test_acc)
                avg_test_loss.append(test_loss)
                avg_train_acc.append(train_acc)
                mean_testaccs.append(mean_testacc)
                mean_trainaccs.append(mean_trainacc)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        for client in self.selected_clients:
            client.report_process_client(domain=client.data_name, algorithm=self.algorithm, test_acc=client.test_acc,
                                         test_loss=client.test_loss, comment=self.comment)
        self.report_process(avg_acc, avg_train_loss, glo_acc, avg_test_loss, avg_train_acc, mean_testaccs, mean_trainaccs)

        # self.save_results()
        # self.save_global_model()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.join_clients))
        # active_clients = self.selected_clients
        # print(active_clients)
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                               client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']

            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.upload())
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        server_learning_rate = 1.0
        server_lr_decay_per_round = 1.0
        round = 1
        server_lr = server_learning_rate * (server_lr_decay_per_round ** (round - 1))
        num_participants = len(self.uploaded_models)
        update_direction_state_dict = None
        # agg weights for prototype
        cumsum_per_class = torch.zeros(self.num_class).to(self.device)
        agg_weights_vec_dict = {}
        with torch.no_grad():
            for idx, (client_state_dict, prototype_dict) in enumerate(self.uploaded_models):
                if self.FedNH_server_adv_prototype_agg == False:
                    cumsum_per_class += prototype_dict['count_by_class_full']
                else:
                    mu = prototype_dict['adv_agg_prototype']
                    W = self.global_model.prototype
                    agg_weights_vec_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))

            for key in self.global_model.state_dict().keys():
                # print('key:', key)
                tmp = torch.zeros_like(self.global_model.state_dict()[key]).float()
                for client_idx in range(len(self.uploaded_weights)):
                    tmp += self.uploaded_weights[client_idx] * self.uploaded_models[client_idx][0].state_dict()[key]
                self.global_model.state_dict()[key].data.copy_(tmp)

            avg_prototype = torch.zeros_like(self.global_model.prototype)
            for _, prototype_dict in self.uploaded_models:
                avg_prototype += prototype_dict['scaled_prototype'] / cumsum_per_class.view(-1, 1)

            # normalize prototype
            avg_prototype = F.normalize(avg_prototype, dim=1)
            # update prototype with moving average
            FedNH_smoothing = 0.9
            weight = FedNH_smoothing
            temp = weight * self.global_model.prototype + (1 - weight) * avg_prototype
            # print('agg weight:', weight)
            # normalize prototype again
            self.global_model.prototype.copy_(F.normalize(temp, dim=1))
