import random

from threading import Thread
import time
import copy

import torch

from algorithms.client.clientFedRep import clientRep
from algorithms.server.server import Server
from dataset.digits import prepare_data_digits
from dataset.domainnet import prepare_data_domainnet_customer
from dataset.office_caltech_10 import prepare_data_office


class FedRep(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        if args.dataset == 'digits' or args.dataset == 'office' or args.dataset == 'domainnet':
            self.set_clients_bn(args, clientObj=clientRep)
            # elif args.dataset == 'Cifar':
            #     self.set_clients_cifar100(args, clientObj=clientFedavg)
        else:
            self.set_clients(args, clientObj=clientRep)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.args = args


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
                train_loss, avg_test_acc, test_loss, train_acc, mean_testacc, mean_trainacc = self.evaluate()
                avg_train_loss.append(train_loss)
                avg_acc.append(avg_test_acc)
                avg_test_loss.append(test_loss)
                avg_train_acc.append(train_acc)
                mean_testaccs.append(mean_testacc)
                mean_trainaccs.append(mean_trainacc)
                glo_acc.append(avg_test_acc)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        for client in self.selected_clients:
            client.report_process_client(domain=client.data_name, algorithm=self.algorithm, test_acc=client.test_acc,
                                         test_loss=client.test_loss, comment=self.comment)
        self.report_process(avg_acc, avg_train_loss, glo_acc, avg_test_loss, avg_train_acc, mean_testaccs, mean_trainaccs)

        # self.save_results()
        # self.save_global_model()

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model.base)

    def aggregate_parameters(self):
        for key in self.global_model.base.state_dict().keys():
            tmp = torch.zeros_like(self.global_model.base.state_dict()[key]).float()
            for client_idx in range(len(self.uploaded_weights)):
                tmp += self.uploaded_weights[client_idx] * self.uploaded_models[client_idx].state_dict()[key]
            self.global_model.base.state_dict()[key].data.copy_(tmp)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:

            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
