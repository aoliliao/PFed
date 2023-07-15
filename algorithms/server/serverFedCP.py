import time
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from algorithms.client.clientFedCP import clientFedcp
from algorithms.client.clientFedavg import clientFedavg
from algorithms.server.server import Server
from dataset.digits import prepare_data_digits
from dataset.domainnet import prepare_data_domainnet_customer
from dataset.office_caltech_10 import prepare_data_office


class FedCP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        # self.set_slow_clients()
        self.global_modules = copy.deepcopy(args.model)
        in_dim = list(args.model.base.parameters())[-1].shape[0]
        cs = ConditionalSelection(in_dim, in_dim).to(args.device)
        if args.dataset == 'digits' or args.dataset == 'office' or args.dataset == 'domainnet':
            self.set_clients_bn(args, clientObj=clientFedcp, cs=cs)
        else:
            self.set_clients(args, clientObj=clientFedcp)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.args = args
        self.head = None
        self.cs = None


    def set_clients_bn(self, args, clientObj, cs=None):
        if args.dataset == "office":
            self.train_loaders, self.test_loaders, self.concat_test_dataloader = prepare_data_office(args)
            # name of each dataset
            datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
        elif args.dataset == "digits":
            self.train_loaders, self.test_loaders = prepare_data_digits(args)
            datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
        elif args.dataset == "domainnet":
            # train_loaders, test_loaders = prepare_data_domainnet(args)
            # datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
            datasets = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
            self.train_loaders, self.test_loaders, self.concat_test_dataloader = prepare_data_domainnet_customer(args, datasets)
        # federated setting
        client_num = len(datasets)
        client_weights = [1 / client_num for i in range(client_num)]
        for i in range(client_num):
            train_data_loader = self.train_loaders[i]
            test_data_loader = self.test_loaders[i]
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data_loader),
                               test_samples=len(test_data_loader),
                               train_loader=train_data_loader,
                               test_loader=test_data_loader,
                               data_name=datasets[i],
                               ConditionalSelection=cs
                              )
            self.clients.append(client)

    def train(self):
        avg_acc, avg_train_loss, glo_acc, avg_test_loss = [], [], [], []
        avg_train_acc, mean_testaccs, mean_trainaccs = [], [], []

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # self.send_models()

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
                client.train_cs_model()
                client.generate_upload_head()

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()
            self.global_head()
            self.global_cs()
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

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_modules.base)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_modules.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def add_parameters_global(self, w, client_model_global):
        for server_param, client_param in zip(self.global_model.parameters(), client_model_global.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        for key in self.global_modules.base.state_dict().keys():
            # print('key:',key)
            tmp = torch.zeros_like(self.global_modules.base.state_dict()[key]).float()
            for client_idx in range(len(self.uploaded_weights)):
                tmp += self.uploaded_weights[client_idx] * self.uploaded_models[client_idx].state_dict()[key]
            self.global_modules.base.state_dict()[key].data.copy_(tmp)

        for key in self.global_model.state_dict().keys():
            tmp = torch.zeros_like(self.global_model.state_dict()[key]).float()
            for client_idx in range(len(self.uploaded_weights)):
                tmp += self.uploaded_weights[client_idx] * self.uploaded_models_global[client_idx].state_dict()[key]
            self.global_model.state_dict()[key].data.copy_(tmp)


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_models_global = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.model.base)
            self.uploaded_models_global.append(client.model.model)

    def global_head(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.head_g)

        self.head = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.head.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_head(w, client_model)

        for client in self.selected_clients:
            client.set_head_g(self.head)

    def add_head(self, w, head):
        for server_param, client_param in zip(self.head.parameters(), head.parameters()):
            server_param.data += client_param.data.clone() * w

    def global_cs(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.gate.cs)

        self.cs = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.cs.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_cs(w, client_model)

        for client in self.selected_clients:
            client.set_cs(self.cs)

    def add_cs(self, w, cs):
        for server_param, client_param in zip(self.cs.parameters(), cs.parameters()):
            server_param.data += client_param.data.clone() * w


class ConditionalSelection(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(ConditionalSelection, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim * 2),
            nn.LayerNorm([h_dim * 2]),
            nn.ReLU(),
        )

    def forward(self, x, tau=1, hard=False):
        shape = x.shape
        x = self.fc(x)
        x = x.view(shape[0], 2, -1)
        x = F.gumbel_softmax(x, dim=1, tau=tau, hard=hard)
        return x[:, 0, :], x[:, 1, :]

