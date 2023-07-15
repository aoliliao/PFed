import time

from threading import Thread

import numpy as np
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from algorithms.client.clientFed2 import clientFedours2
from algorithms.client.clientFedavg import clientFedavg
from algorithms.server.server import Server
from dataset.digits import prepare_data_digits
from dataset.domainnet import prepare_data_domainnet, prepare_data_domainnet_customer
from dataset.office_caltech_10 import prepare_data_office
from util.data_util import read_client_data
from util.utils import generate_projection_matrix


class FedOurs2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        QR = False
        self.proj_matrices = generate_projection_matrix(args.num_clients + 1, feature_dim=int(args.model.head.in_features),
                                                        qr=QR)
        if args.dataset == 'digits' or args.dataset == 'office' or args.dataset == 'domainnet':
            self.set_clients_bn(args, clientObj=clientFedours2)
        else:
            self.set_clients(args, clientObj=clientFedours2)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.args = args

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               g_proj=self.proj_matrices[0],
                               p_proj=self.proj_matrices[i+1],
                              )
            self.clients.append(client)

    def set_clients_bn(self, args, clientObj):
        if args.dataset == "office":
            self.train_loaders, self.test_loaders, self.concat_test_dataloader = prepare_data_office(args)
            datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
        elif args.dataset == "digits":
            train_loaders, test_loaders = prepare_data_digits(args)
            datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
        elif args.dataset == "domainnet":
            datasets = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
            # datasets = ['clipart', 'infograph', 'painting']
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
                               g_proj=self.proj_matrices[0],
                               p_proj=self.proj_matrices[i + 1],
                               )
            self.clients.append(client)

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
                test_acc, test_num, auc = self.test_generic_metric(self.num_class, self.device, self.global_model,
                                                                   test_data=self.concat_test_dataloader)
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
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        # for client in self.selected_clients:
        #     client.report_process_client(domain=client.data_name, algorithm=self.algorithm, test_acc=client.test_acc,
        #                                  test_loss=client.test_loss, comment=self.comment)
        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.report_process(avg_acc, avg_train_loss, glo_acc, avg_test_loss, avg_train_acc, mean_testaccs, mean_trainaccs)

        # self.save_results()
        # self.save_global_model()

    # def test_generic_metric(self, num_classes, device, model, test_data=None, proj=None):
    #     if test_data == None:
    #         test_loader_global = self.load_global_test_data(dataset=self.dataset)
    #     else:
    #         test_loader_global = test_data
    #     model.eval()
    #
    #     test_acc = 0
    #     test_num = 0
    #     loss = 0
    #     y_prob = []
    #     y_true = []
    #     with torch.no_grad():
    #         for x, y in test_loader_global:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(device)
    #             else:
    #                 x = x.to(device)
    #             y = y.to(device)
    #
    #             g_proj = torch.Tensor(proj).to(self.device)
    #             g_fea = model.base(x)
    #             g_fea = torch.mm(g_fea, g_proj)
    #             output = model.head(g_fea)
    #
    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #             test_num += y.shape[0]
    #
    #             y_prob.append(output.detach().cpu().numpy())
    #             nc = num_classes
    #             if num_classes == 2:
    #                 nc += 1
    #             lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #             if num_classes == 2:
    #                 lb = lb[:, :2]
    #             y_true.append(lb)
    #
    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = np.concatenate(y_true, axis=0)
    #
    #     auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    #
    #     return test_acc, test_num, auc




