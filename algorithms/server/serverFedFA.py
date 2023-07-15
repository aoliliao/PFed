import time

from threading import Thread

import torch

from algorithms.client.clientFedFA import clientFedFA
from algorithms.client.clientFedavg import clientFedavg
from algorithms.server.server import Server
# from dataset.cifar100 import prepare_data_cifar100
from dataset.digits import prepare_data_digits
from dataset.domainnet import prepare_data_domainnet, prepare_data_domainnet_customer
from dataset.office_caltech_10 import prepare_data_office


class FedFA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        # self.set_slow_clients()
        if args.dataset == 'digits' or args.dataset == 'office' or args.dataset == 'domainnet':
            self.set_clients_bn(args, clientObj=clientFedFA)
        else:
            self.set_clients(args, clientObj=clientFedFA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.args = args


    def train(self):
        avg_acc, avg_train_loss, glo_acc, avg_test_loss = [], [], [], []
        avg_train_acc, mean_testaccs, mean_trainaccs = [], [], []

        for i in range(self.global_rounds+1):
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

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        # for client in self.selected_clients:
        #     client.report_process_client(domain=client.data_name, algorithm=self.algorithm, test_acc=client.test_acc,
        #                                  test_loss=client.test_loss, comment=self.comment)
        self.report_process(avg_acc, avg_train_loss, glo_acc, avg_test_loss, avg_train_acc, mean_testaccs, mean_trainaccs)

    def aggregate_parameters(self):
        for key in self.global_model.state_dict().keys():
            tmp = torch.zeros_like(self.global_model.state_dict()[key]).float()
            for client_idx in range(len(self.uploaded_weights)):
                tmp += self.uploaded_weights[client_idx] * self.uploaded_models[client_idx].state_dict()[key]
            self.global_model.state_dict()[key].data.copy_(tmp)

            if 'running_var_mean_bmic' in key or 'running_var_std_bmic' in key:
                tmp = []
                for client_idx in range(len(self.uploaded_weights)):
                    tmp.append(self.uploaded_models[client_idx].state_dict()[key.replace('running_var_', 'running_')])

                tmp = torch.stack(tmp)
                var = torch.var(tmp)
                self.global_model.state_dict()[key].data.copy_(var)
