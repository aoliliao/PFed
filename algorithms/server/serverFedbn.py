import copy
from threading import Thread
import time
import  os
import numpy as np

from algorithms.client.clientFedbn import clientFedbn
from algorithms.server.server import Server

from dataset.digits import prepare_data_digits
from dataset.domainnet import prepare_data_domainnet
from dataset.office_caltech_10 import prepare_data_office

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class FedBN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        if args.dataset == 'digits' or args.dataset == 'office' or self.dataset == "domainnet":
            self.set_clients_bn(args, clientObj=clientFedbn)
        else:
            self.set_clients(args, clientObj=clientFedbn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.args = args

        # self.load_model()




    def train(self):
        avg_acc, avg_train_loss, glo_acc, avg_test_loss = [], [], [], []
        avg_train_acc, mean_testaccs, mean_trainaccs = [], [], []
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                test_acc, test_num, auc = self.test_generic_metric(self.num_class, self.device, self.global_model, self.concat_test_dataloader)
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
        for client in self.selected_clients:
            client.report_process_client(domain=client.data_name, algorithm=self.algorithm, test_acc=client.test_acc,
                                         test_loss=client.test_loss, comment=self.comment)
        self.report_process(avg_acc, avg_train_loss, glo_acc, avg_test_loss, avg_train_acc, mean_testaccs, mean_trainaccs)


