import copy
import random

import parser
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from torch import nn

from algorithms.server.serverFed13 import FedOurs13
from algorithms.server.serverFed14 import FedOurs14
from algorithms.server.serverFed15 import FedOurs15
from algorithms.server.serverFed16 import FedOurs16
from algorithms.server.serverFedCP import FedCP
from algorithms.server.serverFedNH import FedNH
from algorithms.server.serverFedRod import FedRod
from models.Fed15Model import AlexNetFed15
from util.result_util import set_fixed_seed

from algorithms.client.clientFed7 import DNN_gate
from algorithms.server.serverFed import FedOurs
# from algorithms.server.serverFed1 import FedOursNet
from algorithms.server.serverFed1 import FedOursNet
from algorithms.server.serverFed10 import FedOursNet10
from algorithms.server.serverFed11 import FedOursNet11
from algorithms.server.serverFed12 import FedOurs12
from algorithms.server.serverFed2 import FedOurs2
from algorithms.server.serverFed3 import FedOurs3
from algorithms.server.serverFed4 import FedOursNet4
from algorithms.server.serverFed5 import FedOurs5
from algorithms.server.serverFed6 import FedOursNet6
from algorithms.server.serverFed7 import FedOursNet7
from algorithms.server.serverFed8 import FedOursNet8
from algorithms.server.serverFed9 import FedOurs9
from algorithms.server.serverFedBABU import FedBABU
from algorithms.server.serverFedFA import FedFA
from algorithms.server.serverFedKL import FedOursKL
from algorithms.server.serverFedKL1 import FedOursKL1
from algorithms.server.serverFedKL2 import FedOursKL2
from algorithms.server.serverFedNoPer import FedOursNo
from algorithms.server.serverFedRe import FedOursRe
from algorithms.server.serverFedRep import FedRep
from algorithms.server.serverFedavg import FedAvg
from algorithms.server.serverFedbn import FedBN
# from algorithms.server.serverFedrod import FedROD
from algorithms.server.serverLocal import FedLocal
from algorithms.server.serverMMD import FedOursMMD
from models.AlexNetFA import AlexNetFedFa
from models.LeNet import LeNet, DigitModel, AlexNet, AlexNet1, gate_DNN, AlexNetNH
from models.ResNet import resnet18
from models.model import LocalModel, ClientOurModel, ClientGateModel
from options import args_parser


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


set_fixed_seed()
# torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
hidden_dim = 32


def run(args):

    time_list = []

    # reporter = MemReporter()
    model_str = args.model
    rate = 1
    if args.algorithm == "Fed":
        rate = 1
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "cnn":
            if args.dataset[:5] == "mnist" or args.dataset == "fmnist":
                args.model = LeNet(rate=rate).to(args.device)
            elif args.dataset == "digits":
                args.model = DigitModel().to(args.device)
            elif args.dataset[:5] == "Cifar":
                args.model = DigitModel(num_classes=args.num_classes, dim=8192, rate=rate).to(args.device)
            elif args.dataset == "office":
                args.model = AlexNet().to(args.device)
            elif args.dataset == "domainnet":
                args.model = AlexNet().to(args.device)
        elif model_str == "resnet":
            args.model = resnet18(num_classes=10).to(args.device)
        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        # elif args.algorithm == "FedROD":
        #     head = copy.deepcopy(args.model.fc)
        #     args.model.fc = nn.Identity()
        #     args.model = LocalModel(args.model, head)
        #     server = FedROD(args, i)
        elif args.algorithm == "FedBN":
            server = FedBN(args, i)
        elif args.algorithm == "FedFA":
            args.model = AlexNetFedFa().to(args.device)
            print(args.model)
            server = FedFA(args, i)
        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedRep(args, i)
        elif args.algorithm == "FedNH":
            args.model = AlexNetNH().to(args.device)
            server = FedNH(args, i)
        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedCP(args, i)
        elif args.algorithm == "Fed":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOurs(args, i)
        elif args.algorithm == "Fed1_1":
            args.model = AlexNet1().to(args.device)
            model_base = copy.deepcopy(args.model.features)
            args.head = copy.deepcopy(args.model.classifier)
            args.model.features = nn.Identity()
            args.model.classifier = nn.Identity()
            args.model = ClientOurModel(model_base, args.model, args.head)
            print(args.model)
            server = FedOurs5(args, i)
        elif args.algorithm == "FedBABU":
            # args.model = AlexNet1().to(args.device)
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            print(args.model)
            server = FedBABU(args, i)
        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedRod(args, i)
        elif args.algorithm == "FedRe":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursRe(args, i)
        elif args.algorithm == "FedNet":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursNet(args, i)
        elif args.algorithm == "FedLocal":
            args.model = AlexNetFed15().to(args.device)
            server = FedLocal(args, i)
        elif args.algorithm == "Fed2":
            # args.model = AlexNet1().to(args.device)
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOurs2(args, i)
        elif args.algorithm == "Fed3":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOurs3(args, i)
        elif args.algorithm == "FedNet4":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursNet4(args, i)
        elif args.algorithm == "Fed5":
            g_fea = copy.deepcopy(args.model.gfc)
            args.head = copy.deepcopy(args.model.fc)
            args.model.gfc = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOurs5(args, i)
        elif args.algorithm == "FedNet6":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursNet6(args, i)
        elif args.algorithm == "FedNet7":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursNet7(args, i)
        elif args.algorithm == "FedNo":
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOursNo(args, i)
        elif args.algorithm == "FedNet8":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursNet8(args, i)
        elif args.algorithm == "Fed9":
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOurs9(args, i)
            # g_fea = copy.deepcopy(args.model.fc1)
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc1 = nn.Identity()
            # args.model.fc = nn.Identity()
            # args.model = ClientOurModel(args.model, g_fea, args.head)
            # server = FedOurs9(args, i)
        elif args.algorithm == "FedKL":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursKL(args, i)
        elif args.algorithm == "FedKL1":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursKL1(args, i)
        elif args.algorithm == "FedKL2":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursKL2(args, i)
        elif args.algorithm == "FedMMD":
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOursMMD(args, i)
        elif args.algorithm == "FedNet10":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = ClientOurModel(args.model, g_fea, args.head)
            server = FedOursNet10(args, i)
        elif args.algorithm == "FedNet11":
            g_fea = copy.deepcopy(args.model.fc1)
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc1 = nn.Identity()
            args.model.fc = nn.Identity()
            # gate = gate_DNN(args.head.in_features * 2, 2, device=args.device)
            gate = DNN_gate(args.head.in_features * 2, args.head.in_features, device=args.device)
            args.model = ClientGateModel(args.model, g_fea, args.head, gate)
            server = FedOursNet11(args, i)
        elif args.algorithm == "Fed12":
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOurs12(args, i)
        elif args.algorithm == "Fed13":
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOurs13(args, i)
        elif args.algorithm == "Fed14":
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOurs14(args, i)
        elif args.algorithm == "Fed15":
            args.model = AlexNetFed15().to(args.device)
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            print(args.model)
            server = FedOurs15(args, i)
        elif args.algorithm == "Fed16":
            # args.model = AlexNet1().to(args.device)
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOurs16(args, i)

        else:
            raise NotImplementedError


        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    # average_data(dataset=args.dataset,
    #              algorithm=args.algorithm,
    #              goal=args.goal,
    #              times=args.times,
    #              length=args.global_rounds / args.eval_gap + 1)

    print("All done!")

    # reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    args = args_parser()



    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)


    run(args)

