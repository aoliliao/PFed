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

from algorithms.server.serverFed16 import FedOurs16
from algorithms.server.serverFedCP import FedCP
from algorithms.server.serverFedNH import FedNH
from algorithms.server.serverFedRod import FedRod
from models.Fed15Model import AlexNetFed15
from util.result_util import set_fixed_seed

# from algorithms.server.serverFed1 import FedOursNet
from algorithms.server.serverFedBABU import FedBABU
from algorithms.server.serverFedFA import FedFA
from algorithms.server.serverFedNoPer import FedOursNo
from algorithms.server.serverFedRep import FedRep
from algorithms.server.serverFedavg import FedAvg
from algorithms.server.serverFedbn import FedBN
# from algorithms.server.serverFedrod import FedROD
from algorithms.server.serverLocal import FedLocal
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
        elif args.algorithm == "FedLocal":
            args.model = AlexNetFed15().to(args.device)
            server = FedLocal(args, i)
        elif args.algorithm == "FedNo":
            head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, head)
            server = FedOursNo(args, i)
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
