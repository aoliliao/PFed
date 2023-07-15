import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import ujson
from sklearn.model_selection import train_test_split
from torchvision import datasets
import matplotlib.pyplot as plt

from dataset_util import check


def label_transfer(pre_label):
    pre_label1 = [[] for _ in range(len(pre_label))]
    dic = {19: 11, 29: 15, 0: 4, 11: 14, 1: 1, 86: 5, 90: 18, 28: 3, 23: 10, 31: 11, 39: 5, 96: 17, 82: 2, 17: 9, 71: 10, 8: 18, 97: 8, 80: 16, 74: 16,
           59: 17, 70: 2, 87: 5, 84: 6, 64: 12, 52: 17, 42: 8, 47: 17, 65: 16, 21: 11, 22: 5, 81: 19, 24: 7, 78: 15, 45: 13, 49: 10, 56: 17, 76: 9,
           89: 19, 73: 1, 14: 7, 9: 3, 6: 7, 20: 6, 98: 14, 36: 16, 55: 0, 72: 0, 43: 8, 51: 4, 35: 14, 83: 4, 33: 10, 27: 15, 53: 4, 92: 2, 50: 16,
           15: 11, 18: 7, 46: 14, 75: 12, 38: 11, 66: 12, 77: 13, 69: 19, 95: 0, 99: 13, 93: 15, 4: 0, 61: 3, 94: 6, 68: 9, 34: 12, 32: 1, 88: 8,
           67: 1, 30: 0, 62: 2, 63: 12, 40: 5, 26: 13, 48: 18, 79: 13, 85: 19, 54: 2, 44: 15, 7: 7, 12: 9, 2: 14, 41: 19, 37: 9, 13: 18,
           25: 6, 10: 3, 57: 4, 5: 6, 60: 10, 91: 1, 3: 8, 58: 18, 16: 3}
    for i in range(len(pre_label)):
        pre_label1[i] = dic[pre_label[i]]
    pre_label1 = np.array(pre_label1)
    return pre_label1

def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}
    global_train_x, global_test_x = [], []
    global_train_y, global_test_y = [], []
    global_train_data, global_test_data = [], []

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)
        if i == 0:
            global_train_x, global_test_x = X_train, X_test
            global_train_y, global_test_y = y_train, y_test
        else:
            global_train_x = np.concatenate((global_train_x, X_train))
            global_test_x = np.concatenate((global_test_x, X_test))
            global_train_y = np.concatenate((global_train_y, y_train))
            global_test_y = np.concatenate((global_test_y, y_test))

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    global_train_data.append({'x': global_train_x, 'y': global_train_y})
    global_test_data.append({'x': global_test_x, 'y': global_test_y})
    del X, y


    return train_data, test_data, global_train_data, global_test_data



def save_file(config_path, train_path, test_path, train_data, test_data, global_train_data, global_test_data, num_clients,
              num_classes, statistic, statistic_label, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'Size of samples for sublabels in clients': statistic_label,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")
    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)

    for idx, train_dict in enumerate(global_train_data):
        with open(train_path + 'global' + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(global_test_data):
        with open(test_path + 'global' + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)

    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


def separate_data_cifar20_balance(data, num_clients, num_classes=5,  partition="dir", mapper=[], alpha_parm=1.0):
    print('enter function')
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    y_label = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    statistic_label = [[] for _ in range(num_clients)]
    dataset_content, dataset_label = data

    n_data_per_clnt = len(dataset_label) / 20 / num_clients

    if partition == "dir":

        n_classes = 5
        client_idcs = [[] for _ in range(num_clients)]
        for i in range(len(mapper)):
            print("i:",i)
            clnt_data_list = (np.ones(num_clients) * n_data_per_clnt).astype(int)
            cls_priors = np.random.dirichlet(alpha=[alpha_parm] * num_classes, size=num_clients)
            prior_cumsum = np.cumsum(cls_priors, axis=1)
            idx_list = [np.where(dataset_label == mapper[i][j])[0] for j in range(len(mapper[i]))]
            cls_amount = [len(idx_list[i]) for i in range(len(mapper[i]))]

            while (np.sum(clnt_data_list) != 0):
                # print('i:',i, 'while ing...')
                curr_clnt = np.random.randint(num_clients)
                if clnt_data_list[curr_clnt] <= 0:
                    continue
                clnt_data_list[curr_clnt] -= 1
                curr_prior = prior_cumsum[curr_clnt]
                while True:
                    cls_label = np.argmax(np.random.uniform() <= curr_prior)
                    if cls_amount[cls_label] <= 0:
                        continue
                    cls_amount[cls_label] -= 1
                    client_idcs[curr_clnt] += [idx_list[cls_label][cls_amount[cls_label]]]
                    break

    else:
        raise NotImplementedError

    # assign data
    print('assign')
    for client in range(num_clients):
        idxs = client_idcs[client]
        X[client] = dataset_content[idxs]
        y_label[client] = dataset_label[idxs]
        y[client] = label_transfer(y_label[client],)
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))
        for i in np.unique(y_label[client]):
            statistic_label[client].append((int(i), int(sum(y_label[client] == i))))

    del data
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    print("======================================")
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y_label[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic_label[client]])
        print("-" * 50)
    print("finish!")

    return X, y, statistic, statistic_label

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    mapper = [[4, 30, 55, 72, 95], [1, 32, 67, 73, 91], [54, 62, 70, 82, 92], [9, 10, 16, 28, 61], [0, 51, 53, 57, 83],
         [22, 39, 40, 86, 87], [5, 20, 25, 84, 94],
         [6, 7, 14, 18, 24], [3, 42, 43, 88, 97], [12, 17, 37, 68, 76], [23, 33, 49, 60, 71], [15, 19, 21, 31, 38],
         [34, 63, 64, 66, 75], [26, 45, 77, 79, 99],
         [2, 11, 35, 46, 98], [27, 29, 44, 78, 93], [36, 50, 65, 74, 80], [47, 52, 56, 59, 96], [8, 13, 48, 58, 90],
         [41, 69, 81, 85, 89]]
    CIFAR_PATH = "./cifar100FL"
    N_CLIENTS = 25
    batch_size = 40
    train_size = 0.8  # merge original training set and test set, then split it manually.
    alpha = 1.0 # for Dirichlet distribution

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True, transform=transform)

    train_img = trainset.data
    train_labels = np.array(trainset.targets)
    test_img = testset.data
    test_labels = np.array(testset.targets)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    # input_sz, num_cls = train_data.data[0].shape[0],  len(train_data.classes)

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic, statistic_label = separate_data_cifar20_balance((dataset_image, dataset_label), num_clients=N_CLIENTS, num_classes=5, mapper=mapper, alpha_parm=alpha)
    train_data, test_data, global_train_data, global_test_data = split_data(X, y)


    dir_path = "./cifar100FL/cifar20/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    save_file(config_path, train_path, test_path, train_data, test_data, global_train_data, global_test_data, num_clients=N_CLIENTS, num_classes=20,
              statistic=statistic, statistic_label=statistic_label, niid=True, balance=True, partition="dir")
