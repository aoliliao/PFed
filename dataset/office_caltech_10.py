import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms

from util.result_util import set_fixed_seed


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./dataset/office/office_caltech_10/{}_train.pkl'.format(site),
                                                   allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./dataset/office/office_caltech_10/{}_test.pkl'.format(site),
                                                   allow_pickle=True)

        label_dict = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4, 'laptop_computer': 5,
                      'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else './dataset/office'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def prepare_data_office(args):
    data_base_path = './dataset/office'
    transform_office = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    test_sets = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]
    concat_dataset = ConcatDataset(test_sets)
    # print(len(concat_dataset))
    concat_test_dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=False)

    min_data_len = min(len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset))
    min_data_len = int(min_data_len * 0.5)
    print(min_data_len)
    amazon_trainset = torch.utils.data.Subset(amazon_trainset, list(range(min_data_len)))
    caltech_trainset = torch.utils.data.Subset(caltech_trainset, list(range(min_data_len)))
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))
    webcam_trainset = torch.utils.data.Subset(webcam_trainset, list(range(min_data_len)))
    train_sets = [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch_size, shuffle=True,)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch_size, shuffle=False,)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch_size, shuffle=True,)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch_size, shuffle=False, )

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch_size, shuffle=True,)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch_size, shuffle=False, )

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch_size, shuffle=True,)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch_size, shuffle=False, )

    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]

    return train_loaders, test_loaders, concat_test_dataloader,
