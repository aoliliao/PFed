import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms



class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./dataset/domainNet/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./dataset/domainNet/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)

        label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
                      'windmill': 7, 'wine_glass': 8, 'zebra': 9}

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

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


def prepare_data_domainnet(args):
    data_base_path = './dataset/domainNet'
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    # clipart
    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train)
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train)
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train)
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False)

    min_data_len = min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset),
                       len(real_trainset), len(sketch_trainset))

    min_data_len = int(min_data_len * 0.05)
    # 0.05
    clipart_trainset = torch.utils.data.Subset(clipart_trainset, list(range(min_data_len)))
    infograph_trainset = torch.utils.data.Subset(infograph_trainset, list(range(min_data_len)))
    painting_trainset = torch.utils.data.Subset(painting_trainset, list(range(min_data_len)))
    quickdraw_trainset = torch.utils.data.Subset(quickdraw_trainset, list(range(min_data_len)))
    real_trainset = torch.utils.data.Subset(real_trainset, list(range(min_data_len)))
    sketch_trainset = torch.utils.data.Subset(sketch_trainset, list(range(min_data_len)))

    clipart_train_loader = torch.utils.data.DataLoader(clipart_trainset, batch_size=args.batch_size, shuffle=True,)
    clipart_test_loader = torch.utils.data.DataLoader(clipart_testset, batch_size=args.batch_size, shuffle=False, )

    infograph_train_loader = torch.utils.data.DataLoader(infograph_trainset, batch_size=args.batch_size, shuffle=True,)
    infograph_test_loader = torch.utils.data.DataLoader(infograph_testset, batch_size=args.batch_size, shuffle=False,)

    painting_train_loader = torch.utils.data.DataLoader(painting_trainset, batch_size=args.batch_size, shuffle=True,)
    painting_test_loader = torch.utils.data.DataLoader(painting_testset, batch_size=args.batch_size, shuffle=False,)

    quickdraw_train_loader = torch.utils.data.DataLoader(quickdraw_trainset, batch_size=args.batch_size, shuffle=True,)
    quickdraw_test_loader = torch.utils.data.DataLoader(quickdraw_testset, batch_size=args.batch_size, shuffle=False,)

    real_train_loader = torch.utils.data.DataLoader(real_trainset, batch_size=args.batch_size, shuffle=True,)
    real_test_loader = torch.utils.data.DataLoader(real_testset, batch_size=args.batch_size, shuffle=False, )

    sketch_train_loader = torch.utils.data.DataLoader(sketch_trainset, batch_size=args.batch_size, shuffle=True,)
    sketch_test_loader = torch.utils.data.DataLoader(sketch_testset, batch_size=args.batch_size, shuffle=False,)

    train_loaders = [clipart_train_loader, infograph_train_loader, painting_train_loader, quickdraw_train_loader,
                     real_train_loader, sketch_train_loader]
    test_loaders = [clipart_test_loader, infograph_test_loader, painting_test_loader, quickdraw_test_loader,
                    real_test_loader, sketch_test_loader]

    return train_loaders, test_loaders

def prepare_data_domainnet_customer(args, datasets=[]):
    data_base_path = './dataset/domainNet'
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    train_sets = []
    test_sets = []
    train_loaders = []
    test_loaders = []

    min_data_len = 1e8
    if len(datasets) <= 6:
        for i in range(len(datasets)):
            trainset = DomainNetDataset(data_base_path, site=datasets[i], transform=transform_train)
            testset = DomainNetDataset(data_base_path, site=datasets[i], transform=transform_test, train=False)
            train_sets.append(trainset)
            test_sets.append(testset)
            if len(trainset) < min_data_len:
                min_data_len = len(trainset)

        concat_dataset = ConcatDataset(test_sets)
        print(len(concat_dataset))
        concat_test_dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=32, shuffle=False)
        min_data_len = int(min_data_len * 0.05)
        print(min_data_len)
        for i in range(len(train_sets)):
            train_sets[i] = torch.utils.data.Subset(train_sets[i], list(range(min_data_len)))
            print(len(train_sets[i]))
            train_loaders.append(torch.utils.data.DataLoader(train_sets[i], batch_size=args.batch_size, shuffle=True, ))
            test_loaders.append(torch.utils.data.DataLoader(test_sets[i], batch_size=args.batch_size, shuffle=False, ))



    print(len(train_loaders), len(test_loaders))
    print(len(train_loaders[0]), len(test_loaders[0]))
    return train_loaders, test_loaders, concat_test_dataloader,

