#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from argparse import Namespace
import os
import random
from typing import Tuple, Union, Dict, List
import os, wget, zipfile

import numpy as np
import torch
from torchvision import datasets, transforms
from dataloader_utils import get_celeba, CelebaDataset, get_utkface, UTKFaceDataset

from sampling import get_iid_partition, get_noniid_partition, paramaterise_noniid_distribution


def get_dataset(args: Union[Namespace, Dict]
                ) -> Tuple[datasets.VisionDataset, datasets.VisionDataset, Dict[int, List[int]] , Dict[int, List[int]]]:
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    
    Mean and Std values reference: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data

    """
    if isinstance(args, Namespace):
        args = vars(args)
    if args['dataset'] == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                (0.49139968, 0.48215827 ,0.44653124), 
                (0.24703233, 0.24348505, 0.26158768))]
             )

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args['dataset'] == 'fashionmnist':
        
        data_dir = '../data/fashionmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args['dataset'] == "utkface":
        data_dir = 'data/UTKFace'

        apply_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.49,), (0.23,))
        ])

        # train_dataset, test_dataset, valid_dataset = get_utkface(data_dir, apply_transform)
        train_dataset, test_dataset, valid_dataset = get_utkface(
            data_dir=data_dir, 
            zfile='data/utkface.tar.gz', 
            extract_dir='data', 
            apply_transform=apply_transform,
            label_type=args["label_type"],
        )

    elif args['dataset'] == "celeba":

        data_dir = 'data/celeba'

        mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
        std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

        apply_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        if "label_type" not in args:
            raise ValueError("celebA label-type is missing. Please use 'gender' or 'smiling'.")

        if args["label_type"] not in ["gender", "smiling"]:
            raise ValueError("celebA label-type is wrong. Please use 'gender' or 'smiling'.")

        label_type = args["label_type"]
        train_dataset, test_dataset, valid_dataset = get_celeba(
            data_dir, 
            label_type, 
            apply_transform
        )


    # sample training data amongst users
    if args['iid']:
        train_user_groups = get_iid_partition(train_dataset, args['num_users'])
        test_user_groups = get_iid_partition(test_dataset, args['num_users'])
        
    elif args['dist_noniid']:
        # users receive unequal data within classes
        distribution = paramaterise_noniid_distribution(args['num_users'], args['num_classes'], train_dataset.targets, float(args['dist_noniid']), args['min_proportion'])
        train_user_groups = get_noniid_partition(train_dataset.targets,distribution)
        test_user_groups = get_noniid_partition(test_dataset.targets, distribution)

    return train_dataset, test_dataset, train_user_groups, test_user_groups

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset.  : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def set_seed(seed: int = 42, is_deterministic=False) -> None:

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set

    if is_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

