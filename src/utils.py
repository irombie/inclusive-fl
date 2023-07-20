#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from argparse import Namespace
import os
import random
from typing import Tuple, Union, Dict, Any

import numpy as np
import torch
from torchvision import datasets, transforms

from sampling import get_iid, distribution_noniid


def get_dataset(args: Union[Namespace, Dict]
                ) -> Tuple[datasets.VisionDataset, datasets.VisionDataset, Dict[int, Any] , Dict[int, Any]]:
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
             transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args['dataset'] == 'mnist' or 'fmnist':
        if args['dataset'] == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # sample training data amongst users
    if args['iid']:
        train_user_groups = get_iid(train_dataset, args['num_users'])
        test_user_groups = get_iid(test_dataset, args['num_users'])
        
    elif args['dist_noniid']:
        # users receive unequal data within classes
        train_user_groups = distribution_noniid(train_dataset.targets, args['num_users'], beta=float(args['dist_noniid']))
        test_user_groups = distribution_noniid(test_dataset.targets, args['num_users'], beta=float(args['dist_noniid']))
            

    return train_dataset, test_dataset, train_user_groups, test_user_groups

def exp_details(args):
    print('\nExperimental details:')
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

