#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import random
from argparse import Namespace
from typing import Dict, List, OrderedDict, Tuple, Union

import numpy as np
import copy
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sampling import get_iid_partition, get_noniid_partition, paramaterise_noniid_distribution


def get_dataset(args: Union[Namespace, Dict]
                ) -> Tuple[datasets.VisionDataset, datasets.VisionDataset, datasets.VisionDataset, Dict[int, List[int]] ,Dict[int, List[int]], Dict[int, List[int]]]:
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.

    Mean and Std values reference: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
    :return: train, test, valid dataset. train, test, valid user groups, which is a dictionary mapping a client
        to the dataset indices for tht client. Note that these are disjoint
    """
    if isinstance(args, Namespace):
        args = vars(args)
    if args["dataset"] == "cifar":
        data_dir = "../data/cifar/"
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))])

        train_valid_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
        train_idxs, valid_idxs = train_test_split(np.arange(len(train_valid_dataset)),
                                             test_size=0.1,
                                             random_state=42,
                                             shuffle=True,
                                             stratify=train_valid_dataset.targets)
        train_dataset = Subset(train_valid_dataset, train_idxs)
        valid_dataset = Subset(train_valid_dataset, valid_idxs)
        train_labels = torch.tensor(train_valid_dataset.targets)[train_idxs]
        valid_labels = torch.tensor(train_valid_dataset.targets)[valid_idxs]

    elif args['dataset'] == 'fashionmnist':
        
        data_dir = '../data/fashionmnist/'

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_valid_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        train_idxs, valid_idxs = train_test_split(np.arange(len(train_valid_dataset)),
                                             test_size=0.1,
                                             random_state=42,
                                             shuffle=True,
                                             stratify=train_valid_dataset.targets)
        train_dataset = Subset(train_valid_dataset, train_idxs)
        valid_dataset = Subset(train_valid_dataset, valid_idxs)
        train_labels = train_valid_dataset.targets[train_idxs]
        valid_labels = train_valid_dataset.targets[valid_idxs]

    # sample training data amongst users
    if args['iid']:
        train_user_groups = get_iid_partition(train_dataset, args['num_users'])
        valid_user_groups = get_iid_partition(valid_dataset, args['num_users'])
        test_user_groups = get_iid_partition(test_dataset, args['num_users'])
        
    elif args['dist_noniid']:
        # users receive unequal data within classes
        distribution = paramaterise_noniid_distribution(args['num_users'], args['num_classes'], train_labels, float(args['dist_noniid']), args['min_proportion'])
        train_user_groups = get_noniid_partition(train_labels,distribution)
        valid_user_groups = get_noniid_partition(valid_labels,distribution)
        test_user_groups = get_noniid_partition(test_dataset.targets, distribution)

    return train_dataset, test_dataset, valid_dataset, train_user_groups, test_user_groups, valid_user_groups


def exp_details(args):
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")

    print("    Federated parameters:")
    if args.iid:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
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

def dict_sum(
        list_of_dicts: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        helper function that sums up dictionaries stored in a list.
        Each dictionary is includes the same key-pair combination type.

        :param list_of_dicts: list of dictionaries

        :return: sum_of_dicts: sum of dictionaries from the list
        """
        assert list_of_dicts != None, "List of Dictionaries cannot be None."
        assert len(list_of_dicts) > 0, "Ensure the List of Dictionaries is not empty."

        sum_of_dicts = copy.deepcopy(list_of_dicts[0])
        for key in list(sum_of_dicts.keys()):
            for indx in range(1, len(list_of_dicts)):
                sum_of_dicts[key] += list_of_dicts[indx][key]

        return sum_of_dicts
