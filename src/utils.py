#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import random
from argparse import Namespace
from typing import Dict, List, OrderedDict, Tuple, Union

import numpy as np
import torch
from torchvision import datasets, transforms

from sampling import (get_iid_partition, get_noniid_partition,
                      paramaterise_noniid_distribution)


def get_dataset(
    args: Union[Namespace, Dict]
) -> Tuple[
    datasets.VisionDataset,
    datasets.VisionDataset,
    Dict[int, List[int]],
    Dict[int, List[int]],
]:
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.

    Mean and Std values reference: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data

    """
    if isinstance(args, Namespace):
        args = vars(args)
    if args["dataset"] == "cifar":
        data_dir = "../data/cifar/"
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215827, 0.44653124),
                    (0.24703233, 0.24348505, 0.26158768),
                ),
            ]
        )

        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )

    elif args["dataset"] == "fashionmnist":
        data_dir = "../data/fashionmnist/"

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=apply_transform
        )

    # sample training data amongst users
    if args["iid"]:
        train_user_groups = get_iid_partition(train_dataset, args["num_users"])
        test_user_groups = get_iid_partition(test_dataset, args["num_users"])

    elif args["dist_noniid"]:
        # users receive unequal data within classes
        distribution = paramaterise_noniid_distribution(
            args["num_users"],
            args["num_classes"],
            train_dataset.targets,
            float(args["dist_noniid"]),
            args["min_proportion"],
        )
        train_user_groups = get_noniid_partition(train_dataset.targets, distribution)
        test_user_groups = get_noniid_partition(test_dataset.targets, distribution)

    return train_dataset, test_dataset, train_user_groups, test_user_groups


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


def flatten(model):
    weights = model.state_dict()
    # create flat array
    flat = np.array([])
    for k in weights.keys():
        flat = np.concatenate((flat, weights[k].cpu().numpy().flatten()))
    return flat


def updateFromNumpyFlatArray(flat_arr, model):
    start = 0
    new_glob = OrderedDict()
    model_dict = model.state_dict()
    for k in model_dict.keys():
        size = 1
        for dim in model_dict[k].shape:
            size *= dim
        shaped = np.reshape(flat_arr[start : start + size].copy(), model_dict[k].shape)
        new_glob[k] = torch.from_numpy(shaped)
        start = start + size

    model.load_state_dict(new_glob)


def get_bitmask_per_method(
    flat_model: np.ndarray, sparse_ratio: float = 1, sparsification_type: str = "randk"
):
    if sparsification_type == "randk":
        return np.random.choice(
            [0, 1], size=(len(flat_model),), p=[1 - sparse_ratio, sparse_ratio]
        )
    else:
        raise ValueError("Unrecognized sparsification method!")
