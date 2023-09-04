#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import os
import random
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets, transforms

from dataloader_utils import get_celeba, get_utkface
from sampling import (
    get_iid_partition,
    get_noniid_partition,
    paramaterise_noniid_distribution,
)


def get_dataset(
    args: Union[Namespace, Dict]
) -> Tuple[
    datasets.VisionDataset,
    datasets.VisionDataset,
    datasets.VisionDataset,
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
]:
    """Returns train and test datasets and a user group which is a dict where
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
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215827, 0.44653124),
                    (0.24703233, 0.24348505, 0.26158768),
                ),
            ]
        )

        train_valid_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )

        train_idxs, valid_idxs = train_test_split(
            np.arange(len(train_valid_dataset)),
            test_size=0.1,
            random_state=42,
            shuffle=True,
            stratify=train_valid_dataset.targets,
        )

        train_dataset = Subset(train_valid_dataset, train_idxs)
        valid_dataset = Subset(train_valid_dataset, valid_idxs)
        train_labels = torch.tensor(train_valid_dataset.targets)[train_idxs]
        valid_labels = torch.tensor(train_valid_dataset.targets)[valid_idxs]
        test_labels = test_dataset.targets

    elif args["dataset"] == "fashionmnist":
        data_dir = "../data/fashionmnist/"

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_valid_dataset = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.FashionMNIST(
            data_dir, train=False, download=True, transform=apply_transform
        )
        train_idxs, valid_idxs = train_test_split(
            np.arange(len(train_valid_dataset)),
            test_size=0.1,
            random_state=42,
            shuffle=True,
            stratify=train_valid_dataset.targets,
        )
        train_dataset = Subset(train_valid_dataset, train_idxs)
        valid_dataset = Subset(train_valid_dataset, valid_idxs)
        train_labels = train_valid_dataset.targets[train_idxs]
        valid_labels = train_valid_dataset.targets[valid_idxs]
        test_labels = test_dataset.targets

    elif args["dataset"] == "utkface":
        data_dir = "data/UTKFace"

        apply_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.49,), (0.23,)),
            ]
        )

        # train_dataset, test_dataset, valid_dataset = get_utkface(data_dir, apply_transform)
        train_dataset, test_dataset, valid_dataset = get_utkface(
            data_dir=data_dir,
            zfile="data/utkface.tar.gz",
            extract_dir="data",
            apply_transform=apply_transform,
            label_type=args["utk_label_type"],
        )

        train_labels = train_dataset.dataset.labels
        train_labels = [d[args["utk_label_type"]] for d in train_labels]
        train_images = train_dataset.dataset.images

        valid_labels = valid_dataset.dataset.labels
        valid_labels = [d[args["utk_label_type"]] for d in valid_labels]
        valid_images = valid_dataset.dataset.images

        test_labels = test_dataset.dataset.labels
        test_labels = [d[args["utk_label_type"]] for d in test_labels]
        test_images = test_dataset.dataset.images

    elif args["dataset"] == "celeba":
        data_dir = "data/celeba"

        mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
        std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

        apply_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        label_type = args["celeba_label_type"]
        train_dataset, test_dataset, valid_dataset = get_celeba(
            data_dir, label_type, apply_transform
        )
        train_labels = train_dataset.dataset.labels
        train_labels = [d[args["celeba_label_type"]] for d in train_labels]
        train_images = train_dataset.dataset.images
        # TODO : ADD OTHERS

    # sample training data amongst users
    if args["iid"]:
        train_user_groups = get_iid_partition(train_dataset, args["num_users"])
        valid_user_groups = get_iid_partition(valid_dataset, args["num_users"])
        test_user_groups = get_iid_partition(test_labels, args["num_users"])

    elif args["dist_noniid"]:
        # users receive unequal data within classes
        distribution = paramaterise_noniid_distribution(
            args["num_users"],
            args["num_classes"],
            train_labels,
            float(args["dist_noniid"]),
            args["min_proportion"],
        )
        train_user_groups = get_noniid_partition(train_labels, distribution)
        valid_user_groups = get_noniid_partition(valid_labels, distribution)
        test_user_groups = get_noniid_partition(test_labels, distribution)

    return (
        train_dataset,
        test_dataset,
        valid_dataset,
        train_user_groups,
        test_user_groups,
        valid_user_groups,
    )


def exp_details(args):
    print("\nExperimental details:")
    print(f"    Dataset.  : {args.dataset}")
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
    flat_model: np.ndarray,
    sparse_ratio: float = 1,
    sparsification_type: str = "randk",
    choose_from_top_r_percentile: float = 1,
):
    assert (
        choose_from_top_r_percentile >= sparse_ratio
    ), "choose_from_top_r_percentile for rtopk should be larger than sparse_ratio"
    if sparsification_type == "randk":
        num_params = int(sparse_ratio * len(flat_model))
        indices = np.random.choice(len(flat_model), size=num_params, replace=False)
        bitmask = np.zeros_like(flat_model)
        np.put(bitmask, indices, 1)
        return bitmask
    elif sparsification_type == "topk":
        num_params = int(sparse_ratio * len(flat_model))
        max_indices = np.argpartition(np.absolute(flat_model), -num_params)[
            -num_params:
        ]
        bitmask = np.zeros_like(flat_model)
        np.put(bitmask, max_indices, 1)
        return bitmask
    elif sparsification_type == "rtopk":
        num_params = int(choose_from_top_r_percentile * len(flat_model))
        max_indices = np.argpartition(np.absolute(flat_model), -num_params)[
            -num_params:
        ]
        sparse_max_indices = np.random.choice(
            max_indices, size=int(sparse_ratio * len(flat_model)), replace=False
        )
        bitmask = np.zeros_like(flat_model)
        np.put(bitmask, sparse_max_indices, 1)
        return bitmask
    else:
        raise ValueError("Unrecognized sparsification method!")


def temperatured_softmax(client_losses, softmax_temperature):
    """Calulate a softmax distribution across client losses with
    temperature
    """
    client_losses = client_losses / softmax_temperature
    return np.exp(client_losses - np.max(client_losses)) / np.sum(
        np.exp(client_losses - np.max(client_losses))
    )
