#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from collections import defaultdict
import numpy as np
import torch
from typing import Union, List, Dict


def get_iid_partition(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def paramaterise_noniid_distribution(
    num_users: int,
    num_classes: int,
    dataset_labels: Union[torch.Tensor, List],
    beta: float,
    min_proportion: float = 0,
):
    """
    Sample from dirichlet distribution to give non-iid distribution for users.

    :param num_users: number of users
    :param num_classes: number of classes
    :param dataset_labels: list or tensor of shape (num_samples,)
    :param beta: determines amount of non-iid
    :param min_proportion: minimum proportion of the dataset per user

    :return: array of shape (num_classes, num_users), where each row is a distribution
        over the users for a specific class
    """
    if isinstance(dataset_labels, list):
        dataset_labels = torch.tensor(dataset_labels)
    class_weights = np.zeros((num_classes,))
    for class_number in range(num_classes):
        class_weights[class_number] = (dataset_labels == class_number).sum() / len(
            dataset_labels
        )

    if min_proportion >= 1 / num_users:
        return ValueError(
            f"min_proportion per user must be less than {1/num_users} for a dataset with {num_users} in it"
        )
    class_sample_distribution = np.zeros((num_classes, num_users))
    dataset_proportion_per_user = np.zeros(num_users)
    while dataset_proportion_per_user.min() <= min_proportion:
        class_sample_distribution = np.random.dirichlet(
            np.repeat(beta, num_users), num_classes
        )
        dataset_proportion_per_user = (class_weights @ class_sample_distribution) / (
            class_weights.sum()
        )

    return class_sample_distribution


def get_noniid_partition(
    dataset_labels: torch.Tensor, distribution: Union[torch.Tensor, List]
) -> Dict[int, List[int]]:
    """
    Get samples assigned to each user

    :param dataset_labels: list or tensor of shape (num_samples,)
    :param distribution: array of shape (num_classes, num_users), where each row is a distribution
        over the users for a specific class

    :return: dictionary where each key is a user index, and each item is a list of sample idxs
        for that user
    """
    if isinstance(dataset_labels, list):
        dataset_labels = torch.tensor(dataset_labels)
    num_classes, num_users = distribution.shape
    sample_idxs = [torch.where(dataset_labels == i)[0] for i in range(num_classes)]
    users_data = defaultdict(list)
    for i in range(num_classes):
        num_class_samples = len(sample_idxs[i])
        sample_user_idx = np.random.choice(
            num_users, num_class_samples, p=distribution[i]
        )
        for user_idx, sample_idx in zip(sample_user_idx, sample_idxs[i]):
            users_data[user_idx].append(sample_idx.item())

    return users_data
