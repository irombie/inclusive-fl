#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11

import copy
import os
import random
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch


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
    assert list_of_dicts is not None, "List of Dictionaries cannot be None."
    assert len(list_of_dicts) > 0, "Ensure the List of Dictionaries is not empty."

    sum_of_dicts = copy.deepcopy(list_of_dicts[0])
    for key in list(sum_of_dicts.keys()):
        for indx in range(1, len(list_of_dicts)):
            sum_of_dicts[key] += list_of_dicts[indx][key]

    return sum_of_dicts


def flatten(model, is_dict=False):
    if not is_dict:
        weights = model.state_dict()
    else:
        weights = model
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
    if sparsification_type == "randk":
        num_params = int(sparse_ratio * len(flat_model))
        indices = np.random.choice(len(flat_model), size=num_params, replace=False)
        bitmask = np.zeros_like(flat_model)
        np.put(bitmask, indices, 1)
        return bitmask

    if sparsification_type == "topk":
        num_params = int(sparse_ratio * len(flat_model))
        max_indices = np.argpartition(np.absolute(flat_model), -num_params)[-num_params:]
        bitmask = np.zeros_like(flat_model)
        np.put(bitmask, max_indices, 1)
        return bitmask

    if sparsification_type == "rtopk":
        choose_from_top_r_percentile = max(choose_from_top_r_percentile, sparse_ratio)

        num_params = int(choose_from_top_r_percentile * len(flat_model))
        max_indices = np.argpartition(np.absolute(flat_model), -num_params)[-num_params:]
        sparse_max_indices = np.random.choice(max_indices, size=int(sparse_ratio * len(flat_model)), replace=False)
        bitmask = np.zeros_like(flat_model)
        np.put(bitmask, sparse_max_indices, 1)
        return bitmask

    raise ValueError("Unrecognized sparsification method!")


def temperatured_softmax(client_losses, softmax_temperature):
    """Calulate a softmax distribution across client losses with
    temperature
    """
    client_losses = client_losses / softmax_temperature
    max_client_loss = np.max(client_losses)
    return np.exp(client_losses - max_client_loss) / np.sum(np.exp(client_losses - max_client_loss))


def linearly_interpolated_softmax(client_losses, max_sparsity, min_sparsity, temperature):
    """
    This method is an extension to the temperatured softmax, which addresses the failiure case when
    not enough parameters are sent by each cases (as is done with custom_exponential_sparsity).

    Namely, we linearly interpolate between a minimum and maximum sparsity per client based on how poorly they are
    performing on the validation set.
    Namely:
        client_sparsity = p_c*max_sparsity + (1-p_c)*min_sparsity, where p_c is derived from the temperatured softmax
        as above.
    As such, each client's sparsity will be bounded between the specified minimum and maximum sparsity, with
    the worst performing client being allocated the largest parameter budget because of the softmax function.
    """
    assert max_sparsity > min_sparsity
    client_prob_masses = temperatured_softmax(client_losses, temperature)
    return [max_sparsity * client_prob + (1 - client_prob) * min_sparsity for client_prob in client_prob_masses]


def custom_exponential_sparsity(client_losses, max_sparsity, min_sparsity, temperature):
    """
    Calculate sparse ratio for each client based on an exponential formula

    The sparse ratio is bounded between max_sparsity and min_sparsity,
    such that the client with the highest loss has max_sparsity

    Note that whilst the sparse ratio will be bounded between max_sparsity and
    min_sparsity, actually achieving min_sparsity is probably infeasible, as this will
    be achieved when client_loss-max_loss is -infinity.
    In future, this formula can probably be ammended such that
    min_sparsity is acheived when the client loss is 0.
    """
    assert max_sparsity > min_sparsity, "Max sparsity must be less than min sparsity"
    client_losses = client_losses / temperature
    max_client_loss = np.max(client_losses)
    return (max_sparsity - min_sparsity) * np.exp(client_losses - max_client_loss) + min_sparsity


def normalize(data_tensor):
    """re-scale image values to [-1, 1]"""
    return (data_tensor / 255.0) * 2.0 - 1.0
