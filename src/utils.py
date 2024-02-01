#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import os
import glob
import random
import shutil
import sys
import scipy
import tarfile
import zipfile
from argparse import Namespace
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Union

import json
import numpy as np
import pandas as pd
import torch
import wget
from parse import parse
from PIL import Image
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


def exp_details(args):
    exp_table = PrettyTable()
    exp_table.field_names = ["Experiment Parameter", "Value"]
    exp_table.add_row(["FL Algorithm", args.fl_method])
    exp_table.add_row(["Dataset", args.dataset])
    exp_table.add_row(["Model", args.model])
    exp_table.add_row(
        [
            "Device",
            torch.device(
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_built() else "cpu")
            ),
        ]
    )
    exp_table.add_row(["Learning Rate", args.lr])
    exp_table.add_row(["Global Rounds", args.epochs])
    exp_table.add_row(["Local Epochs", args.local_ep])
    exp_table.add_row(["Local Batch Size", args.local_bs])
    exp_table.add_row(["Number of Users", args.num_users])
    exp_table.add_row(["Fraction of Users", args.frac])
    exp_table.add_row(["Data Distribution Type", "IID" if args.iid else "Non-IID"])

    print(exp_table)
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


## Sampling methods [IID, Non-IID using Dirichlet distribution]
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


# UKTFace


class UTKFaceDataset(Dataset):
    def __init__(
        self, directory, zfile, extract_dir, transform, label_type="ethnicity"
    ):
        """
        Returns utkface dataset downloaded from link https://susanqq.github.io/UTKFace/.
        Download the aligned and cropped dataset (107 MB) and add it to the data folder
        with name utkface.tar.gz.
        Other helper references: https://github.com/AryaHassanli/

        :params directory: directory where the images are located
        :params zfile: relative path from home folder to the zip file stored under data
        :params extract_dir: main directory where the UTKFace folder will be stored
        :params transform: image transformation for UTKFace

        :returns: dataset that can be used for training UTKFace
        """
        self.directory = directory
        self.transform = transform
        self.label_type = label_type
        self.labels = []
        self.images = []

        if os.path.isdir(directory) and len(os.listdir(directory)) > 0:
            print("UTK Already Exists on", self.directory, " / We will use it!")
        else:
            print("Could not find UTK on", directory)
            print("Looking for ", zfile)
            if os.path.exists(zfile):
                print(zfile, "is found. Trying to extract:")
                try:
                    tar = tarfile.open(zfile, "r:gz")
                    tar.extractall(path=extract_dir)
                    tar.close()
                    print("Successfully extracted")
                except tarfile.TarError:
                    sys.exit("Extract Failed!")
            else:
                sys.exit("UTK Zip file not found!")

        for i, file in enumerate(os.listdir(extract_dir + "/UTKFace")):
            file_labels = parse("{age}_{gender}_{ethnicity}_{}.jpg", file)
            if file_labels is not None:
                if int(file_labels["age"]) > 120 or int(file_labels["gender"]) > 1:
                    continue

                image = Image.open(os.path.join(extract_dir + "/UTKFace", file))
                image = self.transform(image)

                self.images.append(image)
                self.labels.append(
                    {
                        # "age": self.convert_age_to_range(int(file_labels["age"])),
                        "gender": int(file_labels["gender"]),
                        "ethnicity": int(file_labels["ethnicity"]),
                    }
                )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]

        try:  # accepts age/gender/ethnicity labels
            labels = self.labels[idx][self.label_type]
        except:
            print("Wrong Label Type provided")
            return

        return image, labels


def get_utkface(data_dir, zfile, extract_dir, apply_transform, label_type="ethnicity"):
    """
    Returns train/test/validation utkface datasets.

    :params data_dir: directory where the images are located
    :params zfile: relative path from home folder to the zip file stored under data
    :params extract_dir: main directory where the UTKFace folder will be stored
    :params apply_transform: image transformation for UTKFace

    :returns: train/test/validation utkface datasets that can be used for training UTKFace
    """
    dataset = UTKFaceDataset(
        directory=data_dir,
        zfile=zfile,
        extract_dir=extract_dir,
        transform=apply_transform,
        label_type=label_type,
    )

    train_len = int(len(dataset) * 0.8)
    validate_len = int(len(dataset) * 0.1)
    test_len = int(len(dataset) - train_len - validate_len)

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [
            train_len,
            validate_len,
            test_len,
        ],
    )

    return train_dataset, test_dataset, valid_dataset


def normalize_tin_val_folder_structure(
    path, images_folder="images", annotations_file="val_annotations.txt"
):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError("Validation folder is empty.")
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()


class TinyImageNet(ImageFolder):
    """Dataset for TinyImageNet-200"""

    base_folder = "tiny-imagenet-200"
    zip_md5 = "90528d7ca1a48142e341f4ef8d21d0de"
    splits = ("train", "val")
    filename = "tiny-imagenet-200.zip"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(self, root, split="train", download=False, **kwargs):
        self.data_root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        super().__init__(self.split_folder, **kwargs)

    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def download(self):
        if self._check_exists():
            return
        download_and_extract_archive(
            self.url,
            self.data_root,
            filename=self.filename,
            remove_finished=True,
            md5=self.zip_md5,
        )
        assert "val" in self.splits
        normalize_tin_val_folder_structure(os.path.join(self.dataset_folder, "val"))

class VehicleDataset(Dataset):
    def __init__(self):
        self.X = self.preprocess(np.load('data/vehicle/data/X.npy'))
        self.X = torch.from_numpy(self.X)
        self.targets = torch.from_numpy(np.load('data/vehicle/data/Y.npy'))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.targets[idx]
    
    def preprocess(self, x):
        means = np.mean(x, axis=0)
        std = np.std(x, axis=0)

        x = (x - means) * 1.0 / std
        where_are_NaNs = np.isnan(x)
        x[where_are_NaNs] = 0
        return x
    
class GroupedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])


def get_dataset(
    args: Union[Namespace, Dict]
) -> Tuple[
    Union[datasets.VisionDataset, Dataset],
    Union[datasets.VisionDataset, Dataset],
    Union[datasets.VisionDataset, Dataset],
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

    if args.dataset == "fashiomnist" or args.dataset == "cifar":
        args.num_classes = 10
    elif args.dataset == "utkface":
        args.num_classes = 5
    elif args.dataset == "tiny-imagenet":
        args.num_classes = 200
    elif args.dataset == "vehicle":
        args.num_classes = 100
    else:
        raise ValueError("Unrecognized dataset!")

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
        data_dir = "../data"

        apply_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5959, 0.4562, 0.3906), (0.2591, 0.2312, 0.2268)
                ),
            ]
        )

        # train_dataset, test_dataset, valid_dataset = get_utkface(data_dir, apply_transform)
        train_dataset, test_dataset, valid_dataset = get_utkface(
            data_dir=data_dir,
            zfile="../data/UTKFace.tar.gz",
            extract_dir="../data",
            apply_transform=apply_transform,
            label_type="ethnicity",
        )

        train_labels = train_dataset.dataset.labels
        train_labels = [d["ethnicity"] for d in train_labels]
        train_images = train_dataset.dataset.images

        valid_labels = valid_dataset.dataset.labels
        valid_labels = [d["ethnicity"] for d in valid_labels]
        valid_images = valid_dataset.dataset.images

        test_labels = test_dataset.dataset.labels
        test_labels = [d["ethnicity"] for d in test_labels]
        test_images = test_dataset.dataset.images

    elif args["dataset"] == "tiny-imagenet":
        data_dir = "../data"

        apply_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        train_valid_dataset = TinyImageNet(
            data_dir, split="train", download=True, transform=apply_transform
        )
        test_dataset = TinyImageNet(
            data_dir, split="val", download=True, transform=apply_transform
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

    elif args["dataset"] == "vehicle":
        train_test_dataset = VehicleDataset()

        train_valid_idxs, test_idxs = train_test_split(
            np.arange(len(train_test_dataset)),
            test_size=0.1,
            random_state=42,
            shuffle=True,
            stratify=train_test_dataset.targets.numpy(),
        )

        train_split = int(len(train_valid_idxs)*0.9)
        train_idxs, valid_idxs = train_valid_idxs[:train_split], train_valid_idxs[train_split:]

        train_dataset = Subset(train_test_dataset, train_idxs)
        valid_dataset = Subset(train_test_dataset, valid_idxs)
        test_dataset = Subset(train_test_dataset, test_idxs)
        train_labels = train_test_dataset.targets[train_idxs]
        valid_labels = train_test_dataset.targets[valid_idxs]
        test_labels = train_test_dataset.targets[test_idxs]
    

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
    elif sparsification_type == "topk":
        num_params = int(sparse_ratio * len(flat_model))
        max_indices = np.argpartition(np.absolute(flat_model), -num_params)[
            -num_params:
        ]
        bitmask = np.zeros_like(flat_model)
        np.put(bitmask, max_indices, 1)
        return bitmask
    elif sparsification_type == "rtopk":
        if choose_from_top_r_percentile < sparse_ratio:
            choose_from_top_r_percentile = sparse_ratio
        # assert (
        #     choose_from_top_r_percentile >= sparse_ratio
        # ), "choose_from_top_r_percentile for rtopk should be larger than sparse_ratio"
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
