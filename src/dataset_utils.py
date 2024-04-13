#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import json
import os
import random
import shutil
import sys
import tarfile
import zipfile
import gdown
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
import pandas as pd
import torch
import wandb
import wget
from parse import parse
from PIL import Image
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

from fastargs import get_current_config
from fastargs.decorators import param, section


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
                self.labels.append(float(file_labels["ethnicity"]))
                '''      #{
                        # "age": self.convert_age_to_range(int(file_labels["age"])),
                        #"gender": int(file_labels["gender"]),
                        ,
                    #}
                #)
                '''
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        labels = self.labels[idx]

        return image, labels

## class definitions
class SyntheticDataset(Dataset):
    """Synthetic dataset generated using the function generate_synthetic_data"""

    def __init__(self, num_clients, num_classes, num_features):
        """
        Returns synthetic dataset at the given path.

        :params path: path to the synthetic data file
        """
        X, y = self.fetch_synthetic_data()
        self.n_users = len(X)
        self.n_samples_per_user = [len(x) for x in X]
        self.n_samples = sum(self.n_samples_per_user)
        self.cumulative_samples = np.cumsum([0] + self.n_samples_per_user)
        self.user_idx = {
            i: np.arange(start, end).tolist()
            for i, (start, end) in enumerate(
                zip(self.cumulative_samples, self.cumulative_samples[1:])
            )
        }
        self.X = np.concatenate(X).astype(np.float32)
        self.y = np.concatenate(y).astype(np.int64)
        self.validate_data(self.X, self.y)

    def __len__(self) -> int:
        return self.n_samples

    def fetch_synthetic_data(self, num_clients, num_classes, num_features, url: str, path: str | Path):
        """
        Fetch the synthetic data from the given URL and save it to the given path.

        :param url: URL to fetch the synthetic data from
        :param path: path to save the synthetic data to
        """
        synthetic_data_url = f"https://drive.google.com/uc?id={args['gdrive_id']}"

        path = Path(path).resolve() / "synthetic_data.zip"
        if not path.exists():
            gdown.download(synthetic_data_url, path.as_posix())

        with zipfile.ZipFile(data_zip, "r") as zip_ref:
            fname = f"data/synthetic_data_nusers_{args['num_clients']}_nclasses_{args['num_classes']}_ndims_{args['num_features']}.json"
            with zip_ref.open(fname) as f:
                data = json.load(f)
                X = [np.array(x) for x in data["X"]]
                y = [np.array(y) for y in data["y"]]

        return X, y
        

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get an item from the dataset.

        :param idx: index of the item
        :return: a tuple containing the input data and the label
        """
        return self.X[idx], self.y[idx]

    def split(
        self, valid_ratio: float = 0.1, test_ratio: float = 0.1
    ) -> tuple["SyntheticDataset", "SyntheticDataset", "SyntheticDataset"]:
        """
        Split the data into training and validation sets.

        :param valid_ratio: proportion of the data to include in the validation set
        :param test_ratio: proportion of the data to include in the test set
        :return: training and validation datasets
        """
        X_train, y_train = [], []
        X_valid, y_valid = [], []
        X_test, y_test = [], []
        for idx in self.user_idx.values():
            valid_size = int(len(idx) * valid_ratio)
            test_size = int(len(idx) * test_ratio)

            train_idx = idx[: -valid_size - test_size]
            valid_idx = idx[-valid_size - test_size : -test_size]
            test_idx = idx[-test_size:]

            X_train.append(self.X[train_idx])
            y_train.append(self.y[train_idx])
            X_valid.append(self.X[valid_idx])
            y_valid.append(self.y[valid_idx])
            X_test.append(self.X[test_idx])
            y_test.append(self.y[test_idx])

        train_dataset = SyntheticDataset(X_train, y_train)
        valid_dataset = SyntheticDataset(X_valid, y_valid)
        test_dataset = SyntheticDataset(X_test, y_test)
        return train_dataset, valid_dataset, test_dataset

    @staticmethod
    def validate_data(X: list[np.ndarray], y: list[np.ndarray]) -> bool:
        """
        Validate the synthetic data.

        :param X: input data
        :param y: labels
        :return: whether the data is valid
        """
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(y) != 0
        assert len(X) == len(y)

    @classmethod
    def load_from_path(cls, path: str | Path) -> "SyntheticDataset":
        """
        Load the synthetic dataset from the given path.

        :param path: path to the synthetic data file
        :return: the synthetic dataset
        """
        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        X = [np.asarray(x) for x in data["X"]]
        y = [np.asarray(y) for y in data["y"]]
        return cls(X, y)
    

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

    def normalize_tin_val_folder_structure(self, path, images_folder="images", annotations_file="val_annotations.txt"):
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

def prepare_fashionMNIST(data_dir, seed=42):
    apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

    train_valid_dataset = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=apply_transform
    )

    train_idxs, valid_idxs = train_test_split(
        np.arange(len(train_valid_dataset)),
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=train_valid_dataset.targets
    )
    
    train_dataset = Subset(train_valid_dataset, train_idxs)
    valid_dataset = Subset(train_valid_dataset, valid_idxs)

    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)
    

    return train_dataset, test_dataset, valid_dataset

def prepare_cifar10(data_dir, seed=42):

    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transforms_test = transforms.Compose([transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    train_valid_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transforms_train
        )

    train_idxs, valid_idxs = train_test_split(
        np.arange(len(train_valid_dataset)),
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=train_valid_dataset.targets,
    )
    train_dataset = Subset(train_valid_dataset, train_idxs)
    valid_dataset = Subset(train_valid_dataset, valid_idxs)

    test_dataset = datasets.CIFAR10( data_dir, train=False, download=True, transform=transforms_test)


    return train_dataset, test_dataset, valid_dataset

def prepare_utkface(self, seed=42):
    """
    Returns train/test/validation utkface datasets.

    :params data_dir: directory where the images are located
    :params zfile: relative path from home folder to the zip file stored under data
    :params extract_dir: main directory where the UTKFace folder will be stored
    :params apply_transform: image transformation for UTKFace

    :returns: train/test/validation utkface datasets that can be used for training UTKFace
    """
    generator = torch.Generator().manual_seed(seed)

    apply_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5959, 0.4562, 0.3906), (0.2591, 0.2312, 0.2268)
                ),
            ]
        )

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


    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, validate_len, test_len,], generator=generator)

    return train_dataset, test_dataset, valid_dataset

def prepare_synthetic(self, num_clients, num_classes, num_features, seed=42):
    train_dataset, test_dataset, valid_dataset = SyntheticDataset(num_clients, num_classes, num_features).split()
    return train_dataset, test_dataset, valid_dataset

class FLDataset(Dataset):
    def __init__(self, config):
        self.config = get_current_config()
        self.train_dataset, self.test_dataset, self.valid_dataset = self.get_dataset()

    @param('dataset.dataset_name')
    @param('training_params.seed')
    @param('dataset.data_dir')
    @param('fl_parameters.num_clients')
    @param('dataset.num_classes')
    @param('dataset.num_features')
    def get_dataset(self, dataset_name, seed, data_dir, num_clients, num_classes, num_features):
        if dataset_name.lower() == 'cifar10':
            return prepare_cifar10(data_dir, seed)
        elif dataset_name.lower() == 'fashionmnist':
            return prepare_fashionMNIST(data_dir, seed)
        elif dataset_name.lower() == 'utkface':
            return prepare_utkface(data_dir, seed)
        elif dataset_name.lower() == 'synthetic':
            return prepare_synthetic(data_dir, seed, num_clients, num_classes, num_features)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

    @param('split_params.split_type')
    def get_client_groups(self, split_type):
        if split_type == "iid":
            train_user_groups = self.get_iid_partition(dataset=self.train_dataset)
            valid_user_groups = self.get_iid_partition(dataset=self.valid_dataset)
            test_user_groups = self.get_iid_partition(dataset=self.test_dataset)

        elif split_type == "noniid":
            distribution = self.generate_noniid_distribution(self.train_dataset)
            train_user_groups = self.get_noniid_partition(self.train_dataset, distribution)
            valid_user_groups = self.get_noniid_partition(self.valid_dataset, distribution)
            test_user_groups = self.get_noniid_partition(self.test_dataset, distribution)


        elif split_type == "maj_min":
            train_user_groups = self.get_noniid_partition(self.train_dataset.targets, distribution)
            valid_user_groups = self.get_noniid_partition(self.valid_dataset.targets, distribution)
            test_user_groups = self.get_noniid_partition(self.test_dataset.targets, distribution)
        
            wandb.log(
                {
                    "majority_classes": majority_classes,
                    "minority_classes": minority_classes,
                    "majority_users": majority_users,
                    "minority_users": minority_users,
                }
            )

        else:
            raise ValueError(f"Split type {self.split_type} not supported")
        
        return train_user_groups, valid_user_groups, test_user_groups


    @param('fl_parameters.num_clients')
    def get_iid_partition(self, num_clients, dataset):
        """
        Sample I.I.D. client data from dataset
        :param dataset:
        :param num_clients:
        :return: dict of image index
        """
        num_items = int(len(dataset) / num_clients)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_clients):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users 

    @param('fl_parameters.num_clients')
    @param('dataset.num_classes')
    @param('split_params.dirichlet_param')
    @param('split_params.min_proportion')
    def generate_noniid_distribution(self, num_clients: int, num_classes: int, beta: float, min_proportion: float, dataset: Dataset):
        """
        Sample from dirichlet distribution to give non-iid distribution for users.

        :param num_clients: number of clients
        :param num_classes: number of total classes
        :param dataset_labels: list or tensor of shape (num_samples,)
        :param beta: determines amount of non-iid
        :param min_proportion: minimum proportion of the dataset per user

        :return: array of shape (num_classes, num_clients), where each row is a distribution
            over the users for a specific class
        """

        dataset_labels = dataset.targets
        if isinstance(dataset_labels, list):
            dataset_labels = torch.tensor(dataset_labels)
        class_weights = np.zeros((num_classes,))
        for class_number in range(num_classes):
            class_weights[class_number] = (dataset_labels == class_number).sum() / len(
                dataset_labels
            )

        if min_proportion >= 1 / num_clients:
            return ValueError(
                f"min_proportion per user must be less than {1/num_clients} for a dataset with {num_clients} in it"
            )
        class_sample_distribution = np.zeros((num_classes, num_clients))
        dataset_proportion_per_user = np.zeros(num_clients)
        while dataset_proportion_per_user.min() <= min_proportion:
            class_sample_distribution = np.random.dirichlet(
                np.repeat(beta, num_clients), num_classes
            )
            dataset_proportion_per_user = (class_weights @ class_sample_distribution) / (
                class_weights.sum()
            )

        return class_sample_distribution

    @param('fl_parameters.num_clients')
    @param('dataset.num_classes')
    @param('split_params.majority_proportion')
    @param('split_params.majority_minority_overlap')
    def split_majority_minority(self, num_clients: int, num_classes: int, majority_proportion: float, overlap: float):
        """
        Split the classes and users into a majority and minority group

        :param num_clients: The number of users
        :param num_classes: The number of classes
        :param majority proportion: the proportion of users in the majority group
            and the proportion of classes in the majority group
        :param overlap: the extent of overlap between the majority and minority groups
            when 0 there is no overlap, when 1 there is complete overlap and the majority group=minority group

        :return: A Tuple consisting of:
            - A numpy array of shape (num_classes, num_clients) giving a distribution of users over each class.
            - A numpy array denoting the majority classes
            - A numpy array denoting the minority classes
            - A numpy array denoting the majority users
            - A numpy array denoting the minority users
        """

        # To ensure each user should have roughly the same number of samples, 'majority_proportion' is used to
        # split both the classes and users into majority and minority groups, with the majority users more likely
        # to possess samples from the majority classes than the minority classes.
        num_majority_classes = round(majority_proportion * num_classes)
        num_majority_users = round(majority_proportion * num_clients)
        num_minority_users = num_clients - num_majority_users

        # We define a distribution over the users for each class. This will tell us how to probabilistically distribute
        # the samples of that class to the users using 'get_noniid_partition'. The ith row of the 'distribution'
        # object corresponds to the distribution for the ith class.
        distribution = np.zeros((num_classes, num_clients))

        # The probability of a sample from a majority or minority class being assigned to either a majority or minority
        # user, is determined by 'overlap'. 'overlap' represents the probability that a sample is assigned uniformly
        # at random across all users, rather than being assigned uniformly across it's group (ie majority or minority).

        uniform_distribution = np.ones((num_classes, num_clients)) / num_clients

        grouped_distribution = np.zeros((num_classes, num_clients))
        grouped_distribution[:num_majority_classes, :num_majority_users] = (
            1 / num_majority_users
        )
        grouped_distribution[num_majority_classes:, num_majority_users:] = (
            1 / num_minority_users
        )
        # this sums to 1 across users for each class, because uniform_distribution and grouped_distribution both
        # sum to 1 across users for each class
        distribution = uniform_distribution * overlap + (1 - overlap) * grouped_distribution
        assert (
            distribution.sum(axis=1).round(3) == 1
        ).all(), "distribution must sum to 1 across users for each class"

        # The order of the users doesn't matter, however which classes are chosen for each groups is important and must be able to vary.
        permutation = np.random.permutation(num_classes)

        return (
            distribution[np.argsort(permutation)],
            permutation[:num_majority_classes],
            permutation[num_majority_classes:],
            np.arange(num_majority_users),
            np.arange(num_majority_users, num_clients),
        )
            


        














