#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from pathlib import Path
from typing import Tuple

import json
import os
import shutil
import sys
import tarfile
import zipfile
import gdown

from fastargs import get_current_config
from fastargs.decorators import param
from parse import parse
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import numpy as np
import torch
import wandb

from harness_params import get_current_params

get_current_params()

class UTKFaceDataset(Dataset):
    def __init__(
        self, directory, zfile, extract_dir, transform, label_type="ethnicity"
    ):
        """
        ** DATASET HAS TO BE DOWNLOADED FIRST**
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
                ## ignore age values larger 120 and gender values that are not 0 or 1 -- this is just to ensure there are no errors
                ## does not come up as the current dataset only supports the ethnicity task
                if int(file_labels["age"]) > 120 or int(file_labels["gender"]) > 1:
                    continue

                image = Image.open(os.path.join(extract_dir + "/UTKFace", file))
                image = self.transform(image)

                self.images.append(image)
                self.labels.append(float(file_labels["ethnicity"]))

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

    def fetch_synthetic_data(
        self, num_clients, num_classes, num_features, url: str, path: str | Path
    ):
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

    @param("split_params.combine_train_val")
    def split(
        self, combine_train_val: bool = False, valid_ratio: float = 0.1, test_ratio: float = 0.1
    ) -> tuple["SyntheticDataset", "SyntheticDataset", "SyntheticDataset"]:
        """
        Split the data into training and validation sets.

        :param combine_train_val: whether to combine the training and validation sets
        :param valid_ratio: proportion of the data to include in the validation set
        :param test_ratio: proportion of the data to include in the test set
        :return: training and validation datasets
        """
        X_train, y_train = [], []
        X_test, y_test = [], []

        if combine_train_val:
            for idx in self.user_idx.values():
                test_size = int(len(idx) * test_ratio)
                train_idx = idx[: -test_size]
                test_idx = idx[-test_size:]

                X_train.append(self.X[train_idx])
                y_train.append(self.y[train_idx])
                X_test.append(self.X[test_idx])
                y_test.append(self.y[test_idx])

            train_dataset = SyntheticDataset(X_train, y_train)
            test_dataset = SyntheticDataset(X_test, y_test)
            return train_dataset, test_dataset
        else:
            X_valid, y_valid = [], []
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

    def normalize_tin_val_folder_structure(
        self, path, images_folder="images", annotations_file="val_annotations.txt"
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
        self.normalize_tin_val_folder_structure(
            os.path.join(self.dataset_folder, "val")
        )




