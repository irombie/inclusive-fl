from collections import defaultdict

import numpy as np
import torch
from fastargs import get_current_config
from fastargs.decorators import param
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from general_utils import normalize
from harness_params import get_current_params
from src.dataset_defs import SyntheticDataset, UTKFaceDataset


get_current_params()


class FLDataset(Dataset):
    def __init__(self):
        self.config = get_current_config()
        self.train_dataset, self.test_dataset, self.valid_dataset = self.get_dataset()

    @param("dataset.dataset_name")
    @param("training_params.seed")
    @param("dataset.data_dir")
    @param("fl_parameters.num_clients")
    @param("dataset.num_classes")
    @param("dataset.num_features")
    def get_dataset(self, dataset_name, seed, data_dir, num_clients, num_classes, num_features):
        if dataset_name.lower() == "cifar10":
            return prepare_cifar10(data_dir, seed=seed)
        elif dataset_name.lower() == "fashionmnist":
            return prepare_fashionMNIST(data_dir, seed=seed)
        elif dataset_name.lower() == "utkface":
            return prepare_utkface(data_dir, seed=seed)
        elif dataset_name.lower() == "synthetic":
            return prepare_synthetic(
                data_dir,
                seed=seed,
                num_clients=num_clients,
                num_classes=num_classes,
                num_features=num_features,
            )
        elif dataset_name.lower() == "svhn":
            return prepare_SVHN(
                data_dir,
                seed=seed,
                extra=self.config.get().SVHN_data.extra,
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @param("split_params.split_type")
    @param("split_params.combine_train_val")
    def get_client_groups(self, split_type, combine_train_val):
        valid_user_groups = None
        if split_type == "iid":
            train_user_groups = self.get_iid_partition(dataset=self.train_dataset)
            test_user_groups = self.get_iid_partition(dataset=self.test_dataset)
            if not combine_train_val:
                valid_user_groups = self.get_iid_partition(dataset=self.valid_dataset)

        elif split_type == "non-iid":
            distribution = self.generate_noniid_distribution(dataset=self.train_dataset)
            train_user_groups = self.get_noniid_partition(dataset=self.train_dataset, distribution=distribution)
            test_user_groups = self.get_noniid_partition(dataset=self.test_dataset, distribution=distribution)
            if not combine_train_val:
                valid_user_groups = self.get_noniid_partition(dataset=self.valid_dataset, distribution=distribution)

        elif split_type == "majority_minority":
            (
                distribution,
                majority_classes,
                minority_classes,
                majority_users,
                minority_users,
            ) = self.split_majority_minority()

            train_user_groups = self.get_noniid_partition(dataset=self.train_dataset, distribution=distribution)
            test_user_groups = self.get_noniid_partition(dataset=self.test_dataset, distribution=distribution)
            if not combine_train_val:
                valid_user_groups = self.get_noniid_partition(dataset=self.valid_dataset, distribution=distribution)

            """wandb.log(
                {
                    "majority_classes": majority_classes,
                    "minority_classes": minority_classes,
                    "majority_users": majority_users,
                    "minority_users": minority_users,
                }

            )
            """

        else:
            raise ValueError(f"Split type {split_type} not supported")

        return train_user_groups, test_user_groups, valid_user_groups

    @param("fl_parameters.num_clients")
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

    @param("fl_parameters.num_clients")
    @param("dataset.num_classes")
    def get_noniid_partition(self, num_clients, num_classes, dataset, distribution):
        """
        Get samples assigned to each user

        :param num_clients: number of clients
        :param num_classes: number of classes
        :param dataset: dataset object
        :param distribution: array of shape (num_classes, num_clients), where each row is a distribution
            over the users for a specific class

        :return: dictionary where each key is a user index, and each item is a list of sample idxs
            for that user
        """

        dataset_labels = dataset.targets
        if isinstance(dataset_labels, list):
            dataset_labels = torch.tensor(dataset_labels)
        sample_idxs = [torch.where(dataset_labels == i)[0] for i in range(num_classes)]
        users_data = defaultdict(list)
        for i in range(num_classes):
            num_class_samples = len(sample_idxs[i])
            sample_user_idx = np.random.choice(num_clients, num_class_samples, p=distribution[i])
            for user_idx, sample_idx in zip(sample_user_idx, sample_idxs[i]):
                users_data[user_idx].append(sample_idx.item())

        return users_data

    @param("fl_parameters.num_clients")
    @param("dataset.num_classes")
    @param("split_params.dirichlet_param")
    @param("split_params.min_proportion")
    def generate_noniid_distribution(
        self,
        num_clients: int,
        num_classes: int,
        dirichlet_param: float,
        min_proportion: float,
        dataset: Dataset,
    ):
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
            class_weights[class_number] = (dataset_labels == class_number).sum() / len(dataset_labels)

        if min_proportion >= 1 / num_clients:
            raise ValueError(
                f"min_proportion per user must be less than {1 / num_clients} for a dataset with {num_clients} in it"
            )
        class_sample_distribution = np.zeros((num_classes, num_clients))
        dataset_proportion_per_user = np.zeros(num_clients)
        while dataset_proportion_per_user.min() <= min_proportion:
            class_sample_distribution = np.random.dirichlet(np.repeat(dirichlet_param, num_clients), num_classes)
            dataset_proportion_per_user = (class_weights @ class_sample_distribution) / (class_weights.sum())

        return class_sample_distribution

    @param("fl_parameters.num_clients")
    @param("dataset.num_classes")
    @param("split_params.majority_proportion")
    @param("split_params.overlap")
    def split_majority_minority(
        self,
        num_clients: int,
        num_classes: int,
        majority_proportion: float,
        overlap: float,
    ):
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
        # object correspondataset to the distribution for the ith class.
        distribution = np.zeros((num_classes, num_clients))

        # The probability of a sample from a majority or minority class being assigned to either a majority or minority
        # user, is determined by 'overlap'. 'overlap' represents the probability that a sample is assigned uniformly
        # at random across all users, rather than being assigned uniformly across it's group (ie majority or minority).

        uniform_distribution = np.ones((num_classes, num_clients)) / num_clients

        grouped_distribution = np.zeros((num_classes, num_clients))
        grouped_distribution[:num_majority_classes, :num_majority_users] = 1 / num_majority_users
        grouped_distribution[num_majority_classes:, num_majority_users:] = 1 / num_minority_users
        # this sums to 1 across users for each class, because uniform_distribution and grouped_distribution both
        # sum to 1 across users for each class
        distribution = uniform_distribution * overlap + (1 - overlap) * grouped_distribution
        assert (distribution.sum(axis=1).round(3) == 1).all(), "distribution must sum to 1 across users for each class"

        # The order of the users doesn't matter, however which classes are chosen
        # for each groups is important and must be able to vary.
        permutation = np.random.permutation(num_classes)

        return (
            distribution[np.argsort(permutation)],
            permutation[:num_majority_classes],
            permutation[num_majority_classes:],
            np.arange(num_majority_users),
            np.arange(num_majority_users, num_clients),
        )


@param("split_params.combine_train_val")
def prepare_fashionMNIST(data_dir, combine_train_val, seed=42):
    apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_valid_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)

    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)

    if combine_train_val:
        return train_valid_dataset, test_dataset, None

    train_idxs, valid_idxs = train_test_split(
        np.arange(len(train_valid_dataset)),
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=train_valid_dataset.targets,
    )

    train_dataset = Subset(train_valid_dataset, train_idxs)
    valid_dataset = Subset(train_valid_dataset, valid_idxs)
    train_dataset.targets = train_valid_dataset.targets[train_idxs]
    valid_dataset.targets = train_valid_dataset.targets[valid_idxs]

    return train_dataset, test_dataset, valid_dataset


@param("split_params.combine_train_val")
def prepare_cifar10(data_dir, combine_train_val, seed=42):
    transforms_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transforms_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_valid_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms_train)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms_test)

    if combine_train_val:
        return train_valid_dataset, test_dataset, None

    train_idxs, valid_idxs = train_test_split(
        np.arange(len(train_valid_dataset)),
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=train_valid_dataset.targets,
    )

    train_dataset = Subset(train_valid_dataset, train_idxs)
    valid_dataset = Subset(train_valid_dataset, valid_idxs)

    train_dataset.targets = torch.tensor(train_valid_dataset.targets)[train_idxs]
    valid_dataset.targets = torch.tensor(train_valid_dataset.targets)[valid_idxs]

    return train_dataset, test_dataset, valid_dataset


@param("split_params.combine_train_val")
@param("dataset.data_dir")
@param("dataset.zfile")
@param("dataset.extract_dir")
@param("dataset.label_type")
def prepare_utkface(self, combine_train_val, data_dir, zfile, extract_dir, label_type="ethnicity", seed=42):
    """
    Returns train/test/validation utkface datasets.

    :params data_dir: directory where the images are located
    :params zfile: relative path from home folder to the zip file stored under data
    :params extract_dir: main directory where the UTKFace folder will be stored
    :params apply_transform: image transformation for UTKFace
    :params label_type: type of label to use (default: ethnicity)
    :params seed: random seed

    :returns: train/test/validation utkface datasets that can be used for training UTKFace
    """
    # Remove unused generator variable
    # generator = torch.Generator().manual_seed(seed)

    apply_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5959, 0.4562, 0.3906), (0.2591, 0.2312, 0.2268)),
        ]
    )

    dataset = UTKFaceDataset(
        directory=data_dir,
        zfile=zfile,
        extract_dir=extract_dir,
        transform=apply_transform,
        label_type=label_type,
    )

    train_idxs, valid_idxs = train_test_split(
        np.arange(len(dataset)),
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=dataset.targets,
    )

    train_dataset = Subset(dataset, train_idxs)
    valid_dataset = Subset(dataset, valid_idxs) if not combine_train_val else None

    new_train_idxs, test_idxs = train_test_split(
        np.arange(len(train_dataset)),
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=train_dataset.targets,
    )

    test_dataset = Subset(train_dataset, test_idxs)

    if combine_train_val:
        train_dataset = Subset(train_dataset, train_idxs)
        train_dataset.targets = torch.tensor(dataset.labels)[train_idxs]
    else:
        train_dataset = Subset(train_dataset, new_train_idxs)
        train_dataset.targets = torch.tensor(dataset.labels)[new_train_idxs]

    valid_dataset.targets = torch.tensor(dataset.labels)[valid_idxs]
    test_dataset.targets = torch.tensor(dataset.labels)[test_idxs]

    return train_dataset, test_dataset, valid_dataset


@param("split_params.combine_train_val")
def prepare_synthetic(self, num_clients, num_classes, num_features, combine_train_val, seed=42):
    train_dataset, test_dataset, valid_dataset = SyntheticDataset(num_clients, num_classes, num_features).split()
    return train_dataset, test_dataset, valid_dataset


@param("split_params.combine_train_val")
def prepare_SVHN(data_dir, combine_train_val, extra=False, seed=42):
    apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: normalize(x))])

    train_valid_dataset = datasets.SVHN(data_dir, split="train", download=True, transform=apply_transform)
    test_dataset = datasets.SVHN(data_dir, split="test", download=True, transform=apply_transform)

    """if extra:
        extra_train_dataset = datasets.SVHN(
            data_dir, split="extra", download=True, transform=apply_transform
        )
        # NB: ConcatDataset preserves the order and just adjusts the index accordingly.
        train_valid_dataset = ConcatDataset([train_valid_dataset, extra_train_dataset])
        full_train_labels = np.concatenate(
            (full_train_labels, extra_train_dataset.labels)
        )
    """
    if combine_train_val:
        return train_valid_dataset, test_dataset, None

    train_idxs, valid_idxs = train_test_split(
        np.arange(len(train_valid_dataset)),
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=train_valid_dataset.labels,
    )

    train_dataset = Subset(train_valid_dataset, train_idxs)
    valid_dataset = Subset(train_valid_dataset, valid_idxs)

    train_dataset.targets = torch.tensor(train_valid_dataset.labels)[train_idxs]
    valid_dataset.targets = torch.tensor(train_valid_dataset.labels)[valid_idxs]

    test_dataset = datasets.SVHN(data_dir, split="test", download=True, transform=apply_transform)

    test_dataset.targets = torch.tensor(test_dataset.labels)

    return train_dataset, test_dataset, valid_dataset
