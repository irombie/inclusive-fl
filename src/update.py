#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from typing import Dict, OrderedDict, Type

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        try:
            image, label = self.dataset[self.idxs[item]]
        except:
            # print("IN EXCEPT -- OR UTKFACE")
            image = self.dataset.dataset.images[self.idxs[item]]
            label = [d["ethnicity"] for d in self.dataset.dataset.labels][
                self.idxs[item]
            ]
        return image, label


class LocalUpdate:
    """
    A base class for local updates. In the training process of most federated
    learning algorithms, each client will perform a local update on its local
    dataset. This class provides a common interface for local updates.

    Many FedLearn algos share common set of steps which are implemented in
    this class. If a FedLearn algo requires a different set of steps, it can
    override the methods in this class.
    """

    def __init__(
        self,
        args,
        train_dataset,
        test_dataset,
        valid_dataset,
        train_idxs,
        test_idxs,
        valid_idxs,
        logger,
        global_model,
    ):
        self.args = args
        self.logger = logger
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs
        self.trainloader = DataLoader(
            DatasetSplit(train_dataset, train_idxs),
            batch_size=self.args.local_bs,
            shuffle=True,
        )
        self.testloader = DataLoader(
            DatasetSplit(test_dataset, test_idxs),
            batch_size=self.args.local_bs,
            shuffle=False,
        )
        self.validloader = DataLoader(
            DatasetSplit(valid_dataset, valid_idxs),
            batch_size=self.args.local_bs,
            shuffle=False,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_built() else 'cpu'))

        self.criterion = nn.NLLLoss().to(self.device)

        self.global_model = global_model

    def configure_optimizer(self, model):
        """
        Configures the optimizer for the local updates.
        """
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.99
        )
        return optimizer

    def calculate_loss(self, model, images, labels):
        """
        Calculates the loss for the local updates.
            :param model: local model
            :param images: images from the local dataset
            :param labels: labels from the local dataset

            :return loss: the loss value
        """
        log_probs = model(images)

        loss = self.criterion(log_probs, labels)
        return loss

    def update_weights(self, model, global_round, client_id=None):
        """
        Performs the local updates and returns the updated model.
            :param model: local model
            :param global_round: the step number of current global round
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = self.configure_optimizer(model)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                loss = self.calculate_loss(model, images, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                batch_loss.append(loss.item())
            avg_loss_per_local_training = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(avg_loss_per_local_training)
            # self.logger.log({f'local model train loss for user {self.user_id} ': avg_loss_per_local_training})

        return model, sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, dataset_type: str):
        """Returns the inference accuracy and loss."""
        if dataset_type == "test":
            loader = self.testloader
        elif dataset_type == "train":
            loader = self.trainloader
        elif dataset_type == "valid":
            loader = self.validloader
        else:
            raise ValueError("dataset_type must be one of test, train, valid")
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)

            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss / len(loader)


class LocalUpdateSparsified(LocalUpdate):
    def __init__(
        self,
        args,
        train_dataset,
        valid_dataset,
        test_dataset,
        train_idxs,
        valid_idxs,
        test_idxs,
        logger,
        global_model,
    ):
        super().__init__(
            args,
            train_dataset,
            valid_dataset,
            test_dataset,
            train_idxs,
            valid_idxs,
            test_idxs,
            logger,
            global_model,
        )
        self.sparsification_ratio = args.sparsification_ratio

    def update_weights(self, model, global_round, client_id=None, **kwargs):
        """
        Performs the local updates and returns the updated model.
            :param model: local model
            :param global_round: the step number of current global round
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = self.configure_optimizer(model)
        glob_flat = utils.flatten(model)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                loss = self.calculate_loss(model, images, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                batch_loss.append(loss.item())
            avg_loss_per_local_training = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(avg_loss_per_local_training)
            # self.logger.log({f'local model train loss for user {self.user_id} ': avg_loss_per_local_training})

        sparse_ratio = self.sparsification_ratio
        if kwargs["sparsification_percentage"] is not None:
            sparse_ratio = kwargs["sparsification_percentage"]
        flat = utils.flatten(model)
        diff_flat = flat - glob_flat
        bitmask = utils.get_bitmask_per_method(
            flat_model=diff_flat,
            sparse_ratio=sparse_ratio,
            sparsification_type=self.args.sparsification_type,
            choose_from_top_r_percentile=self.args.choose_from_top_r_percentile,
        )
        diff_flat *= bitmask
        return model, diff_flat, bitmask, sum(epoch_loss) / len(epoch_loss)


class FedProxLocalUpdate(LocalUpdate):
    """
    FedProx Local Update. This is a subclass of LocalUpdate. It overrides the
    calculate_loss method to include the proximal term.
    """

    def calculate_loss(self, model, images, labels):
        """
        The proximal term is added to the loss function. The proximal term is
        calculated as:
            proximal_term = Σ(||w - w_t||^2)
        """
        if self.args.mu is None:
            raise ValueError(
                "mu argument must be passed as arugument for fl_method=FedProx"
            )
        fedprox_term = 0.0
        proximal_term = 0.0

        for w, w_t in zip(model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)

        fedprox_term = (self.args.mu / 2) * proximal_term
        return super().calculate_loss(model, images, labels) + fedprox_term


class qFedAvgLocalUpdate(LocalUpdate):
    """
    qFedAvg Local Update. This is a subclass of LocalUpdate. It overrides the
    calculate_loss method to include the updated qFedAvg Loss.
    Reference: https://arxiv.org/pdf/1905.10497.pdf
    """

    def update_weights(self, model, global_round, client_id=None):
        """
        Performs the qFedAvg local updates and returns the updated model.
            :param model: local model
            :param global_round: the step number of current global round
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = self.configure_optimizer(model)

        base_model = copy.deepcopy(model.state_dict())

        training_losses = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            train_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                loss = self.calculate_loss(model, images, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                batch_loss.append(loss.item())
            avg_loss_per_local_training = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(avg_loss_per_local_training)

            train_loss /= len(self.trainloader.dataset)
            training_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            # qFedAvg update to the model
            F = sum(training_losses) / len(training_losses)
            if self.args.q is None:
                raise ValueError(
                    "q argument must be passed as argument for fl_method=qFedAvg"
                )
            F += self.args.eps
            Fq = torch.pow(torch.tensor(F, dtype=torch.float32), self.args.q)
            L = 1.0 / self.args.lr

            delta_weights, delta, h, h_expanded = (
                OrderedDict(),
                OrderedDict(),
                OrderedDict(),
                OrderedDict(),
            )

            for key in list(model.state_dict().keys()):
                # Line 6 calculations qFedAvg algorithm
                delta_weights[key] = L * (base_model[key] - model.state_dict()[key])
                delta[key] = Fq * delta_weights[key]

                # Lemma 3 in the qFedAvg paper provides the connection between the Local
                # Lipchitz constant at q=0 and when q>0. It is used to estimate the learning rate
                # as the learning rate is set as the inverse of lipschitz constant.
                size = 1
                h[key] = Fq * (
                    (
                        self.args.q
                        * torch.linalg.norm(torch.flatten(delta_weights[key]), ord=2)
                        ** 2
                    )
                    / F
                    + L
                )

                for dim in model.state_dict()[key].shape:
                    size *= dim
                h_expanded[key] = torch.full((size,), h[key])

            return (
                utils.flatten(delta, is_dict=True),
                utils.flatten(h_expanded, is_dict=True),
                model,
                sum(epoch_loss) / len(epoch_loss),
            )


NAME_TO_LOCAL_UPDATE: Dict[str, Type[LocalUpdate]] = {
    "FedAvg": LocalUpdate,
    "FedProx": FedProxLocalUpdate,
    "FedBN": LocalUpdate,
    "TestLossWeighted": LocalUpdate,
    "FedSyn": LocalUpdateSparsified,
    "qFedAvg": qFedAvgLocalUpdate,
}


def test_inference(args, model, test_dataset):
    """Returns the test accuracy and loss."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_built() else 'cpu'))

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss / len(testloader)


def get_local_update(
    args,
    train_dataset,
    test_dataset,
    valid_dataset,
    train_idxs,
    test_idxs,
    valid_idxs,
    logger,
    global_model,
) -> LocalUpdate:
    """
    Get local update from federated learning method name and return the
    local update object.

    :param args: Arguments object containing configurations passed
                    as arguments to the program call

    :param train_dataset: Dataset object containing the training data
    :param test_dataset: Dataset object containing the test data

    :param idxs: List of indices of the training data assigned to the
                    local update

    :param logger: Logger object to log the local update
    :return: Local update object
    """
    if args.fl_method in NAME_TO_LOCAL_UPDATE:
        return NAME_TO_LOCAL_UPDATE[args.fl_method](
            args=args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            valid_dataset=valid_dataset,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            valid_idxs=valid_idxs,
            logger=logger,
            global_model=global_model,
        )
    else:
        raise ValueError(
            f"Unsupported federated learning method name {args.fl_method} for local update."
        )
