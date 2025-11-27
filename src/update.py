#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.11

import copy
from typing import Dict, OrderedDict, Type

from fastargs import get_current_config
from fastargs.decorators import param
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import general_utils
from harness_params import get_current_params

get_current_params()


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
        self, train_dataset, test_dataset, valid_dataset, train_idxs, test_idxs, valid_idxs, logger, global_model, proportion
    ):

        self.config = get_current_config()
        self.logger = logger
        self.proportion = proportion

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_built() else "cpu")
        )

        self.criterion = nn.NLLLoss().to(self.device)

        self.global_model = global_model
        self.train_dataset, self.test_dataset, self.valid_dataset = (
            train_dataset,
            test_dataset,
            valid_dataset,
        )
        self.train_idxs, self.test_idxs, self.valid_idxs = (
            train_idxs,
            test_idxs,
            valid_idxs,
        )
        self.trainloader, self.testloader, self.validloader = self.get_local_loaders()

    @param("client_parameters.local_bs")
    @param("split_params.combine_train_val")
    def get_local_loaders(self, local_bs, combine_train_val):
        if combine_train_val:
            n_samples = torch.ceil(torch.tensor(len(self.train_idxs) * self.proportion)).int()
            train_idx = torch.randperm(len(self.train_idxs))[:n_samples]
        else:
            train_idx = self.train_idxs
        self.client_train_dataset = Subset(self.train_dataset, list(train_idx))
        self.client_test_dataset = Subset(self.test_dataset, list(self.test_idxs))
        if not combine_train_val:
            self.client_valid_dataset = Subset(self.train_dataset, list(self.valid_idxs))

        trainloader = DataLoader(
            self.client_train_dataset,
            batch_size=local_bs,
            shuffle=True,
        )
        testloader = DataLoader(
            self.client_test_dataset,
            batch_size=local_bs,
            shuffle=False,
        )
        validloader = (
            None
            if combine_train_val
            else DataLoader(
                self.client_valid_dataset,
                batch_size=local_bs,
                shuffle=False,
            )
        )

        return trainloader, testloader, validloader

    @param("client_parameters.local_lr")
    def configure_optimizer(self, local_lr, model):
        """
        Configures the optimizer for the local updates.
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=local_lr, momentum=0.5)  # 0.5 momentum? why?
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

    @param("client_parameters.local_epochs")
    def update_weights(self, local_epochs, model, global_round):
        """
        Performs the local updates and returns the updated model.
            :param model: local model
            :param global_round: the step number of current global round
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = self.configure_optimizer(model=model)

        for iter in range(local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                loss = self.calculate_loss(model=model, images=images, labels=labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
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

        for images, labels in loader:
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
    @param("client_parameters.local_epochs")
    @param("fl_parameters.sparsification_type")
    @param("fl_parameters.choose_from_top_r_percentile")
    def update_weights(
        self,
        local_epochs,
        model,
        global_round,
        sparsification_ratio,
        sparsification_type,
        choose_from_top_r_percentile,
    ):
        """
        Performs the local updates and returns the updated model.
            :param model: local model
            :param global_round: the step number of current global round
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = self.configure_optimizer(model=model)
        glob_flat = general_utils.flatten(model)

        for iter_ in range(local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                loss = self.calculate_loss(model, images, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print(
                        "| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            global_round,
                            iter_,
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

        sparse_ratio = sparsification_ratio

        flat = general_utils.flatten(model)
        diff_flat = flat - glob_flat
        bitmask = general_utils.get_bitmask_per_method(
            flat_model=diff_flat,
            sparse_ratio=sparse_ratio,
            sparsification_type=sparsification_type,
            choose_from_top_r_percentile=choose_from_top_r_percentile,
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
        mu = self.config["fl_parameters.mu"]
        if mu is None:
            raise ValueError("mu argument must be passed as arugument for fl_method=FedProx")
        fedprox_term = 0.0
        proximal_term = 0.0

        for w, w_t in zip(model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        fedprox_term = (mu / 2) * proximal_term
        return super().calculate_loss(model, images, labels) + fedprox_term


class qFedAvgLocalUpdate(LocalUpdate):
    """
    qFedAvg Local Update. This is a subclass of LocalUpdate. It overrides the
    calculate_loss method to include the updated qFedAvg Loss.
    Reference: https://arxiv.org/pdf/1905.10497.pdf
    """

    @param("fl_parameters.q")
    @param("client_parameters.local_lr")
    @param("fl_parameters.eps")
    @param("client_parameters.local_epochs")
    def update_weights(self, q, local_lr, eps, local_epochs, model, global_round):
        """
        Performs the qFedAvg local updates and returns the updated model.
            :param model: local model
            :param global_round: the step number of current global round
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = self.configure_optimizer(model=model)

        base_model = copy.deepcopy(model.state_dict())

        training_losses = []
        for iter in range(local_epochs):
            batch_loss = []
            train_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                loss = self.calculate_loss(model, images, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

                if batch_idx % 10 == 0:
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
            if q is None:
                raise ValueError("q argument must be passed as argument for fl_method=qFedAvg")
            F += eps
            Fq = torch.pow(torch.tensor(F, dtype=torch.float32), q)
            L = 1.0 / local_lr

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
                h[key] = Fq * ((q * torch.norm(torch.flatten(delta_weights[key]), 2) ** 2) / F + L)

                for dim in model.state_dict()[key].shape:
                    size *= dim
                h_expanded[key] = torch.full((size,), h[key])

            return (
                general_utils.flatten(delta, is_dict=True),
                general_utils.flatten(h_expanded, is_dict=True),
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


def test_inference(model, test_dataset):
    """Returns the test accuracy and loss."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_built() else "cpu"))

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for images, labels in testloader:
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
    fl_method, train_dataset, test_dataset, valid_dataset, train_idxs, test_idxs, valid_idxs, logger, global_model, proportion
) -> LocalUpdate:
    """
    Get local update from federated learning method name and return the
    local update object.


    :param train_dataset: Dataset object containing the training data
    :param test_dataset: Dataset object containing the test data

    :param idxs: List of indices of the training data assigned to the
                    local update

    :param logger: Logger object to log the local update
    :return: Local update object
    """
    if fl_method in NAME_TO_LOCAL_UPDATE:
        return NAME_TO_LOCAL_UPDATE[fl_method](
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            valid_dataset=valid_dataset,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            valid_idxs=valid_idxs,
            logger=logger,
            global_model=global_model,
            proportion=proportion,
        )
    raise ValueError(f"Unsupported federated learning method name {fl_method} for local update.")
