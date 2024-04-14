# -*- coding: utf-8 -*-
# Python version: 3.11

import copy
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import wandb
from models import (
    VGG,
    CNN_FashionMNIST,
    ResNet9,
    ResNet18,
    SmallCNN,
    MLP,
    LogisticRegression,
)

from general_utils import (
    custom_exponential_sparsity,
    flatten,
    set_seed,
    updateFromNumpyFlatArray,
)

from dataset_utils import FLDataset

from fastargs import get_current_config
from fastargs.decorators import param, section
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser

from harness_params import get_current_params

from global_updates import get_global_update 
from update import get_local_update, test_inference

get_current_params()


if sys.version_info[0:2] != (3, 11):
        print()
        raise RuntimeError(
            f"Code requires python 3.11. You are using {sys.version_info[0:2]}. Please update your conda env and install requirements.txt on the new env."
        )

class FLTrainingHarness:
    def __init__(self):
        self.config = get_current_config()
        self.device =  torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_built() else "cpu"))
        self.global_model = self.init_global_model()
        self.train_user_groups, self.test_user_groups, self.valid_user_groups = self.get_data_splits()
        self.global_update = self.get_global_update()
        

    @param('model.model_name')
    @param('dataset.num_classes')
    @param('dataset.num_features')
    def init_global_model(self, model_name, num_classes, num_features=None):
        model_definition = globals()[model_name]

        if num_features != 0:
            model = model_definition(num_classes=num_classes, num_features=num_features)
        else:
            model = model_definition(num_classes=num_classes)
        model.to(self.device)
 
        return model

    def get_data_splits(self):
        main_ds = FLDataset()
        main_ds.get_dataset()
        self.train_dataset, self.test_dataset, self.valid_dataset = main_ds.train_dataset, main_ds.test_dataset, main_ds.valid_dataset
        train_user_groups, test_user_groups, valid_user_groups = main_ds.get_client_groups()

        return train_user_groups, test_user_groups, valid_user_groups
    
    @param('fl_parameters.fl_method')
    def get_global_update(self, fl_method):
        global_update = get_global_update(fl_method=fl_method, model=self.global_model)
        return global_update
    
    @param('fl_parameters.fl_method')
    def get_local_update(self, fl_method, client_idx):
        local_update = get_local_update(
            fl_method=fl_method,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            valid_dataset=self.valid_dataset,
            train_idxs=self.train_user_groups[client_idx],
            test_idxs=self.test_user_groups[client_idx],
            valid_idxs=self.valid_user_groups[client_idx],
            logger=False,
            global_model=self.global_model,
        )

        return local_update

    @param('fl_parameters.use_fair_sparsification')
    @param('fl_parameters.sparsification_ratio')
    def run_fedsyn_client(self, use_fair_sparsification, sparsification_ratio, client_idx, local_update, epoch):
        
        local_model = copy.deepcopy(self.global_model)
        if use_fair_sparsification:
            assert self.client_prob_dist is not None
            sparsification_ratio = self.client_prob_dist[client_idx]


        w, flat_update, bitmask, _ = local_update.update_weights(model=local_model, sparsification_ratio=sparsification_ratio, global_round=epoch)
        
        acc, loss = local_update.inference(model=w, dataset_type="train")

        return w, flat_update, bitmask, loss, acc, local_model

    def run_qfedavg_client(self, local_update, epoch):
        local_model = copy.deepcopy(self.global_model)
        delta, h, w, _ = local_update.update_weights(model=local_model, global_round=epoch)
        acc, loss = local_update.inference(model=w, dataset_type="train")
        return delta, h, w, loss, acc, local_model

    def run_generic_client(self, local_update, epoch):
        local_model = copy.deepcopy(self.global_model)
        w, _ = local_update.update_weights(model=local_model, global_round=epoch)
        acc, loss = local_update.inference(model=w, dataset_type="train")
        return w, loss, acc, local_model

    @param('fl_parameters.num_clients')
    @param('fl_parameters.frac')
    @param('fl_parameters.fl_method')
    @param('fl_parameters.use_fair_sparsification')
    def train_global_round(self, num_clients, frac, fl_method, use_fair_sparsification, epoch):
        global_flat = flatten(self.global_model)
        m = max(int(frac * num_clients), 1)

        self.idxs_users = list(np.random.choice(range(num_clients), m, replace=False))
        
        local_weights_sum, local_bitmasks_sum, local_delta_sum, local_h_sum = (
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
        )

        valid_losses = self.compute_client_prob_dist()

        client_metrics = {i : {'train_loss': [], 'train_acc' : [], 'test_loss' : [], 'test_acc' : []} for i in range(num_clients)}

        for client_idx in self.idxs_users:
            local_update = self.get_local_update(client_idx=client_idx)
            if fl_method == "FedSyn":
                w, flat_update, bitmask, loss, acc, local_model = self.run_fedsyn_client(client_idx=client_idx, local_update=local_update, epoch=epoch)
                local_weights_sum += flat_update
                local_bitmasks_sum += bitmask
                
            elif fl_method == "qFedAvg":
                delta, h, w, loss, acc, local_model = self.run_qfedavg_client(local_update=local_update, epoch=epoch)
                local_delta_sum += delta
                local_h_sum += h
                local_bitmasks_sum += np.ones_like(local_bitmasks_sum)
            else:
                w, loss, acc, local_model = self.run_generic_client(local_update=local_update, epoch=epoch)
                local_weights_sum += flatten(local_model)
                local_bitmasks_sum += np.ones_like(local_bitmasks_sum)

            test_acc, test_loss = local_update.inference(model=w, dataset_type="test")

            client_metrics[client_idx]['train_loss'].append(loss)
            client_metrics[client_idx]['train_acc'].append(acc)
            client_metrics[client_idx]['test_loss'].append(test_loss)
            client_metrics[client_idx]['test_acc'].append(test_acc)
        ## global step
        if fl_method == "FedSyn":
            global_weights = self.global_update.aggregate_weights(local_weights_sum=local_weights_sum, global_model=self.global_model, local_bitmasks_sum=local_bitmasks_sum)
            updateFromNumpyFlatArray(flat_arr=global_weights, model=self.global_model)
        elif fl_method == "qFedAvg":
            global_weights = self.global_update.aggregate_weights(self.global_model, local_delta_sum, local_h_sum)
            updateFromNumpyFlatArray(flat_arr=global_weights, model=self.global_model)
        else:
            global_weights = self.global_update.aggregate_weights(len(self.idxs_users), local_weights_sum, valid_losses,)
            self.global_update.update_global_model(self.global_model, global_weights)

        return global_weights, local_bitmasks_sum, client_metrics

    def consolidate_local_metrics(self, bitmasks):
        train_loss, train_acc = [], []
        test_loss, test_acc = [], []
        valid_loss, valid_acc = [], []

        for client_idx in self.idxs_users:
            local_update = self.get_local_update(client_idx=client_idx)
            acc, loss = local_update.inference(model=self.global_model, dataset_type="train")
            train_loss.append(loss)
            train_acc.append(acc)

            acc, loss = local_update.inference(model=self.global_model, dataset_type="test")
            test_loss.append(loss)
            test_acc.append(acc)

            acc, loss = local_update.inference(model=self.global_model, dataset_type="valid")
            valid_loss.append(loss)
            valid_acc.append(acc)
        

        val_stddev = np.std(valid_loss)
        test_stddev = np.std(test_loss)
        train_stddev = np.std(train_loss)

        params_sent_stddev = np.std(bitmasks)
        params_sent_mean = np.mean(bitmasks)

        avg_test_acc = np.mean(test_acc)
        avg_test_loss = np.mean(test_loss)

        print(f"Average test accuracy: {avg_test_acc}")
        print(f"Average test loss: {avg_test_loss}")
        print(f"Test loss stddev: {test_stddev}")
        print(f"Test loss stddev: {test_stddev}")
        print(f'Params sent mean: {params_sent_mean}')
        print(f'Params sent stddev: {params_sent_stddev}')



    @param('fl_parameters.fairness_temperature')
    @param('fl_parameters.min_sparsification_ratio')
    @param('fl_parameters.sparsification_ratio')
    def compute_client_prob_dist(self, fairness_temperature, min_sparsification_ratio, sparsification_ratio):
        valid_losses = []
        for client_idx in self.idxs_users:
            local_update = self.get_local_update(client_idx=client_idx)
            acc, loss = local_update.inference(model=self.global_model, dataset_type="valid")
            valid_losses.append(loss)
        valid_losses = np.array(valid_losses)
        client_prob_dist = custom_exponential_sparsity(valid_losses, sparsification_ratio, min_sparsification_ratio, fairness_temperature)

        self.client_prob_dist = {client_idx: client_prob_dist[i] for i, client_idx in enumerate(self.idxs_users)}
        
        return valid_losses
        

    @param('fl_parameters.use_fair_sparsification')
    def run_client_validation_step(self, client_idx):
        local_update = self.get_local_update(client_idx=client_idx)
        acc, loss = local_update.inference(model=self.global_model, dataset_type="valid")
        return acc, loss

    @param('global_parameters.global_rounds')
    def run_federated_training(self, global_rounds):
        self.get_data_splits()

        for epoch in range(global_rounds):
            global_weights, bitmasks, _ = self.train_global_round(epoch=epoch)
            self.consolidate_local_metrics(bitmasks=bitmasks)
    
if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    harness = FLTrainingHarness()
    harness.run_federated_training()

