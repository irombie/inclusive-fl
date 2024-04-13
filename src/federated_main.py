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
    CNNFashion_Mnist,
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

from global_updates import get_global_update ## currently in work
#from update import get_local_update, test_inference

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
        self.splits = self.get_data_splits()

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
        train_user_groups, test_user_groups, valid_user_groups = main_ds.get_client_groups()

        return train_user_groups, test_user_groups, valid_user_groups
    
    @param('fl_parameters.fl_method')
    def global_update(self, fl_method):
        global_update = get_global_update(fl_method=fl_method, model=self.global_model)
        return global_update
    
    def get_local_update(self, client_idx):
        local_update = get_local_update(
            args=self.config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            valid_dataset=valid_dataset,
            train_idxs=train_user_groups[client_idx],
            test_idxs=test_user_groups[client_idx],
            valid_idxs=valid_user_groups[client_idx],
            logger=False,
            global_model=global_model,
        )

        return local_update

    @param('fl_parameters.use_fair_sparsification')
    @param('fl_parameters.sparsification_ratio')
    def run_fedsyn_client(self, use_fair_sparsification, sparsification_ratio, client_idx, local_update, epoch):
        local_model = copy.deepcopy(self.global_model)
        if use_fair_sparsification:
            sparsification_ratio = self.client_prob_dist[client_idx]

        w, flat_update, bitmask, _ = local_update.update_weights(local_model=local_model, sparsification_ratio=sparsification_ratio, global_round=epoch)
        
        acc, loss = local_update.inference(model=w, dataset_type="train")

        return w, flat_update, bitmask, loss, acc

    def run_qfedavg_client(self, local_update, epoch):
        local_model = copy.deepcopy(self.global_model)
        delta, h, w, _ = local_update.update_weights(local_model=local_model, global_round=epoch)
        acc, loss = local_update.inference(model=w, dataset_type="train")
        return delta, h, w, loss, acc

    def run_generic_client(self, local_update, epoch):
        local_model = copy.deepcopy(self.global_model)
        w, _ = local_update.update_weights(local_model=local_model, global_round=epoch)
        acc, loss = local_update.inference(model=w, dataset_type="train")
        return w, loss, acc

    @param('fl_parameters.num_clients')
    @param('fl.paramsters.frac')
    @param('fl_parameters.fl_method')
    def train_global_round(self, num_clients, frac, fl_method, epoch):
        m = max(int(frac * num_clients), 1)
        idxs_users = np.random.choice(range(num_clients), m, replace=False)
        local_weights_sum, local_bitmasks_sum, local_delta_sum, local_h_sum = (
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
        )
        client_train_loss, client_train_acc = [], []
        client_test_loss, client_test_acc = [], []
        client_valid_loss, client_valid_acc = [], []
        for client_idx in num_clients:
            local_update = self.get_local_update(client_idx)
            if fl_method == "FedSyn":
                w, flat_update, bitmask, loss, acc = self.run_fedsyn_client(client_idx, local_update, epoch)
                local_weights_sum += flat_update
                local_bitmasks_sum += bitmask
            elif fl_method == "qFedAvg":
                delta, h, w, loss, acc = self.run_qfedavg_client(local_update, epoch)
                local_delta_sum += delta
                local_h_sum += h
                local_bitmasks_sum += np.ones_like(local_bitmasks_sum)
            else:
                w, loss, acc = self.run_generic_client(local_update, epoch)
                local_weights_sum += flatten(local_model)
                local_bitmasks_sum += np.ones_like(local_bitmasks_sum)
        
        ## global step
        if fl_method == "FedSyn":
            global_w = self.global_update.aggregate_weights(local_weights_sum, self.global_model, local_bitmasks_sum)
            updateFromNumpyFlatArray(global_w, self.global_model)
        elif fl_method == "qFedAvg":
            global_weights = self.global_update.aggregate_weights(self.global_model, local_delta_sum, local_h_sum)
            updateFromNumpyFlatArray(global_weights, self.global_model)
        else:
            global_weights = self.global_update.aggregate_weights(local_weights_sum, valid_losses, len(idxs_users))
            self.global_update.update_global_model(self.global_model, global_weights)


    def consolidate_metrics(self):


    @param('fl_parameters.use_fair_sparsification')
    def run_client_validation_step(self, client_idx):
        local_update = self.get_local_update(train_dataset=self.train_dataset,
                                             test_dataset=self.test_dataset,
                                             valid_dataset=self.valid_dataset,
                                             train_idxs=self.train_user_groups[client_idx],
                                             test_idxs=self.test_user_groups[client_idx],
                                             valid_idxs=self.valid_user_groups[client_idx],
                                             logger=self.logger,
                                             global_model=self.global_model)
            
        acc, loss = local_update.inference(model=self.global_model, dataset_type="valid")
        return acc, loss

        



    
if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    #harness = FLTrainingHarness()
    #harness.get_data_splits()
    #globalupdate = harness.global_update()
    ### Testing code
    #harness = FLTrainingHarness()
    #harness.get_data_splits()

