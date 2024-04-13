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
    @param('fl_parameters.fl_method')
    def global_update(self, fl_method):
        global_update = get_global_update(fl_method=fl_method, model=self.global_model)
        return global_update
    
    def get_local_update(self):
        local_update = get_local_update(
            args=self.config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            valid_dataset=valid_dataset,
            train_idxs=train_user_groups[c],
            test_idxs=test_user_groups[c],
            valid_idxs=valid_user_groups[c],
            logger=run,
            global_model=global_model,
        )

        return local_update
    
    def get_data_splits(self):
        main_ds = FLDataset()
        main_ds.get_dataset()
        train_user_groups, test_user_groups, valid_user_groups = main_ds.get_client_groups()

        return train_user_groups, test_user_groups, valid_user_groups

    @param('fl_parameters.num_clients')
    @param('fl_parameters.frac')
    def train_global_round(self, num_clients, frac):
        m = max(int(frac * num_clients), 1)
        idxs_users = np.random.choice(range(num_clients), m, replace=False)
        global_flat = general_utils.flatten(self.global_model)
        local_weights_sum, local_bitmasks_sum, local_delta_sum, local_h_sum = (
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
        )
    
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
    
    def run_client_test_step(self, )

    #def train_one_round(self):
    #def evaluate -- need to think about how this would work in the context of the different algorithms in the codebase

    
if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    harness = FLTrainingHarness()
    harness.get_data_splits()
    #globalupdate = harness.global_update()
    ### Testing code
    #harness = FLTrainingHarness()
    #harness.get_data_splits()

'''

def main():

    list_acc = []

    for epoch in tqdm(range(args.epochs)):
        
        for c in idxs_users:
            # Getting the validation loss for all users' data of the global model
            local_update = get_local_update(
                args=args,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                valid_dataset=valid_dataset,
                train_idxs=train_user_groups[c],
                test_idxs=test_user_groups[c],
                valid_idxs=valid_user_groups[c],
                logger=run,
                global_model=global_model,
            )

            acc, loss = local_update.inference(model=global_model, dataset_type="valid")

            valid_accs.append(acc)
            valid_losses.append(loss)
            # Uncomment to log to wandb if needed
            # run.log({f"local model test loss for user {c}": loss})
            # run.log({f"local model test accuracy for user {c}": acc})

        valid_loss_avg = sum(valid_losses) / len(valid_accs)
        valid_loss.append(valid_loss_avg)
        valid_acc_avg = sum(valid_accs) / len(valid_accs)
        valid_accuracy.append(valid_acc_avg)

        run.log(
            {
                f"Local Model Stddev of Valid Losses": np.std(
                    np.array(valid_losses).flatten()
                )
            }
        )

        client_prob_dist = None
        if args.use_fair_sparsification:

            client_prob_dist = custom_exponential_sparsity(
                np.array(valid_losses),
                args.sparsification_ratio,
                args.min_sparsification_ratio,
                args.fairness_temperature,
            )

            client_prob_dist = {
                idxs_users[i]: client_prob_dist[i] for i in range(len(client_prob_dist))
            }

        for idx in idxs_users:
            local_model = copy.deepcopy(global_model)
            ## Calculate local update
            local_update = get_local_update(
                args=args,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                valid_dataset=valid_dataset,
                train_idxs=train_user_groups[idx],
                test_idxs=test_user_groups[idx],
                valid_idxs=valid_user_groups[idx],
                logger=run,
                global_model=global_model,
            )
            # we might want to separate sparse updates and non-sparse
            # updates into separate classes in the future to avoid ifs
            # of this nature

            if args.fl_method == "FedSyn":
                sparsification_percentage = args.sparsification_ratio
                if args.use_fair_sparsification:
                    sparsification_percentage = client_prob_dist[idx]
                    print(f"Sparsification percentage {sparsification_percentage}")
                # assert sparsification_percentage is not None
                w, flat_update, bitmask, loss = local_update.update_weights(
                    model=local_model,
                    sparsification_percentage=sparsification_percentage,
                    global_round=epoch,
                )

                local_weights_sum += flat_update
                local_bitmasks_sum += bitmask
            elif args.fl_method == "qFedAvg":
                delta, h, w, loss = local_update.update_weights(
                    model=local_model, global_round=epoch
                )
                local_delta_sum += delta
                local_h_sum += h
                local_bitmasks_sum += np.ones_like(local_bitmasks_sum)
            else:
                w, loss = local_update.update_weights(
                    model=local_model, global_round=epoch
                )
                local_weights_sum += flatten(local_model)
                local_bitmasks_sum += np.ones_like(local_bitmasks_sum)

            acc, loss = local_update.inference(model=w, dataset_type="train")
            list_acc.append(acc)
            local_losses.append(copy.deepcopy(loss))
            # Uncomment to log to wandb if needed
            # run.log({f"local model training loss per iteration for user {idx}": loss})
            # run.log({f"local model training accuracy per iteration for user {idx}": acc})

        run.log(
            {
                f"Local Model Stddev of Train Losses": np.std(
                    np.array(local_losses).flatten()
                )
            }
        )

        num_client_params_sent = local_bitmasks_sum
        run.log(
            {
                "Standard deviation of number of parameters sent:": np.std(
                    num_client_params_sent
                )
            }
        )
        run.log(
            {
                "Mean of number of parameters sent:": np.sum(num_client_params_sent)
                / len(idxs_users)
            }
        )

        acc_avg = sum(list_acc) / len(list_acc)
        train_accuracy.append(acc_avg)

        # update global weights
        if args.fl_method == "FedSyn":
            global_w = global_update.aggregate_weights(
                local_weights_sum, global_model, local_bitmasks_sum
            )
            # update models
            updateFromNumpyFlatArray(global_w, global_model)
        elif args.fl_method == "qFedAvg":
            global_weights = global_update.aggregate_weights(
                global_model, local_delta_sum, local_h_sum
            )
            updateFromNumpyFlatArray(global_weights, global_model)
        else:
            global_weights = global_update.aggregate_weights(
                local_weights_sum, valid_losses, len(idxs_users)
            )
            global_update.update_global_model(global_model, global_weights)

        if epoch % int(args.save_every) == 0:
            ckpt_dict["state_dict"] = global_model.state_dict()
            ckpt_dir = os.path.join(args.ckpt_path, "")

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            filename = f"{args.fl_method}_{args.model}_{args.dataset}_global_model_{epoch}_{dt_string}.pt"
            filepath = os.path.join(ckpt_dir, filename)
            torch.save(ckpt_dict, filepath)
        loss_avg = sum(local_losses) / len(local_losses)

        train_loss.append(loss_avg)

        test_losses = []
        global_model.eval()

        test_accs, test_loss = [], []

        # Getting the test loss for all users' data of the global model
        for c in idxs_users:
            local_update = get_local_update(
                args=args,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                valid_dataset=valid_dataset,
                train_idxs=train_user_groups[c],
                test_idxs=test_user_groups[c],
                valid_idxs=valid_user_groups[c],
                logger=run,
                global_model=global_model,
            )

            acc, loss = local_update.inference(model=global_model, dataset_type="test")

            test_accs.append(acc)
            test_losses.append(loss)
            # Uncomment to log to wandb if needed
            # run.log({f"local model test loss for user {c}": loss})
            # run.log({f"local model test accuracy for user {c}": acc})

        test_loss_avg = sum(test_losses) / len(test_accs)
        test_loss.append(test_loss_avg)
        test_acc_avg = sum(test_accs) / len(test_accs)
        test_accuracy.append(test_acc_avg)

        run.log(
            {
                f"Local Model Stddev of Test Losses": np.std(
                    np.array(test_losses).flatten()
                )
            }
        )
        run.log(
            {"client_test_loss_hist": wandb.Histogram(np.array(test_losses).flatten())}
        )
        run.log(
            {
                f"Local Model Stddev of Test Accuracies": np.std(
                    np.array(test_accs).flatten()
                )
            }
        )

        run.log({"Global test accuracy: ": 100 * test_accuracy[-1]})
        run.log({"Global train accuracy: ": 100 * train_accuracy[-1]})
        run.log({"Global valid accuracy: ": 100 * valid_accuracy[-1]})
        run.log({"Global train loss: ": train_loss[-1]})
        run.log({"Global test loss: ": test_loss[-1]})
        run.log({"Global valid loss: ": valid_loss[-1]})

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Loss : {np.mean(np.array(train_loss))}")
            print("Train Accuracy: {:.2f}% \n".format(100 * train_accuracy[-1]))
            print("Test Accuracy: {:.2f}% \n".format(100 * test_accuracy[-1]))
            print("Valid Accuracy: {:.2f}% \n".format(100 * valid_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f" \n Results after {args.epochs} global rounds of training:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))


if __name__ == "__main__":
    try:
        main()
        wandb.finish(exit_code=0)
    except Exception as e:
        print(f"Experiment failed due to {e}")
        print(traceback.format_exc())
        wandb.finish(exit_code=-1)
'''