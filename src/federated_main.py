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
from global_updates import get_global_update
from models import (
    VGG,
    CNNFashion_Mnist,
    ResNet9,
    ResNet18,
    SmallCNN,
    MLP,
    LogisticRegression,
)
from options import args_parser
from update import get_local_update, test_inference
from utils import (
    custom_exponential_sparsity,
    exp_details,
    flatten,
    get_dataset,
    set_seed,
    updateFromNumpyFlatArray,
)


from fastargs import get_current_config
from fastargs.decorators import param, section
from fastargs import Param, Section
from fastargs.validation import And, OneOf

if sys.version_info[0:2] != (3, 11):
        print()
        raise RuntimeError(
            f"Code requires python 3.11. You are using {sys.version_info[0:2]}. Please update your conda env and install requirements.txt on the new env."
        )

Section('model', 'model parameters').params(
    model_name=Param(str, 'Global model architecture (common across devices)', validator=OneOf(['SmallCNN', 'ResNet9', 'ResNet18', 'MLP', 'LogisticRegression', 'VGG']), required=True),
    num_classes=Param(int, 'Number of classes', required=True))

Section('global_paramters', 'global parameters').params(
    global_rounds=Param(int, 'number of rounds of training', required=True),
    num_clients=Param(int, 'number of clients', required=True),
    client_frac=Param(float, 'Client fraction sampled at each round for training', required=True),
    global_lr=Param(float, 'global learning rate', default=1),)

Section('client_paramters', 'general client parameters').params(
    local_epochs=Param(int, 'number of local epochs', default=5),
    local_batch_size=Param(int, 'local batch size', default=64),
    local_lr=Param(float, 'local learning rate', default=0.01))

Section('fl_method', 'federated learning method').params(
    fl_method=Param(str, 'federated learning method', validator=OneOf(['FedAvg', 'FedProx', 'qFedAvg', 'FedSyn']), required=True),
    sparsification_ratio=Param(float, 'sparsification ratio', default=1),
    sparsification_type=Param(str, 'sparsification type', default='randk'),
    choose_from_top_r_percentile=Param(float, 'choose from top r percentile', default=1),
    use_fair_sparsification=Param(int, 'use fair sparsification', default=True),
    fairness_temperature=Param(float, 'fairness temperature', default=1),
    min_sparsification_ratio=Param(float, 'minimum sparsification ratio', default=0),
    mu=Param(float, 'mu value for FedProx', default=None),
    q=Param(float, 'q value for qFedAvg', default=None),
    eps=Param(float, 'eps value for qFedAvg', default=1e-6))


Section('dataset', 'dataset related stuff').params(
    dataset_name=Param(str, 'Name of dataset', required=True),
    distribution=Param(str, 'distribution', validator=OneOf(['iid', 'non_iid', 'majority_minority']), required=True),
    dirichlet_param=Param(float, 'dirichlet param', default=0),
    min_proportion=Param(float, 'minimum proportion', default=0),
    majority_proportion=Param(float, 'majority proportion', default=None),
    majority_minority_overlap=Param(float, 'majority minority overlap', default=None),
    num_features=Param(int, 'number of features', required=True),
    num_classes=Param(int, 'number of classes', required=True),
    gdrive_id=Param(str, 'google drive id', default=None))


Section('training_harness_params', 'parameters configuring the general training harness').params(
    verbose=Param(bool, 'verbose', default=True),
    seed=Param(int, 'random seed', required=True),
    save_every=Param(int, 'save model every x rounds', default=2),
    ckpt_path=Param(str, 'path to save checkpoints', default='./checkpoints/'),)


Section('optimizer', 'optimizer parameters').params(
    lr=Param(float, 'initial learning rate', default=0.1),
    warmup_length=Param(int, 'warmup length', default=10),
    weight_decay=Param(float, 'weight decay', default=1e-4),
    momentum=Param(float, 'momentum', default=0.9),
    nesterov=Param(int, 'use nesterov momentum? (1/0)', default=0),
    scheduler_type=Param(str, 'scheduler type', default='custom_step')
)


Section('training_params', 'harness related stuff').params(
    ckpt_path=Param(str, 'path to save checkpoints', default='checkpoints'),
    epochs=Param(int, 'number of epochs', default=150),
    seed=Param(int, 'random seed', default=42),
    wandb_project=Param(str, 'wandb project name', required=True))

class FLTrainingHarness:
    super(FLTrainingHarness, self).__init__()
    self.config = get_current_config()
    self.device =  torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_built() else "cpu"))
    self.run_name = f"{self.config.fl_method}_{self.config.model_name}_{self.config.dataset_name}_{self.config.global_rounds}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    self.global_model = self.get_model()
    self.train_dataset, self.test_dataset, self.valid_dataset, self.train_user_groups, self.test_user_groups, self.valid_user_groups = self.get_dataset()

    @param('model.model_name')
    @param('model.num_classes')
    @param('model.num_features')
    def get_model(self, model_name, num_classes, num_features):
        model_definition = globals()[model_name]

        if num_features is not None:
            model = model_definition(num_classes=num_classes, num_features=num_features)
        else:
            model = model_definition(num_classes=num_classes)
        model.to(self.device)

        
        return model

    @param('dataset.dataset_name')
    @param('dataset.batch_size')
    @param('dataset.train_image_size')
    @param('dataset.test_image_size')
    @param('dataset.num_workers')
    @param('dataset.distribution')
    @param('dataset.dirichlet_param')
    @param('dataset.min_proportion')
    @param('dataset.majority_proportion')
    @param('dataset.majority_minority_overlap')
    @param('dataset.num_features')
    @param('dataset.num_classes')
    @param('dataset.gdrive_id')
    def get_dataset(self, dataset_name, batch_size, train_image_size, test_image_size, num_workers, distribution, dirichlet_param, min_proportion, majority_proportion, majority_minority_overlap, num_features, num_classes, gdrive_id):
        return FLDataset()




def main():
    
    

    # copy weights
    global_weights = global_model.state_dict()

    global_update = get_global_update(args, global_model, num_users=args.num_users)

    # Training
    train_loss, train_accuracy, test_accuracy, valid_accuracy = [], [], [], []
    print_every = 2

    ### ckpt params
    ckpt_dict = dict()
    ckpt_dict.update(vars(args))
    ckpt_dict["train_ds_splits"] = train_user_groups
    ckpt_dict["test_ds_splits"] = test_user_groups
    ckpt_dict["global_lr"] = args.global_lr
    ckpt_dict["wandb_run_name"] = run_name

    list_acc = []

    for epoch in tqdm(range(args.epochs)):
        local_losses = []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        valid_losses = []
        global_model.eval()

        valid_accs, valid_loss = [], []

        global_flat = flatten(global_model)
        local_weights_sum, local_bitmasks_sum, local_delta_sum, local_h_sum = (
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
        )

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
