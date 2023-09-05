#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import wandb
from global_updates import get_global_update
from models import VGG, CNNFashion_Mnist, ResNet9, ResNet18, SmallCNN
from options import args_parser
from update import get_local_update, test_inference
from utils import (
    exp_details,
    flatten,
    get_dataset,
    set_seed,
    temperatured_softmax,
    updateFromNumpyFlatArray,
)


def main():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath("..")

    args = args_parser()
    exp_details(args)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H_%M")
    run_name = f"{args.fl_method}_{args.dataset}_clients_{args.num_users}_frac_{args.frac}_{args.sparsification_ratio}_{time.time()}"
    args_dict = vars(args)
    tag_list = []
    for k in args_dict:
        tag_list.append(f"{k}:{args_dict[k]}")
    run = wandb.init(project=args.wandb_name, config=args, name=run_name, tags=tag_list)

    if args.gpu and args.device == "cuda":
        device = "cuda"
    elif args.gpu and args.device == "mps":
        device = "mps"
    else:
        device = "cpu"

    set_seed(args.seed, False)

    # load dataset and user groups
    (
        train_dataset,
        test_dataset,
        valid_dataset,
        train_user_groups,
        test_user_groups,
        valid_user_groups,
    ) = get_dataset(args)

    # BUILD MODEL
    if args.dataset == "fashionmnist":
        global_model = CNNFashion_Mnist(args=args)

    elif args.dataset == "cifar":
        if args.model == "small_cnn":
            global_model = SmallCNN(args=args, num_classes=10)
        elif args.model == "vgg11_bn":
            global_model = VGG(num_classes=10, args=args)
        elif args.model == "resnet18":
            global_model = ResNet18(num_classes=10, args=args)
        elif args.model == "resnet9":
            global_model = ResNet9(num_classes=10, args=args)
        else:
            exit("Error: Model not implemented!")

    elif args.dataset == "utkface":
        if args.model == "small_cnn":
            global_model = SmallCNN(args=args, num_classes=5)
        elif args.model == "vgg11_bn":
            global_model = VGG(num_classes=5, args=args)
        elif args.model == "resnet18":
            global_model = ResNet18(num_classes=5, args=args)
        elif args.model == "resnet9":
            global_model = ResNet9(num_classes=5, args=args)
        else:
            exit("Error: Model not implemented!")

    elif args.dataset == "tiny-imagenet":
        if args.model == "small_cnn":
            global_model = SmallCNN(args=args, num_classes=200)
        elif args.model == "vgg11_bn":
            global_model = VGG(num_classes=200, args=args)
        elif args.model == "resnet18":
            global_model = ResNet18(num_classes=200, args=args)
        elif args.model == "resnet9":
            global_model = ResNet9(num_classes=200, args=args)
        else:
            exit("Error: Model not implemented!")
    else:
        exit("Error: Dataset not implemented!")

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

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
    # local_bitmasks = []
    for epoch in tqdm(range(args.epochs)):
        local_losses = []
        local_deltas, local_hs = [], []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        valid_losses = []
        global_model.eval()

        valid_accs, valid_loss = [], []

        global_flat = flatten(global_model)
        local_weights_sum, local_bitmasks_sum = np.zeros_like(
            global_flat
        ), np.zeros_like(global_flat)

        for c in idxs_users:
            ########## Getting the validation loss for all users' data of the global model
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
            client_prob_dist = temperatured_softmax(
                np.array(valid_losses), args.softmax_temperature
            )
            client_prob_dist = {
                idxs_users[i]: client_prob_dist[i] for i in range(len(client_prob_dist))
            }

        for c in idxs_users:
            local_model = copy.deepcopy(global_model)
            ####### Calculate local update
            idx = c
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
            # we might want to separate sparse updates and non-sparse
            # updates into separate classes in the future to avoid ifs
            # of this nature

            if args.fl_method == "FedSyn":
                sparsification_percentage = None
                if args.use_fair_sparsification:
                    sparsification_percentage = client_prob_dist[idx]
                    print(f"Sparsification percentage {sparsification_percentage}")
                    assert sparsification_percentage is not None
                w, flat_update, bitmask, loss = local_update.update_weights(
                    model=local_model,
                    sparsification_percentage=sparsification_percentage,
                    global_round=epoch,
                )
                # local_weights.append(copy.deepcopy(flat_update))
                # local_bitmasks.append(bitmask)
                local_weights_sum += flat_update
                local_bitmasks_sum += bitmask
            elif args.fl_method == "qFedAvg":
                delta, h, w, loss = local_update.update_weights(
                    model=local_model, global_round=epoch
                )
                local_deltas.append(copy.deepcopy(delta))
                local_hs.append(copy.deepcopy(h))
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
        run.log({"Mean of number of parameters sent:": np.mean(num_client_params_sent)})

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
            global_weights = global_update.update_global_model(
                global_model, local_deltas, local_hs
            )
        else:
            global_weights = global_update.aggregate_weights(
                local_weights_sum, valid_losses
            )
            global_update.update_global_model(
                global_model, global_weights, len(idxs_users)
            )

        if epoch % int(args.save_every) == 0:
            ckpt_dict["state_dict"] = global_model.state_dict()
            if not os.path.exists(args.ckpt_path):
                os.makedirs(args.ckpt_path)
            torch.save(
                ckpt_dict,
                f"{args.ckpt_path}/{args.fl_method}_{args.model}_{args.dataset}_global_model_{epoch}_{dt_string}.pt",
            )

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

    # Saving the objects train_loss and train_accuracy:
    file_name = os.path.join(
        os.path.abspath(""),
        "save",
        "objects",
        f"{args.dataset}_{args.model}_ \
                            {args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].pkl",
    )
    with open(file_name, "wb") as f:
        pickle.dump([train_loss, train_accuracy], f)

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))


if __name__ == "__main__":
    main()
