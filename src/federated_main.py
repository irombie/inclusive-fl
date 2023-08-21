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
from models import MLP, VGG, CNNCifar, CNNFashion_Mnist, ResNet18, ResNet50
from options import args_parser
from update import get_local_update, test_inference

from utils import (
    exp_details,
    get_dataset,
    set_seed,
    updateFromNumpyFlatArray,
    temperatured_softmax,
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
    train_dataset, test_dataset, train_user_groups, test_user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == "cnn":
        # Convolutional neural netork
        if args.dataset == "fashionmnist":
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == "cifar":
            global_model = CNNCifar(args=args)

    elif args.model == "mlp":
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)

    elif args.model == "vgg19":
        if args.dataset == "cifar" or args.dataset == "fashionmnist":
            global_model = VGG(num_classes=10, args=args)
        elif args.dataset == "utkface":
            global_model = VGG(num_classes=4, args=args)
        elif args.dataset == "celeba":
            global_model = VGG(num_classes=40, args=args)

    elif args.model == "resnet18":
        if args.dataset == "cifar" or args.dataset == "fashionmnist":
            global_model = ResNet18(num_classes=10, args=args)
        elif args.dataset == "utkface":
            global_model = ResNet18(num_classes=4, args=args)
        elif args.dataset == "celeba":
            global_model = ResNet18(num_classes=40, args=args)

    elif args.model == "resnet50":
        if args.dataset == "cifar" or args.dataset == "fashionmnist":
            global_model = ResNet50(num_classes=10, args=args)
        elif args.dataset == "utkface":
            global_model = ResNet50(num_classes=4, args=args)
        elif args.dataset == "celeba":
            global_model = ResNet50(num_classes=40, args=args)

    else:
        exit("Error: unrecognized model")

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    global_update = get_global_update(args, global_model, num_users=args.num_users)

    # Training
    train_loss, train_accuracy, test_accuracy = [], [], []
    print_every = 2

    ### ckpt params
    ckpt_dict = dict()
    ckpt_dict.update(vars(args))
    ckpt_dict["train_ds_splits"] = train_user_groups
    ckpt_dict["test_ds_splits"] = test_user_groups
    ckpt_dict["global_lr"] = args.global_lr
    ckpt_dict["wandb_run_name"] = run_name

    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_bitmasks = [], [], []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        list_loss = []
        global_model.eval()

        test_accs, test_loss = [], []

        # Getting the test loss for all users' data of the global model
        for c in idxs_users:
            local_update = get_local_update(
                args=args,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                train_idxs=train_user_groups[c],
                test_idxs=test_user_groups[c],
                logger=run,
                global_model=global_model,
            )

            acc, loss = local_update.inference(model=local_models[c], is_test=True)

            test_accs.append(acc)
            list_loss.append(loss)
            # Uncomment to log to wandb if needed
            # run.log({f"local model test loss for user {c}": loss})
            # run.log({f"local model test accuracy for user {c}": acc})

        test_loss_avg = sum(list_loss) / len(test_accs)
        test_loss.append(test_loss_avg)
        test_acc_avg = sum(test_accs) / len(test_accs)
        test_accuracy.append(test_acc_avg)

        client_prob_dist = temperatured_softmax(
            np.array(list_loss), args.softmax_temperature
        )
        client_prob_dist = {
            idxs_users[i]: client_prob_dist[i] for i in range(len(client_prob_dist))
        }
        run.log(
            {
                f"Local Model Stddev of Test Losses": np.std(
                    np.array(list_loss).flatten()
                )
            }
        )

        global_model.train()
        list_acc = []
        for idx in idxs_users:
            local_update = get_local_update(
                args=args,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                train_idxs=train_user_groups[idx],
                test_idxs=test_user_groups[idx],
                logger=run,
                global_model=global_model,
            )
            # we might want to separate sparse updates and non-sparse
            # updates into separate classes in the future to avoid ifs
            # of this nature
            if args.fl_method == "FedSyn":
                w, flat_update, bitmask, loss = local_update.update_weights(
                    model=local_models[idx],
                    global_round=epoch,
                    sparsification_percentage=client_prob_dist[idx],
                )
                local_weights.append(copy.deepcopy(flat_update))
                local_bitmasks.append(bitmask)
            else:
                w, loss = local_update.update_weights(
                    model=local_models[idx], global_round=epoch
                )
                local_weights.append(copy.deepcopy(w.state_dict()))

            acc, loss = local_update.inference(model=w, is_test=False)
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

        acc_avg = sum(list_acc) / len(list_acc)
        train_accuracy.append(acc_avg)

        # update global weights
        if args.fl_method == "FedSyn":
            global_w = global_update.aggregate_weights(
                local_weights, global_model, local_bitmasks
            )
            # update models
            updateFromNumpyFlatArray(global_w, global_model)
            local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]
        else:
            global_weights = global_update.aggregate_weights(local_weights, list_loss)
            global_update.update_global_model(global_model, global_weights)
            global_update.update_local_models(local_models, global_weights)

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

        run.log({"Global test accuracy: ": 100 * test_accuracy[-1]})
        run.log({"Global train accuracy: ": 100 * train_accuracy[-1]})
        run.log({"Global train loss: ": train_loss[-1]})
        run.log({"Global test loss: ": test_loss[-1]})

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Loss : {np.mean(np.array(train_loss))}")
            print("Train Accuracy: {:.2f}% \n".format(100 * train_accuracy[-1]))
            print("Test Accuracy: {:.2f}% \n".format(100 * test_accuracy[-1]))

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

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))


if __name__ == "__main__":
    main()
