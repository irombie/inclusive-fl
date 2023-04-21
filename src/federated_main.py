#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import os
import pickle
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

import wandb
from global_updates import get_global_update
from models import MLP, CNNCifar, CNNFashion_Mnist, CNNMnist
from options import args_parser
from update import LocalUpdate, test_inference
from utils import weighted_average, exp_details, get_dataset

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    
    args = args_parser()
    exp_details(args)

    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y-%H_%M")
    run = wandb.init(project=args.wandb_name, config=args)

    if args.gpu and args.device == "cuda":
        device = "cuda"
    elif args.gpu and args.device == "mps":
        device = "mps"
    else:
        device = "cpu"

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    global_update = get_global_update(args, global_model, num_users=args.num_users)

    # Training
    train_loss, train_accuracy, test_accuracy = [], [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        m = max(int(args.frac * args.num_users), 1)
        print(args.num_users)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        list_loss = []
        global_model.eval()

        test_accs, test_loss = [], []

        # Getting the test loss for all users' data of the global model
        for c in idxs_users:
            local_update = get_local_update(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=run,
                                      global_model=global_model)
            acc, loss = local_update.inference(model=local_models[c], is_test=True)
            
            test_accs.append(acc)
            list_loss.append(loss)
            # Uncomment to log to wandb if needed
            run.log({f"local model test loss for user {c}": loss})
            run.log({f"local model test accuracy for user {c}": acc})

        test_loss_avg = sum(list_loss)/len(test_accs)
        test_loss.append(test_loss_avg)
        test_acc_avg = sum(test_accs)/len(test_accs)
        test_accuracy.append(test_acc_avg)

        global_model.train()
        list_acc = []
        for idx in idxs_users:
            local_update = get_local_update(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=run,
                                      global_model=global_model)
            w, loss = local_update.update_weights(
                model=local_models[idx], global_round=epoch)
            acc, loss = local_update.inference(model=w, is_test=False)
            list_acc.append(acc)
            local_weights.append(copy.deepcopy(w.state_dict()))
            local_losses.append(copy.deepcopy(loss))
            # Uncomment to log to wandb if needed
            run.log({f"local model training loss per iteration for user {idx}": loss})
            run.log({f"local model training accuracy per iteration for user {idx}": acc})

        acc_avg = sum(list_acc)/len(list_acc)
        train_accuracy.append(acc_avg)

        # update global weights
        global_weights = global_update.aggregate_weights(local_weights, list_loss)
        # update models
        global_update.update_global_model(global_model, global_weights)
        global_update.update_local_models(local_models, global_weights)

        loss_avg = sum(local_losses) / len(local_losses)

        train_loss.append(loss_avg)

        run.log({"Global test accuracy: ": 100*test_accuracy[-1]})
        run.log({"Global train accuracy: ": 100*train_accuracy[-1]})
        run.log({"Global train loss: ": train_loss[-1]})
        run.log({"Global test loss: ": test_loss[-1]})

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print('Test Accuracy: {:.2f}% \n'.format(100*test_accuracy[-1]))
            
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    

    # Saving the objects train_loss and train_accuracy:
    file_name = os.path.join(os.path.abspath(""), "save", "objects", f"{args.dataset}_{args.model}_ \
                            {args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].pkl")
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

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