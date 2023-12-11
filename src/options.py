#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of rounds of training"
    )
    parser.add_argument("--num_users", type=int, default=10, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=5, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=64, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    
    parser.add_argument(
        "--fl_method",
        type=str,
        default="FedAvg",
        help="Name of federated learning method to use, \
                        options are FedAvg, FedBN, FedProx, TestLossWeighted",
    )
    # model arguments
    parser.add_argument("--model", type=str, default="cnn", help="model name")

    # other arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashionmnist",
        help="name \
                        of dataset",
    )
    
    parser.add_argument(
        "--iid", type=int, default=1, help="Default set to IID. Set to 0 for non-IID."
    )
    parser.add_argument(
        "--dist_noniid",
        type=float,
        default=0,
        help="whether to use distribution-based label imbalande for  \
                        non-i.i.d setting (use 0 for equal splits)",
    )
    parser.add_argument(
        "--min_proportion",
        type=float,
        default=0,
        help="Minimum proportion of dataset for each user. Used in dist_noniid",
    )
    
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    parser.add_argument("--seed", type=int, help="random seed", required=True)
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="FL",
        help="wandb project name, please set according to the details of your experiment",
    )
    parser.add_argument(
        "--save_every", type=int, default=2, help="save model every x rounds"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/",
        help="path to save checkpoints",
    )
    parser.add_argument("--global_lr", type=float, default=1)
    # Experimentation Flags
    parser.add_argument(
        "--reweight_loss_avg",
        type=int,
        default=0,
        help="To enable reweighted loss averaging or not, set to 1 to enable it",
    )
    parser.add_argument(
        "--sparsification_ratio",
        type=float,
        default=1,
        help="the percentage of model parameters that will be sent",
    )
    parser.add_argument(
        "--sparsification_type",
        type=str,
        default="randk",
        help="Type of sparsification to use.",
    )
    parser.add_argument(
        "--choose_from_top_r_percentile",
        type=float,
        default=1,
        help="the ratio of r from the rtopk method",
    )
    parser.add_argument(
        "--use_fair_sparsification",
        type=int,
        default=True,
        help="Activate fair sparsification on methods.",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=1,
        help="The temperature used in fairness sparsification.",
    )

    # arguments for FedProx
    parser.add_argument("--mu", type=float, default=None, help="mu value for FedProx")

    # arguments for qFedAvg
    parser.add_argument("--q", type=float, default=None, help="q value for qFedAvg")
    parser.add_argument("--eps", type=float, default=1e-6, help="eps value for qFedAvg")

    args = parser.parse_args()
    return args
