#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument("--fl_method", type=str, default="FedAvg", help="Name of federated learning method to use, \
                        options are FedAvg, FedBN, FedProx, TestLossWeighted")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='fashionmnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--device', default=None, help="To use cuda, set \
                        device to cuda. To use MPS, set device to mps.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--dist_noniid', type=float, default=0,
                        help='whether to use distribution-based label imbalande for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--min_proportion', type=float, default=0,
                        help='Minimum proportion of dataset for each user. Used in dist_noniid')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, help='random seed', required=True)
    parser.add_argument("--wandb_name", type=str, default = "FL", help="wandb project name, please set according to the details of your experiment")
    parser.add_argument('--save_every', type=int, default=2, help='save model every x rounds')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/', help='path to save checkpoints')
    # Experimentation Flags
    parser.add_argument("--reweight_loss_avg", type=int, default=0, help="To enable reweighted loss averaging or not, set to 1 to enable it")

    # arguments for FedProx
    parser.add_argument('--mu', type=float, default=None, help="mu value for FedProx")
    args = parser.parse_args()
    return args
