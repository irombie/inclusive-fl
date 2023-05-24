from typing import List
import os
import random
import argparse

# Python additional
import numpy as np
import tqdm
from prettytable import PrettyTable
from pyhessian import hessian
import pandas as pd
import wandb

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import Dataset, DataLoader

# Torchvision
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from utils import get_dataset_for_metrics, set_seed
from update import DatasetSplit
### Model imports
from models import CNNCifar, CNNFashion_Mnist, CNNMnist, MLP

class MetricHarness:
    def __init__(self, harness_params):
        self.harness_params = harness_params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.net = harness_params["model"]
        self.criterion = nn.CrossEntropyLoss()
        self.net.to(self.device)

        if self.harness_params["dataset"] == "cifar":
            self.classes = (
                "plane",
                "car",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            )
        
        elif self.harness_params["dataset"] == "mnist":
            self.classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        
        elif self.harness_params["dataset"] == "fmnist":
            ## classes in fashion mnist
            self.classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                            "Sneaker", "Bag", "Ankle boot")
        
        self.num_classes = 10
               

    def compute_accuracy_metrics(self, testloader):
        self.net.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        class_correct = list(0.0 for i in range(self.num_classes))
        class_total = list(0.0 for i in range(self.num_classes))
        class_loss = list(0.0 for i in range(self.num_classes))
        print('len testloader', len(testloader))
        with torch.no_grad():
            for ims, labels in tqdm.tqdm(testloader):

                outputs = self.net(ims)
                loss = self.criterion(outputs, labels)

                _, pred = torch.max(outputs, 1)

                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += pred[i].eq(label).item()
                    class_total[label] += 1
                    class_loss[label] += self.criterion(
                        outputs[i].unsqueeze(0), labels[i].unsqueeze(0)
                    ).item()
                test_loss += loss.item()
                test_correct += pred.eq(labels).sum().item()
                test_total += labels.size(0)

        test_loss /= len(testloader)
        test_accuracy = (test_correct / test_total) * 100
        class_accuracy = np.divide(class_correct, class_total) * 100
        class_loss = np.divide(class_loss, class_total)

        return test_loss, class_loss, test_accuracy, class_accuracy

    def compute_grad_norm(self, testloader):
        class_total = list(0.0 for i in range(self.num_classes))
        class_gradnorm = list(0.0 for i in range(self.num_classes))

        self.net.eval()
        for ims, labels in tqdm.tqdm(testloader):

            outputs = self.net(ims)
            for i in range(self.num_classes):
                if len(labels[labels == i]) > 0:
                    self.net.zero_grad()
                    group_loss = self.criterion(
                        outputs[labels == i], labels[labels == i]
                    )
                    grads = autograd.grad(
                        group_loss, self.net.parameters(), retain_graph=True
                    )
                    sub_norm = torch.norm(
                        torch.stack([torch.norm(g)
                                    for g in grads if g is not None])
                    ).item()
                    self.net.zero_grad(set_to_none=True)
                    class_total[i] += len(labels[labels == i])
                    class_gradnorm[i] += sub_norm

        return np.divide(class_gradnorm, np.multiply(class_total, len(testloader)))

    def compute_hessian(self, testloader):

        class_total = list(0.0 for i in range(self.num_classes))
        class_eigen = list(0.0 for i in range(self.num_classes))
        for ims, labels in tqdm.tqdm(testloader):

            for i in range(self.num_classes):
                if ims[labels == i].shape[0] > 0:
                    sub_hessian_comp = hessian(
                        self.net,
                        self.criterion,
                        data=(ims[labels == i], labels[labels == i]),
                        cuda=True if torch.cuda.is_available() else False,
                    )
                    (
                        top_eigenvalues,
                        top_eigenvector,
                    ) = sub_hessian_comp.eigenvalues()
                    class_total[i] += len(labels[labels == i])
                    class_eigen[i] += top_eigenvalues[-1]

        return np.divide(class_eigen, np.multiply(class_total, len(testloader)))

    def compute_decision_boundary_distance(self, testloader):
        self.net.eval()
        class_total = list(0.0 for i in range(self.num_classes))
        class_distances = list(0.0 for i in range(self.num_classes))

        with torch.no_grad():
            for ims, labels in tqdm.tqdm(testloader):

                for i in range(self.num_classes):
                    
                    if len(labels[labels == i]) > 0:
                        outputs = self.net(ims[labels == i])
                        softmax_outputs = torch.sort(
                            F.softmax(outputs, dim=1), dim=1, descending=True
                        )[0].cpu()
                        class_distances[i] += (
                            (softmax_outputs[:, 0] -
                             softmax_outputs[:, 1]).sum().cpu()
                        )
                        class_total[i] += len(labels[labels == i])
        return np.divide(class_distances, class_total)

    def compute_robust_loss(self, testloader):
        noise_list = []
        noise_std_list = [0.01, 0.05]
        self.net.eval()

        with torch.no_grad():
            for noise_std in noise_std_list:
                noise_dict = {}
                class_correct = list(0.0 for i in range(self.num_classes))
                class_total = list(0.0 for i in range(self.num_classes))
                class_loss = list(0.0 for i in range(self.num_classes))

                for ims, labels in tqdm.tqdm(testloader):
                    for i in range(self.num_classes):

                        if len(labels[labels == i]) > 0:
                            class_ims, class_labels = (
                                ims[labels == i],
                                labels[labels == i],
                            )
                            class_ims += torch.randn_like(class_ims) * \
                                noise_std
                            class_outputs = self.net(class_ims)

                            _, class_pred = torch.max(class_outputs, 1)

                            class_correct[i] += class_pred.eq(
                                class_labels).sum().item()
                            class_loss[i] += self.criterion(
                                class_outputs, class_labels
                            ).item()
                            class_total[i] += len(class_labels)

                class_accuracy = np.divide(class_correct, class_total) * 100
                class_loss = np.divide(class_loss, len(testloader))
                noise_dict[f"{noise_std}_class_accuracy"] = class_accuracy
                noise_dict[f"{noise_std}_class_loss"] = class_loss
                noise_list.append(noise_dict)
        return noise_list


def compute_metrics(harness_params):
    set_seed(int(harness_params['seed']), False)

    testloader = harness_params['testloader']
    experiment_seed = harness_params['seed']
    # Model Definition
    """Model Definition"""

    # Harness init
    metric_harness = MetricHarness(harness_params)
    
    results_list = []
    if harness_params["compute_accuracy"]:
        print("Computing Accuracy...")
        print("==" * 100)
        (
            test_loss,
            class_loss,
            test_accuracy,
            class_accuracy,
        ) = metric_harness.compute_accuracy_metrics(testloader)

    if harness_params["compute_grad_norms"]:
        print("Computing Grad Norms...")
        print("==" * 100)
        grad_norms = metric_harness.compute_grad_norm(testloader)

    if harness_params['compute_hessian_eigenvalues']:
        print("Computing Hessian Eigenvalues...")
        print("==" * 100)
        hessians = metric_harness.compute_hessian(testloader)

    if harness_params["compute_decision_boundary_distances"]:
        print("Computing Decision Boundary Distances...")
        print("==" * 100)
        decision_boundary_dists = metric_harness.compute_decision_boundary_distance(
            testloader
        )

    if harness_params["compute_robustness_metrics"]:
        print("Computing Robust Loss and Accuracy...")
        print("==" * 100)
        robust_loss = metric_harness.compute_robust_loss(testloader)

    for class_idx in range(harness_params["num_classes"]):
        class_dict = {}
        class_dict['global_lr'] = harness_params['global_lr']
        class_dict['iid'] = harness_params['iid']
        class_dict['num_users'] = harness_params['num_users']
        class_dict['local_epoch'] = harness_params['local_ep']
        class_dict['local_bs'] = harness_params['local_bs']
        class_dict['local_lr'] = harness_params['local_lr']
        class_dict['training_rounds'] = harness_params['training_rounds']
        class_dict['frac'] = harness_params['frac']
        class_dict['seed'] = harness_params['seed']
        class_dict['arch'] = harness_params['arch']
        class_dict['dataset'] = harness_params['dataset']
        if harness_params["compute_accuracy"]:
            class_dict["class"] = metric_harness.classes[class_idx]
            class_dict[f"test_loss_{experiment_seed}"] = test_loss
            class_dict[f"class_loss_{experiment_seed}"] = class_loss[class_idx]
            class_dict[f"test_accuracy_{experiment_seed}"] = test_accuracy
            class_dict[f"class_accuracy_{experiment_seed}"] = class_accuracy[class_idx]

        if harness_params["compute_grad_norms"]:
            class_dict[f"grad_norms_{experiment_seed}"] = grad_norms[class_idx]

        if harness_params['compute_hessian_eigenvalues']:
            class_dict[f"hessian_eigen_{experiment_seed}"] = hessians[class_idx]

        if harness_params["compute_decision_boundary_distances"]:
            class_dict[f"distance_to_decision_{experiment_seed}"] = decision_boundary_dists[
                class_idx
            ]

        if harness_params["compute_robustness_metrics"]:
            class_dict[f"robust_class_loss_0.01_{experiment_seed}"] = robust_loss[0][
                "0.01_class_loss"
            ][class_idx]
            class_dict[f"robust_class_loss_0.05_{experiment_seed}"] = robust_loss[1][
                "0.05_class_loss"
            ][class_idx]
            class_dict[f"robust_class_acc_0.01_{experiment_seed}"] = robust_loss[0][
                "0.01_class_accuracy"
            ][class_idx]
            class_dict[f"robust_class_acc_0.05_{experiment_seed}"] = robust_loss[1][
                "0.05_class_accuracy"
            ][class_idx]

        results_list.append(class_dict)

    return results_list

def train_test(dataset, idxs):
    """
    Returns test dataloader for a given dataset, for computing metrics.
    """
    # split indexes for train, validation, and test (80, 10, 10)
    testloader = DataLoader(DatasetSplit(dataset, idxs),
                            batch_size=int(len(idxs_test)/10), shuffle=False)
    return testloader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment for Tooling Fairness")

    parser.add_argument('-model_ckpt_path', '--model_ckpt',
                        help='Path to checkpoint', required=True)
    parser.add_argument('-results_path', '--results_path',
                        help='Path to save the results CSV', required=True)
    parser.add_argument('-compute_accuracy_metrics', '--compute_accuracy',
                        help='Toggles computation of accuracy metrics', action='store_true')
    parser.add_argument('-compute_grad_norms', '--compute_grad_norms',
                        help='Toggles computation of grad norms', action='store_true')
    parser.add_argument('-compute_hessian_eigenvalues', '--compute_hessian_eigenvalues',
                        help='Toggles computation of hessian eigenvalues', action='store_true')
    parser.add_argument('-compute_decision_boundary_distances', '--compute_decision_boundary_distances',
                        help='Toggles computation of decision boundary distances', action='store_true')
    parser.add_argument('-compute_robustness_metrics', '--compute_robustness_metrics',
                        help='Toggles computation of robustness metrics', action='store_true')
    parser.add_argument('-log_to_wandb', '--log_to_wandb', action='store_true')

    args = parser.parse_args()
    harness_params = vars(args)
    
    print(harness_params)

    # Experiment Parameters
    harness_params["batch_size"] = 512

    ckpt = torch.load(harness_params['model_ckpt'], map_location=torch.device('cuda') if torch.cuda.is_available() else 'cpu')

    harness_params['global_lr'] = ckpt['global_lr']
    harness_params['iid'] = ckpt['iid']
    harness_params['num_users'] = ckpt['num_users']
    harness_params['local_ep'] = ckpt['local_epoch']
    harness_params['local_bs'] = ckpt['local_bs']
    harness_params['local_lr'] = ckpt['local_lr']
    harness_params['training_rounds'] = ckpt['training_rounds']
    harness_params['frac'] = ckpt['frac']
    harness_params['seed'] = ckpt['seed']
    harness_params['arch'] = ckpt['arch']
    harness_params['dataset'] = ckpt['dataset']
    harness_params['arch'] = ckpt['arch']
    wandb_proj_name = ckpt['wandb_proj_name']
    wandb_run_name = ckpt['wandb_run_name']
    
    test_user_groups = ckpt['test_ds_splits']
    num_users = ckpt['num_users']
    iid = ckpt['iid']
    dist_noniid = ckpt['dist_non_iid']
    unequal = ckpt['unequal']

    num_classes = 10

    arg_dict = {'dataset' : harness_params['dataset'], 'arch' : harness_params['arch'], 'test_user_groups' : test_user_groups, 'num_users' : num_users, 'iid' : iid, 'num_classes' : num_classes, 'unequal' : unequal, 'dist_noniid' : dist_noniid}

    args.num_classes = 10
    _, test_dataset, __, test_num_groups = get_dataset_for_metrics(arg_dict)
    
    if harness_params['dataset'] == 'cifar':
        len_in = 3*32*32
    elif harness_params['dataset'] == 'mnist':
        len_in = 28*28
    elif harness_params['dataset'] == 'fmnist':
        len_in = 28*28

        

    if harness_params['arch'] == 'cnn':
        if  harness_params['dataset'] == 'cifar':
            model = CNNCifar(args=args)
        elif harness_params['dataset'] == 'mnist':
            model = CNNMnist(args=args)
        elif harness_params['dataset'] == 'fmnist':
            model = CNNFashion_Mnist(args=args)

    elif harness_params['arch'] == 'mlp':
        model = MLP(dim_in=len_in, dim_hidden=64,
                            dim_out=num_classes)

    harness_params["num_classes"] = num_classes

    model.load_state_dict(ckpt['state_dict'])
    
    harness_params["model"] = model

    for group in range(len(test_num_groups)):
        testloader = train_test(test_dataset, test_num_groups[group])
        harness_params['testloader'] = testloader
        my_table = PrettyTable()
        my_table.field_names = ['Algorithm', 'Model Name', 'Dataset Name', 'Seed', 'Compute Accuracy?', 'Compute Grad Norm?',
                                'Compute Hessian Eigenvalues?', 'Compute Decision Boundary Distances?', 'Compute Robustness Metrics?']

        csv_path = f"{harness_params['results_path']}/FedProx_{harness_params['dataset']}_{harness_params['arch']}_client_{group}.csv"

        if not os.path.exists(csv_path):
            df = pd.DataFrame()
        else:
            df = pd.read_csv(
                csv_path, index_col=False
            )

        my_table.add_row(['FedProx', f"{harness_params['arch']}", f"{harness_params['dataset']}", ckpt['seed'], harness_params["compute_accuracy"],
                        harness_params["compute_grad_norms"], harness_params["compute_hessian_eigenvalues"], harness_params["compute_decision_boundary_distances"], harness_params["compute_robustness_metrics"]])
        print(my_table)
        results_list = compute_metrics(harness_params)

        df_temp = pd.DataFrame(results_list)

        df = pd.concat([df, df_temp], axis=1)

        df.to_csv(csv_path, index=False)
        if args.log_to_wandb:
            run = wandb.init(project=wandb_proj_name, name=wandb_run_name)
            wandb_table = wandb.Table(dataframe=df)
            run.log({f'{csv_path.split('/')[-1]}': wandb_table})
