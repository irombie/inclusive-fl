import argparse
import itertools
import os
import subprocess
import time

import torch
import yaml
from loguru import logger
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--config-file", "-config", type=str, required=False)

MODEL_IDX = 0
DATASET_IDX = 7

IGNORE_EXPERIMENTS = [
    ("resnet18", "cifar"),
    ("vgg11_bn", "cifar"),
    ("small_cnn", "tiny-imagenet"),
]


def parse_yml(path: str = "scripts/configs_irem.yml"):
    with open(path, "r") as stream:
        try:
            logger.info("Parsing YAML file.")
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML file: {exc}")
            return None


def generate_command_args(combination):
    fl_method = combination[11]

    command_args = {
        "--model": combination[0],
        "--lr": combination[1],
        "--local_ep": combination[2],
        "--local_bs": combination[3],
        "--frac": combination[4],
        "--distribution": combination[5],
        "--dirichlet_param": combination[6],
        "--dataset": combination[7],
        "--seed": combination[8],
        "--num_users": combination[9],
        "--epochs": combination[10],
        "--fl_method": fl_method,
        "--wandb_name": f"test_suite_{combination[7]}_{combination[0]}",
    }

    if fl_method == "FedProx":
        command_args["--mu"] = combination[12]
    elif fl_method == "qFedAvg":
        command_args["--q"] = combination[12]
        command_args["--epochs"] = combination[13]
    elif fl_method == "FedSyn":
        command_args["--sparsification_ratio"] = combination[12]
        command_args["--use_fair_sparsification"] = combination[15]
        if combination[15] == 1:
            command_args["--min_sparsification_ratio"] = combination[13]
            command_args["fairness_temperature"] = combination[16]
        command_args["--sparsification_type"] = combination[14]

    return command_args


def main():
    args = parser.parse_args()

    configs = parse_yml(path=args.config_file)
    if configs is None:
        raise Exception("Unable to read config file!")

    # TODO: Fix this. Always false: "sparsification_type" == "rtopk"??
    # if "sparsification_ratio" in configs and "sparsification_type" == "rtopk":
    #     configs["choose_from_top_r_percentile"] = [
    #         1.5 * float(num) for num in configs["sparsification_ratio"]
    #     ]

    # Generate all parameter combinations
    parameter_combinations = []
    for fl_method in configs["fl_method"]:
        if fl_method == "FedProx":
            parameter_combinations += list(
                itertools.product(
                    configs["model"],
                    configs["lr"],
                    configs["local_ep"],
                    configs["local_bs"],
                    configs["frac"],
                    configs["distribution"],
                    configs["dirichlet_param"],
                    configs["dataset"],
                    configs["seed"],
                    configs["num_users"],
                    configs["epochs"],
                    [fl_method],
                    configs["mu"],
                )
            )

        elif fl_method == "qFedAvg":
            parameter_combinations += list(
                itertools.product(
                    configs["model"],
                    configs["lr"],
                    configs["local_ep"],
                    configs["local_bs"],
                    configs["frac"],
                    configs["distribution"],
                    configs["dirichlet_param"],
                    configs["dataset"],
                    configs["seed"],
                    configs["num_users"],
                    configs["epochs"],
                    [fl_method],
                    configs["q"],
                    configs["epochs"],
                )
            )
        elif fl_method == "FedAvg":
            parameter_combinations += list(
                itertools.product(
                    configs["model"],
                    configs["lr"],
                    configs["local_ep"],
                    configs["local_bs"],
                    configs["frac"],
                    configs["distribution"],
                    configs["dirichlet_param"],
                    configs["dataset"],
                    configs["seed"],
                    configs["num_users"],
                    configs["epochs"],
                    [fl_method],
                )
            )

        else:
            parameter_combinations += list(
                itertools.product(
                    configs["model"],
                    configs["lr"],
                    configs["local_ep"],
                    configs["local_bs"],
                    configs["frac"],
                    configs["distribution"],
                    configs["dirichlet_param"],
                    configs["dataset"],
                    configs["seed"],
                    configs["num_users"],
                    configs["epochs"],
                    [fl_method],
                    configs["sparsification_ratio"],
                    configs["min_sparsification_ratio"],
                    configs["sparsification_type"],
                    configs["use_fair_sparsification"],
                    configs["fairness_temperature"],
                )
            )

    # Remove unwanted experiments:
    final_list = []
    for exp in parameter_combinations:
        if not (exp[MODEL_IDX], exp[DATASET_IDX]) in IGNORE_EXPERIMENTS:
            final_list.append(exp)

    parameter_combinations = final_list

    # Launch experiments
    for i, combination in enumerate(tqdm(parameter_combinations)):
        command_args = generate_command_args(combination)
        command = [f"{k}={v}" for k, v in command_args.items()]
        command.insert(0, f"{os.getcwd()}/src/federated_main.py")  # + command
        command.insert(
            0, "python3"
        )  # might need to change to python depending on how it is aliased in your machine

        logger.info(
            f'Launching experiment {i+1}/{len(parameter_combinations)}: {" ".join(command)}'
        )

        subprocess.run(command)


if __name__ == "__main__":
    main()
