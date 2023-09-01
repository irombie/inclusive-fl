import itertools
import subprocess
import time

import yaml
from tqdm import tqdm
from loguru import logger

def parse_yml(path: str = "scripts/configs.yml"):
    with open(path, "r") as stream:
        try:
            logger.info("Parsing YAML file.")
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML file: {exc}")
            return None
        
def generate_command_args(combination, gpu, device, timestamp):
    fl_method = combination[11]

    command_args = {
        "python3": "src/federated_main.py",
        "--gpu": gpu,
        "--device": device,
        "--model": combination[0],
        "--lr": combination[1],
        "--local_ep": combination[2],
        "--local_bs": combination[3],
        "--frac": combination[4],
        "--iid": combination[5],
        "--alpha": combination[6],
        "--dataset": combination[7],
        "--seed": combination[8],
        "--num_users": combination[9],
        "--epochs": combination[10],
        "--fl_method": fl_method,
        "--wandb_name": f"test_suite_{timestamp}",
    }

    if fl_method == "FedProx":
        command_args["--mu"] = combination[12]
    elif fl_method == "qFedAvg":
        command_args["--q"] = combination[12]
        command_args["--eps"] = combination[13]
    elif fl_method in ["FedAvg", "FedSyn"]:
        command_args["--sparsification_ratio"] = combination[12]
        command_args["--sparsification_type"] = combination[13]
        command_args["--use_fair_sparsification"] = combination[14]

    return command_args


def main():
    gpu = None
    device = "cpu"
    val = input("do u have gpu😈 (y or n): ")
    if val == "y":
        gpu = 0
        device = "cuda"
    elif val == "n":
        val_device = input("do u wanna use mac mps thingy (y or n):")
        if val_device == "y":
            device = "mps"
        elif val_device != "n":
            logger.warning("i dont understand ur answer, so assigning u cpu ☠️")
    else:
        raise Exception("do u have gpu or not? pls specify")

    configs = parse_yml()
    if configs is None:
        raise Exception("Unable to read config file!")

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
                    configs["iid"],
                    configs["alpha"],
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
                    configs["iid"],
                    configs["alpha"],
                    configs["dataset"],
                    configs["seed"],
                    configs["num_users"],
                    configs["epochs"],
                    [fl_method],
                    configs["q"],
                    configs["eps"],
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
                    configs["iid"],
                    configs["alpha"],
                    configs["dataset"],
                    configs["seed"],
                    configs["num_users"],
                    configs["epochs"],
                    [fl_method],
                    configs["sparsification_ratio"],
                    configs["sparsification_type"],
                    configs["use_fair_sparsification"],
                )
            )

    timestamp = time.time()

    # Launch experiments
    for i, combination in enumerate(tqdm(parameter_combinations)):
        command_args = generate_command_args(combination, fl_method, gpu, device, timestamp)
        command = [f"{k}={v}" for k, v in command_args.items()]

        logger.info(f'Launching experiment {i+1}/{len(parameter_combinations)}: {" ".join(command)}')

        subprocess.run(command)

if __name__ == "__main__":
    main()

