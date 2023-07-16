import itertools
import subprocess
import time

import yaml
from tqdm import tqdm

# Define the parameter combinations

def parse_yml(path: str = "scripts/configs.yml"):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

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
            print("i dont understand ur answer, so assigning u cpu ☠️")
    else:
        raise Exception("do u have gpu or not? pls specify")

    configs = parse_yml()
    if configs is None:
        raise Exception("Unable to read config file!")

    # Generate all parameter combinations
    parameter_combinations = list(itertools.product(
        configs["lr"],
        configs["local_ep"],
        configs["frac"],
        configs["iid"],
        configs["dist_noniid"],
        configs["mu"],
        configs["dataset"],
        configs["reweight_loss_avg"],
        configs["seed"],
        configs["num_users"],
        configs["epochs"],
        configs["fl_method"]
    ))

    # Parse command-line arguments

    # Launch experiments
    for i, combination in tqdm(enumerate(parameter_combinations)):
        lr, local_ep, frac, iid, dist_noniid, mu, dataset, reweight_test_loss, seed, num_users, epochs, fl_method = combination

        # Apply conditionals
        if iid == 0:
            dist_noniid = 1
        if dist_noniid != 0:
            iid = 0
        if fl_method != "FedProx":
            mu = 0
        # Generate command-line arguments
        command = [
        'python3',
        'src/federated_main.py',
        f'--gpu={gpu}',
        f'--device={device}',
        '--model=cnn',
        f'--num_users={num_users}',
        f'--epochs={epochs}',
        f'--wandb_name=test_suite_{time.time()}',
        f'--lr={lr}',
        f'--local_ep={local_ep}',
        f'--frac={frac}',
        f'--iid={iid}',
        f'--dist_noniid={dist_noniid}',
        f'--mu={mu}',
        f'--dataset={dataset}',
        f'--fl_method={fl_method}',
        f'--reweight_loss_avg={reweight_test_loss}',
        f'--seed={seed}'
    ]


        # Launch the experiment as a subprocess
        print(f'Launching experiment {i+1}/{len(parameter_combinations)}: {" ".join(command)}')
        subprocess.run(command)

if __name__ == "__main__":
    main()