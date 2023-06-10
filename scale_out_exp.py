import argparse
import itertools
import subprocess
import os

# Define the parameter combinations

parser = argparse.ArgumentParser()
parser.add_argument('--algo', help='Path to the script to be executed')
args = parser.parse_args()

lr_values = [0.01]
local_ep_values = [2, 5]
frac_values = [0.4, 1]
iid_values = [0, 1]
dist_noniid_values = [0.3, 0.5]
if args.algo == 'FedProx':
    mu_values = [0.01, 0.001]
else:
    mu_values = [0]
dataset_values = ['cifar']
reweight_test_loss_values = [0, 1]

# Generate all parameter combinations
parameter_combinations = list(itertools.product(
    lr_values,
    local_ep_values,
    frac_values,
    iid_values,
    dist_noniid_values,
    mu_values,
    dataset_values,
    reweight_test_loss_values
))

# Parse command-line arguments

# Launch experiments
for i, combination in enumerate(parameter_combinations):
    lr, local_ep, frac, iid, dist_noniid, mu, dataset, reweight_test_loss = combination

    # Apply conditionals
    if iid == 0:
        dist_noniid = 1
    if dist_noniid != 0:
        iid = 0
    # Generate command-line arguments
    command = [
    'python',
    'src/federated_main.py',
    '--gpu=0',
    '--device=cuda',
    '--model=cnn',
    '--num_users=20',
    '--epochs=30',
    '--wandb_name=\'scale_out\'',
    f'--lr={lr}',
    f'--local_ep={local_ep}',
    f'--frac={frac}',
    f'--iid={iid}',
    f'--dist_noniid={dist_noniid}',
    f'--mu={mu}',
    f'--dataset={dataset}',
    f'--fl_method={args.algo}',
    f'--reweight_loss_avg={reweight_test_loss}',
    '--seed=1234'
]


    # Launch the experiment as a subprocess
    print(f'Launching experiment {i+1}/{len(parameter_combinations)}: {" ".join(command)}')
    subprocess.run(command)
