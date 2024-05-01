# Inclusive Federated Learning (PyTorch)

Codebase for experimentation on Federated learning algorithms including but not limited to standard averaging schemes, fairness based techniques and th like. The primary aim of building this code-base is to provide an easily extensible and consistent way to experiment with, extend and evaluate federated learning algorithms.

The focus of our research was device-fairness.

You will find support for the following algorithms in the repository:
1. FedAvg
2. FedProx
3. qFedAvg
4. FedSyn (our method)

The following datasets are supported:
1. FashionMNIST
2. CIFAR10
3. UTKFace (Ethnicity)
4. Tiny-Imagenet

We also provide relevant dataset utils for 
- IID Data Distribution
- Non-IID Data distribution (Dirichlet parameterization)
- Majority - Minority IID distribution

## Setup

Create a conda environment from the `environment_droplet.yml` via running the command 

```conda env create --name <envname> --file=environment_droplet.yml```

Then, activate the environment by running `conda activate <envname>`.

## Contributing
1. Install precommit hooks via `pre-commit install`. Once installed, this step can be skipped.
2. Commit to a new branch, not to main. `git add . ; git commit -m "<commit_msg>"
3. If precommit does not let you commit because it wants to format the file, repeat the above steps. You will now commit the formatted files.
4. Push the commits via `git push -u origin <branch_name>`. Once you commit and push once to a branch, you can push by just typing `git push`. 😼

## Running the experiments

### Generic Experiments with the code

1. The parameters can be supplied via a `config.yaml` an example of which is available in src/configs/
2. You can also manually override parameters provided in the config via the command line as

```bash

python src/harness.py --config configs/fedavg_fmnist.yaml

## in case you want to over-ride a parameter

python src/harness.py --config configs/fedavg_fmnist.yaml --local_parameters.local_ep 3
```
Make sure your wandb settings are configured properly so that the experiment results are being logged onto our [team space called `inclusive-fl`](https://wandb.ai/inclusive-fl). 

Federated experiment involves training a global model using many local models.
