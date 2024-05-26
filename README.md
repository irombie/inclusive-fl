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
0. Make sure your env is up to date by installing the requirements into your env via `pip install -r requirements.txt`
1. Clone/pull the repository. `git pull origin main`
2. Install precommit hooks via `pre-commit install`. Once installed, this step can be skipped.
3. Switch to a new branch, something other than main, and possibly appropriately named so it reflects what the changes are about. `git checkout -b <branch_name>`
4. Make desired changes. 
5. Commit to the new branch. `git add . ; git commit -m "<commit_msg>"
6. If precommit does not let you commit because it wants to format the file, repeat the above steps. You will now commit the formatted files.
7. Push the commits via git push -u origin <branch_name>`. Once you commit and push once to a branch, you can push by just typing `git push`. 😼
8. Create a pull request (PR) by heading to the repo's website and clicking on `compare & pull request`.
9. Give an easy to understand name and provide explanation on what the PR does. 
10. Ask for reviews from team members. Wait for a day for people to give feedback. After that, once your PR gets approved by someone, merge to `main`.
11. YOU are AMAZING! 🥳🎉 Thank you for being such a valuable team member 💙

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

## Creating sweeps

You can create a hyperparameter sweep or an experimentation sweep on wandb easily. First of all, create a (set of) single experiment config(s) in the appropriate directory under `configs/`. Currently, we organize experiments first by the dataset name and then by the model name.

If you are planning to do an experiment, create an experimentation sweep config under `sweeps/experiment_sweeps`. You can give it an appropriate project name to identify the experiment easily later on. This type of config will run a set of configs. 

Alternatively, you can run a hyperparameter sweep on a single experiment config. Create the sweep config under `sweeps/hparam_sweeps/`. Add the path of the config you want to run in the `command` section of the config and specify the grids for each hyperparameter in the config. You can refer to the existing configs if needed.

Once you have the sweep config prepared, running the sweep is as easy as running the command `wandb sweep <CONFIG_PATH>`. Then, wandb will prompt you with a command to kick off the sweep. Copy and paste it in your terminal and you are done! 