# Inclusive Federated Learning (PyTorch)

Experiments are produced on Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.


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

1. To check which parameters are there to configure, head to `options.py` 👀
2. If you add a new parameter there, make sure to add it to the README below so others can be informed!
3. Whenever you are creating a new experiment set, **create a new project with a unique name** by running the code with the flag `--wandb_name <name>`. If you are running experiments in that set, **create runs within that project** by using the same name with that flag. 
4. Make sure your wandb settings are configured properly so that the experiment results are being logged onto our [team space called `inclusive-fl`](https://wandb.ai/inclusive-fl). 
5. Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```bash
python src/federated_main.py --model=small_cnn --dataset=cifar --iid=1 --epochs=10
```
You can change the default values of other parameters to simulate different conditions. 

### Scale out (Large Scale Experiments)
1. Define a .yaml file and upload it to scripts/ so we can be clear about what experiment each of y'all are running (to ensure no gaps/overlap).
2. You can find a template for the yaml in the scripts/ folder, make a copy and edit as required.
3. Ensure that your wandb project is linked to the group project and not your personal ID to avoid challenges in aggregating.
*To run the grid based on your experiments*

```bash
python src/scale_out_exp.py --config-file <YOUR CONFIG FILE>
```

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'fashionmnist'. Options: 'fashionmnist', 'cifar', 'tiny-imagenet', 'utkface' (only ethnicity)
* ```--model:```    Default: 'small_cnn'. Options: 'small_cnn', 'resnet9', 'resnet18', 'vgg11_bn'
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

## Federated Parameters 
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--dist_noniid:```  Used in non-iid setting (--iid=0). Option to give each user the proportion of each class samples according to Dirichlet distribution. Default set to 0 to omit this option. Set to 1 to select this option. (The default value of other non-IID option for unequal splits --unequal=0 must stay unchanged).

## Algorithm specific parameters
* ```--fl_method:``` Name of method to use. Currently supports "FedAvg", "FedProx", "FedBN" and "TestLossWeighted"
* ```--mu:``` mu value for FedProx
* ```--sparsification_ratio``` The ratio of parameters that will be sent from the clients to the server at each round. 
* and some others :D