# Inclusive Federated Learning (PyTorch)

Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.


## Setup

Create a conda environment from the `environment_droplet.yml` via running the command 

```conda env create --name <envname> --file=environment_droplet.yml```

Then, activate the environment by running `conda activate <envname>`.

## Contributing
1. Clone/pull the repository. `git pull origin main`
2. Switch to a new branch, something other than main, and possibly appropriately named so it reflects what the changes are about. `git checkout -b <branch_name>`
3. Make desired changes. 
4. Commit and push to the new branch. `git add . ; git commit -m "<commit_msg>"; git push -u origin <branch_name>`. Once you commit and push once to a branch, you can push by just typing `git push`. 😼
5. Create a pull request (PR) by heading to the repo's website and clicking on `compare & pull request`.
6. Give an easy to understand name and provide explanation on what the PR does. 
7. Ask for reviews from team members. Wait for a day for people to give feedback. After that, once your PR gets approved by someone, merge to `main`.
8. YOU are AMAZING! 🥳🎉 Thank you for being such a valuable team member 💙

## Running the experiments
1. To check which parameters are there to configure, head to `options.py` 👀
2. If you add a new parameter there, make sure to add it to the README below so others can be informed!
3. Whenever you are creating a new experiment set, **create a new project with a unique name** by running the code with the flag `--wandb_name <name>`. If you are running experiments in that set, **create runs within that project** by using the same name with that flag. 
4. Make sure your wandb settings are configured properly so that the experiment results are being logged onto our [team space called `inclusive-fl`](https://wandb.ai/inclusive-fl). 
5. Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
You can change the default values of other parameters to simulate different conditions. 

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters 
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting (--iid must be set to 0). Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.
* ```--dist_noniid:```  Used in non-iid setting (--iid must be set to 0). Option to assign each user proportion of each class samples according to Dirichlet distribution. Default set to 0 to omit this option. Set to 1 to select this option. Option to be selected concurrent non-IID option of unequal splits must be set to 0 (--unequal=0).

## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
