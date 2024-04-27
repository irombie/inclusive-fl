from fastargs import Param, Section
from fastargs.validation import And, OneOf


def get_current_params():
    Section("model", "model parameters").params(
        model_name=Param(
            And(
                str,
                OneOf(
                    [
                        "SmallCNN",
                        "ResNet9",
                        "ResNet18",
                        "MLP",
                        "LogisticRegression",
                        "VGG",
                        "CNNFashionMNIST",
                    ]
                ),
            ),
            "Global model architecture (common across devices)",
            required=True,
        ),
        num_features=Param(int, "Number of features", default=None),
        device=Param(str, "device to run the model on", default=None),
    )

    Section("global_parameters", "global parameters").params(
        global_rounds=Param(int, "number of rounds of training", required=True),
        client_frac=Param(
            float, "Client fraction sampled at each round for training", required=True
        ),
        global_lr=Param(float, "global learning rate", default=1),
    )

    Section("client_parameters", "general client parameters").params(
        local_epochs=Param(int, "number of local epochs", default=5),
        local_bs=Param(int, "local batch size", default=64),
        local_lr=Param(float, "local learning rate", default=0.01),
    )

    Section("dataset", "dataset parameters").params(
        dataset_name=Param(
            And(
                str,
                OneOf(
                    [
                        "CIFAR10",
                        "CIFAR100",
                        "FashionMNIST",
                        "MNIST",
                        "Synthetic",
                        "SVHN",
                    ]
                ),
            ),
            "dataset name",
            required=True,
        ),
        data_dir=Param(str, "path to data directory", default="./data/"),
        num_classes=Param(int, "Number of classes", required=True),
        num_features=Param(int, "Number of features", default=0),
    )
    Section("SVHN_data").enable_if(
        lambda cfg: cfg["dataset.dataset_name"] == "SVHN"
    ).params(
        extra=Param(bool, "Whether to add the extra SVHN training data", default=False)
    )

    Section("fl_parameters", "FL Training parameters").params(
        num_clients=Param(int, "number of clients", required=True),
        frac=Param(float, "fraction of clients used per round", default=0.1),
        save_every=Param(int, "save model every x rounds", default=2),
        ckpt_path=Param(str, "path to save checkpoints", default="./checkpoints/"),
        seed=Param(int, "random seed", default=42),
        fl_method=Param(
            And(str, OneOf(["FedAvg", "FedProx", "qFedAvg", "FedSyn"])),
            "federated learning method",
            required=True,
        ),
        sparsification_ratio=Param(float, "sparsification ratio", default=0.2),
        sparsification_type=Param(str, "sparsification type", default="randk"),
        choose_from_top_r_percentile=Param(
            float, "choose from top r percentile", default=1
        ),
        use_fair_sparsification=Param(bool, "use fair sparsification", default=False),
        fairness_temperature=Param(float, "fairness temperature", default=1),
        min_sparsification_ratio=Param(
            float, "minimum sparsification ratio", default=0
        ),
        mu=Param(float, "mu value for FedProx", default=0.01),
        q=Param(float, "q value for qFedAvg", default=None),
        eps=Param(float, "eps value for qFedAvg", default=1e-6),
        beta=Param(float, "beta value for qFedAvg", default=0.5),
    )

    Section("split_params", "parameters for splitting the dataset").params(
        split_type=Param(
            And(str, OneOf(["iid", "non-iid", "majority_minority"])),
            "split type",
            required=True,
        ),
        overlap=Param(
            float, "overlap between majority and minority classes", default=0.1
        ),
        majority_proportion=Param(float, "proportion of majority class", default=0.8),
        min_proportion=Param(float, "proportion of minority class", required=True),
        dirichlet_param=Param(float, "dirichlet parameter", default=10),
    )

    Section("training_params", "harness related stuff").params(
        ckpt_path=Param(str, "path to save checkpoints", default="checkpoints"),
        seed=Param(int, "random seed", default=42),
        wandb_project=Param(str, "wandb project name"),
    )
