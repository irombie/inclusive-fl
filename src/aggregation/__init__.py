import torch
from typing import Dict, Type

from .base import AbstractGlobalUpdate
from .mean_weight import MeanWeights
from .qfedavg import qFedAvgGlobalUpdate
from .mean_weight_nobn import MeanWeightsNoBatchNorm
from .mean_weight_sparsified import MeanWeightsSparsified
from .avg_weight_with_test_loss import AverageWeightsWithTestLoss


NAME_TO_GLOBAL_UPDATE: Dict[str, Type[AbstractGlobalUpdate]] = {
    "FedAvg": MeanWeights,
    "FedBN": MeanWeightsNoBatchNorm,
    "FedProx": MeanWeights,
    "TestLossWeighted": AverageWeightsWithTestLoss,
    "FedSyn": MeanWeightsSparsified,
    "qFedAvg": qFedAvgGlobalUpdate,
}


def get_global_update(args, model: torch.nn.Module, **kwargs) -> AbstractGlobalUpdate:
    """
    Get global update from federated learning method name

    :param federated_learning_method: name of federated learning method.
        Currently supports, FedAvg, FedBN

    :param model: global model used for initialising global update class

    :return: initialised global update object
    """
    if args.fl_method in NAME_TO_GLOBAL_UPDATE:
        return NAME_TO_GLOBAL_UPDATE[args.fl_method](args, model, **kwargs)
    else:
        raise ValueError(
            f"Unsupported federated learning method name {args.fl_method} for global update."
        )
