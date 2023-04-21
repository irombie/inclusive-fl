from typing import Dict, List, Tuple, Type
import torch
import copy
from abc import ABC, abstractmethod
import numpy as np

import numpy as np
from dataclasses import dataclass

class AbstractGlobalUpdate(ABC):
    """
    Abstract class which provides common interface
        for implementing global updates, which are methods to
        update the global model.

    Each global update class must have an "aggregate_weights" method
    """

    def __init__(self, args, model: torch.nn.Module, **kwargs):
        """
        :param model: global model object
        """
        self.args = args
        self.server_params = None
        self.clients_param = None
        pass

    @abstractmethod
    def aggregate_weights(
        self,
        local_model_weights: List[Dict[str, torch.Tensor]],
        test_losses: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Method to aggregate local model weights to return global
            model weights

        :param local_model_weights: list of state dictionaries, where each element is a state
            dictionary, which maps model attributes to parameter tensors

        :param test_losses: list of local model test losses. Some methods require this,
            eg weighted average using test loss

        :return: global model state dictionary
        """
        pass

    def update_global_model(
        self, global_model: torch.nn.Module, global_weights: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update global model with global weights

        :param global_model: pytorch global model object
        :param global_weights: global model state dictionary after aggregation
        """
        global_model.load_state_dict(global_weights)

    @staticmethod
    def update_local_models(
        local_models: List[torch.nn.Module], global_weights: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update local models with global weights
        In this case, updating each local model to be exactly the
            same as global model

        :param local_models: list of pytorch local models
        :param global_weights: global model state dictionary after aggregation
        """
        for model in local_models:
            model.load_state_dict(global_weights)


class MeanWeights(AbstractGlobalUpdate):
    """Aggregate weights by taking the mean, used by FedAvg."""

    def aggregate_weights(
        self,
        local_model_weights: List[Dict[str, torch.Tensor]],
        test_losses: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the mean of the weights.

        All local models and global model are assumed to have the
        same architecture, and hence the same keys in the state dict

        :param local_model_weights: list of state dictionaries, where each element is a state
            dictionary, which maps model attributes to parameter tensors

        :return: global model state dictionary,
            which is the average of all local models provided
        """
        weights_scalar = np.divide(test_losses, np.sum(test_losses))
        w_avg = copy.deepcopy(local_model_weights[0])
        for key in w_avg.keys():
            for i in range(1, len(local_model_weights)):
                w_avg[key] += local_model_weights[i][key] 

            if self.args.reweight_loss_avg==1:
                w_avg[key] *= weights_scalar[i]
            else:
                w_avg[key] = torch.div(w_avg[key], len(local_model_weights))
        return w_avg


class MeanWeightsNoBatchNorm(AbstractGlobalUpdate):
    """Fed BN method. See https://arxiv.org/abs/2102.07623"""

    def __init__(self, args, model: torch.nn.Module):
        super().__init__(args, model)
        batchnorm_layers = self._find_batchnorm_layers(model)
        assert (
            len(batchnorm_layers) != 0
        ), "No batch norm layers found, cannot use FedBN."
        self.batchnorm_layers = batchnorm_layers

    @staticmethod
    def _find_batchnorm_layers(model: torch.nn.Module) -> Tuple[str, ...]:
        """
        Find the name of layers in model which are batch norm layers.

        :param model: pytorch model

        :return: tuple of batch norm layer names
        """
        batch_norm_layers = []
        for module_name, module in model.named_modules():
            if isinstance(
                module,
                (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
            ):
                batch_norm_layers.append(module_name)
        return tuple(batch_norm_layers)

    def aggregate_weights(
        self,
        local_model_weights: List[Dict[str, torch.Tensor]],
        test_losses: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Averaging weights, but ignoring any batch norm layers

        :param local_model_weights: list of state dictionaries, where each element is a state
            dictionary, which maps model attributes to parameter tensors

        :return: global model state dictionary,
        """
        weights_scalar = np.divide(test_losses, np.sum(test_losses))
        w_avg = copy.deepcopy(local_model_weights[0])
        keys = list(w_avg.keys())
        for key in keys:
            if any(substring in key for substring in self.batchnorm_layers):
                del w_avg[key]
            else:
                for i in range(1, len(local_model_weights)):
                    w_avg[key] += local_model_weights[i][key]
                    
                if self.args.reweight_loss_avg==1:
                    w_avg[key] *= weights_scalar[i]
                else:
                    w_avg[key] = torch.div(w_avg[key], len(local_model_weights))
        return w_avg

    @staticmethod
    def update_global_model(
        global_model: torch.nn.Module, global_weights: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update global model with global weights

        Due to absense of batch norm keys, strict=False is used in loading
            of state dict

        :param global_model: pytorch global model object
        :param global_weights: global model state dictionary after aggregation
        """
        global_model.load_state_dict(global_weights, strict=False)

    @staticmethod
    def update_local_models(
        local_models: List[torch.nn.Module], global_weights: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update local models with global weights
        In this case, updating local models and preserving batch norm layers

        :param local_models: list of pytorch local models
        :param global_weights: global model state dictionary after aggregation
        """
        for model in local_models:
            model.load_state_dict(global_weights, strict=False)


class AverageWeightsWithTestLoss(AbstractGlobalUpdate):
    """Aggregate weights by using weighted average based on test loss."""

    def aggregate_weights(
        self,
        local_model_weights: List[Dict[str, torch.Tensor]],
        test_losses: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the weighted average of the weights with respect to the test loss.

        All local models and global model are assumed to have the
        same architecture, and hence the same keys in the state dict

        :param local_model_weights: list of state dictionaries, where each element is a state
            dictionary, which maps model attributes to parameter tensors

        :param test_losses: list of local model test losses

        :return: global model state dictionary
        """
        weights_scalar = np.divide(test_losses, np.sum(test_losses))
        model_layers = local_model_weights[0].keys()
        w_avg = {}
        # Loop through layers in model
        for key in model_layers:
            # Loop through each users losses
            for i in range(len(local_model_weights)):
                if key in w_avg:
                    w_avg[key] += local_model_weights[i][key] * weights_scalar[i]
                else:
                    w_avg[key] = local_model_weights[i][key] * weights_scalar[i]
        return w_avg


NAME_TO_GLOBAL_UPDATE: Dict[str, Type[AbstractGlobalUpdate]] = {
    "FedAvg": MeanWeights,
    "FedBN": MeanWeightsNoBatchNorm,
    "FedProx": MeanWeights,
    "TestLossWeighted": AverageWeightsWithTestLoss,
}

def get_global_update(
    args, model: torch.nn.Module, **kwargs
) -> AbstractGlobalUpdate:
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
