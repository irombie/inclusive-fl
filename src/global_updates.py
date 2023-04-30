from typing import Dict, List, Tuple, Type
import torch
import copy
from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass

class AbstractGlobalUpdate(ABC):
    """
    Abstract class which provides common interface
        for implementing global updates, which are methods to
        update the global model.

    Each global update class must have an "aggregate_weights" method
    """

    def __init__(self, model: torch.nn.Module):
        """
        :param model: global model object
        """
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
        w_avg = copy.deepcopy(local_model_weights[0])
        for key in w_avg.keys():
            for i in range(1, len(local_model_weights)):
                w_avg[key] += local_model_weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(local_model_weights))
        return w_avg


class MeanWeightsNoBatchNorm(AbstractGlobalUpdate):
    """Fed BN method. See https://arxiv.org/abs/2102.07623"""

    def __init__(self, model: torch.nn.Module):
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
        w_avg = copy.deepcopy(local_model_weights[0])
        keys = list(w_avg.keys())
        for key in keys:
            if any(substring in key for substring in self.batchnorm_layers):
                del w_avg[key]
            else:
                for i in range(1, len(local_model_weights)):
                    w_avg[key] += local_model_weights[i][key]
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



@dataclass
class ScaffoldParams:
    """Class for keeping track of Scaffold Parameters."""
    control: dict
    delta_control: dict
    delta_y: dict

class ScaffoldMeanWeights(AbstractGlobalUpdate):
    def __init__(self, model: torch.nn.Module, num_users: int):
        self.global_model = model
        self.num_users = num_users
        
        self.server_params = ScaffoldParams({}, {}, {})
        for k, v in self.global_model.named_parameters():
            self.server_params.control[k] = torch.zeros_like(v.data)
            self.server_params.delta_control[k] = torch.zeros_like(v.data)
            self.server_params.delta_y[k] = torch.zeros_like(v.data)
        
        self.clients_param = []
        for i in range(self.num_users):
            temp = copy.deepcopy(self.server_params)
            self.clients_param.append(temp)
    
    def aggregate_weights(
        self,
        local_model_weights: List[Dict[str, torch.Tensor]],
        test_losses: List[float],
    ) -> Dict[str, torch.Tensor]:        
        # compute
        self.x = {}
        self.c = {}
        
        # init
        for k, v in local_model_weights[0].items():
            self.x[k] = torch.zeros_like(v.data)
            self.c[k] = torch.zeros_like(v.data)

        for j in range(len(local_model_weights)):
            for k, v in local_model_weights[j].items():
                self.x[k] += torch.div(self.clients_param[j].delta_y[k], len(local_model_weights))  # averaging
                self.c[k] += torch.div(self.clients_param[j].delta_control[k], len(local_model_weights))  # averaging

        # Update server's control variables
        for k, v in self.global_model.named_parameters():
            self.server_params.control[k].data += self.c[k].data / (len(local_model_weights) / self.num_users)

        # Update global model weights
        for k, v in self.global_model.named_parameters():
            v.data += self.x[k].data  # lr=1

        return self.global_model.state_dict()
        

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
    "Scaffold": ScaffoldMeanWeights,
    "TestLossWeighted": AverageWeightsWithTestLoss,
}

def get_global_update(
    federated_learning_method: str, model: torch.nn.Module, **kwargs
) -> AbstractGlobalUpdate:
    """
    Get global update from federated learning method name

    :param federated_learning_method: name of federated learning method.
        Currently supports, FedAvg, FedBN

    :param model: global model used for initialising global update class

    :return: initialised global update object
    """
    if federated_learning_method in NAME_TO_GLOBAL_UPDATE:
        return NAME_TO_GLOBAL_UPDATE[federated_learning_method](model, **kwargs)
    else:
        raise ValueError(
            f"Unsupported federated learning method name {federated_learning_method} for global update."
        )
