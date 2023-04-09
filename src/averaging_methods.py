from typing import Dict, List, Tuple, Type
import torch
import copy
from abc import ABC, abstractmethod


class AbstractAvgMethod(ABC):
    """
    Abstract class which provides common interface 
        for implementing averaging methods.

    Each averaging method class must have an "average_weights" method
    """

    def __init__(self, model: torch.nn.Module):
        """
        :param model: global model object
        """
        pass

    @abstractmethod
    def average_weights(
        self, local_model_weights: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Method to aggregate local model weights to return global
            model weights

        :param local_model_weights: list of state dictionaries, where each element is a state
            dictionary, which maps model attributes to parameter tensors

        :return: global model state dictionary
        """
        pass


class FedAvg(AbstractAvgMethod):
    """Fed Avg implementation."""

    def average_weights(
        self, local_model_weights: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the average of the weights.

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


class FedBN(AbstractAvgMethod):
    """Fed BN method. See https://arxiv.org/abs/2102.07623"""

    def __init__(self, model: torch.nn.Module):
        batchnorm_layers = self._find_batchnorm_layers(model)
        assert len(batchnorm_layers) !=0, "No batch norm layers found, cannot use FedBN."
        self.batchnorm_layers = batchnorm_layers

    @staticmethod
    def _find_batchnorm_layers(model: torch.nn.Module) -> Tuple[str, ...]:
        """
        Find the name of layers in model which batch norm layers.

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

    def average_weights(
        self, local_model_weights: List[Dict[str, torch.Tensor]]
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


AVG_METHOD_NAME_TO_CLASS: Dict[str, Type[AbstractAvgMethod]] = {
    "FedAvg": FedAvg,
    "FedBN": FedBN,
}
