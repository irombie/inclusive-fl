import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from fastargs import get_current_config
from fastargs.decorators import param

import utils


class AbstractGlobalUpdate(ABC):
    """
    Abstract class which provides common interface
        for implementing global updates, which are methods to
        update the global model.

    Each global update class must have an "aggregate_weights" method
    """

    def __init__(self, model: torch.nn.Module, **kwargs):
        """
        :param model: global model object
        """
        self.server_params = None
        self.clients_param = None
        self.config = get_current_config()

    @abstractmethod
    def aggregate_weights(
        self,
        local_model_weights: List[Dict[str, torch.Tensor]],
        test_losses: List[float],
        **kwargs,
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

    def update_global_model(
        self,
        global_model: torch.nn.Module,
        global_weights: Dict[str, torch.Tensor],
        **kwargs,
    ) -> None:
        """
        Update global model with global weights

        :param global_model: pytorch global model object
        :param global_weights: global model state dictionary after aggregation
        """
        global_model.load_state_dict(global_weights)

    @staticmethod
    def update_local_models(local_models: List[torch.nn.Module], global_weights: Dict[str, torch.Tensor]) -> None:
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
        num_clients,
        local_weights_sum,
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
        return np.divide(local_weights_sum, num_clients)

    def update_global_model(self, global_model, global_weights, **kwargs) -> None:
        utils.updateFromNumpyFlatArray(global_weights, global_model)


class MeanWeightsSparsified(AbstractGlobalUpdate):
    """Aggregate weights by taking the mean, used by FedSyn."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model)

    @param("global_parameters.global_lr")
    def aggregate_weights(
        self,
        global_lr,
        local_weights_sum,
        global_model,
        local_bitmasks_sum,
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
        weigted_local_model_sum = np.divide(
            local_weights_sum,
            local_bitmasks_sum,
            out=np.zeros_like(local_weights_sum),
            where=local_bitmasks_sum != 0,
        )
        flat_glob = utils.flatten(global_model)
        return flat_glob + global_lr * weigted_local_model_sum


class MeanWeightsNoBatchNorm(AbstractGlobalUpdate):
    """Fed BN method. See https://arxiv.org/abs/2102.07623"""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        batchnorm_layers = self._find_batchnorm_layers(model)
        assert len(batchnorm_layers) != 0, "No batch norm layers found, cannot use FedBN."
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

    @param("fl_parameters.reweight_loss_avg")
    def aggregate_weights(
        self,
        reweight_loss_avg,
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

                if reweight_loss_avg:
                    w_avg[key] *= weights_scalar[i]
                else:
                    w_avg[key] = torch.div(w_avg[key], len(local_model_weights))
        return w_avg

    @staticmethod
    def update_global_model(global_model: torch.nn.Module, global_weights: Dict[str, torch.Tensor]) -> None:
        """
        Update global model with global weights

        Due to absense of batch norm keys, strict=False is used in loading
            of state dict

        :param global_model: pytorch global model object
        :param global_weights: global model state dictionary after aggregation
        """
        global_model.load_state_dict(global_weights, strict=False)

    @staticmethod
    def update_local_models(local_models: List[torch.nn.Module], global_weights: Dict[str, torch.Tensor]) -> None:
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
            for i, weights in enumerate(local_model_weights):
                if key in w_avg:
                    w_avg[key] += weights[key] * weights_scalar[i]
                else:
                    w_avg[key] = weights[key] * weights_scalar[i]
        return w_avg


class qFedAvgGlobalUpdate(AbstractGlobalUpdate):
    """Aggregate weights by using average based on the qFedAvg loss."""

    @staticmethod
    def aggregate_weights(
        global_model: torch.nn.Module,
        delta_sum: np.array,
        h_sum: np.array,
    ) -> Dict[str, torch.Tensor]:
        """
        Update global model with global weights

        Due to absense of batch norm keys, strict=False is used in loading
            of state dict

        :param global_model: pytorch global model object
        :param local_deltas: list of delta computed in qFedAvg Algorithm 2
        :param local_hs: list of h computed in qFedAvg Algorithm 2

        :return: updated global model with qFedAvg
        """
        weigted_local_model_sum = np.divide(
            delta_sum,
            h_sum,
            out=np.zeros_like(delta_sum),
            where=h_sum != 0,
        )
        flat_glob = utils.flatten(global_model)
        return flat_glob - weigted_local_model_sum


NAME_TO_GLOBAL_UPDATE: Dict[str, Type[AbstractGlobalUpdate]] = {
    "FedAvg": MeanWeights,
    "FedBN": MeanWeightsNoBatchNorm,
    "FedProx": MeanWeights,
    "TestLossWeighted": AverageWeightsWithTestLoss,
    "FedSyn": MeanWeightsSparsified,
    "qFedAvg": qFedAvgGlobalUpdate,
}


@param("fl_parameters.fl_method")
def get_global_update(fl_method, model: torch.nn.Module, **kwargs) -> AbstractGlobalUpdate:
    """
    Get global update from federated learning method name

    :param federated_learning_method: name of federated learning method.
        Currently supports, FedAvg, FedBN

    :param model: global model used for initialising global update class

    :return: initialised global update object
    """
    if fl_method in NAME_TO_GLOBAL_UPDATE:
        return NAME_TO_GLOBAL_UPDATE[fl_method](model, **kwargs)
    else:
        raise ValueError(f"Unsupported federated learning method name {fl_method} for global update.")
