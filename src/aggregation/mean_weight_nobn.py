import copy
import torch
import numpy as np
from typing import Dict, List, Tuple

from .base import AbstractGlobalUpdate

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

                if self.args.reweight_loss_avg == 1:
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