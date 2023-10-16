import torch
import numpy as np
from typing import Dict, List

from .base import AbstractGlobalUpdate

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
