import torch
import utils
import numpy as np
from typing import Dict, List

from .base import AbstractGlobalUpdate

class MeanWeights(AbstractGlobalUpdate):
    """Aggregate weights by taking the mean, used by FedAvg."""

    def aggregate_weights(
        self,
        local_weights_sum,
        test_losses: List[float],
        num_users,
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
        return np.divide(local_weights_sum, num_users)

    def update_global_model(self, global_model, global_weights, **kwargs) -> None:
        utils.updateFromNumpyFlatArray(global_weights, global_model)
