import torch
import utils
import numpy as np
from typing import Dict

from .base import AbstractGlobalUpdate

class MeanWeightsSparsified(AbstractGlobalUpdate):
    """Aggregate weights by taking the mean, used by FedSyn."""

    def __init__(self, args, model: torch.nn.Module, **kwargs):
        super().__init__(args, model)
        self.global_learning_rate = args.global_lr

    def aggregate_weights(
        self,
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
        return flat_glob + self.global_learning_rate * weigted_local_model_sum
