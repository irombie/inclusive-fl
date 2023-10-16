import torch
import utils
import numpy as np
from typing import Dict

from .base import AbstractGlobalUpdate

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