import torch
from typing import Dict, List
from abc import ABC, abstractmethod

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
        pass

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