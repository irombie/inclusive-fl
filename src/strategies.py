import torch
import copy

class BaseStrategy:
    def __init__(self):
        pass

    def average_weights(self, weights):
        """Return the average of the weights."""
        pass

    def local_loss(self, loss):
        """Return the local loss."""
        pass

class FedAvg(BaseStrategy):
    def average_weights(self, weights):
        """Return the average of the weights."""
        w_avg = copy.deepcopy(weights[0])
        for key in w_avg.keys():
            for i in range(1, len(weights)):
                w_avg[key] += weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(weights))
        return w_avg

    def local_loss(self, loss):
        return loss
    
class FedProx(BaseStrategy):
    def __init__(self, local_params, global_params):
        self.mu = 0.5
        self.local_params = local_params
        self.global_params = global_params

    def average_weights(self, weights):
        """Return the average of the weights."""
        w_avg = copy.deepcopy(weights[0])
        for key in w_avg.keys():
            for i in range(1, len(weights)):
                w_avg[key] += weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(weights))
        return w_avg
    
    def local_loss(self, loss):
        fedprox_term = 0.0
        proximal_term = 0.0
        
        for w, w_t in zip(self.local_params, self.global_params):
            proximal_term += (w - w_t).norm(2)
        
        fedprox_term = (self.args.mu / 2) * proximal_term
        return loss + fedprox_term
    
AVG_METHOD_NAME_TO_CLASS = {
    "FedAvg": FedAvg,
    "FedProx": FedProx,
}