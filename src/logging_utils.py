import datetime

from fastargs import get_current_config

import wandb


class WandbLogger:
    def __init__(self) -> None:
        self.now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.config = get_current_config()
        self.flat_config = self.parse_config(self.config)
        self.project_name = f"fl-{self.config['model.model_name']}-{self.config['dataset.dataset_name']}"
        fl_method = self.config["fl_parameters.fl_method"]
        num_clients = self.config["fl_parameters.num_clients"]
        frac = self.config["fl_parameters.frac"]
        sparsification_ratio = self.config["fl_parameters.sparsification_ratio"]
        self.run_name = f"{fl_method}-clients_{num_clients}-frac_{frac}-sparsification_ratio_{sparsification_ratio}-{self.now}"
        self.tags = [f"{k}:{v}" for k, v in self.flat_config.items()]
        self.run = wandb.init(
            project=self.project_name,
            config=self.flat_config,
            name=self.run_name,
            tags=self.tags,
        )

    def log(self, metrics: dict, **kwargs) -> None:
        self.run.log(metrics)

    def close(self, exit_code, **kwargs) -> None:
        self.run.finish(exit_code=exit_code)

    @staticmethod
    def parse_config(config):
        """Parses the fastargs config into a dictionary."""
        return {"-".join(param): value for param, value in config.content.items()}
