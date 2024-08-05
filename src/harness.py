from argparse import ArgumentParser
from pathlib import Path
import copy
import sys
import traceback

from fastargs import get_current_config, set_current_config
from fastargs.decorators import param
import numpy as np
import torch

from general_utils import (
    custom_exponential_sparsity,
    linearly_interpolated_softmax,
    flatten,
    updateFromNumpyFlatArray,
)

from fl_dataset import FLDataset
from global_updates import get_global_update
from harness_params import get_current_params
from update import get_local_update
import logging_utils
import models


get_current_params()


class FLTrainingHarness:
    def __init__(self):
        self.config = get_current_config()
        self.device = self.config["model.device"] or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_built() else "cpu")
        )
        self.global_model = self.init_global_model()
        (
            self.train_user_groups,
            self.test_user_groups,
            self.valid_user_groups,
        ) = self.get_data_splits()
        self.global_update = self.get_global_update()
        self.client_prob_dist = None
        self.client_indices = None

        self.logger: logging_utils.WandbLogger
        self._ckpt_dir = None

    @property
    def ckpt_dir(self):
        if self._ckpt_dir is None:
            self._ckpt_dir = (
                Path(self.config["fl_parameters.ckpt_path"]).resolve()
                / self.logger.project_name
                / self.logger.run_name
            )
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        return self._ckpt_dir

    def run(self):
        self.logger = logging_utils.WandbLogger()
        try:
            for epoch in range(self.config["global_parameters.global_rounds"]):
                client_metrics = self.train_stage(epoch=epoch)
                self.logger.log(client_metrics)
                self.save_model(epoch=epoch)
                test_metrics = self.test_stage()
                self.logger.log(test_metrics)
            self.logger.close(exit_code=0)
        except Exception as ex:
            print("Experiment failed with exception: ", ex)
            print(traceback.format_exc())
            self.logger.close(exit_code=-1)

    @param("fl_parameters.save_every")
    def save_model(self, save_every, epoch):
        if epoch % self.config["fl_parameters.save_every"] == 0:
            torch.save(
                self.global_model.state_dict(),
                self.ckpt_dir / f"global_model_{epoch}.pt",
            )

    @param("model.model_name")
    @param("dataset.num_classes")
    @param("dataset.num_features")
    def init_global_model(self, model_name, num_classes, num_features=None):
        model_cls = getattr(models, model_name)
        if num_features:
            model = model_cls(num_classes=num_classes, num_features=num_features)
        else:
            model = model_cls(num_classes=num_classes)
        return model.to(self.device)

    def get_data_splits(self):
        main_ds = FLDataset()
        self.train_dataset, self.test_dataset, self.valid_dataset = (
            main_ds.train_dataset,
            main_ds.test_dataset,
            main_ds.valid_dataset,
        )
        (
            train_user_groups,
            test_user_groups,
            valid_user_groups,
        ) = main_ds.get_client_groups()

        return train_user_groups, test_user_groups, valid_user_groups

    @param("fl_parameters.fl_method")
    def get_global_update(self, fl_method):
        global_update = get_global_update(fl_method=fl_method, model=self.global_model)
        return global_update

    @param("fl_parameters.fl_method")
    def get_local_update(self, fl_method, client_idx, proportion=1.0):
        local_update = get_local_update(
            fl_method=fl_method,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            valid_dataset=self.valid_dataset,
            train_idxs=self.train_user_groups[client_idx],
            test_idxs=self.test_user_groups[client_idx],
            valid_idxs=self.valid_user_groups[client_idx],
            logger=False,
            global_model=self.global_model,
            proportion=proportion,
        )

        return local_update

    @param("fl_parameters.use_fair_sparsification")
    @param("fl_parameters.sparsification_ratio")
    def run_fedsyn_client(
        self,
        use_fair_sparsification,
        sparsification_ratio,
        client_idx,
        local_update,
        epoch,
    ):

        local_model = copy.deepcopy(self.global_model)
        if use_fair_sparsification:
            assert self.client_prob_dist is not None
            sparsification_ratio = self.client_prob_dist[client_idx]

        w, flat_update, bitmask, _ = local_update.update_weights(
            model=local_model,
            sparsification_ratio=sparsification_ratio,
            global_round=epoch,
        )

        acc, loss = local_update.inference(model=w, dataset_type="train")

        return w, flat_update, bitmask, loss, acc, local_model

    def run_qfedavg_client(self, local_update, epoch):
        local_model = copy.deepcopy(self.global_model)
        delta, h, w, _ = local_update.update_weights(
            model=local_model, global_round=epoch
        )
        acc, loss = local_update.inference(model=w, dataset_type="train")
        return delta, h, w, loss, acc, local_model

    def run_generic_client(self, local_update, epoch):
        local_model = copy.deepcopy(self.global_model)
        w, _ = local_update.update_weights(model=local_model, global_round=epoch)
        acc, loss = local_update.inference(model=w, dataset_type="train")
        return w, loss, acc, local_model

    @param("fl_parameters.num_clients")
    @param("fl_parameters.frac")
    @param("fl_parameters.fl_method")
    def train_stage(self, num_clients, frac, fl_method, epoch):
        self.global_model.train()

        global_flat = flatten(self.global_model)
        m = max(int(frac * num_clients), 1)

        self.client_indices = np.random.choice(
            range(num_clients), m, replace=False
        ).tolist()

        local_weights_sum, local_bitmasks_sum, local_delta_sum, local_h_sum = (
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
            np.zeros_like(global_flat),
        )

        client_metrics = {}
        valid_losses, valid_accs = self.compute_client_prob_dist()
        client_metrics["valid/loss_mean"] = np.mean(valid_losses)
        client_metrics["valid/loss_std"] = np.std(valid_losses)
        client_metrics["valid/acc_mean"] = np.mean(valid_accs)
        client_metrics["valid/acc_std"] = np.std(valid_accs)

        train_losses = []
        train_accs = []
        train_params_sent = []
        for client_idx in self.client_indices:
            local_update = self.get_local_update(client_idx=client_idx)
            if fl_method == "FedSyn":
                (
                    w,
                    flat_update,
                    bitmask,
                    train_loss,
                    train_acc,
                    local_model,
                ) = self.run_fedsyn_client(
                    client_idx=client_idx, local_update=local_update, epoch=epoch
                )
                local_weights_sum += flat_update
                local_bitmasks_sum += bitmask
                local_bitmasks = bitmask
            elif fl_method == "qFedAvg":
                (
                    delta,
                    h,
                    w,
                    train_loss,
                    train_acc,
                    local_model,
                ) = self.run_qfedavg_client(local_update, epoch)
                local_delta_sum += delta
                local_h_sum += h
                local_bitmasks_sum += np.ones_like(local_bitmasks_sum)
                local_bitmasks = np.ones_like(local_bitmasks_sum)
            else:
                w, train_loss, train_acc, local_model = self.run_generic_client(
                    local_update, epoch
                )
                local_weights_sum += flatten(w)
                local_bitmasks_sum += np.ones_like(local_bitmasks_sum)
                local_bitmasks = np.ones_like(local_bitmasks_sum)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            train_params_sent.append(np.sum(local_bitmasks))

        client_metrics["train/loss_mean"] = np.mean(train_losses)
        client_metrics["train/loss_std"] = np.std(train_losses)
        client_metrics["train/acc_mean"] = np.mean(train_accs)
        client_metrics["train/acc_std"] = np.std(train_accs)
        client_metrics["train/params_sent_mean"] = np.mean(train_params_sent)
        client_metrics["train/params_sent_std"] = np.std(train_params_sent)

        if fl_method == "FedSyn":
            global_weights = self.global_update.aggregate_weights(
                local_weights_sum=local_weights_sum,
                local_bitmasks_sum=local_bitmasks_sum,
                global_model=self.global_model,
            )
            updateFromNumpyFlatArray(flat_arr=global_weights, model=self.global_model)
        elif fl_method == "qFedAvg":
            global_weights = self.global_update.aggregate_weights(
                self.global_model, local_delta_sum, local_h_sum
            )
            updateFromNumpyFlatArray(flat_arr=global_weights, model=self.global_model)
        else:
            global_weights = self.global_update.aggregate_weights(
                len(self.client_indices), local_weights_sum, valid_losses
            )
            self.global_update.update_global_model(self.global_model, global_weights)

        return client_metrics

    @param("fl_parameters.fairness_function")
    @param("fl_parameters.fairness_temperature")
    @param("fl_parameters.min_sparsification_ratio")
    @param("fl_parameters.sparsification_ratio")
    @param("split_params.combine_train_val")
    @param("split_params.fairness_proportion")
    def compute_client_prob_dist(
        self,
        fairness_function,
        fairness_temperature,
        min_sparsification_ratio,
        sparsification_ratio,
        combine_train_val,
        fairness_proportion,
    ):
        self.global_model.eval()
        valid_losses = []
        valid_accs = []
        for client_idx in self.client_indices:
            if combine_train_val:
                local_update = self.get_local_update(client_idx=client_idx, proportion=fairness_proportion)
                acc, loss = local_update.inference(
                    model=self.global_model, dataset_type="train"
                )
            else:
                local_update = self.get_local_update(client_idx=client_idx)
                acc, loss = local_update.inference(
                    model=self.global_model, dataset_type="valid"
                )
            valid_losses.append(loss)
            valid_accs.append(acc)
        valid_losses = np.array(valid_losses)
        if fairness_function == "custom-exp":
            client_prob_dist = custom_exponential_sparsity(
                valid_losses,
                sparsification_ratio,
                min_sparsification_ratio,
                fairness_temperature,
            )
        elif fairness_function == "linear-interpolate":
            client_prob_dist = linearly_interpolated_softmax(
                valid_losses,
                sparsification_ratio,
                min_sparsification_ratio,
                fairness_temperature,
            )
        self.client_prob_dist = {
            client_idx: client_prob_dist[i]
            for i, client_idx in enumerate(self.client_indices)
        }
        self.global_model.train()
        return valid_losses, valid_accs

    @param("fl_parameters.num_clients")
    def test_stage(self, num_clients):
        self.global_model.eval()
        test_metrics = {}

        test_losses = []
        test_accs = []
        for client_idx in range(num_clients):
            local_update = self.get_local_update(client_idx=client_idx)
            acc, loss = local_update.inference(
                model=self.global_model, dataset_type="test"
            )
            test_losses.append(loss)
            test_accs.append(acc)

        test_metrics["global/test_loss_mean"] = np.mean(test_losses)
        test_metrics["global/test_loss_std"] = np.std(test_losses)
        test_metrics["global/test_acc_mean"] = np.mean(test_accs)
        test_metrics["global/test_acc_std"] = np.std(test_accs)

        return test_metrics


if __name__ == "__main__":
    if sys.version_info[0:2] != (3, 11):
        raise RuntimeError(
            f"Code requires python 3.11. You are using {sys.version_info[0:2]}. Please update your conda env and install requirements.txt on the new env."
        )

    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    config.summary()
    set_current_config(config)
    harness = FLTrainingHarness()
    harness.run()
