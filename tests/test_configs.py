"""Tests for configuration loading and validation."""

from pathlib import Path
import sys

import pytest


class TestConfigLoading:
    """Test suite for configuration file loading."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """Reset configuration between tests."""
        # This runs before each test
        yield
        # Cleanup after test if needed

    def test_fedavg_config_loading(self, config_dir):
        """Test FedAvg configuration loading."""
        config_path = config_dir / "cifar10" / "resnet9" / "fedavg.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        sys.argv = ["pytest", "--config-file", str(config_path)]

        from argparse import ArgumentParser

        from fastargs import get_current_config

        from src.harness_params import get_current_params

        get_current_params()
        config = get_current_config()
        parser = ArgumentParser()
        config.augment_argparse(parser)
        config.collect_argparse_args(parser)
        config.validate(mode="stderr")

        assert config["fl_parameters.fl_method"] == "FedAvg"
        assert config["model.model_name"] == "ResNet9"
        assert config["dataset.dataset_name"] == "CIFAR10"

    def test_qfedavg_config_loading(self, config_dir):
        """Test qFedAvg configuration loading."""
        config_path = config_dir / "cifar10" / "resnet9" / "qfedavg.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        sys.argv = ["pytest", "--config-file", str(config_path)]

        from argparse import ArgumentParser

        from fastargs import get_current_config

        from src.harness_params import get_current_params

        get_current_params()
        config = get_current_config()
        parser = ArgumentParser()
        config.augment_argparse(parser)
        config.collect_argparse_args(parser)
        config.validate(mode="stderr")

        assert config["fl_parameters.fl_method"] == "qFedAvg"
        assert config["fl_parameters.q"] == 0.1
        assert config["model.model_name"] == "ResNet9"

    def test_fedsyn_fair_config_loading(self, config_dir):
        """Test FedSyn fair sparsification configuration loading."""
        config_path = config_dir / "cifar10" / "resnet9" / "fedsyn_fair_sparsification.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        sys.argv = ["pytest", "--config-file", str(config_path)]

        from argparse import ArgumentParser

        from fastargs import get_current_config

        from src.harness_params import get_current_params

        get_current_params()
        config = get_current_config()
        parser = ArgumentParser()
        config.augment_argparse(parser)
        config.collect_argparse_args(parser)
        config.validate(mode="stderr")

        assert config["fl_parameters.fl_method"] == "FedSyn"
        assert config["fl_parameters.use_fair_sparsification"] is True

    def test_all_cifar10_configs_exist(self, config_dir):
        """Test that all expected CIFAR10 configs exist."""
        cifar10_dir = config_dir / "cifar10" / "resnet9"

        expected_configs = [
            "fedavg.yaml",
            "qfedavg.yaml",
            "fedsyn_fair_sparsification.yaml",
            "fedsyn_unfair_sparsification.yaml",
        ]

        for config_name in expected_configs:
            config_path = cifar10_dir / config_name
            assert config_path.exists(), f"Expected config not found: {config_path}"

    def test_fashionmnist_config_exists(self, config_dir):
        """Test that FashionMNIST config exists."""
        config_path = config_dir / "fashionmnist" / "qfedavg.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_svhn_config_exists(self, config_dir):
        """Test that SVHN config exists."""
        config_path = config_dir / "svhn" / "qfedavg.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
