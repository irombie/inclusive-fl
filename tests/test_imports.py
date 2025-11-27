"""Tests for project module imports."""


class TestProjectImports:
    """Test suite for project module imports."""

    def test_harness_import(self):
        """Test that harness module can be imported."""
        from src import harness

        assert hasattr(harness, "FLTrainingHarness")

    def test_models_import(self):
        """Test that models module can be imported."""
        from src import models

        assert hasattr(models, "ResNet9")
        assert hasattr(models, "SmallCNN")
        assert hasattr(models, "CNNFashionMNIST")

    def test_fl_dataset_import(self):
        """Test that fl_dataset module can be imported."""
        from src import fl_dataset

        assert hasattr(fl_dataset, "FLDataset")

    def test_global_updates_import(self):
        """Test that global_updates module can be imported."""
        from src import global_updates

        assert hasattr(global_updates, "AbstractGlobalUpdate")
        assert hasattr(global_updates, "MeanWeights")

    def test_update_import(self):
        """Test that update module can be imported."""
        from src import update

        assert hasattr(update, "LocalUpdate")

    def test_general_utils_import(self):
        """Test that general_utils module can be imported."""
        from src import general_utils

        assert hasattr(general_utils, "flatten")
        assert hasattr(general_utils, "updateFromNumpyFlatArray")
        assert hasattr(general_utils, "get_bitmask_per_method")

    def test_logging_utils_import(self):
        """Test that logging_utils module can be imported."""
        from src import logging_utils

        assert hasattr(logging_utils, "WandbLogger")

    def test_harness_params_import(self):
        """Test that harness_params module can be imported."""
        from src import harness_params

        assert hasattr(harness_params, "get_current_params")

    def test_dataset_defs_import(self):
        """Test that dataset_defs module can be imported."""
        from src import dataset_defs

        assert hasattr(dataset_defs, "UTKFaceDataset")
        assert hasattr(dataset_defs, "SyntheticDataset")

    def test_all_imports_together(self):
        """Test that all modules can be imported together without conflicts."""
        from src import (
            dataset_defs,
            fl_dataset,
            general_utils,
            global_updates,
            harness,
            harness_params,
            logging_utils,
            models,
            update,
        )

        assert harness is not None
        assert models is not None
        assert fl_dataset is not None
        assert global_updates is not None
        assert update is not None
        assert general_utils is not None
        assert logging_utils is not None
        assert harness_params is not None
        assert dataset_defs is not None
