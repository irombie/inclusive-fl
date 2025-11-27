"""Tests for model architectures."""

import pytest
import torch


class TestModelInstantiation:
    """Test suite for model instantiation."""

    def test_resnet9_instantiation(self):
        """Test ResNet9 model instantiation."""
        from src import models

        model = models.ResNet9(num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_resnet18_instantiation(self):
        """Test ResNet18 model instantiation."""
        from src import models

        model = models.ResNet18(num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_vgg_instantiation(self):
        """Test VGG model instantiation."""
        from src import models

        model = models.VGG(num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_small_cnn_instantiation(self):
        """Test SmallCNN model instantiation."""
        from src import models

        model = models.SmallCNN(num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_cnn_fashionmnist_instantiation(self):
        """Test CNNFashionMNIST model instantiation."""
        from src import models

        model = models.CNNFashionMNIST(num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_mlp_instantiation(self):
        """Test MLP model instantiation."""
        from src import models

        model = models.MLP(num_classes=10, num_features=784)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_logistic_regression_instantiation(self):
        """Test LogisticRegression model instantiation."""
        from src import models

        model = models.LogisticRegression(num_features=784, num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_forward_pass(self):
        """Test that models can perform forward pass."""
        from src import models

        # Test with ResNet9
        model = models.ResNet9(num_classes=10)
        model.eval()

        # Create dummy input (batch_size=2, channels=3, height=32, width=32)
        dummy_input = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 10)

    @pytest.mark.parametrize("num_classes", [5, 10, 100])
    def test_model_different_num_classes(self, num_classes):
        """Test models with different numbers of classes."""
        from src import models

        model = models.ResNet9(num_classes=num_classes)
        model.eval()

        dummy_input = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, num_classes)
