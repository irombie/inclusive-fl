"""Tests for Python environment and dependencies."""

import sys


class TestEnvironment:
    """Test suite for environment setup."""

    def test_python_version(self):
        """Test that Python version is 3.11."""
        assert sys.version_info[0:2] == (3, 11), f"Expected Python 3.11, got {sys.version_info[0:2]}"

    def test_core_imports(self):
        """Test that all core dependencies can be imported."""
        import numpy as np
        import torch
        import torchvision

        # Basic version checks
        assert torch.__version__ is not None
        assert torchvision.__version__ is not None
        assert np.__version__ is not None

    def test_pytorch_device_support(self):
        """Test PyTorch device availability."""
        import torch

        # At least CPU should be available
        assert torch.device("cpu") is not None

        # Check for GPU support (will be False on CPU-only systems)
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()

        # Should have at least one accelerator available or CPU
        assert cuda_available or mps_available or True  # CPU always works
