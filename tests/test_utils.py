"""Tests for utility functions."""

import numpy as np
import pytest
import torch


class TestGeneralUtils:
    """Test suite for general utility functions."""

    def test_flatten_model(self):
        """Test model flattening."""
        from src import models
        from src.utils import flatten

        model = models.SmallCNN(num_classes=10)
        flat = flatten(model)

        assert isinstance(flat, np.ndarray)
        assert len(flat) > 0
        assert flat.dtype == np.float32 or flat.dtype == np.float64

    def test_update_from_flat_array(self):
        """Test updating model from flat array."""
        from src import models
        from src.utils import flatten, updateFromNumpyFlatArray

        model = models.SmallCNN(num_classes=10)
        original_flat = flatten(model)

        # Create new random parameters
        new_flat = np.random.randn(len(original_flat))

        # Update model
        updateFromNumpyFlatArray(new_flat, model)

        # Verify update worked
        updated_flat = flatten(model)

        # Check that the model was actually updated (not same as original)
        assert not np.allclose(updated_flat, original_flat)

        # The vast majority of parameters should match very closely
        abs_diff = np.abs(updated_flat - new_flat)
        # 99% of parameters should have tiny differences (float precision)
        percentile_99 = np.percentile(abs_diff, 99)
        assert percentile_99 < 1e-6, f"99th percentile of differences: {percentile_99}"

        # Mean difference should be very small
        assert np.mean(abs_diff) < 1e-5, f"Mean absolute difference: {np.mean(abs_diff)}"

    @pytest.mark.parametrize("sparse_ratio", [0.1, 0.5, 0.9])
    def test_randk_sparsification(self, sparse_ratio):
        """Test random-k sparsification."""
        from src.utils import get_bitmask_per_method

        flat_array = np.random.randn(1000)
        bitmask = get_bitmask_per_method(flat_array, sparse_ratio=sparse_ratio, sparsification_type="randk")

        assert bitmask.shape == flat_array.shape
        expected_params = int(sparse_ratio * len(flat_array))
        actual_params = int(np.sum(bitmask))
        assert actual_params == expected_params

    @pytest.mark.parametrize("sparse_ratio", [0.1, 0.5, 0.9])
    def test_topk_sparsification(self, sparse_ratio):
        """Test top-k sparsification."""
        from src.utils import get_bitmask_per_method

        flat_array = np.random.randn(1000)
        bitmask = get_bitmask_per_method(flat_array, sparse_ratio=sparse_ratio, sparsification_type="topk")

        assert bitmask.shape == flat_array.shape
        expected_params = int(sparse_ratio * len(flat_array))
        actual_params = int(np.sum(bitmask))
        assert actual_params == expected_params

    def test_rtopk_sparsification(self):
        """Test random top-k sparsification."""
        from src.utils import get_bitmask_per_method

        flat_array = np.random.randn(1000)
        bitmask = get_bitmask_per_method(
            flat_array,
            sparse_ratio=0.3,
            sparsification_type="rtopk",
            choose_from_top_r_percentile=0.5,
        )

        assert bitmask.shape == flat_array.shape
        expected_params = int(0.3 * len(flat_array))
        actual_params = int(np.sum(bitmask))
        assert actual_params == expected_params

    def test_invalid_sparsification_type(self):
        """Test that invalid sparsification type raises error."""
        from src.utils import get_bitmask_per_method

        flat_array = np.random.randn(1000)

        with pytest.raises(ValueError, match="Unrecognized sparsification method"):
            get_bitmask_per_method(flat_array, sparse_ratio=0.5, sparsification_type="invalid_method")

    def test_temperatured_softmax(self):
        """Test temperatured softmax function."""
        from src.utils import temperatured_softmax

        losses = np.array([1.0, 2.0, 3.0, 4.0])
        result = temperatured_softmax(losses, softmax_temperature=1.0)

        assert result.shape == losses.shape
        assert np.allclose(np.sum(result), 1.0)  # Should sum to 1
        assert np.all(result > 0)  # All probabilities should be positive

    def test_custom_exponential_sparsity(self):
        """Test custom exponential sparsity function."""
        from src.utils import custom_exponential_sparsity

        losses = np.array([1.0, 2.0, 3.0, 4.0])
        result = custom_exponential_sparsity(losses, max_sparsity=0.8, min_sparsity=0.2, temperature=1.0)

        assert result.shape == losses.shape
        assert np.all(result >= 0.2)  # All should be >= min
        assert np.all(result <= 0.8)  # All should be <= max

    def test_linearly_interpolated_softmax(self):
        """Test linearly interpolated softmax function."""
        from src.utils import linearly_interpolated_softmax

        losses = np.array([1.0, 2.0, 3.0, 4.0])
        result = linearly_interpolated_softmax(losses, max_sparsity=0.8, min_sparsity=0.2, temperature=1.0)

        assert len(result) == len(losses)
        for r in result:
            assert r >= 0.2  # All should be >= min
            assert r <= 0.8  # All should be <= max

    def test_set_seed(self):
        """Test seed setting function."""
        from src.utils import set_seed

        set_seed(42)
        rand1 = np.random.rand()

        set_seed(42)
        rand2 = np.random.rand()

        assert rand1 == rand2  # Should get same random number with same seed

    def test_dict_sum(self):
        """Test dictionary summing function."""
        from src.utils import dict_sum

        dict1 = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
        dict2 = {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([7.0, 8.0])}

        result = dict_sum([dict1, dict2])

        assert torch.allclose(result["a"], torch.tensor([6.0, 8.0]))
        assert torch.allclose(result["b"], torch.tensor([10.0, 12.0]))

    def test_normalize(self):
        """Test normalize function for image data."""
        from src.utils import normalize

        # Create dummy image tensor (0-255 range)
        img = torch.tensor([0.0, 127.5, 255.0])
        normalized = normalize(img)

        # Should be in [-1, 1] range
        assert torch.allclose(normalized, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-6)
