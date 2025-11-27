# Inclusive-FL Test Suite

This directory contains the pytest test suite for the inclusive-fl project.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_models.py
```

### Run specific test class
```bash
pytest tests/test_models.py::TestModelInstantiation
```

### Run specific test
```bash
pytest tests/test_models.py::TestModelInstantiation::test_resnet9_instantiation
```

### Run tests matching a pattern
```bash
pytest -k "config"  # Run all tests with "config" in the name
```

### Run with coverage (requires pytest-cov)
```bash
pytest --cov=src --cov-report=html
```

## Test Organization

- **test_environment.py** - Tests for Python environment and dependencies
- **test_imports.py** - Tests for module imports
- **test_models.py** - Tests for model architectures
- **test_configs.py** - Tests for configuration loading
- **test_utils.py** - Tests for utility functions
- **test_structure.py** - Tests for repository structure

## Test Fixtures

Common fixtures are defined in `conftest.py`:
- `project_root_dir` - Path to project root
- `config_dir` - Path to configs directory
- `src_dir` - Path to src directory

## Adding New Tests

1. Create a new test file with the `test_` prefix
2. Define test classes with the `Test` prefix
3. Define test methods with the `test_` prefix
4. Use pytest fixtures for common setup
5. Use parametrize for testing multiple scenarios

Example:
```python
import pytest

class TestNewFeature:
    def test_basic_functionality(self):
        assert True

    @pytest.mark.parametrize("value", [1, 2, 3])
    def test_with_parameters(self, value):
        assert value > 0
```

## Installing Test Dependencies

The base project dependencies include pytest. To install additional testing tools:

```bash
pip install pytest-cov pytest-xdist
```

- `pytest-cov` - For coverage reports
- `pytest-xdist` - For parallel test execution
