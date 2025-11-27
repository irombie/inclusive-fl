"""Pytest configuration and fixtures for inclusive-fl tests."""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_dir():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_dir(project_root_dir):
    """Return the configs directory."""
    return project_root_dir / "configs"


@pytest.fixture(scope="session")
def src_dir(project_root_dir):
    """Return the src directory."""
    return project_root_dir / "src"
