"""Tests for repository structure and file organization."""

import pytest


class TestRepositoryStructure:
    """Test suite for repository structure."""

    def test_required_directories_exist(self, project_root_dir):
        """Test that all required directories exist."""
        required_dirs = ["src", "configs", "scripts", "sweeps"]

        for dirname in required_dirs:
            dir_path = project_root_dir / dirname
            assert dir_path.exists(), f"Required directory '{dirname}' does not exist"
            assert dir_path.is_dir(), f"'{dirname}' exists but is not a directory"

    def test_required_files_exist(self, project_root_dir):
        """Test that all required files exist."""
        required_files = [
            "README.md",
            "pyproject.toml",
            "LICENSE",
        ]

        for filename in required_files:
            file_path = project_root_dir / filename
            assert file_path.exists(), f"Required file '{filename}' does not exist"
            assert file_path.is_file(), f"'{filename}' exists but is not a file"

    def test_source_files_exist(self, src_dir):
        """Test that all main source files exist."""
        source_files = [
            "harness.py",
            "models.py",
            "fl_dataset.py",
            "update.py",
            "global_updates.py",
            "dataset_defs.py",
            "utils.py",
            "harness_params.py",
        ]

        for filename in source_files:
            file_path = src_dir / filename
            assert file_path.exists(), f"Source file '{filename}' does not exist"
            assert file_path.is_file(), f"'{filename}' exists but is not a file"

    def test_config_directories_exist(self, config_dir):
        """Test that config directories exist."""
        config_dirs = ["cifar10", "fashionmnist", "svhn"]

        for dirname in config_dirs:
            dir_path = config_dir / dirname
            assert dir_path.exists(), f"Config directory '{dirname}' does not exist"

    def test_source_files_are_valid_python(self, src_dir):
        """Test that all source files have valid Python syntax."""
        import py_compile

        source_files = [
            "harness.py",
            "models.py",
            "fl_dataset.py",
            "update.py",
            "global_updates.py",
            "dataset_defs.py",
            "harness_params.py",
            "utils.py",
        ]

        for filename in source_files:
            file_path = src_dir / filename
            try:
                py_compile.compile(str(file_path), doraise=True)
            except py_compile.PyCompileError as e:
                pytest.fail(f"Syntax error in {filename}: {e}")
