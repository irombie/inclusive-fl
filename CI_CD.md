# CI/CD Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) setup for the Inclusive FL project.

## Overview

The project uses GitHub Actions for automated testing, code quality checks, building, and deployment. The CI/CD pipeline is triggered on every push and pull request to ensure code quality and prevent regressions.

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:** Push to `main`/`develop`, Pull Requests

**Jobs:**
- **Test**: Runs the full test suite on Python 3.11
  - Installs dependencies
  - Runs pytest with coverage
  - Uploads coverage to Codecov

- **Lint**: Code quality checks
  - Black formatting check
  - isort import sorting check
  - Flake8 linting

- **Security**: Security vulnerability scanning
  - Runs safety check on dependencies

- **Build**: Package building
  - Builds distribution packages
  - Uploads artifacts

### 2. Tests Workflow (`.github/workflows/tests.yml`)

**Triggers:** Push, Pull Requests, Daily at 2 AM UTC, Manual

**Jobs:**
- **test-matrix**: Runs tests in parallel by test group
  - Splits tests into: environment, imports, models, configs, utils, structure
  - Faster feedback on specific failures

- **test-full**: Complete test suite
  - Runs all tests with parallel execution (pytest-xdist)
  - Generates comprehensive coverage reports
  - Uploads coverage to Codecov

### 3. Code Quality Workflow (`.github/workflows/code-quality.yml`)

**Triggers:** Push to `main`/`develop`, Pull Requests

**Jobs:**
- **black**: Code formatting check
- **isort**: Import sorting check
- **flake8**: Linting
- **mypy**: Static type checking

### 4. Docker Workflow (`.github/workflows/docker.yml`)

**Triggers:** Push to `main`, Tags starting with `v*`, Pull Requests

**Jobs:**
- **build-and-push**:
  - Builds Docker image
  - Pushes to GitHub Container Registry (ghcr.io)
  - Tags with version, branch, and SHA

### 5. Release Workflow (`.github/workflows/release.yml`)

**Triggers:** Tags starting with `v*`, Manual

**Jobs:**
- **build-and-release**:
  - Builds distribution packages
  - Creates GitHub Release
  - Optionally publishes to PyPI (commented out by default)

### 6. Documentation Workflow (`.github/workflows/docs.yml`)

**Triggers:** Push to `main`, Manual

**Jobs:**
- **deploy-docs**:
  - Builds documentation with MkDocs
  - Optionally deploys to GitHub Pages

## Setup Instructions

### 1. Repository Secrets

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- `CODECOV_TOKEN`: Token for uploading coverage to Codecov (optional)
- `PYPI_API_TOKEN`: Token for publishing to PyPI (optional, for releases)

### 2. Enable GitHub Actions

1. Go to repository Settings → Actions → General
2. Set "Actions permissions" to "Allow all actions and reusable workflows"
3. Enable "Read and write permissions" for workflows

### 3. Branch Protection Rules

Recommended settings for `main` branch:
- Require pull request reviews before merging
- Require status checks to pass before merging:
  - `Test on Python 3.11`
  - `Lint and Format Check`
  - `Full Test Suite`
- Require branches to be up to date before merging

### 4. Pre-commit Hooks (Local Development)

Install pre-commit hooks for local development:

```bash
pip install pre-commit
pre-commit install
```

This will run formatting and linting checks before each commit.

## Running CI Checks Locally

### Tests
```bash
pytest -v --cov=src --cov-report=term
```

### Code Formatting
```bash
black --check src/ tests/
isort --check-only src/ tests/
```

### Linting
```bash
flake8 src/ tests/
```

### Type Checking
```bash
mypy src/ --ignore-missing-imports
```

### Run All Checks
```bash
# Format code
black src/ tests/
isort src/ tests/

# Run tests
pytest -v --cov=src

# Lint
flake8 src/ tests/
```

## Docker

### Build Image Locally
```bash
docker build -t inclusive-fl:latest .
```

### Run Container
```bash
# Interactive mode
docker run -it inclusive-fl:latest bash

# Run experiment
docker run -v $(pwd)/data:/app/data inclusive-fl:latest \
  python src/harness.py --config configs/cifar10/resnet9/fedavg.yaml
```

### Pull from GitHub Container Registry
```bash
docker pull ghcr.io/irombie/inclusive-fl:main
```

## Creating a Release

1. Update version in `pyproject.toml`
2. Commit changes: `git commit -am "Bump version to X.Y.Z"`
3. Create and push tag:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```
4. GitHub Actions will automatically:
   - Run all tests
   - Build packages
   - Create GitHub Release
   - Build and push Docker image

## Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/irombie/inclusive-fl/actions/workflows/ci.yml/badge.svg)](https://github.com/irombie/inclusive-fl/actions/workflows/ci.yml)
[![Tests](https://github.com/irombie/inclusive-fl/actions/workflows/tests.yml/badge.svg)](https://github.com/irombie/inclusive-fl/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/irombie/inclusive-fl/branch/main/graph/badge.svg)](https://codecov.io/gh/irombie/inclusive-fl)
[![Code Quality](https://github.com/irombie/inclusive-fl/actions/workflows/code-quality.yml/badge.svg)](https://github.com/irombie/inclusive-fl/actions/workflows/code-quality.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
```

## Troubleshooting

### Tests Fail in CI but Pass Locally

- Ensure you're using Python 3.11
- Check that `WANDB_MODE=offline` is set
- Verify all dependencies are in `pyproject.toml`

### Docker Build Fails

- Check `.dockerignore` isn't excluding required files
- Verify `pyproject.toml` has all dependencies
- Test build locally first

### Coverage Upload Fails

- Ensure `CODECOV_TOKEN` is set in repository secrets
- Check that `coverage.xml` is being generated

## Maintenance

### Updating Dependencies

1. Update versions in `pyproject.toml`
2. Test locally: `pip install -e .[dev]`
3. Run full test suite: `pytest -v`
4. Update pre-commit hooks: `pre-commit autoupdate`
5. Commit and push

### Updating GitHub Actions

1. Check for newer versions of actions
2. Update `rev` values in workflows
3. Test in a separate branch first

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Codecov Documentation](https://docs.codecov.com/)
