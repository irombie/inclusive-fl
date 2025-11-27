# CI/CD Integration - Quick Start Guide

## ✅ What Was Set Up

Your repository now has a complete CI/CD pipeline with:

### 🔄 GitHub Actions Workflows (6 workflows)

1. **CI Workflow** (`ci.yml`)
   - Runs on every push/PR to main/develop
   - Tests, linting, security scanning, and builds

2. **Tests Workflow** (`tests.yml`)
   - Parallel test execution by test group
   - Full test suite with coverage reports
   - Runs daily at 2 AM UTC

3. **Code Quality** (`code-quality.yml`)
   - Black formatting checks
   - isort import sorting
   - Flake8 linting
   - MyPy type checking

4. **Docker** (`docker.yml`)
   - Builds Docker images
   - Pushes to GitHub Container Registry
   - Tags with version/branch/SHA

5. **Release** (`release.yml`)
   - Creates releases on version tags
   - Builds distribution packages
   - Ready for PyPI publishing

6. **Documentation** (`docs.yml`)
   - Builds docs with MkDocs
   - Ready for GitHub Pages deployment

### 📦 Files Created

```
.github/
├── workflows/
│   ├── ci.yml                    # Main CI pipeline
│   ├── tests.yml                 # Test suite
│   ├── code-quality.yml          # Linting & formatting
│   ├── docker.yml                # Docker builds
│   ├── release.yml               # Release automation
│   └── docs.yml                  # Documentation
├── ISSUE_TEMPLATE/
│   ├── bug_report.md             # Bug report template
│   └── feature_request.md        # Feature request template
└── PULL_REQUEST_TEMPLATE.md      # PR template

Dockerfile                         # Container definition
.dockerignore                      # Docker build exclusions
.flake8                           # Flake8 configuration
CI_CD.md                          # Comprehensive CI/CD docs
```

### 📝 Updated Files

- `README.md` - Added CI/CD status badges
- `pyproject.toml` - Added tool configurations (black, isort, pytest, mypy)
- `.pre-commit-config.yaml` - Enhanced with more hooks

## 🚀 Next Steps

### 1. Push to GitHub (if not already done)

```bash
git add .
git commit -m "Add CI/CD pipeline with GitHub Actions"
git push origin main
```

### 2. Enable GitHub Actions

1. Go to your repository on GitHub
2. Click "Actions" tab
3. Enable workflows if prompted

### 3. (Optional) Set Up Codecov

1. Sign up at [codecov.io](https://codecov.io)
2. Add your repository
3. Copy the token
4. Add to repository: Settings → Secrets → New secret
   - Name: `CODECOV_TOKEN`
   - Value: [your token]

### 4. (Optional) Configure Branch Protection

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - ✅ Require pull request reviews
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date

### 5. Local Development Setup

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Install dev dependencies:

```bash
pip install -e .[dev]
```

## 🧪 Testing the CI/CD Pipeline

### Trigger Workflows

1. **Push to main**: Triggers CI, Tests, Code Quality
2. **Create PR**: Triggers all checks
3. **Create tag** (`v1.0.0`): Triggers release workflow
4. **Manual trigger**: Go to Actions → Select workflow → Run workflow

### Expected Behavior

- ✅ All tests should pass (51 tests)
- ✅ Code quality checks run (may show warnings, won't fail)
- ✅ Docker image builds successfully
- ✅ Coverage report uploaded (if Codecov configured)

## 📊 Monitoring

### GitHub Actions Status

View in your repository:
- Actions tab shows all workflow runs
- Green checkmark = passed
- Red X = failed
- Yellow dot = in progress

### Status Badges

Your README now shows badges for:
- CI pipeline status
- Test suite status
- Code quality status
- Python version
- License

## 🐳 Using Docker

### Pull Latest Image

```bash
docker pull ghcr.io/irombie/inclusive-fl:main
```

### Run Experiment in Docker

```bash
docker run -v $(pwd)/data:/app/data \
  -e WANDB_MODE=offline \
  ghcr.io/irombie/inclusive-fl:main \
  python src/harness.py --config configs/cifar10/resnet9/fedavg.yaml
```

## 🔧 Common Tasks

### Run Tests Locally (same as CI)

```bash
pytest -v --cov=src --cov-report=term
```

### Format Code

```bash
black src/ tests/
isort src/ tests/
```

### Check Before Committing

```bash
# Or just commit - pre-commit hooks will run automatically
pre-commit run --all-files
```

### Create a Release

```bash
# 1. Update version in pyproject.toml
# 2. Commit
git commit -am "Bump version to 1.0.0"

# 3. Tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# 4. Push
git push origin v1.0.0
```

## 📚 Documentation

- **CI_CD.md** - Comprehensive CI/CD documentation
- **tests/README.md** - Testing guide
- **VERIFICATION_REPORT.md** - Setup verification
- **RUN_VERIFICATION.md** - Runtime verification

## ✨ Features

Your CI/CD pipeline now provides:

✅ **Automated Testing** - Every commit tested
✅ **Code Quality** - Consistent formatting and style
✅ **Security Scanning** - Dependency vulnerability checks
✅ **Coverage Tracking** - Know what's tested
✅ **Docker Support** - Containerized experiments
✅ **Release Automation** - Easy versioning
✅ **Documentation** - Auto-generated docs
✅ **Parallel Execution** - Fast feedback

## 🎯 Current Status

All components are configured and ready to use:
- ✅ 6 GitHub Actions workflows
- ✅ Pre-commit hooks configured
- ✅ Docker support
- ✅ Issue/PR templates
- ✅ Testing infrastructure (51 tests passing)
- ✅ Code quality tools configured

## 💡 Tips

1. **Always use the venv pytest**: `.venv/bin/pytest`
2. **Enable branch protection** for production branches
3. **Review failed checks** in the Actions tab
4. **Use pre-commit hooks** to catch issues early
5. **Monitor coverage trends** over time

## 🆘 Troubleshooting

### Workflows not running?
- Check Actions tab → Enable workflows
- Verify file permissions in `.github/workflows/`

### Tests fail in CI but pass locally?
- Ensure Python 3.11
- Check `WANDB_MODE=offline` is set
- Verify dependencies match

### Docker build fails?
- Test locally: `docker build -t test .`
- Check `.dockerignore` isn't excluding needed files
