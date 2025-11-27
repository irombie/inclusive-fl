# CI/CD Deployment Commands

## 🚀 Ready to Deploy!

All CI/CD components have been verified and are ready to deploy.

**Verification Status**: ✅ 43/43 checks passed

---

## Step 1: Review Changes

```bash
# See what files were added/modified
git status

# Review the changes
git diff README.md
git diff pyproject.toml
git diff .pre-commit-config.yaml
```

---

## Step 2: Stage All CI/CD Files

```bash
# Stage all new and modified files
git add .github/
git add tests/
git add Dockerfile .dockerignore
git add .flake8 pytest.ini
git add pyproject.toml
git add .pre-commit-config.yaml
git add README.md
git add CI_CD.md CI_CD_QUICKSTART.md CI_CD_SUMMARY.md
git add verify_cicd_setup.sh
git add .gitignore
```

---

## Step 3: Commit Changes

```bash
git commit -m "Add comprehensive CI/CD pipeline with GitHub Actions

Features:
- 6 GitHub Actions workflows (CI, tests, quality, docker, release, docs)
- 51 passing pytest tests covering environment, imports, models, configs, utils
- Docker containerization with ghcr.io integration
- Code quality tools (black, isort, flake8, mypy)
- GitHub templates (PR, 2 issue templates)
- Pre-commit hooks for local validation
- Comprehensive documentation (CI_CD.md, quickstart, summary)
- Setup verification script

Workflows:
- CI: Main pipeline with test/lint/security/build jobs
- Tests: Matrix testing + full suite with coverage
- Code Quality: Black, isort, flake8, mypy checks
- Docker: Build and push to GitHub Container Registry
- Release: Automated releases on version tags
- Docs: Documentation building (MkDocs ready)

Testing:
- 51 tests passing locally
- Coverage reporting configured
- Runtime verified with FL training (2 rounds, loss: 2.45→1.38)

Configuration:
- Python 3.11 with PyTorch 2.2.2 (MPS support)
- All tools configured in pyproject.toml
- Docker multi-stage builds
- Pre-commit hooks installed and working"
```

---

## Step 4: Push to GitHub

```bash
# Push to main branch
git push origin main

# Or if you're on a different branch
git push origin <your-branch-name>
```

---

## Step 5: Monitor First CI Run

1. **Go to GitHub**: Navigate to your repository
2. **Click "Actions" tab**: View all workflow runs
3. **Watch the workflows execute**:
   - CI Workflow (~10-15 min)
   - Tests Workflow (~8-12 min)
   - Code Quality Workflow (~2-3 min)
   - Docker Workflow (~5-7 min, if on main/develop)

---

## Expected Results

### ✅ What Should Happen

1. **CI Workflow** starts immediately
   - Test job: Downloads CIFAR10, runs 51 tests
   - Lint job: Runs black, isort, flake8, mypy
   - Security job: Runs Bandit security scan
   - Build job: Validates Docker build

2. **Tests Workflow** runs in parallel
   - 6 parallel test groups
   - Full suite job combines all
   - Coverage uploaded to Codecov (if configured)

3. **Code Quality Workflow** runs checks
   - May show warnings (won't fail)
   - Informational only on first run

4. **Docker Workflow** (on main/develop only)
   - Builds Docker image
   - Pushes to ghcr.io/irombie/inclusive-fl

### 🎯 Success Indicators

- All workflows show green checkmarks ✅
- Badges in README update to "passing"
- Docker image available at `ghcr.io/irombie/inclusive-fl:main`
- Coverage report available (if Codecov configured)

---

## Troubleshooting First Run

### If Tests Fail

```bash
# Run locally first to verify
.venv/bin/pytest -v

# Check specific test
.venv/bin/pytest tests/test_models.py -v

# Verify environment
.venv/bin/python -c "import torch; print(torch.__version__)"
```

### If Lint Warnings Appear

Code quality checks are set to continue on error (informational):

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check locally
flake8 src/ tests/
```

### If Docker Fails

```bash
# Test Docker build locally
docker build -t inclusive-fl:test .

# Check .dockerignore isn't excluding needed files
cat .dockerignore
```

### If Coverage Upload Fails

This is expected if `CODECOV_TOKEN` isn't set:

```bash
# Optional: Add token to repository secrets
# Settings → Secrets → Actions → New repository secret
# Name: CODECOV_TOKEN
# Value: <token from codecov.io>
```

---

## Optional Configurations

### 1. Enable Branch Protection

Go to: Settings → Branches → Add rule

Protect `main` branch:
- ✅ Require pull request reviews
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- Select status checks: `test`, `lint`, `security`

### 2. Configure Codecov

1. Sign up at [codecov.io](https://codecov.io)
2. Add repository
3. Copy token
4. Add to secrets: `CODECOV_TOKEN`

### 3. Configure PyPI Publishing (for releases)

1. Get API token from PyPI
2. Add to secrets: `PYPI_TOKEN`
3. Create release tag: `git tag v1.0.0 && git push origin v1.0.0`

### 4. Set up GitHub Pages (for docs)

1. Go to Settings → Pages
2. Source: GitHub Actions
3. Push to trigger docs workflow

---

## Post-Deployment Checklist

- [ ] Push to GitHub completed
- [ ] Actions tab shows workflows running
- [ ] All workflows completed successfully (green checkmarks)
- [ ] README badges updated
- [ ] Docker image available (if pushed to main/develop)
- [ ] Coverage report visible (if Codecov configured)
- [ ] Pre-commit hooks working locally
- [ ] Team notified of CI/CD setup

---

## Daily Usage

### Before Committing

```bash
# Pre-commit hooks run automatically
git add <files>
git commit -m "message"

# Or run manually
pre-commit run --all-files
```

### Running Tests

```bash
# All tests
.venv/bin/pytest -v

# With coverage
.venv/bin/pytest -v --cov=src --cov-report=html

# Specific test
.venv/bin/pytest tests/test_models.py::test_resnet9 -v
```

### Code Formatting

```bash
# Format all code
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/
mypy src/
```

### Docker Usage

```bash
# Build locally
docker build -t inclusive-fl:dev .

# Run experiment
docker run -v $(pwd)/data:/app/data \
  -e WANDB_MODE=offline \
  inclusive-fl:dev \
  python src/harness.py --config configs/cifar10/resnet9/fedavg.yaml

# Interactive session
docker run -it inclusive-fl:dev bash
```

---

## Documentation Links

- **CI_CD.md** - Comprehensive CI/CD documentation
- **CI_CD_QUICKSTART.md** - Quick start guide
- **CI_CD_SUMMARY.md** - Complete setup summary
- **tests/README.md** - Testing guide
- **VERIFICATION_REPORT.md** - Initial setup verification
- **RUN_VERIFICATION.md** - Runtime verification

---

## Support

If you encounter issues:

1. Check GitHub Actions logs for detailed error messages
2. Review documentation files
3. Run verification script: `./verify_cicd_setup.sh`
4. Test locally before pushing: `.venv/bin/pytest -v`

---

**🎉 Your CI/CD pipeline is ready to deploy!**

Run the commands above to push your changes and activate the automation.
