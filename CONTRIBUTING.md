# Contributing

Thanks for your interest in SessionIQ. Contributions are welcome.

## Setup

```bash
git clone https://github.com/nicolasallerponte/sessioniq.git
cd sessioniq
uv sync
uv run pre-commit install
```

## Workflow

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes
3. Run checks: `uv run ruff check . && uv run ruff format --check .`
4. Run tests: `uv run pytest tests/`
5. Commit using [Conventional Commits](https://www.conventionalcommits.org): `feat(scope): description`
6. Open a pull request against `main`

## Commit convention

```
feat(pipeline): add brand entropy feature
fix(intent): handle empty session edge case
docs: update quickstart instructions
chore: bump lightgbm to 4.5
```

## What to contribute

- Additional session features
- Alternative model architectures
- Evaluation metrics (NDCG, Hit@k for recommender)
- Tests for uncovered modules
- Documentation improvements

## What not to contribute

- Changes that break the temporal split logic
- Dependencies that require a GPU at inference time
- Anything that adds a mandatory external API call

## Code style

Ruff with default settings. Line length 88. E501 ignored in `pipeline/features.py` and `llm/*.py`.

## Questions

Open an issue or start a discussion on GitHub.
