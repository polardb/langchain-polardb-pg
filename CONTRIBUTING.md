# Contributing to langchain-polardb-pg

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/<OWNER>/<REPO>.git
cd langchain-polardb-pg
```

2. Install development dependencies:

```bash
pip install -e ".[dev,cloud]"
```

3. Run tests:

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

## Code Style

- Follow PEP 8 conventions
- Use type annotations for all public APIs
- Maximum line length: 88 characters (Black default)
- Use `ruff` for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Project Structure

```
src/langchain_polardb_pg/
├── __init__.py          # Public API exports
├── engine.py            # PolarDBPGEngine (connection management)
├── embeddings.py        # PolarDBPGEmbeddings (dual-mode embedding)
├── model_manager.py     # PolarDBPGModelManager (model lifecycle)
├── vectorstores.py      # PolarDBPGVector (vector store)
└── utils/
    ├── cloud_api.py     # Alibaba Cloud OpenAPI integration
    ├── extensions.py    # PostgreSQL extension management
    └── polar_ai_ext.py  # polar_ai extension helpers
```

## Testing

### Unit Tests

Unit tests mock all database interactions and can run without a live instance:

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

### End-to-End Tests

E2E tests require a live PolarDB instance. They are **not** part of the standard test suite and are excluded from CI. To run them locally:

1. Set environment variables:

```bash
export POLARDB_CLUSTER_ID="pc-xxxxx"
export POLARDB_DATABASE="ai_test"
export POLARDB_USER="your_user"
export POLARDB_PASSWORD="your_password"
export ALIBABA_CLOUD_ACCESS_KEY_ID="your_ak"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_sk"
```

2. Run e2e scripts directly.

## Pull Request Guidelines

1. **Fork & Branch**: Create a feature branch from `main`.
2. **Tests**: Add unit tests for new functionality. Ensure all existing tests pass.
3. **Type Annotations**: All public methods must have complete type annotations.
4. **Docstrings**: Use Google-style docstrings for all public classes and methods.
5. **Commit Messages**: Use conventional commits format:
   - `feat: add new feature`
   - `fix: correct bug in ...`
   - `docs: update README`
   - `test: add tests for ...`
6. **No Credentials**: Never commit API keys, passwords, or connection strings.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include Python version, `langchain-polardb-pg` version, and PolarDB cluster edition.
- For bugs, include a minimal reproducible example.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
