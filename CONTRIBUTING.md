# Contributing to langchain-polardb-pg

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/polardb/langchain-polardb-pg.git
cd langchain-polardb-pg
```

2. Install development dependencies:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,cloud]"
```

3. Run tests:

```bash
make test
```

## Code Style

- Follow PEP 8 conventions
- Use type annotations for all public APIs
- Maximum line length: 88 characters (Black default)
- Use `ruff` for linting and formatting:

```bash
make lint
make format-check
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

Unit tests live under `tests/unit_tests` and mock all database interactions.
They can run without a live PolarDB instance:

```bash
make test
```

### LangChain Standard Integration Tests

LangChain standard tests live under `tests/integration_tests` and require a
live PolarDB instance. If the required environment variables are not set, these
tests skip automatically.

1. Set environment variables:

```bash
export POLARDB_CLUSTER_ID="pc-xxxxx"
export POLARDB_DATABASE="ai_test"
export POLARDB_USER="your_user"
export POLARDB_PASSWORD="your_password"
export POLARDB_NETWORK_TYPE="Public"
export ALIBABA_CLOUD_ACCESS_KEY_ID="your_ak"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_sk"
export DASHSCOPE_API_KEY="your_dashscope_key"
```

2. Run the standard tests:

```bash
make integration-test
```

### Distribution Checks

Before publishing a release, verify that the package builds and that its
metadata passes PyPI checks:

```bash
make build
make check-dist
```

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
