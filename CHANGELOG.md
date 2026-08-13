# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-13

### Changed

- Improved README release readiness with CI, PyPI, Python version, and license badges.
- Documented runtime environment variables and PolarDB requirements.
- Updated the quick start to detect embedding vector dimensions dynamically.

## [0.1.0] - 2026-06-16

### Added

- **PolarDBPGEngine**: extends community `PGEngine` with `from_instance()` for auto-discovering PolarDB cluster endpoints and attributes via Alibaba Cloud OpenAPI.
- **PolarDBPGEmbeddings**: dual-mode in-database embedding
  - `ai_text_embedding` mode: inline embedding in SQL (Standard edition)
  - `call_model` mode: embedding via `ai_callmodel` (Enterprise edition + AI node)
  - Auto mode resolution based on cluster attributes
  - `embed_query_inline` for zero-round-trip embedding in VectorStore operations
- **PolarDBPGModelManager**: model lifecycle management (create, alter, drop, set_token, call_model, list, get) aligned with `polar_ai` kernel function signatures.
- **PolarDBPGVector**: extends community `PGVectorStore`, narrows engine type to `PolarDBPGEngine`.
- **Utils**: cloud API integration (endpoint discovery, cluster attributes), extension management helpers.
- **get_ai_api_key()**: unified AI key retrieval from cluster metadata or environment variables.
- Full unit test suite.
- Apache License 2.0.
