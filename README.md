# langchain-polardb-pg

LangChain integration for **Alibaba Cloud PolarDB for PostgreSQL** — providing in-database embeddings, vector store, and model management powered by the `polar_ai` extension.

## Features

- **In-Database Embeddings** — compute text embeddings directly inside PolarDB, eliminating external API calls and network round-trips.
- **Dual Embedding Modes** — automatically selects the optimal path based on your cluster edition:
  - `ai_text_embedding` mode (Standard edition): inline embedding in a single SQL statement
  - `call_model` mode (Enterprise edition + AI node): embedding via `ai_callmodel`
- **Vector Store** — full-featured similarity search, MMR, filtering, built on community `langchain-postgres`.
- **Model Management** — register, configure, and manage AI models through Python APIs.
- **Auto-Discovery** — `from_instance()` resolves cluster endpoints and capabilities via Alibaba Cloud OpenAPI.

## Installation

```bash
pip install langchain-polardb-pg
```

With cloud auto-discovery support (recommended):

```bash
pip install langchain-polardb-pg[cloud]
```

## Quick Start

```python
from langchain_polardb_pg import PolarDBPGEngine, PolarDBPGEmbeddings, PolarDBPGVector

# 1. Connect to your PolarDB instance
engine = PolarDBPGEngine.from_instance(
    cluster_id="pc-xxxxx",
    database="mydb",
    user="user",
    password="password",
    access_key_id="your-ak",
    access_key_secret="your-sk",
)

# 2. Create in-database embeddings (no external API needed!)
embeddings = PolarDBPGEmbeddings.create_sync(engine)

# 3. Initialize vector store table
engine.init_vectorstore_table("my_docs", vector_size=1536)

# 4. Create vector store
store = PolarDBPGVector.create_sync(engine, embeddings, table_name="my_docs")

# 5. Add documents & search
store.add_texts(["PolarDB is a cloud-native database", "LangChain is an AI framework"])
results = store.similarity_search("cloud database", k=2)
```

## Embedding Modes

| Mode | Edition | How it works | Vector Dim |
|------|---------|--------------|------------|
| `ai_text_embedding` | Standard (Beijing) | Inline SQL — embedding computed inside INSERT/SELECT | 1536 |
| `call_model` | Enterprise + AI Node | Two-step — `ai_callmodel` then INSERT/SELECT | 1024 |

The mode is **auto-resolved** from your cluster attributes. You can also specify it explicitly:

```python
from langchain_polardb_pg.embeddings import EmbeddingMode

embeddings = PolarDBPGEmbeddings.create_sync(engine, mode=EmbeddingMode.AI_TEXT_EMBEDDING)
```

## Components

### PolarDBPGEngine

Extends the community `PGEngine` with:
- `from_instance()` — auto-discover endpoints via Alibaba Cloud OpenAPI
- `cluster_attribute` — cached cluster metadata (region, edition, AI capabilities)
- `get_ai_api_key()` — unified AI key retrieval (from cluster or environment)

### PolarDBPGEmbeddings

Implements LangChain `Embeddings` interface:
- `embed_query(text)` / `embed_documents(texts)` — standard embedding APIs
- `embed_query_inline(text)` — returns a SQL expression for inline embedding (ai_text_embedding mode only)

### PolarDBPGVector

Inherits all capabilities from community `PGVectorStore`:
- `add_documents()` / `add_texts()` — with in-database embedding
- `similarity_search()` / `similarity_search_with_score()`
- `max_marginal_relevance_search()` — diversity-aware retrieval
- `delete()` — by IDs or metadata filter

### PolarDBPGModelManager

Model lifecycle management:
- `create_model()` / `alter_model()` / `drop_model()`
- `set_model_token()` / `call_model()`
- `list_models()` / `get_model()`

## Connection Methods

```python
# Method 1: Auto-discovery (recommended)
engine = PolarDBPGEngine.from_instance(
    cluster_id="pc-xxxxx",
    database="mydb", user="user", password="pass",
    access_key_id="ak", access_key_secret="sk",
)

# Method 2: Direct connection string
engine = PolarDBPGEngine.from_connection_string(
    "postgresql+asyncpg://user:pass@host:5432/mydb"
)

# Method 3: From existing SQLAlchemy engine
engine = PolarDBPGEngine.from_engine(existing_async_engine)
```

## Requirements

- Python >= 3.9
- PolarDB for PostgreSQL with `polar_ai` extension
- Dependencies: `langchain-postgres`, `langchain-core`, `sqlalchemy[asyncio]`, `asyncpg`

## Development

Install local development dependencies:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,cloud]"
```

Run unit tests:

```bash
make test
```

Run LangChain standard vector store integration tests with a live PolarDB
instance:

```bash
make integration-test
```

Build and validate distribution metadata:

```bash
make build
make check-dist
```

## License

[Apache License 2.0](LICENSE)
