# Examples

This directory contains runnable examples demonstrating `langchain-polardb-pg` usage.

## Prerequisites

1. A PolarDB for PostgreSQL instance with `polar_ai` extension enabled.
2. Install the package: `pip install langchain-polardb-pg[cloud]`
3. Set environment variables:

```bash
export POLARDB_CLUSTER_ID="pc-xxxxx"
export POLARDB_DATABASE="postgres"
export POLARDB_USER="your_user"
export POLARDB_PASSWORD="your_password"
export ALIBABA_CLOUD_ACCESS_KEY_ID="your_ak"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_sk"
export POLARDB_NETWORK_TYPE="Public"  # or "Private"
```

## Examples

| File | Description |
|------|-------------|
| `quickstart.py` | End-to-end vector store workflow in 5 steps |
| `model_management.py` | Full model lifecycle (create, alter, token, drop) |

## Running

```bash
python examples/quickstart.py
python examples/model_management.py
```
