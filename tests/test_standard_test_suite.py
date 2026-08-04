"""LangChain standard test suite for PolarDBPGVector.

These tests verify that PolarDBPGVector conforms to the LangChain VectorStore
interface contract using the official langchain-tests standard suite.

Requires a live PolarDB instance. Set environment variables:
    POLARDB_CLUSTER_ID, POLARDB_DATABASE, POLARDB_USER, POLARDB_PASSWORD,
    ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET

Run with:
    PYTHONPATH=src python -m pytest tests/test_standard_test_suite.py -v
"""

import os
import uuid
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_postgres.v2.engine import Column
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_polardb_pg.embeddings import EmbeddingMode, PolarDBPGEmbeddings
from langchain_polardb_pg.engine import PolarDBPGEngine
from langchain_polardb_pg.model_manager import PolarDBPGModelManager
from langchain_polardb_pg.vectorstores import PolarDBPGVector

DEFAULT_TABLE = "test_std_" + uuid.uuid4().hex[:8]

# Skip all tests if env vars are not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("POLARDB_CLUSTER_ID"),
    reason="POLARDB_CLUSTER_ID not set, skipping standard test suite",
)


def _get_engine() -> PolarDBPGEngine:
    """Create engine from environment variables."""
    return PolarDBPGEngine.from_instance(
        cluster_id=os.environ["POLARDB_CLUSTER_ID"],
        database=os.environ.get("POLARDB_DATABASE", "ai_test"),
        user=os.environ["POLARDB_USER"],
        password=os.environ["POLARDB_PASSWORD"],
        access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
        access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
        network_type=os.environ.get("POLARDB_NETWORK_TYPE", "Public"),
    )


@pytest.mark.filterwarnings("ignore")
class TestStandardSuite(VectorStoreIntegrationTests):
    """LangChain standard VectorStore integration tests for PolarDB."""

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for standard tests."""
        engine = _get_engine()

        # Setup embeddings with token
        manager = PolarDBPGModelManager.create_sync(
            engine, auto_create_extension=True
        )
        mode = PolarDBPGEmbeddings._resolve_mode(engine)
        model_name = PolarDBPGEmbeddings._default_model_for_mode(mode)

        if mode == EmbeddingMode.AI_TEXT_EMBEDDING:
            token = os.environ.get("DASHSCOPE_API_KEY", "")
            if token:
                manager.set_model_token(model_name, token)
        else:
            keys = engine.get_ai_api_key(source="cluster")
            if keys:
                manager.set_model_token(model_name, keys[0])

        embeddings = PolarDBPGEmbeddings.create_sync(engine)

        # Determine vector size
        sample = embeddings.embed_query("test")
        vector_size = len(sample)

        # Create test table with TEXT id_column (standard tests use string IDs)
        table_name = f"test_std_{uuid.uuid4().hex[:8]}"
        engine.init_vectorstore_table(
            table_name=table_name,
            vector_size=vector_size,
            id_column=Column("langchain_id", "TEXT", nullable=False),
            overwrite_existing=True,
        )

        store = PolarDBPGVector.create_sync(
            engine=engine,
            embedding_service=embeddings,
            table_name=table_name,
        )

        try:
            yield store
        finally:
            engine.drop_table(table_name)
