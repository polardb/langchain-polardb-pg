"""Quick Start Example — PolarDB + LangChain Vector Store in 5 Steps.

This example demonstrates the core workflow:
1. Connect to a PolarDB instance
2. Create in-database embeddings
3. Initialize a vector store table
4. Add documents
5. Perform similarity search

Prerequisites:
- A PolarDB for PostgreSQL instance with polar_ai extension
- pip install langchain-polardb-pg[cloud]

Usage:
    Set environment variables, then run:
    $ python examples/quickstart.py
"""

import os

from langchain_core.documents import Document

from langchain_polardb_pg import PolarDBPGEmbeddings, PolarDBPGEngine, PolarDBPGVector
from langchain_polardb_pg.model_manager import PolarDBPGModelManager


def main():
    # ---- Step 1: Connect to PolarDB ----
    print("Step 1: Connecting to PolarDB...")
    engine = PolarDBPGEngine.from_instance(
        cluster_id=os.environ["POLARDB_CLUSTER_ID"],
        database=os.environ.get("POLARDB_DATABASE", "postgres"),
        user=os.environ["POLARDB_USER"],
        password=os.environ["POLARDB_PASSWORD"],
        access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
        access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
        network_type=os.environ.get("POLARDB_NETWORK_TYPE", "Public"),
    )
    print(f"  Connected! Region: {engine.cluster_attribute.region_id}")

    # ---- Step 2: Create Embeddings ----
    print("\nStep 2: Creating in-database embeddings...")
    # Ensure model token is set (one-time setup)
    manager = PolarDBPGModelManager.create_sync(engine, auto_create_extension=True)
    embeddings = PolarDBPGEmbeddings.create_sync(engine)
    print(f"  Mode: {embeddings.mode.value}, Model: {embeddings.model_name}")

    # Get vector dimension for table creation
    sample_vec = embeddings.embed_query("test")
    vector_size = len(sample_vec)
    print(f"  Vector dimension: {vector_size}")

    # ---- Step 3: Initialize Vector Store Table ----
    table_name = "quickstart_docs"
    print(f"\nStep 3: Initializing table '{table_name}'...")
    engine.init_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        overwrite_existing=True,
    )
    print("  Table created.")

    # ---- Step 4: Create Vector Store & Add Documents ----
    print("\nStep 4: Adding documents...")
    store = PolarDBPGVector.create_sync(engine, embeddings, table_name=table_name)

    documents = [
        Document(
            page_content="PolarDB is a cloud-native relational database by Alibaba Cloud.",
            metadata={"source": "docs", "topic": "database"},
        ),
        Document(
            page_content="LangChain is a framework for building AI-powered applications.",
            metadata={"source": "docs", "topic": "ai"},
        ),
        Document(
            page_content="Vector databases enable efficient semantic similarity search.",
            metadata={"source": "blog", "topic": "database"},
        ),
        Document(
            page_content="The polar_ai extension brings AI capabilities inside PostgreSQL.",
            metadata={"source": "docs", "topic": "database"},
        ),
    ]
    ids = store.add_documents(documents)
    print(f"  Added {len(ids)} documents.")

    # ---- Step 5: Similarity Search ----
    print("\nStep 5: Searching for 'cloud native database'...")
    results = store.similarity_search("cloud native database", k=3)
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content[:60]}...")
        print(f"      metadata: {doc.metadata}")

    # Search with score
    print("\n  With scores:")
    scored = store.similarity_search_with_score("AI framework", k=2)
    for doc, score in scored:
        print(f"  score={score:.4f} | {doc.page_content[:50]}...")

    # ---- Cleanup ----
    print("\nCleanup: Dropping test table...")
    engine.drop_table(table_name)
    print("Done!")


if __name__ == "__main__":
    main()
