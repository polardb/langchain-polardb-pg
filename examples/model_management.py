"""Model Management Example — PolarDBPGModelManager.

This example demonstrates the full model lifecycle:
1. Connect to PolarDB
2. List registered models
3. Create a custom model
4. Set model token
5. Call model (inference)
6. Alter model attributes
7. Drop model

Prerequisites:
- A PolarDB for PostgreSQL instance with polar_ai extension
- pip install langchain-polardb-pg[cloud]

Usage:
    Set environment variables, then run:
    $ python examples/model_management.py
"""

import os

from langchain_polardb_pg import PolarDBPGEngine
from langchain_polardb_pg.model_manager import PolarDBPGModelManager


def main():
    # ---- Connect ----
    print("Connecting to PolarDB...")
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

    # ---- Create Manager ----
    manager = PolarDBPGModelManager.create_sync(engine, auto_create_extension=True)

    # ---- 1. List Models ----
    print("\n1. Listing registered models...")
    models = manager.list_models()
    print(f"   Found {len(models)} model(s):")
    for m in models:
        print(f"   - {m.model_id} (provider={m.model_provider}, type={m.model_type})")

    # ---- 2. Create a Model ----
    print("\n2. Creating a custom model...")
    model_id = "example/my_embedding_model"
    created = manager.create_model(
        model_id=model_id,
        model_url="https://api.example.com/v1/embeddings",
        model_provider="example_provider",
        model_type="text_embedding",
        model_name="my-embedding-v1",
        model_config={"dimension": 768, "max_tokens": 512},
    )
    print(f"   Created: {created}")

    # ---- 3. Get Model Info ----
    print("\n3. Getting model info...")
    model = manager.get_model(model_id)
    if model:
        print(f"   ID: {model.model_id}")
        print(f"   URL: {model.model_url}")
        print(f"   Provider: {model.model_provider}")
        print(f"   Config: {model.model_config}")

    # ---- 4. Set Token ----
    print("\n4. Setting model token...")
    token_set = manager.set_model_token(model_id, "sk-example-token-12345")
    print(f"   Token set: {token_set}")

    # ---- 5. Alter Model ----
    print("\n5. Altering model attributes...")
    altered = manager.alter_model(
        model_id=model_id,
        model_provider="updated_provider",
        model_config={"dimension": 1024, "max_tokens": 1024},
    )
    print(f"   Altered: {altered}")

    # Verify changes
    model = manager.get_model(model_id)
    if model:
        print(f"   New provider: {model.model_provider}")
        print(f"   New config: {model.model_config}")

    # ---- 6. Drop Model ----
    print("\n6. Dropping model...")
    dropped = manager.drop_model(model_id)
    print(f"   Dropped: {dropped}")

    # Verify deletion
    model = manager.get_model(model_id)
    print(f"   Model exists after drop: {model is not None}")

    # ---- Using get_ai_api_key ----
    print("\n7. Getting AI API keys...")
    try:
        keys = engine.get_ai_api_key(source="cluster")
        print(f"   Cluster keys: {len(keys)} key(s) available")
    except RuntimeError as e:
        print(f"   Cluster keys: {e}")

    # From environment variable
    os.environ.setdefault("POLARDB_AI_API_KEY", "sk-demo-key")
    env_keys = engine.get_ai_api_key(source="env")
    print(f"   Env keys: {env_keys}")

    print("\nDone!")


if __name__ == "__main__":
    main()
