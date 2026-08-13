# Copyright 2026 Alibaba Cloud PolarDB Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for PolarDBPGEngine."""

import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_polardb_pg.engine import PolarDBPGEngine


class TestPolarDBPGEngineFromConnectionString:
    """Tests for PolarDBPGEngine.from_connection_string()."""

    def test_creates_engine_from_url_string(self):
        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        assert engine is not None
        assert engine._pool is not None
        assert engine._loop is not None

    def test_inherits_from_pgengine(self):
        from langchain_postgres import PGEngine

        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        assert isinstance(engine, PGEngine)
        assert isinstance(engine, PolarDBPGEngine)

    def test_passes_kwargs_to_create_async_engine(self):
        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb",
            pool_size=10,
            max_overflow=20,
        )
        assert engine is not None


class TestPolarDBPGEngineFromEngine:
    """Tests for PolarDBPGEngine.from_engine()."""

    def test_creates_from_existing_async_engine(self):
        from sqlalchemy.ext.asyncio import create_async_engine

        async_engine = create_async_engine(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        engine = PolarDBPGEngine.from_engine(async_engine)
        assert engine is not None
        assert engine._pool is async_engine


class TestPolarDBPGEngineFromInstance:
    """Tests for PolarDBPGEngine.from_instance()."""

    @patch("langchain_polardb_pg.engine.resolve_polardb_endpoint")
    @patch("langchain_polardb_pg.engine.describe_polardb_cluster_attribute")
    def test_works_without_region(self, mock_describe, mock_resolve):
        # Region is optional: the centralized OpenAPI endpoint resolves the
        # cluster by ID, so omitting region must not raise.
        from langchain_polardb_pg.utils.cloud_api import (
            PolarDBClusterAttribute,
            PolarDBEndpointInfo,
        )

        mock_describe.return_value = PolarDBClusterAttribute(
            region_id="cn-hangzhou",
            category="Normal",
            dbversion="16",
            dbtype="PostgreSQL",
            engine="POLARDB",
        )
        mock_resolve.return_value = PolarDBEndpointInfo(
            host="pc-bp1234.rwlb.polardb-pg.rds.aliyuncs.com",
            port="5432",
            endpoint_type="Cluster",
            network_type="Private",
            endpoint_id="pe-bp1234",
        )

        env_backup = os.environ.pop("POLARDB_REGION", None)
        try:
            engine = PolarDBPGEngine.from_instance(
                cluster_id="pc-bp1234567890",
                database="mydb",
                user="testuser",
                password="testpass",
                access_key_id="LTAI5tFake",
                access_key_secret="HzLBFake",
            )
            assert engine is not None
            assert mock_resolve.call_args[1]["region"] is None
        finally:
            if env_backup:
                os.environ["POLARDB_REGION"] = env_backup

    def test_raises_without_credentials(self):
        env_backup_id = os.environ.pop("ALIBABA_CLOUD_ACCESS_KEY_ID", None)
        env_backup_secret = os.environ.pop("ALIBABA_CLOUD_ACCESS_KEY_SECRET", None)
        try:
            with pytest.raises(
                ValueError, match="Alibaba Cloud credentials are required"
            ):
                PolarDBPGEngine.from_instance(
                    cluster_id="pc-bp1234567890",
                    database="mydb",
                    user="testuser",
                    password="testpass",
                    region="cn-hangzhou",
                )
        finally:
            if env_backup_id:
                os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"] = env_backup_id
            if env_backup_secret:
                os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"] = env_backup_secret

    @patch("langchain_polardb_pg.engine.resolve_polardb_endpoint")
    @patch("langchain_polardb_pg.engine.describe_polardb_cluster_attribute")
    def test_from_instance_with_mock(self, mock_describe, mock_resolve):
        from langchain_polardb_pg.utils.cloud_api import (
            PolarDBClusterAttribute,
            PolarDBEndpointInfo,
        )

        mock_describe.return_value = PolarDBClusterAttribute(
            region_id="cn-hangzhou",
            category="Normal",
            dbversion="16",
            dbtype="PostgreSQL",
            engine="POLARDB",
        )
        mock_resolve.return_value = PolarDBEndpointInfo(
            host="pc-bp1234.rwlb.polardb-pg.rds.aliyuncs.com",
            port="5432",
            endpoint_type="Cluster",
            network_type="Private",
            endpoint_id="pe-bp1234",
        )

        engine = PolarDBPGEngine.from_instance(
            cluster_id="pc-bp1234567890",
            database="mydb",
            user="testuser",
            password="testpass",
            region="cn-hangzhou",
            access_key_id="LTAI5tFake",
            access_key_secret="HzLBFake",
        )

        assert engine is not None
        assert isinstance(engine, PolarDBPGEngine)
        mock_resolve.assert_called_once_with(
            cluster_id="pc-bp1234567890",
            region="cn-hangzhou",
            access_key_id="LTAI5tFake",
            access_key_secret="HzLBFake",
            endpoint_type="Cluster",
            network_type="Private",
        )

    @patch("langchain_polardb_pg.engine.resolve_polardb_endpoint")
    @patch("langchain_polardb_pg.engine.describe_polardb_cluster_attribute")
    def test_from_instance_reads_region_from_env(self, mock_describe, mock_resolve):
        from langchain_polardb_pg.utils.cloud_api import (
            PolarDBClusterAttribute,
            PolarDBEndpointInfo,
        )

        mock_describe.return_value = PolarDBClusterAttribute(
            region_id="cn-shanghai",
            category="Normal",
            dbversion="16",
            dbtype="PostgreSQL",
            engine="POLARDB",
        )
        mock_resolve.return_value = PolarDBEndpointInfo(
            host="pc-bp1234.rwlb.polardb-pg.rds.aliyuncs.com",
            port="5432",
            endpoint_type="Cluster",
            network_type="Private",
            endpoint_id="pe-bp1234",
        )

        with patch.dict(os.environ, {"POLARDB_REGION": "cn-shanghai"}):
            engine = PolarDBPGEngine.from_instance(
                cluster_id="pc-bp1234567890",
                database="mydb",
                user="testuser",
                password="testpass",
                access_key_id="LTAI5tFake",
                access_key_secret="HzLBFake",
            )

        assert engine is not None
        mock_resolve.assert_called_once()
        call_kwargs = mock_resolve.call_args[1]
        assert call_kwargs["region"] == "cn-shanghai"


def _make_cluster_attribute():
    """Build a sample PolarDBClusterAttribute with AI info populated."""
    from langchain_polardb_pg.utils.cloud_api import PolarDBClusterAttribute

    return PolarDBClusterAttribute(
        region_id="cn-hangzhou",
        category="Normal",
        dbversion="16",
        dbtype="PostgreSQL",
        engine="POLARDB",
        ai_type="DLNode",
        ai_creating_time="2026-06-02T12:36:07Z",
        ai_free_mode="ON",
        api_keys=["sk-test-key"],
    )


class TestPolarDBPGEngineClusterAttributeCaching:
    """Tests for cluster attribute caching across the three creation paths."""

    @patch("langchain_polardb_pg.engine.resolve_polardb_endpoint")
    @patch("langchain_polardb_pg.engine.describe_polardb_cluster_attribute")
    def test_from_instance_caches_cluster_attribute(self, mock_describe, mock_resolve):
        from langchain_polardb_pg.utils.cloud_api import PolarDBEndpointInfo

        mock_describe.return_value = _make_cluster_attribute()
        mock_resolve.return_value = PolarDBEndpointInfo(
            host="pc-bp1234.rwlb.polardb-pg.rds.aliyuncs.com",
            port="5432",
            endpoint_type="Cluster",
            network_type="Private",
            endpoint_id="pe-bp1234",
        )

        engine = PolarDBPGEngine.from_instance(
            cluster_id="pc-bp1234567890",
            database="mydb",
            user="testuser",
            password="testpass",
            region="cn-hangzhou",
            access_key_id="LTAI5tFake",
            access_key_secret="HzLBFake",
        )

        cached = engine.cluster_attribute
        assert cached is not None
        assert cached.region_id == "cn-hangzhou"
        assert cached.ai_type == "DLNode"
        assert cached.ai_free_mode == "ON"
        assert cached.api_keys == ["sk-test-key"]
        assert cached.has_ai_node is True

    def test_from_connection_string_has_no_cluster_attribute(self):
        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        assert engine.cluster_attribute is None

    def test_from_engine_reuse_polardb_engine_inherits_cache(self):
        source = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        source._cluster_attribute = _make_cluster_attribute()

        reused = PolarDBPGEngine.from_engine(source)

        assert reused._pool is source._pool
        assert reused.cluster_attribute is source.cluster_attribute
        assert reused.cluster_attribute.ai_type == "DLNode"
        assert reused.cluster_attribute.api_keys == ["sk-test-key"]

    def test_from_engine_raw_async_engine_has_no_cache(self):
        from sqlalchemy.ext.asyncio import create_async_engine

        async_engine = create_async_engine(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        engine = PolarDBPGEngine.from_engine(async_engine)

        assert engine._pool is async_engine
        assert engine.cluster_attribute is None

    def test_from_engine_explicit_cluster_attribute_overrides_inherited(self):
        from langchain_polardb_pg.utils.cloud_api import PolarDBClusterAttribute

        source = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        source._cluster_attribute = _make_cluster_attribute()

        override = PolarDBClusterAttribute(
            region_id="cn-beijing",
            category="SENormal",
            dbversion="14",
            dbtype="PostgreSQL",
            engine="POLARDB",
        )
        reused = PolarDBPGEngine.from_engine(source, cluster_attribute=override)

        assert reused.cluster_attribute is override
        assert reused.cluster_attribute.region_id == "cn-beijing"
        assert reused.cluster_attribute.ai_type == ""


class TestPolarDBPGEngineGetAiApiKey:
    """Tests for PolarDBPGEngine.get_ai_api_key()."""

    def _make_engine_with_cluster_attribute(self):
        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        engine._cluster_attribute = _make_cluster_attribute()
        return engine

    def test_cluster_source_returns_api_keys_list(self):
        engine = self._make_engine_with_cluster_attribute()
        keys = engine.get_ai_api_key(source="cluster")
        assert keys == ["sk-test-key"]

    def test_cluster_source_is_default(self):
        engine = self._make_engine_with_cluster_attribute()
        assert engine.get_ai_api_key() == ["sk-test-key"]

    def test_cluster_source_returns_copy(self):
        engine = self._make_engine_with_cluster_attribute()
        keys = engine.get_ai_api_key(source="cluster")
        keys.append("mutated")
        assert engine.cluster_attribute.api_keys == ["sk-test-key"]

    def test_cluster_source_without_cache_raises(self):
        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        with pytest.raises(RuntimeError, match="No cluster attribute is cached"):
            engine.get_ai_api_key(source="cluster")

    def test_env_source_reads_default_env_var(self):
        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        with patch.dict(os.environ, {"POLARDB_AI_API_KEY": "sk-from-env"}):
            assert engine.get_ai_api_key(source="env") == ["sk-from-env"]

    def test_env_source_custom_env_var(self):
        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        with patch.dict(os.environ, {"MY_AI_KEY": "sk-custom"}):
            keys = engine.get_ai_api_key(source="env", env_var="MY_AI_KEY")
            assert keys == ["sk-custom"]

    def test_env_source_returns_empty_when_unset(self):
        engine = PolarDBPGEngine.from_connection_string(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        env_backup = os.environ.pop("POLARDB_AI_API_KEY", None)
        try:
            assert engine.get_ai_api_key(source="env") == []
        finally:
            if env_backup is not None:
                os.environ["POLARDB_AI_API_KEY"] = env_backup

    def test_unknown_source_raises(self):
        engine = self._make_engine_with_cluster_attribute()
        with pytest.raises(ValueError, match="Unknown source 'bogus'"):
            engine.get_ai_api_key(source="bogus")


class TestPolarDBPGEngineDirectConstruction:
    """Test that direct construction is prevented."""

    def test_direct_construction_raises(self):
        with pytest.raises(Exception, match="Only create class through"):
            PolarDBPGEngine(
                key="wrong_key",
                pool=MagicMock(),
                loop=None,
                thread=None,
            )
