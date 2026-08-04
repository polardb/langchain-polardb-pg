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

"""PolarDB for PostgreSQL Engine for LangChain."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import Future
from threading import Thread
from typing import Any, Mapping, Optional

from langchain_postgres import PGEngine
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from .utils.cloud_api import (
    AI_API_KEY_ENV_VAR,
    PolarDBClusterAttribute,
    describe_polardb_cluster_attribute,
    get_ai_api_key_from_env,
    resolve_polardb_endpoint,
)
from .version import __version__

USER_AGENT = "langchain-polardb-pg/" + __version__


class PolarDBPGEngine(PGEngine):
    """A class for managing connections to a PolarDB for PostgreSQL database.

    Extends PGEngine with PolarDB-specific connection support:
    - from_instance(): Auto-discover connection address via Alibaba Cloud OpenAPI
    """

    # Cluster attributes resolved during from_instance(); None for the other
    # constructors where attributes are not fetched from the OpenAPI.
    _cluster_attribute: Optional[PolarDBClusterAttribute] = None

    @property
    def cluster_attribute(self) -> Optional[PolarDBClusterAttribute]:
        """Cached cluster attributes resolved during from_instance().

        Returns the PolarDBClusterAttribute captured when the engine was
        created via from_instance() (region, edition/category, db version).
        Returns None for engines built via from_connection_string() or
        from_engine(), where cluster metadata is not fetched.

        Consumers (e.g. embeddings) can read this to make region/edition
        aware decisions without re-querying the OpenAPI.
        """
        return self._cluster_attribute

    def get_ai_api_key(
        self,
        source: str = "cluster",
        env_var: str = AI_API_KEY_ENV_VAR,
    ) -> list[str]:
        """Get AI service API key(s) for authorizing polar_ai model calls.

        Provides a single entry point for obtaining the AI authorization key
        from one of two sources, selected by ``source``:

        - ``"cluster"``: the keys provisioned on the cluster, read from the
          cached cluster_attribute.api_keys (populated by from_instance()).
        - ``"env"``: a key from an environment variable (default
          POLARDB_AI_API_KEY), kept separate from the Alibaba Cloud AK/SK.

        A list is always returned. The cluster source may expose multiple
        keys; the caller decides which one to use. The env source returns at
        most one key (wrapped in a list for a uniform return shape).

        Args:
            source: Where to read the key from. One of "cluster" or "env".
            env_var: Environment variable name to read when source="env".
                Defaults to POLARDB_AI_API_KEY.

        Returns:
            A list of API key strings (possibly empty if none are available).

        Raises:
            ValueError: If ``source`` is not "cluster" or "env".
            RuntimeError: If source="cluster" but no cluster_attribute is
                cached (e.g. the engine was built via from_connection_string()
                or from_engine() without cluster metadata).
        """
        if source == "cluster":
            if self._cluster_attribute is None:
                raise RuntimeError(
                    "No cluster attribute is cached on this engine, so "
                    "AI API keys cannot be read from the cluster. Build the "
                    "engine via from_instance(), or use source='env'."
                )
            return list(self._cluster_attribute.api_keys)
        if source == "env":
            return get_ai_api_key_from_env(env_var)
        raise ValueError(
            f"Unknown source '{source}'. Expected 'cluster' or 'env'."
        )

    @classmethod
    def _ensure_background_loop(cls) -> None:
        """Ensure a background event loop is running for sync/async bridging."""
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()

    @classmethod
    def from_instance(
        cls,
        cluster_id: str,
        database: str,
        user: str,
        password: str,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        endpoint_type: str = "Cluster",
        network_type: str = "Private",
        engine_args: Mapping[str, Any] = {},
    ) -> PolarDBPGEngine:
        """Create a PolarDBPGEngine by auto-discovering the connection address.

        Uses Alibaba Cloud OpenAPI (DescribeDBClusterEndpoints) to resolve
        the cluster's connection host and port automatically.

        Args:
            cluster_id: PolarDB cluster ID (e.g. "pc-bp1234567890").
            database: Database name.
            user: Database user name.
            password: Database password.
            region: Optional Alibaba Cloud region hint (e.g. "cn-hangzhou").
                Not required: the centralized OpenAPI endpoint resolves the
                cluster by ID regardless of region. Falls back to env var
                POLARDB_REGION.
            access_key_id: Alibaba Cloud Access Key ID.
                Falls back to env var ALIBABA_CLOUD_ACCESS_KEY_ID.
            access_key_secret: Alibaba Cloud Access Key Secret.
                Falls back to env var ALIBABA_CLOUD_ACCESS_KEY_SECRET.
            endpoint_type: Endpoint type. One of "Cluster", "Primary", "Custom".
            network_type: Network type. One of "Private", "Public".
            engine_args: Additional keyword arguments passed to
                sqlalchemy.ext.asyncio.create_async_engine.

        Returns:
            PolarDBPGEngine instance.

        Raises:
            ValueError: If no matching endpoint is found.
            ImportError: If alibabacloud SDK is not installed.
        """
        # Validate credentials early, before calling the SDK
        from .utils.cloud_api import _get_credentials

        resolved_key_id, resolved_key_secret = _get_credentials(
            access_key_id, access_key_secret
        )

        # Region is optional. The centralized OpenAPI endpoint
        # (polardb.aliyuncs.com) routes by cluster_id, so the region does
        # not need to be known in advance; it is provided only as a hint.
        region_hint = region or os.environ.get("POLARDB_REGION")

        cluster_attribute = describe_polardb_cluster_attribute(
            cluster_id=cluster_id,
            region=region_hint,
            access_key_id=resolved_key_id,
            access_key_secret=resolved_key_secret,
        )

        # PolarDBPGEngine only supports PolarDB for PostgreSQL. Fail fast with
        # a clear message if the cluster is a different engine (e.g. MySQL),
        # rather than letting the postgresql+asyncpg driver fail obscurely
        # against a non-PostgreSQL port later.
        if cluster_attribute.dbtype != "PostgreSQL":
            raise ValueError(
                f"Cluster '{cluster_id}' is a '{cluster_attribute.dbtype}' "
                f"instance, but PolarDBPGEngine only supports PolarDB for "
                f"PostgreSQL. Please provide a PostgreSQL cluster ID."
            )

        endpoint_info = resolve_polardb_endpoint(
            cluster_id=cluster_id,
            region=region_hint,
            access_key_id=resolved_key_id,
            access_key_secret=resolved_key_secret,
            endpoint_type=endpoint_type,
            network_type=network_type,
        )

        connection_url = URL.create(
            drivername="postgresql+asyncpg",
            username=user,
            password=password,
            host=endpoint_info.host,
            port=int(endpoint_info.port),
            database=database,
        )

        cls._ensure_background_loop()
        engine = create_async_engine(connection_url, **engine_args)
        instance = cls(
            PGEngine._PGEngine__create_key,  # type: ignore[attr-defined]
            engine,
            cls._default_loop,
            cls._default_thread,
        )
        instance._cluster_attribute = cluster_attribute
        return instance

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        cluster_attribute: Optional[PolarDBClusterAttribute] = None,
    ) -> PolarDBPGEngine:
        """Create a PolarDBPGEngine from an existing engine.

        Supports reusing an engine created via from_instance() while keeping
        its cached cluster attributes:

        - If ``engine`` is a PolarDBPGEngine, its underlying connection pool
          and cached cluster_attribute (region, edition, AI info) are reused.
        - If ``engine`` is a raw SQLAlchemy AsyncEngine, it is used as-is.

        The ``cluster_attribute`` argument, when provided, always takes
        precedence over any inherited value, letting callers attach or
        override cluster metadata explicitly.

        Args:
            engine: An existing PolarDBPGEngine or SQLAlchemy AsyncEngine.
            loop: Optional event loop for async/sync bridging. When reusing a
                PolarDBPGEngine, its loop is inherited if not provided.
            cluster_attribute: Optional cluster attributes to attach. Overrides
                any value inherited from a reused PolarDBPGEngine.

        Returns:
            PolarDBPGEngine instance.
        """
        inherited_attribute: Optional[PolarDBClusterAttribute] = None
        if isinstance(engine, PolarDBPGEngine):
            inherited_attribute = engine._cluster_attribute
            loop = loop or engine._loop
            underlying_engine = engine._pool
        else:
            underlying_engine = engine

        instance = cls(
            PGEngine._PGEngine__create_key,  # type: ignore[attr-defined]
            underlying_engine,
            loop,
            None,
        )
        instance._cluster_attribute = cluster_attribute or inherited_attribute
        return instance
