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

"""Vector store for PolarDB for PostgreSQL.

``PolarDBPGVector`` inherits every capability from the community
``langchain_postgres.v2.PGVectorStore`` (add/delete, similarity search,
MMR, hybrid search and index management) while narrowing the engine type
to :class:`~langchain_polardb_pg.engine.PolarDBPGEngine`.

Usage::

    from langchain_polardb_pg import PolarDBPGEngine, PolarDBPGEmbeddings, PolarDBPGVector

    engine = PolarDBPGEngine.from_instance(cluster_id=..., database=..., ...)
    embeddings = PolarDBPGEmbeddings.create_sync(engine)
    store = PolarDBPGVector.create_sync(engine, embeddings, table_name="docs")
"""

from __future__ import annotations

from langchain_postgres.v2.vectorstores import PGVectorStore

from .engine import PolarDBPGEngine


class PolarDBPGVector(PGVectorStore):
    """PolarDB for PostgreSQL Vector Store class.

    A thin subclass of the community ``PGVectorStore`` that narrows the engine
    type to :class:`~langchain_polardb_pg.engine.PolarDBPGEngine` and keeps the
    PolarDB integration namespace self-contained.

    No method is overridden. ``create`` / ``create_sync`` and every search /
    add / delete capability are inherited as-is. The inherited factory methods
    construct via ``cls(...)``, so they already return ``PolarDBPGVector``
    instances. PolarDB's value (in-database and inline embedding) lives in the
    engine and embeddings layers; the community vector store consumes them
    transparently — including the inline ``embed_query_inline`` fast path, which
    it detects by duck-typing the embedding service.

    Usage::

        from langchain_polardb_pg import (
            PolarDBPGEngine, PolarDBPGEmbeddings, PolarDBPGVector,
        )

        engine = PolarDBPGEngine.from_instance(cluster_id=..., database=..., ...)
        embeddings = PolarDBPGEmbeddings.create_sync(engine)
        store = PolarDBPGVector.create_sync(engine, embeddings, table_name="docs")
    """

    _engine: PolarDBPGEngine
