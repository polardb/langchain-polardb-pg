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

"""Unit tests for the shared polar_ai extension helpers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_polardb_pg.utils.polar_ai_ext import (
    POLAR_AI_EXTENSION_NAME,
    create_polar_ai_extension,
    ensure_polar_ai_extension,
    polar_ai_extension_exists,
)


def _make_engine_with_first(first_value):
    """Build a mock AsyncEngine whose execute().first() returns first_value."""
    result = MagicMock()
    result.first = MagicMock(return_value=first_value)

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=result)
    conn.commit = AsyncMock()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    engine = MagicMock()
    engine.connect = MagicMock(return_value=ctx)
    return engine, conn


class TestPolarAIExtensionExists:
    @pytest.mark.asyncio
    async def test_returns_true_when_present(self):
        engine, _ = _make_engine_with_first((1,))
        assert await polar_ai_extension_exists(engine) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_absent(self):
        engine, _ = _make_engine_with_first(None)
        assert await polar_ai_extension_exists(engine) is False


class TestCreatePolarAIExtension:
    @pytest.mark.asyncio
    async def test_executes_create_and_commits(self):
        engine, conn = _make_engine_with_first(None)
        await create_polar_ai_extension(engine)
        conn.execute.assert_awaited_once()
        conn.commit.assert_awaited_once()
        executed_sql = str(conn.execute.call_args[0][0])
        assert "CREATE EXTENSION IF NOT EXISTS" in executed_sql
        assert POLAR_AI_EXTENSION_NAME in executed_sql

    @pytest.mark.asyncio
    async def test_wraps_failure_with_guidance(self):
        engine, conn = _make_engine_with_first(None)
        conn.execute = AsyncMock(side_effect=Exception("permission denied"))
        with pytest.raises(RuntimeError, match="Failed to create"):
            await create_polar_ai_extension(engine)


class TestEnsurePolarAIExtension:
    @pytest.mark.asyncio
    async def test_returns_when_already_present(self):
        engine, conn = _make_engine_with_first((1,))
        await ensure_polar_ai_extension(engine)
        # Only the existence check runs; no CREATE/commit.
        conn.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_missing_and_no_auto_create(self):
        engine, _ = _make_engine_with_first(None)
        with pytest.raises(RuntimeError, match="is not installed"):
            await ensure_polar_ai_extension(engine, auto_create=False)

    @pytest.mark.asyncio
    async def test_creates_when_missing_then_present(self):
        # First existence check -> absent; create; re-check -> present.
        absent_result = MagicMock()
        absent_result.first = MagicMock(return_value=None)
        present_result = MagicMock()
        present_result.first = MagicMock(return_value=(1,))

        conn = AsyncMock()
        conn.execute = AsyncMock(
            side_effect=[absent_result, MagicMock(), present_result]
        )
        conn.commit = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)
        engine = MagicMock()
        engine.connect = MagicMock(return_value=ctx)

        await ensure_polar_ai_extension(engine, auto_create=True)
        conn.commit.assert_awaited_once()
