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

"""Unit tests for PolarDBPGModelManager."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_polardb_pg.model_manager import PolarDBPGModel, PolarDBPGModelManager
from tests.unit_tests.helpers import run_as_sync_raising, run_as_sync_returning


class TestPolarDBPGModel:
    """Tests for PolarDBPGModel dataclass."""

    def test_create_model_with_required_fields(self):
        model = PolarDBPGModel(
            model_id="test_model",
            model_url="https://api.example.com",
            model_provider="dashscope",
            model_type="text_embedding",
        )
        assert model.model_id == "test_model"
        assert model.model_url == "https://api.example.com"
        assert model.model_provider == "dashscope"
        assert model.model_type == "text_embedding"
        assert model.model_name is None
        assert model.model_config == {}
        assert model.model_headers_fn is None
        assert model.model_in_transform_fn is None
        assert model.model_out_transform_fn is None

    def test_create_model_with_all_fields(self):
        model = PolarDBPGModel(
            model_id="_dashscope/text_embedding/v2",
            model_url="https://api.example.com",
            model_name="dashscope_embed",
            model_config={"version": "v2"},
            model_provider="dashscope",
            model_type="text_embedding",
            model_headers_fn="polar_ai.headers_fn",
            model_in_transform_fn="polar_ai.in_fn",
            model_out_transform_fn="polar_ai.out_fn",
        )
        assert model.model_id == "_dashscope/text_embedding/v2"
        assert model.model_url == "https://api.example.com"
        assert model.model_name == "dashscope_embed"
        assert model.model_config["version"] == "v2"
        assert model.model_provider == "dashscope"
        assert model.model_type == "text_embedding"
        assert model.model_in_transform_fn == "polar_ai.in_fn"


class TestPolarDBPGModelManagerConstruction:
    """Tests for PolarDBPGModelManager construction."""

    def test_direct_construction_raises(self):
        mock_engine = MagicMock()
        with pytest.raises(Exception, match="Only create class through"):
            PolarDBPGModelManager(key="wrong_key", engine=mock_engine)


class TestPolarDBPGModelManagerCreateSync:
    """Tests for PolarDBPGModelManager.create_sync()."""

    @patch.object(
        PolarDBPGModelManager,
        "_PolarDBPGModelManager__avalidate",
        new_callable=AsyncMock,
    )
    def test_create_sync_validates_extension(self, mock_validate):
        mock_engine = MagicMock()
        mock_engine._run_as_sync = run_as_sync_returning(None)

        manager = PolarDBPGModelManager.create_sync(mock_engine)
        assert manager is not None
        mock_engine._run_as_sync.assert_called_once()

    def test_create_sync_raises_on_missing_extension(self):
        mock_engine = MagicMock()
        mock_engine._run_as_sync = run_as_sync_raising(
            RuntimeError("The 'polar_ai' extension is not installed")
        )

        with pytest.raises(RuntimeError, match="polar_ai"):
            PolarDBPGModelManager.create_sync(mock_engine)


class TestPolarDBPGModelManagerOperations:
    """Tests for model CRUD operations."""

    def _create_manager_with_mock_engine(self):
        """Helper to create a manager with mocked engine."""
        mock_engine = MagicMock()
        mock_engine._run_as_sync = run_as_sync_returning(None)
        mock_engine._run_as_async = AsyncMock(return_value=None)
        manager = PolarDBPGModelManager(
            PolarDBPGModelManager._PolarDBPGModelManager__create_key,
            mock_engine,
        )
        return manager, mock_engine

    def test_create_model_calls_run_as_sync(self):
        manager, mock_engine = self._create_manager_with_mock_engine()
        manager.create_model(
            model_id="test_model",
            model_url="https://api.example.com",
            model_provider="dashscope",
            model_type="text_embedding",
        )
        mock_engine._run_as_sync.assert_called_once()

    def test_drop_model_calls_run_as_sync(self):
        manager, mock_engine = self._create_manager_with_mock_engine()
        manager.drop_model(model_id="test_model")
        mock_engine._run_as_sync.assert_called_once()

    def test_alter_model_calls_run_as_sync(self):
        manager, mock_engine = self._create_manager_with_mock_engine()
        manager.alter_model(model_id="test_model", model_url="https://new.url")
        mock_engine._run_as_sync.assert_called_once()

    def test_set_model_token_calls_run_as_sync(self):
        manager, mock_engine = self._create_manager_with_mock_engine()
        manager.set_model_token(model_id="test_model", model_token="sk-abc123")
        mock_engine._run_as_sync.assert_called_once()

    def test_list_models_calls_run_as_sync(self):
        manager, mock_engine = self._create_manager_with_mock_engine()
        mock_engine._run_as_sync = run_as_sync_returning([])
        result = manager.list_models()
        assert result == []
        mock_engine._run_as_sync.assert_called_once()

    def test_get_model_calls_run_as_sync(self):
        manager, mock_engine = self._create_manager_with_mock_engine()
        # get_model delegates to list_models, which returns a list; an empty
        # list means "not found" so get_model returns None.
        mock_engine._run_as_sync = run_as_sync_returning([])
        result = manager.get_model(model_id="nonexistent")
        assert result is None
        mock_engine._run_as_sync.assert_called_once()


class TestPolarDBPGModelManagerCreateModelSQL:
    """Tests for __acreate_model SQL construction and return value."""

    def _make_manager_capturing_sql(self, created_value=True):
        """Build a manager whose connection captures executed SQL/params.

        Returns (manager, captured) where captured is a dict populated with
        the executed 'query' and 'params' after acreate_model runs.
        """
        captured: dict = {}

        row = {"created": created_value} if created_value is not None else None
        result = MagicMock()
        result.mappings.return_value.fetchone.return_value = row

        async def fake_execute(query, params=None):
            captured["query"] = str(query)
            captured["params"] = params
            return result

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=fake_execute)
        conn.commit = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine._pool.connect = MagicMock(return_value=ctx)

        async def run_coro(coro):
            return await coro

        mock_engine._run_as_async = run_coro

        manager = PolarDBPGModelManager(
            PolarDBPGModelManager._PolarDBPGModelManager__create_key,
            mock_engine,
        )
        return manager, captured

    @pytest.mark.asyncio
    async def test_returns_created_boolean(self):
        manager, _ = self._make_manager_capturing_sql(created_value=True)
        created = await manager.acreate_model(
            model_id="m",
            model_url="https://u",
            model_provider="dashscope",
            model_type="text_embedding",
        )
        assert created is True

    @pytest.mark.asyncio
    async def test_returns_false_when_already_exists(self):
        manager, _ = self._make_manager_capturing_sql(created_value=False)
        created = await manager.acreate_model(
            model_id="m",
            model_url="https://u",
            model_provider="dashscope",
            model_type="text_embedding",
        )
        assert created is False

    @pytest.mark.asyncio
    async def test_uses_named_parameter_binding(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.acreate_model(
            model_id="m",
            model_url="https://u",
            model_provider="dashscope",
            model_type="text_embedding",
            model_name="My Model",
        )
        sql = captured["query"]
        assert "model_id => :model_id" in sql
        assert "model_name => :model_name" in sql
        assert "SELECT created FROM polar_ai.ai_createmodel(" in sql
        assert captured["params"]["model_name"] == "My Model"

    @pytest.mark.asyncio
    async def test_serializes_dict_model_config(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.acreate_model(
            model_id="m",
            model_url="https://u",
            model_provider="dashscope",
            model_type="text_embedding",
            model_config={"dim": 1024, "nested": {"a": 1}},
        )
        config_param = captured["params"]["model_config"]
        assert isinstance(config_param, str)
        assert json.loads(config_param) == {"dim": 1024, "nested": {"a": 1}}

    @pytest.mark.asyncio
    async def test_rejects_unknown_optional_argument(self):
        manager, _ = self._make_manager_capturing_sql()
        with pytest.raises(ValueError, match="Unknown argument 'bogus'"):
            await manager.acreate_model(
                model_id="m",
                model_url="https://u",
                model_provider="dashscope",
                model_type="text_embedding",
                bogus="x",
            )

    @pytest.mark.asyncio
    async def test_casts_regprocedure_args(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.acreate_model(
            model_id="m",
            model_url="https://u",
            model_provider="Alibaba",
            model_type="comment",
            model_in_transform_fn="polar_ai._in_fn(text,text)",
            model_out_transform_fn="polar_ai._out_fn(text,jsonb)",
        )
        sql = captured["query"]
        assert (
            "model_in_transform_fn => CAST(:model_in_transform_fn AS regprocedure)"
            in sql
        )
        assert (
            "model_out_transform_fn => CAST(:model_out_transform_fn AS regprocedure)"
            in sql
        )
        assert (
            captured["params"]["model_in_transform_fn"] == "polar_ai._in_fn(text,text)"
        )


class TestPolarDBPGModelManagerAlterModelSQL:
    """Tests for __aalter_model SQL construction and return value."""

    def _make_manager_capturing_sql(self, result_value=True):
        """Build a manager whose connection captures executed SQL/params."""
        captured: dict = {}

        row = {"result": result_value} if result_value is not None else None
        result = MagicMock()
        result.mappings.return_value.fetchone.return_value = row

        async def fake_execute(query, params=None):
            captured["query"] = str(query)
            captured["params"] = params
            return result

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=fake_execute)
        conn.commit = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine._pool.connect = MagicMock(return_value=ctx)

        async def run_coro(coro):
            return await coro

        mock_engine._run_as_async = run_coro

        manager = PolarDBPGModelManager(
            PolarDBPGModelManager._PolarDBPGModelManager__create_key,
            mock_engine,
        )
        return manager, captured

    @pytest.mark.asyncio
    async def test_returns_boolean_result(self):
        manager, _ = self._make_manager_capturing_sql(result_value=True)
        result = await manager.aalter_model(model_id="m", model_url="https://new")
        assert result is True

    @pytest.mark.asyncio
    async def test_uses_named_parameter_binding(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.aalter_model(model_id="m", model_provider="openai")
        sql = captured["query"]
        assert "model_id => :model_id" in sql
        assert "model_provider => :model_provider" in sql
        assert "polar_ai.ai_altermodel(" in sql
        assert captured["params"]["model_provider"] == "openai"

    @pytest.mark.asyncio
    async def test_serializes_dict_model_config(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.aalter_model(model_id="m", model_config={"dim": 512})
        config_param = captured["params"]["model_config"]
        assert isinstance(config_param, str)
        assert json.loads(config_param) == {"dim": 512}

    @pytest.mark.asyncio
    async def test_only_model_id_is_required(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.aalter_model(model_id="m")
        sql = captured["query"]
        assert sql.count("=>") == 1
        assert captured["params"] == {"model_id": "m"}

    @pytest.mark.asyncio
    async def test_rejects_unknown_optional_argument(self):
        manager, _ = self._make_manager_capturing_sql()
        with pytest.raises(ValueError, match="Unknown argument 'bogus'"):
            await manager.aalter_model(model_id="m", bogus="x")

    @pytest.mark.asyncio
    async def test_casts_regprocedure_args(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.aalter_model(
            model_id="m",
            model_headers_fn="polar_ai._headers_fn(text)",
        )
        sql = captured["query"]
        assert "model_headers_fn => CAST(:model_headers_fn AS regprocedure)" in sql


class TestPolarDBPGModelManagerSetModelTokenSQL:
    """Tests for __aset_model_token SQL construction and return value."""

    def _make_manager_capturing_sql(self, result_value=True):
        """Build a manager whose connection captures executed SQL/params."""
        captured: dict = {}

        row = {"result": result_value} if result_value is not None else None
        result = MagicMock()
        result.mappings.return_value.fetchone.return_value = row

        async def fake_execute(query, params=None):
            captured["query"] = str(query)
            captured["params"] = params
            return result

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=fake_execute)
        conn.commit = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine._pool.connect = MagicMock(return_value=ctx)

        async def run_coro(coro):
            return await coro

        mock_engine._run_as_async = run_coro

        manager = PolarDBPGModelManager(
            PolarDBPGModelManager._PolarDBPGModelManager__create_key,
            mock_engine,
        )
        return manager, captured

    @pytest.mark.asyncio
    async def test_returns_boolean_result(self):
        manager, _ = self._make_manager_capturing_sql(result_value=True)
        result = await manager.aset_model_token(model_id="m", model_token="sk-xyz")
        assert result is True

    @pytest.mark.asyncio
    async def test_uses_named_parameter_binding(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.aset_model_token(model_id="m", model_token="sk-xyz")
        sql = captured["query"]
        assert "model_id => :model_id" in sql
        assert "model_token => :model_token" in sql
        assert "polar_ai.ai_setmodeltoken(" in sql
        assert captured["params"] == {"model_id": "m", "model_token": "sk-xyz"}


class TestPolarDBPGModelManagerListModelsSQL:
    """Tests for __alist_models filter SQL construction."""

    def _make_manager_capturing_sql(self, rows=None):
        """Build a manager whose connection captures executed SQL/params."""
        captured: dict = {}

        result = MagicMock()
        result.mappings.return_value.fetchall.return_value = rows or []

        async def fake_execute(query, params=None):
            captured["query"] = str(query)
            captured["params"] = params
            return result

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=fake_execute)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine._pool.connect = MagicMock(return_value=ctx)

        async def run_coro(coro):
            return await coro

        mock_engine._run_as_async = run_coro

        manager = PolarDBPGModelManager(
            PolarDBPGModelManager._PolarDBPGModelManager__create_key,
            mock_engine,
        )
        return manager, captured

    @pytest.mark.asyncio
    async def test_no_filter_selects_all(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.alist_models()
        sql = captured["query"]
        assert "WHERE" not in sql
        assert "FROM polar_ai._ai_models" in sql
        assert "ORDER BY model_seq" in sql
        assert captured["params"] == {}

    @pytest.mark.asyncio
    async def test_single_filter(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.alist_models(model_id="m1")
        sql = captured["query"]
        assert "WHERE model_id = :model_id" in sql
        assert captured["params"] == {"model_id": "m1"}

    @pytest.mark.asyncio
    async def test_multiple_filters_combined_with_and(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.alist_models(model_provider="Alibaba", model_type="embedding")
        sql = captured["query"]
        assert "model_provider = :model_provider" in sql
        assert "model_type = :model_type" in sql
        assert " AND " in sql
        assert captured["params"] == {
            "model_provider": "Alibaba",
            "model_type": "embedding",
        }

    @pytest.mark.asyncio
    async def test_get_model_returns_single(self):
        rows = [{"model_id": "m1", "model_config": {"token": "sk"}}]
        manager, _ = self._make_manager_capturing_sql(rows=rows)
        model = await manager.aget_model("m1")
        assert model is not None
        assert model.model_id == "m1"

    @pytest.mark.asyncio
    async def test_get_model_returns_none_when_absent(self):
        manager, _ = self._make_manager_capturing_sql(rows=[])
        model = await manager.aget_model("nope")
        assert model is None


class TestPolarDBPGModelManagerCallModelSQL:
    """Tests for __acall_model SQL construction and jsonb result parsing.

    call_model first reads the model token (model_config ->> 'token') and
    only then invokes ai_callmodel. The mock therefore serves two queries:
    the first execute returns a token row, the second returns the call
    result. captured["query"]/captured["params"] always reflect the LAST
    (ai_callmodel) execute so existing SQL assertions keep working.
    """

    def _make_manager_capturing_sql(self, result_value=None, token="sk-token"):
        """Build a manager whose connection captures executed SQL/params.

        The pre-call token lookup now reuses the unified list query, so the
        first execute returns a full model row (with model_config carrying the
        token under the 'token' key). The second execute returns the
        ai_callmodel result. captured reflects the LAST (ai_callmodel) execute.

        Args:
            result_value: The 'result' value the ai_callmodel query returns.
                None means no row (function returned NULL / no row).
            token: The token stored in the model's model_config. Use a
                non-empty value to let the call proceed; "" to exercise the
                empty-token guard; None to simulate a missing model (the list
                query returns no rows).
        """
        captured: dict = {}

        if token is None:
            # No model row -> model not found.
            model_rows = []
        else:
            model_rows = [{"model_id": "m", "model_config": {"token": token}}]
        token_result = MagicMock()
        token_result.mappings.return_value.fetchall.return_value = model_rows

        call_row = {"result": result_value} if result_value is not None else None
        call_result = MagicMock()
        call_result.mappings.return_value.fetchone.return_value = call_row

        calls = {"n": 0}

        async def fake_execute(query, params=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # First execute is the token lookup (unified list query).
                return token_result
            captured["query"] = str(query)
            captured["params"] = params
            return call_result

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=fake_execute)
        conn.commit = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine._pool.connect = MagicMock(return_value=ctx)

        async def run_coro(coro):
            return await coro

        mock_engine._run_as_async = run_coro

        manager = PolarDBPGModelManager(
            PolarDBPGModelManager._PolarDBPGModelManager__create_key,
            mock_engine,
        )
        return manager, captured

    @pytest.mark.asyncio
    async def test_casts_string_payload_to_text(self):
        manager, captured = self._make_manager_capturing_sql(result_value={"ok": True})
        await manager.acall_model("m", "hello")
        sql = captured["query"]
        assert "polar_ai.ai_callmodel(:model_id, CAST(:payload AS text))" in sql
        assert captured["params"] == {"model_id": "m", "payload": "hello"}

    @pytest.mark.asyncio
    async def test_infers_sql_type_from_python_value(self):
        # Each call uses a fresh manager because the mock's per-connection
        # execute counter distinguishes the token lookup from the call query.
        manager, captured = self._make_manager_capturing_sql(result_value={"ok": True})
        await manager.acall_model("m", 42)
        assert "CAST(:payload AS bigint)" in captured["query"]

        manager, captured = self._make_manager_capturing_sql(result_value={"ok": True})
        await manager.acall_model("m", True)
        assert "CAST(:payload AS boolean)" in captured["query"]

        manager, captured = self._make_manager_capturing_sql(result_value={"ok": True})
        await manager.acall_model("m", 1.5)
        assert "CAST(:payload AS double precision)" in captured["query"]

    @pytest.mark.asyncio
    async def test_explicit_payload_type_overrides_inference(self):
        manager, captured = self._make_manager_capturing_sql(result_value={"ok": True})
        await manager.acall_model("m", "hi", payload_type="jsonb")
        assert "CAST(:payload AS jsonb)" in captured["query"]

    @pytest.mark.asyncio
    async def test_raises_when_token_is_empty(self):
        manager, _ = self._make_manager_capturing_sql(
            result_value={"ok": True}, token=""
        )
        with pytest.raises(ValueError, match="has an empty token"):
            await manager.acall_model("m", "hi")

    @pytest.mark.asyncio
    async def test_raises_when_model_not_found(self):
        manager, _ = self._make_manager_capturing_sql(
            result_value={"ok": True}, token=None
        )
        with pytest.raises(ValueError, match="was not found"):
            await manager.acall_model("m", "hi")

    @pytest.mark.asyncio
    async def test_returns_parsed_dict(self):
        manager, _ = self._make_manager_capturing_sql(
            result_value={"embedding": [0.1, 0.2]}
        )
        out = await manager.acall_model("m", "hi")
        assert out == {"embedding": [0.1, 0.2]}

    @pytest.mark.asyncio
    async def test_parses_json_string_result(self):
        manager, _ = self._make_manager_capturing_sql(result_value='{"a": 1}')
        out = await manager.acall_model("m", "hi")
        assert out == {"a": 1}

    @pytest.mark.asyncio
    async def test_returns_none_when_no_row(self):
        manager, _ = self._make_manager_capturing_sql(result_value=None)
        out = await manager.acall_model("m", "hi")
        assert out is None

    @pytest.mark.asyncio
    async def test_passes_through_non_json_string(self):
        manager, _ = self._make_manager_capturing_sql(result_value="plain")
        out = await manager.acall_model("m", "hi")
        assert out == "plain"


class TestPolarDBPGModelManagerDropModelSQL:
    """Tests for __adrop_model SQL construction and return value."""

    def _make_manager_capturing_sql(self, result_value=True):
        """Build a manager whose connection captures executed SQL/params."""
        captured: dict = {}

        row = {"result": result_value} if result_value is not None else None
        result = MagicMock()
        result.mappings.return_value.fetchone.return_value = row

        async def fake_execute(query, params=None):
            captured["query"] = str(query)
            captured["params"] = params
            return result

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=fake_execute)
        conn.commit = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine._pool.connect = MagicMock(return_value=ctx)

        async def run_coro(coro):
            return await coro

        mock_engine._run_as_async = run_coro

        manager = PolarDBPGModelManager(
            PolarDBPGModelManager._PolarDBPGModelManager__create_key,
            mock_engine,
        )
        return manager, captured

    @pytest.mark.asyncio
    async def test_returns_boolean_result(self):
        manager, _ = self._make_manager_capturing_sql(result_value=True)
        result = await manager.adrop_model("m")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_dropped(self):
        manager, _ = self._make_manager_capturing_sql(result_value=False)
        result = await manager.adrop_model("m")
        assert result is False

    @pytest.mark.asyncio
    async def test_uses_named_parameter_binding(self):
        manager, captured = self._make_manager_capturing_sql()
        await manager.adrop_model("m")
        sql = captured["query"]
        assert "polar_ai.ai_dropmodel(:model_id)" in sql
        assert captured["params"] == {"model_id": "m"}
