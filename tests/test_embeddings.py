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

"""Unit tests for PolarDBPGEmbeddings."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings

from langchain_polardb_pg.embeddings import (
    AI_TEXT_EMBEDDING_DEFAULT_MODEL,
    CALL_MODEL_DEFAULT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingMode,
    PolarDBPGEmbeddings,
)


def _make_embeddings(
    engine, model_name="my_model", mode=EmbeddingMode.AI_TEXT_EMBEDDING
):
    """Build an instance via the private create key for unit testing."""
    return PolarDBPGEmbeddings(
        PolarDBPGEmbeddings._PolarDBPGEmbeddings__create_key,
        engine,
        model_name,
        mode,
    )


def _attribute(
    *, beijing=False, standard=False, enterprise=False, ai_node=False
):
    """Build a cluster attribute stub exposing the resolution properties."""
    attribute = MagicMock()
    attribute.is_beijing_region = beijing
    attribute.is_standard_edition = standard
    attribute.is_enterprise_edition = enterprise
    attribute.has_ai_node = ai_node
    return attribute


class TestPolarDBPGEmbeddingsConstruction:
    """Tests for PolarDBPGEmbeddings construction."""

    def test_direct_construction_raises(self):
        with pytest.raises(Exception, match="Only create class through"):
            PolarDBPGEmbeddings(
                key="wrong_key",
                engine=MagicMock(),
                model_name="test_model",
                mode=EmbeddingMode.AI_TEXT_EMBEDDING,
            )

    def test_implements_embeddings_interface(self):
        embeddings = _make_embeddings(MagicMock(), "test_model")
        assert isinstance(embeddings, Embeddings)


class TestPolarDBPGEmbeddingsModeResolution:
    """Tests for _resolve_mode based on cluster attributes."""

    def test_no_attribute_falls_back_to_ai_text_embedding(self):
        engine = MagicMock()
        engine.cluster_attribute = None
        assert (
            PolarDBPGEmbeddings._resolve_mode(engine)
            == EmbeddingMode.AI_TEXT_EMBEDDING
        )

    def test_beijing_standard_uses_ai_text_embedding(self):
        engine = MagicMock()
        engine.cluster_attribute = _attribute(beijing=True, standard=True)
        assert (
            PolarDBPGEmbeddings._resolve_mode(engine)
            == EmbeddingMode.AI_TEXT_EMBEDDING
        )

    def test_enterprise_with_ai_node_uses_call_model(self):
        engine = MagicMock()
        engine.cluster_attribute = _attribute(enterprise=True, ai_node=True)
        assert (
            PolarDBPGEmbeddings._resolve_mode(engine)
            == EmbeddingMode.CALL_MODEL
        )

    def test_enterprise_without_ai_node_falls_back(self):
        engine = MagicMock()
        engine.cluster_attribute = _attribute(enterprise=True, ai_node=False)
        assert (
            PolarDBPGEmbeddings._resolve_mode(engine)
            == EmbeddingMode.AI_TEXT_EMBEDDING
        )

    def test_default_model_for_mode(self):
        assert (
            PolarDBPGEmbeddings._default_model_for_mode(
                EmbeddingMode.AI_TEXT_EMBEDDING
            )
            == AI_TEXT_EMBEDDING_DEFAULT_MODEL
        )
        assert (
            PolarDBPGEmbeddings._default_model_for_mode(
                EmbeddingMode.CALL_MODEL
            )
            == CALL_MODEL_DEFAULT_MODEL
        )

    def test_validate_rejects_unknown_ai_text_embedding_model(self):
        with pytest.raises(ValueError, match="not supported"):
            PolarDBPGEmbeddings._validate_model_for_mode(
                "some_random_model", EmbeddingMode.AI_TEXT_EMBEDDING
            )

    def test_validate_accepts_any_call_model_model(self):
        # Should not raise for arbitrary models in call_model mode.
        PolarDBPGEmbeddings._validate_model_for_mode(
            "custom/model", EmbeddingMode.CALL_MODEL
        )


class TestPolarDBPGEmbeddingsCreateSync:
    """Tests for PolarDBPGEmbeddings.create_sync()."""

    def test_create_sync_raises_on_nonexistent_model(self):
        mock_engine = MagicMock()
        mock_engine.cluster_attribute = None
        mock_engine._run_as_sync = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="does not exist"):
            PolarDBPGEmbeddings.create_sync(
                mock_engine, model_name=DEFAULT_EMBEDDING_MODEL
            )

    def test_create_sync_succeeds_with_existing_model(self):
        mock_engine = MagicMock()
        mock_engine.cluster_attribute = None
        mock_engine._run_as_sync = MagicMock(return_value=True)

        embeddings = PolarDBPGEmbeddings.create_sync(
            mock_engine, model_name=DEFAULT_EMBEDDING_MODEL
        )
        assert embeddings is not None
        assert embeddings.model_name == DEFAULT_EMBEDDING_MODEL

    def test_default_model_name_ai_text_embedding(self):
        mock_engine = MagicMock()
        mock_engine.cluster_attribute = _attribute(beijing=True, standard=True)
        mock_engine._run_as_sync = MagicMock(return_value=True)

        embeddings = PolarDBPGEmbeddings.create_sync(mock_engine)
        assert embeddings.model_name == AI_TEXT_EMBEDDING_DEFAULT_MODEL
        assert embeddings.mode == EmbeddingMode.AI_TEXT_EMBEDDING

    def test_default_model_name_call_model(self):
        mock_engine = MagicMock()
        mock_engine.cluster_attribute = _attribute(
            enterprise=True, ai_node=True
        )
        mock_engine._run_as_sync = MagicMock(return_value=True)

        embeddings = PolarDBPGEmbeddings.create_sync(mock_engine)
        assert embeddings.model_name == CALL_MODEL_DEFAULT_MODEL
        assert embeddings.mode == EmbeddingMode.CALL_MODEL


class TestPolarDBPGEmbeddingsInline:
    """Tests for embed_query_inline()."""

    def test_embed_query_inline_returns_sql_expression(self):
        embeddings = _make_embeddings(MagicMock(), "my_model")
        result = embeddings.embed_query_inline("hello world")
        assert "polar_ai.ai_text_embedding" in result
        assert "my_model" in result
        assert "hello world" in result
        assert "::vector" in result

    def test_embed_query_inline_dollar_quotes_single_quotes(self):
        embeddings = _make_embeddings(MagicMock(), "my_model")
        result = embeddings.embed_query_inline("it's a test")
        # Dollar-quoting keeps single quotes verbatim inside the literal.
        assert "$polarai$it's a test$polarai$" in result

    def test_embed_query_inline_dollar_quotes_model_name(self):
        embeddings = _make_embeddings(MagicMock(), "model'name")
        result = embeddings.embed_query_inline("test")
        assert "$polarai$model'name$polarai$" in result

    def test_embed_query_inline_resists_dollar_tag_collision(self):
        embeddings = _make_embeddings(MagicMock(), "my_model")
        # Content containing the default tag must force a longer, unique tag
        # so the literal cannot be terminated early.
        result = embeddings.embed_query_inline("evil $polarai$ payload")
        assert "$polaraix$evil $polarai$ payload$polaraix$" in result

    def test_embed_query_inline_hidden_in_call_model_mode(self):
        # Outside ai_text_embedding mode the method must be hidden via
        # AttributeError so the community vector store's
        # getattr(..., "embed_query_inline", None) detection falls back to the
        # regular embed-then-insert path instead of splicing inline SQL.
        embeddings = _make_embeddings(
            MagicMock(), "custom/model", EmbeddingMode.CALL_MODEL
        )
        with pytest.raises(AttributeError, match="only available in the"):
            embeddings.embed_query_inline("hello world")

    def test_embed_query_inline_undetectable_in_call_model_mode(self):
        # The community capability check uses getattr with a default; in
        # call_model mode it must resolve to the default (None), i.e. the
        # store sees no inline capability.
        embeddings = _make_embeddings(
            MagicMock(), "custom/model", EmbeddingMode.CALL_MODEL
        )
        inline_func = getattr(embeddings, "embed_query_inline", None)
        assert inline_func is None


class TestPolarDBPGEmbeddingsModelExists:
    """Tests for __amodel_exists reusing the unified model query."""

    @staticmethod
    def _model_with_token(token):
        model = MagicMock()
        model.model_config = {"author_type": "token", "token": token}
        return model

    @pytest.mark.asyncio
    async def test_model_exists_uses_aget_model_by_model_id(self):
        # __amodel_exists must call aget_model with model_id as a positional
        # argument (the param was renamed from model_name to model_id). This
        # guards against the call breaking on the renamed parameter.
        embeddings = _make_embeddings(MagicMock(), "my_model")

        mock_manager = MagicMock()
        mock_manager.aget_model = AsyncMock(
            return_value=self._model_with_token("sk-abc")
        )

        with patch(
            "langchain_polardb_pg.embeddings.PolarDBPGModelManager.create",
            new=AsyncMock(return_value=mock_manager),
        ):
            exists = await embeddings._PolarDBPGEmbeddings__amodel_exists()

        assert exists is True
        mock_manager.aget_model.assert_awaited_once_with("my_model")

    @pytest.mark.asyncio
    async def test_model_with_empty_token_raises(self):
        # A registered model with an empty token must raise, since embedding
        # calls would otherwise fail at runtime.
        embeddings = _make_embeddings(MagicMock(), "my_model")

        mock_manager = MagicMock()
        mock_manager.aget_model = AsyncMock(
            return_value=self._model_with_token("")
        )

        with patch(
            "langchain_polardb_pg.embeddings.PolarDBPGModelManager.create",
            new=AsyncMock(return_value=mock_manager),
        ):
            with pytest.raises(ValueError, match="token is empty"):
                await embeddings._PolarDBPGEmbeddings__amodel_exists()

    @pytest.mark.asyncio
    async def test_model_missing_falls_back_to_builtin_probe(self):
        embeddings = _make_embeddings(
            MagicMock(), "my_model", EmbeddingMode.AI_TEXT_EMBEDDING
        )

        mock_manager = MagicMock()
        mock_manager.aget_model = AsyncMock(return_value=None)

        with patch(
            "langchain_polardb_pg.embeddings.PolarDBPGModelManager.create",
            new=AsyncMock(return_value=mock_manager),
        ), patch.object(
            PolarDBPGEmbeddings,
            "_PolarDBPGEmbeddings__atry_builtin_model",
            new=AsyncMock(return_value=True),
        ):
            exists = await embeddings._PolarDBPGEmbeddings__amodel_exists()

        assert exists is True
        mock_manager.aget_model.assert_awaited_once_with("my_model")

    @pytest.mark.asyncio
    async def test_call_model_missing_does_not_probe_builtin(self):
        # In call_model mode a missing registration must not fall back to the
        # ai_text_embedding builtin probe.
        embeddings = _make_embeddings(
            MagicMock(), "custom/model", EmbeddingMode.CALL_MODEL
        )

        mock_manager = MagicMock()
        mock_manager.aget_model = AsyncMock(return_value=None)

        with patch(
            "langchain_polardb_pg.embeddings.PolarDBPGModelManager.create",
            new=AsyncMock(return_value=mock_manager),
        ):
            exists = await embeddings._PolarDBPGEmbeddings__amodel_exists()

        assert exists is False


class TestPolarDBPGEmbeddingsSync:
    """Tests for sync embed methods."""

    def test_embed_query_calls_run_as_sync(self):
        mock_engine = MagicMock()
        mock_engine._run_as_sync = MagicMock(return_value=[0.1, 0.2, 0.3])

        embeddings = _make_embeddings(mock_engine, "my_model")
        result = embeddings.embed_query("test text")
        assert result == [0.1, 0.2, 0.3]
        mock_engine._run_as_sync.assert_called_once()

    def test_embed_documents_calls_run_as_sync(self):
        mock_engine = MagicMock()
        mock_engine._run_as_sync = MagicMock(
            return_value=[[0.1, 0.2], [0.3, 0.4]]
        )

        embeddings = _make_embeddings(mock_engine, "my_model")
        result = embeddings.embed_documents(["text1", "text2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_engine._run_as_sync.assert_called_once()


class TestPolarDBPGEmbeddingsCallModelParse:
    """Tests for the call_model response parsing helper."""

    def _embeddings(self):
        return _make_embeddings(
            MagicMock(), CALL_MODEL_DEFAULT_MODEL, EmbeddingMode.CALL_MODEL
        )

    def test_parse_bare_vector_array(self):
        embeddings = self._embeddings()
        parsed = embeddings._PolarDBPGEmbeddings__parse_call_model_response(
            [0.1, 0.2, 0.3]
        )
        assert parsed == [0.1, 0.2, 0.3]

    def test_parse_bare_vector_array_from_json_string(self):
        embeddings = self._embeddings()
        parsed = embeddings._PolarDBPGEmbeddings__parse_call_model_response(
            "[0.1, 0.2, 0.3]"
        )
        assert parsed == [0.1, 0.2, 0.3]

    def test_parse_failure_envelope_raises_with_message(self):
        embeddings = self._embeddings()
        response = {"success": "false", "errMessage": "Unknown model."}
        with pytest.raises(RuntimeError, match="Unknown model."):
            embeddings._PolarDBPGEmbeddings__parse_call_model_response(response)

    def test_parse_unexpected_object_raises(self):
        embeddings = self._embeddings()
        response = {"foo": "bar"}
        with pytest.raises(RuntimeError, match="Unexpected ai_callmodel"):
            embeddings._PolarDBPGEmbeddings__parse_call_model_response(response)


class TestPolarDBPGEmbeddingsCallModelBody:
    """Tests that the call_model path builds the correct request body."""

    @staticmethod
    def _conn_factory(captured):
        class _Result:
            def mappings(self):
                return self

            def fetchone(self):
                return {"response": [0.5, 0.6]}

        class _Conn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def execute(self, _sql, params):
                captured.update(params)
                return _Result()

        return _Conn

    @pytest.mark.asyncio
    async def test_call_model_embed_builds_body_and_parses(self):
        captured = {}
        mock_engine = MagicMock()
        mock_engine._pool.connect = MagicMock(
            return_value=self._conn_factory(captured)()
        )

        embeddings = _make_embeddings(
            mock_engine, CALL_MODEL_DEFAULT_MODEL, EmbeddingMode.CALL_MODEL
        )
        vector = await embeddings._PolarDBPGEmbeddings__acall_model_embed(
            ["hello"]
        )

        assert vector == [0.5, 0.6]
        import json as _json

        body = _json.loads(captured["body"])
        assert body["input"] == ["hello"]
        assert body["model"] == embeddings.call_model_body_model
        assert body["dimensions"] == embeddings.call_model_dimension
        assert body["deploymentName"] == embeddings.call_model_deployment_name

    @pytest.mark.asyncio
    async def test_call_model_embed_empty_raises(self):
        mock_engine = MagicMock()
        embeddings = _make_embeddings(
            mock_engine, CALL_MODEL_DEFAULT_MODEL, EmbeddingMode.CALL_MODEL
        )
        with pytest.raises(ValueError, match="at least one text"):
            await embeddings._PolarDBPGEmbeddings__acall_model_embed([])

    @pytest.mark.asyncio
    async def test_call_model_embed_multiple_warns_and_uses_first(self):
        captured = {}
        mock_engine = MagicMock()
        mock_engine._pool.connect = MagicMock(
            return_value=self._conn_factory(captured)()
        )

        embeddings = _make_embeddings(
            mock_engine, CALL_MODEL_DEFAULT_MODEL, EmbeddingMode.CALL_MODEL
        )
        with pytest.warns(UserWarning, match="does not support batched"):
            vector = await embeddings._PolarDBPGEmbeddings__acall_model_embed(
                ["first", "second"]
            )

        assert vector == [0.5, 0.6]
        import json as _json

        body = _json.loads(captured["body"])
        assert body["input"] == ["first"]
