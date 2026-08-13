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

"""PolarDB for PostgreSQL in-database Embeddings for LangChain.

Computes text embeddings directly inside the database using the polar_ai
extension, eliminating network round-trips from the application.

Two execution modes are supported, selected automatically from the cluster
attributes:

- ``ai_text_embedding`` mode: calls ``polar_ai.ai_text_embedding(content,
  model_id)`` and passes the user text straight through. Available on
  Beijing-region Standard-edition clusters. This is also the universal
  fallback used when cluster attributes are not cached (e.g. embedded
  scenarios) and the mode used by ``embed_query_inline``.

- ``call_model`` mode: calls ``polar_ai.ai_callmodel`` against a registered
  model. For the default model ``_polar4ai/_polar4ai_text_embedding`` the
  request body and response are wrapped/parsed following the
  ``polar_ai._get_batch_embeddings`` contract. Available on Enterprise-edition
  clusters that have an AI node.
"""

from __future__ import annotations

import json
import warnings
from enum import Enum
from typing import Any

from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from .engine import PolarDBPGEngine
from .model_manager import PolarDBPGModelManager
from .utils.polar_ai_ext import ensure_polar_ai_extension


class EmbeddingMode(str, Enum):
    """Execution mode for computing embeddings inside PolarDB."""

    AI_TEXT_EMBEDDING = "ai_text_embedding"
    CALL_MODEL = "call_model"


# Default model for the ai_text_embedding mode (Beijing Standard edition and
# the universal fallback). Kept as DEFAULT_EMBEDDING_MODEL for compatibility.
DEFAULT_EMBEDDING_MODEL = "_dashscope/text_embedding/text_embedding_v2"
AI_TEXT_EMBEDDING_DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL

# Models supported by the ai_text_embedding mode.
AI_TEXT_EMBEDDING_MODELS = (
    "_dashscope/text_embedding/text_embedding_v2",
    "_dashscope/text_embedding/text_embedding_v3",
)

# Default registered model for the call_model mode (Enterprise + AI node).
CALL_MODEL_DEFAULT_MODEL = "_polar4ai/_polar4ai_text_embedding"

# Defaults for the call_model request body, mirroring
# polar_ai._get_batch_embeddings.
CALL_MODEL_DEFAULT_BODY_MODEL = "Qwen-8B-embedding"
CALL_MODEL_DEFAULT_DIMENSION = 1024
CALL_MODEL_DEFAULT_DEPLOYMENT_NAME = "RAG"


class PolarDBPGEmbeddings(Embeddings):
    """PolarDB in-database embeddings via polar_ai AI_Text_Embedding.

    Computes text embeddings inside the database using registered AI models,
    avoiding the need for external embedding API calls from the application.
    """

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: PolarDBPGEngine,
        model_name: str,
        mode: EmbeddingMode,
        call_model_body_model: str = CALL_MODEL_DEFAULT_BODY_MODEL,
        call_model_dimension: int = CALL_MODEL_DEFAULT_DIMENSION,
        call_model_deployment_name: str = CALL_MODEL_DEFAULT_DEPLOYMENT_NAME,
    ) -> None:
        """PolarDBPGEmbeddings constructor.

        Args:
            key: Prevent direct constructor usage.
            engine: PolarDBPGEngine connection pool.
            model_name: The registered model name for embedding generation.
            mode: Execution mode (ai_text_embedding or call_model).
            call_model_body_model: For call_model mode, the underlying embedding
                model name placed in the request body's "model" field.
            call_model_dimension: For call_model mode, the embedding dimension.
            call_model_deployment_name: For call_model mode, the deployment name.

        Raises:
            Exception: If called directly instead of via create()/create_sync().
        """
        if key != PolarDBPGEmbeddings.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self.model_name = model_name
        self.mode = mode
        self.call_model_body_model = call_model_body_model
        self.call_model_dimension = call_model_dimension
        self.call_model_deployment_name = call_model_deployment_name

    def __getattribute__(self, name: str) -> Any:
        """Hide ``embed_query_inline`` unless inline embedding is supported.

        The community vector store detects inline embedding capability via
        ``getattr(embedding_service, "embed_query_inline", None)`` and uses it
        whenever the attribute is callable. Inline embedding only works in the
        ai_text_embedding mode, so outside that mode we raise AttributeError
        here. This makes the community ``getattr`` fall back to its default
        (the regular embed-then-insert path) and also gives a clear error if a
        caller accesses the method directly.
        """
        if name == "embed_query_inline":
            mode = object.__getattribute__(self, "mode")
            if mode != EmbeddingMode.AI_TEXT_EMBEDDING:
                raise AttributeError(
                    "embed_query_inline is only available in the "
                    "ai_text_embedding mode (Beijing Standard edition or "
                    f"unknown region), not in the '{mode.value}' mode."
                )
        return object.__getattribute__(self, name)

    @classmethod
    async def create(
        cls,
        engine: PolarDBPGEngine,
        model_name: str | None = None,
        mode: EmbeddingMode | None = None,
        auto_create_extension: bool = False,
        call_model_body_model: str = CALL_MODEL_DEFAULT_BODY_MODEL,
        call_model_dimension: int = CALL_MODEL_DEFAULT_DIMENSION,
        call_model_deployment_name: str = CALL_MODEL_DEFAULT_DEPLOYMENT_NAME,
    ) -> PolarDBPGEmbeddings:
        """Create a PolarDBPGEmbeddings instance (async).

        Ensures the polar_ai extension is available, resolves the execution
        mode and default model from the cluster attributes when not given,
        then validates that the specified model exists.

        Args:
            engine: PolarDBPGEngine connection pool.
            model_name: The registered model name. When None, the default model
                for the resolved mode is used.
            mode: Execution mode. When None, it is resolved from the cluster
                attributes (Beijing Standard edition -> ai_text_embedding;
                Enterprise edition with an AI node -> call_model; otherwise the
                ai_text_embedding fallback).
            auto_create_extension: When True, attempt to create the polar_ai
                extension if missing (requires a privileged account).
            call_model_body_model: For call_model mode, the underlying embedding
                model name placed in the request body's "model" field.
            call_model_dimension: For call_model mode, the embedding dimension.
            call_model_deployment_name: For call_model mode, the deployment name.

        Returns:
            PolarDBPGEmbeddings instance.

        Raises:
            ValueError: If the resolved model does not exist or the model is
                incompatible with the resolved mode.
        """
        resolved_mode = mode or cls._resolve_mode(engine)
        resolved_model = model_name or cls._default_model_for_mode(resolved_mode)
        cls._validate_model_for_mode(resolved_model, resolved_mode)

        embeddings = cls(
            cls.__create_key,
            engine,
            resolved_model,
            resolved_mode,
            call_model_body_model,
            call_model_dimension,
            call_model_deployment_name,
        )
        await engine._run_as_async(
            ensure_polar_ai_extension(engine._pool, auto_create=auto_create_extension)
        )
        model_exists = await engine._run_as_async(embeddings.__amodel_exists())
        if not model_exists:
            raise ValueError(
                f"Model '{resolved_model}' does not exist. "
                f"Use PolarDBPGModelManager to register it, or use a default "
                f"built-in model."
            )
        return embeddings

    @classmethod
    def create_sync(
        cls,
        engine: PolarDBPGEngine,
        model_name: str | None = None,
        mode: EmbeddingMode | None = None,
        auto_create_extension: bool = False,
        call_model_body_model: str = CALL_MODEL_DEFAULT_BODY_MODEL,
        call_model_dimension: int = CALL_MODEL_DEFAULT_DIMENSION,
        call_model_deployment_name: str = CALL_MODEL_DEFAULT_DEPLOYMENT_NAME,
    ) -> PolarDBPGEmbeddings:
        """Create a PolarDBPGEmbeddings instance (sync).

        See :meth:`create` for argument semantics.

        Returns:
            PolarDBPGEmbeddings instance.

        Raises:
            ValueError: If the resolved model does not exist or the model is
                incompatible with the resolved mode.
        """
        resolved_mode = mode or cls._resolve_mode(engine)
        resolved_model = model_name or cls._default_model_for_mode(resolved_mode)
        cls._validate_model_for_mode(resolved_model, resolved_mode)

        embeddings = cls(
            cls.__create_key,
            engine,
            resolved_model,
            resolved_mode,
            call_model_body_model,
            call_model_dimension,
            call_model_deployment_name,
        )
        engine._run_as_sync(
            ensure_polar_ai_extension(engine._pool, auto_create=auto_create_extension)
        )
        model_exists = engine._run_as_sync(embeddings.__amodel_exists())
        if not model_exists:
            raise ValueError(
                f"Model '{resolved_model}' does not exist. "
                f"Use PolarDBPGModelManager to register it, or use a default "
                f"built-in model."
            )
        return embeddings

    # ---- Mode resolution ----

    @staticmethod
    def _resolve_mode(engine: PolarDBPGEngine) -> EmbeddingMode:
        """Resolve the execution mode from the cluster attributes.

        - Beijing Standard edition -> ai_text_embedding.
        - Enterprise edition with an AI node -> call_model.
        - No cached attributes (embedded scenarios) or anything else ->
          ai_text_embedding fallback.
        """
        attribute = engine.cluster_attribute
        if attribute is None:
            return EmbeddingMode.AI_TEXT_EMBEDDING
        if attribute.is_beijing_region and attribute.is_standard_edition:
            return EmbeddingMode.AI_TEXT_EMBEDDING
        if attribute.is_enterprise_edition and attribute.has_ai_node:
            return EmbeddingMode.CALL_MODEL
        return EmbeddingMode.AI_TEXT_EMBEDDING

    @staticmethod
    def _default_model_for_mode(mode: EmbeddingMode) -> str:
        """Return the default registered model name for a mode."""
        if mode == EmbeddingMode.CALL_MODEL:
            return CALL_MODEL_DEFAULT_MODEL
        return AI_TEXT_EMBEDDING_DEFAULT_MODEL

    @staticmethod
    def _validate_model_for_mode(model_name: str, mode: EmbeddingMode) -> None:
        """Validate that a model name is usable in the given mode.

        Only enforces the closed set of ai_text_embedding models; call_model
        accepts any registered model since users may bring their own.
        """
        if (
            mode == EmbeddingMode.AI_TEXT_EMBEDDING
            and model_name not in AI_TEXT_EMBEDDING_MODELS
        ):
            supported = ", ".join(AI_TEXT_EMBEDDING_MODELS)
            raise ValueError(
                f"Model '{model_name}' is not supported by the "
                f"ai_text_embedding mode. Supported models: {supported}."
            )

    # ---- LangChain Embeddings interface ----

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text (sync).

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return self._engine._run_as_sync(self.__aembed_query(text))

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query text (async).

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return await self._engine._run_as_async(self.__aembed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts (sync).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self._engine._run_as_sync(self.__aembed_documents(texts))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts (async).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await self._engine._run_as_async(self.__aembed_documents(texts))

    def embed_query_inline(self, query: str) -> str:
        """Return a SQL expression for inline embedding computation.

        This is used by VectorStore to compute embeddings directly in SQL
        queries, avoiding a separate embedding API call.

        Unlike the execute paths, this method returns a SQL *fragment* that
        the caller splices into a larger statement, so it has no execution
        point and cannot use bind parameters. To prevent injection it wraps
        both literals in PostgreSQL dollar-quoting with a collision-free tag,
        which removes any dependency on single-quote escaping.

        Inline embedding is only available in the ai_text_embedding mode
        (Beijing Standard edition, or when the cluster region is unknown).
        The call_model mode wraps its request body and parses the response in
        Python, so it has no single SQL expression to splice and inline
        embedding is unsupported there.

        Args:
            query: Text to embed.

        Returns:
            SQL expression string that computes the embedding inline.

        Raises:
            ValueError: If the embeddings instance is not in the
                ai_text_embedding mode.
        """
        if self.mode != EmbeddingMode.AI_TEXT_EMBEDDING:
            raise ValueError(
                "embed_query_inline is only supported in the "
                "ai_text_embedding mode (Beijing Standard edition or unknown "
                f"region), not in the '{self.mode.value}' mode."
            )
        quoted_query = self.__dollar_quote(query)
        quoted_model = self.__dollar_quote(self.model_name)
        return (
            f"polar_ai.ai_text_embedding({quoted_query}::text, "
            f"{quoted_model}::text)::vector"
        )

    @staticmethod
    def __dollar_quote(value: str) -> str:
        """Wrap a string as a PostgreSQL dollar-quoted literal.

        Picks a tag that does not occur within the value, so the literal
        cannot be terminated early by attacker-controlled content.
        """
        tag = "polarai"
        while f"${tag}$" in value:
            tag += "x"
        return f"${tag}${value}${tag}$"

    # ---- Model validation ----

    async def amodel_exists(self) -> bool:
        """Check if the embedding model exists (async)."""
        return await self._engine._run_as_async(self.__amodel_exists())

    def model_exists(self) -> bool:
        """Check if the embedding model exists (sync)."""
        return self._engine._run_as_sync(self.__amodel_exists())

    # ---- Private async implementations ----

    async def __amodel_exists(self) -> bool:
        """Check if the embedding model exists and is usable in polar_ai.

        Reuses the unified model query (PolarDBPGModelManager.aget_model,
        backed by list_models) to look the model up by its model_id. When the
        model is registered, its token (stored in model_config['token']) must
        not be empty, otherwise embedding calls would fail at runtime. For the
        ai_text_embedding mode, fall back to probing the AI_Text_Embedding
        function directly, since some built-in models may be callable without
        an explicit row in the metadata table.

        Raises:
            ValueError: If the model is registered but its token is empty.
        """
        manager = await PolarDBPGModelManager.create(self._engine)
        model = await manager.aget_model(self.model_name)
        if model is not None:
            token = (model.model_config or {}).get("token", "")
            if not token:
                raise ValueError(
                    f"Model '{self.model_name}' is registered but its token "
                    "is empty. Set it with "
                    "PolarDBPGModelManager.set_model_token() before use."
                )
            return True
        if self.mode == EmbeddingMode.AI_TEXT_EMBEDDING:
            return await self.__atry_builtin_model()
        return False

    async def __atry_builtin_model(self) -> bool:
        """Try to invoke the model to check if it's a valid built-in model."""
        test_query = text(
            "SELECT polar_ai.ai_text_embedding(CAST('test' AS text), "
            "CAST(:model_name AS text));"
        )
        try:
            async with self._engine._pool.connect() as conn:
                await conn.execute(test_query, {"model_name": self.model_name})
            return True
        except Exception:
            return False

    async def __aembed_query(self, query: str) -> list[float]:
        """Compute embedding for a single query text, dispatched by mode."""
        if self.mode == EmbeddingMode.CALL_MODEL:
            return await self.__acall_model_embed([query])
        return await self.__atext_embedding_query(query)

    async def __aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for multiple texts, dispatched by mode.

        Neither mode supports server-side batching for these models, so the
        texts are embedded one at a time.
        """
        if not texts:
            return []
        results: list[list[float]] = []
        for single_text in texts:
            if self.mode == EmbeddingMode.CALL_MODEL:
                results.append(await self.__acall_model_embed([single_text]))
            else:
                results.append(await self.__atext_embedding_query(single_text))
        return results

    # ---- ai_text_embedding mode ----

    async def __atext_embedding_query(self, query: str) -> list[float]:
        """Compute one embedding via polar_ai.ai_text_embedding.

        Note: polar_ai.ai_text_embedding signature is:
            ai_text_embedding(content text, model_id text DEFAULT '...')
        Content comes first, model_id second.
        """
        sql = text(
            "SELECT polar_ai.ai_text_embedding(CAST(:query AS text), "
            "CAST(:model_name AS text));"
        )

        async with self._engine._pool.connect() as conn:
            result = await conn.execute(
                sql, {"query": query, "model_name": self.model_name}
            )
            row = result.mappings().fetchone()

        if row is None:
            raise RuntimeError(
                f"AI_Text_Embedding returned no result for model '{self.model_name}'."
            )

        first_value = next(iter(row.values()))
        return self.__coerce_vector(first_value)

    # ---- call_model mode ----

    async def __acall_model_embed(self, texts: list[str]) -> list[float]:
        """Compute one embedding via polar_ai.ai_callmodel.

        The polar4ai embedding model currently does not support batched
        embedding: ai_callmodel returns a single real[] vector and ignores
        any input beyond the first element. To avoid silently dropping data,
        this method checks the input length: when more than one text is given,
        it warns and embeds only the first one. The request body follows the
        polar_ai._get_batch_embeddings contract:
            {"model": <body_model>, "input": [<text>],
             "dimensions": <int>, "deploymentName": <name>}

        Args:
            texts: Texts to embed. Only the first one is used.

        Returns:
            The embedding vector for the first text.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("call_model embedding requires at least one text.")
        if len(texts) > 1:
            warnings.warn(
                "ai_callmodel for "
                f"'{self.model_name}' does not support batched embedding; "
                f"{len(texts)} texts were given but only the first will be "
                "embedded.",
                stacklevel=2,
            )
        query = texts[0]
        body = {
            "model": self.call_model_body_model,
            "input": [query],
            "dimensions": self.call_model_dimension,
            "deploymentName": self.call_model_deployment_name,
        }
        sql = text(
            "SELECT polar_ai.ai_callmodel(:model_name, "
            "CAST(:body AS jsonb)) AS response;"
        )

        async with self._engine._pool.connect() as conn:
            result = await conn.execute(
                sql,
                {"model_name": self.model_name, "body": json.dumps(body)},
            )
            row = result.mappings().fetchone()

        if row is None:
            raise RuntimeError(
                f"ai_callmodel returned no result for model '{self.model_name}'."
            )

        return self.__parse_call_model_response(row["response"])

    def __parse_call_model_response(self, response: object) -> list[float]:
        """Parse the ai_callmodel response into a single embedding vector.

        ai_callmodel for the polar4ai embedding model returns either:
        - A nested list [[float, ...]] (batch response with one element), or
        - A flat list [float, ...] (bare real[] vector).
        On error it returns the {"success", "errMessage"} envelope.
        """
        payload = self.__as_json(response)

        if isinstance(payload, dict):
            success = payload.get("success")
            if success is not None and str(success).lower() != "true":
                message = payload.get("errMessage") or payload.get("message")
                raise RuntimeError(f"Embedding error: {message}")
            raise RuntimeError(f"Unexpected ai_callmodel response object: {payload!r}.")

        # Handle nested list [[float, ...]] -> unwrap to first element
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            payload = payload[0]

        return self.__coerce_vector(payload)

    @staticmethod
    def __as_json(value: object) -> object:
        """Normalize an asyncpg jsonb result into a Python object."""
        if isinstance(value, (str, bytes, bytearray)):
            return json.loads(value)
        return value

    @staticmethod
    def __coerce_vector(value: object) -> list[float]:
        """Coerce a jsonb/string/list value into a list of floats."""
        if isinstance(value, (str, bytes, bytearray)):
            value = json.loads(value)
        if isinstance(value, list):
            return [float(item) for item in value]
        return [float(item) for item in list(value)]
