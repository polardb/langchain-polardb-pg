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

"""PolarDB for PostgreSQL AI Model Manager for LangChain.

Manages AI models registered via the polar_ai extension.
Provides CRUD operations that map to PolarDB's AI model management SQL functions:
- AI_CreateModel, AI_AlterModel, AI_DropModel, AI_SetModelToken, AI_CallModel
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from .engine import PolarDBPGEngine
from .utils.polar_ai_ext import ensure_polar_ai_extension


@dataclass
class PolarDBPGModel:
    """Represents an AI model registered in PolarDB via polar_ai extension.

    Fields map 1:1 to the polar_ai._ai_models metadata table. The three
    *_fn columns are PostgreSQL regprocedure references; they are surfaced
    as read-only string fields for inspection and are not meant to be set
    by application code.
    """

    model_id: str
    model_url: str
    model_provider: str
    model_type: str
    model_name: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    model_headers_fn: str | None = None
    model_in_transform_fn: str | None = None
    model_out_transform_fn: str | None = None


class PolarDBPGModelManager:
    """Manage AI models registered via PolarDB's polar_ai extension.

    Wraps the polar_ai SQL functions (AI_CreateModel, AI_AlterModel,
    AI_DropModel, AI_SetModelToken) into a Python interface compatible
    with the LangChain ecosystem.
    """

    __create_key = object()

    # Optional named arguments accepted by polar_ai.ai_createmodel (those with
    # a DEFAULT). Used as a whitelist so callers cannot inject unknown kwargs.
    _CREATE_MODEL_OPTIONAL_ARGS = (
        "model_name",
        "model_config",
        "model_headers_fn",
        "model_in_transform_fn",
        "model_out_transform_fn",
    )

    # Optional named arguments accepted by polar_ai.ai_altermodel (those with
    # a DEFAULT). Only model_id is required; model_url is optional here (unlike
    # ai_createmodel where it is required).
    _ALTER_MODEL_OPTIONAL_ARGS = (
        "model_url",
        "model_provider",
        "model_type",
        "model_name",
        "model_config",
        "model_headers_fn",
        "model_in_transform_fn",
        "model_out_transform_fn",
    )

    # Arguments typed as regprocedure in the kernel. Their bound string value
    # (e.g. "polar_ai._fn(text,jsonb)") must be explicitly cast to
    # regprocedure, because a plain bind parameter is treated as text/unknown
    # and will not implicitly coerce to regprocedure.
    _REGPROCEDURE_ARGS = (
        "model_headers_fn",
        "model_in_transform_fn",
        "model_out_transform_fn",
    )

    def __init__(self, key: object, engine: PolarDBPGEngine) -> None:
        """PolarDBPGModelManager constructor.

        Args:
            key: Prevent direct constructor usage.
            engine: PolarDBPGEngine connection pool.

        Raises:
            Exception: If called directly instead of via create()/create_sync().
        """
        if key != PolarDBPGModelManager.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine

    @classmethod
    async def create(
        cls,
        engine: PolarDBPGEngine,
        auto_create_extension: bool = False,
    ) -> PolarDBPGModelManager:
        """Create a PolarDBPGModelManager instance (async).

        Ensures the polar_ai extension is available.

        Args:
            engine: PolarDBPGEngine connection pool.
            auto_create_extension: When True, attempt to create the polar_ai
                extension if missing (requires a privileged account).

        Returns:
            PolarDBPGModelManager instance.
        """
        manager = cls(cls.__create_key, engine)
        await engine._run_as_async(manager.__avalidate(auto_create_extension))
        return manager

    @classmethod
    def create_sync(
        cls,
        engine: PolarDBPGEngine,
        auto_create_extension: bool = False,
    ) -> PolarDBPGModelManager:
        """Create a PolarDBPGModelManager instance (sync).

        Ensures the polar_ai extension is available.

        Args:
            engine: PolarDBPGEngine connection pool.
            auto_create_extension: When True, attempt to create the polar_ai
                extension if missing (requires a privileged account).

        Returns:
            PolarDBPGModelManager instance.
        """
        manager = cls(cls.__create_key, engine)
        engine._run_as_sync(manager.__avalidate(auto_create_extension))
        return manager

    # ---- Public async API ----

    async def acreate_model(
        self,
        model_id: str,
        model_url: str,
        model_provider: str,
        model_type: str,
        **kwargs: Any,
    ) -> bool:
        """Create (register) an AI model via AI_CreateModel.

        The four positional parameters mirror the required (no-default)
        arguments of polar_ai.ai_createmodel. Optional arguments are passed
        via kwargs and restricted to the function's named optional arguments:
        model_name, model_config, model_headers_fn, model_in_transform_fn,
        model_out_transform_fn. model_config accepts a dict (serialized to
        JSON automatically).

        Args:
            model_id: Unique model identifier.
            model_url: The model service URL.
            model_provider: The provider of the model (e.g. "dashscope", "openai").
            model_type: The model type (e.g. "text_embedding").
            **kwargs: Optional arguments passed to AI_CreateModel.

        Returns:
            True if a new model was created, False if it already existed.

        Raises:
            ValueError: If an unknown optional argument name is provided.
        """
        return await self._engine._run_as_async(
            self.__acreate_model(
                model_id, model_url, model_provider, model_type, **kwargs
            )
        )

    def create_model(
        self,
        model_id: str,
        model_url: str,
        model_provider: str,
        model_type: str,
        **kwargs: Any,
    ) -> bool:
        """Create (register) an AI model via AI_CreateModel (sync).

        Returns:
            True if a new model was created, False if it already existed.
        """
        return self._engine._run_as_sync(
            self.__acreate_model(
                model_id, model_url, model_provider, model_type, **kwargs
            )
        )

    async def aalter_model(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> bool:
        """Alter an existing AI model via AI_AlterModel.

        Only model_id is required; the attributes to change are passed via
        kwargs and restricted to the function's named optional arguments:
        model_url, model_provider, model_type, model_name, model_config,
        model_headers_fn, model_in_transform_fn, model_out_transform_fn.
        model_config accepts a dict (serialized to JSON automatically).

        Args:
            model_id: The id of the model to alter.
            **kwargs: Attributes to modify.

        Returns:
            The boolean result returned by AI_AlterModel.

        Raises:
            ValueError: If an unknown optional argument name is provided.
        """
        return await self._engine._run_as_async(self.__aalter_model(model_id, **kwargs))

    def alter_model(self, model_id: str, **kwargs: Any) -> bool:
        """Alter an existing AI model via AI_AlterModel (sync).

        Returns:
            The boolean result returned by AI_AlterModel.
        """
        return self._engine._run_as_sync(self.__aalter_model(model_id, **kwargs))

    async def adrop_model(self, model_id: str) -> bool:
        """Drop (unregister) an AI model via AI_DropModel.

        Mirrors the kernel signature ai_dropmodel(text) returning boolean.

        Args:
            model_id: The id of the model to drop.

        Returns:
            The boolean result returned by AI_DropModel.
        """
        return await self._engine._run_as_async(self.__adrop_model(model_id))

    def drop_model(self, model_id: str) -> bool:
        """Drop (unregister) an AI model via AI_DropModel (sync).

        Returns:
            The boolean result returned by AI_DropModel.
        """
        return self._engine._run_as_sync(self.__adrop_model(model_id))

    async def aset_model_token(
        self,
        model_id: str,
        model_token: str,
    ) -> bool:
        """Set the token for an AI model via AI_SetModelToken.

        Args:
            model_id: The id of the model.
            model_token: The token/API key to set.

        Returns:
            The boolean result returned by AI_SetModelToken.
        """
        return await self._engine._run_as_async(
            self.__aset_model_token(model_id, model_token)
        )

    def set_model_token(self, model_id: str, model_token: str) -> bool:
        """Set the token for an AI model via AI_SetModelToken (sync).

        Returns:
            The boolean result returned by AI_SetModelToken.
        """
        return self._engine._run_as_sync(self.__aset_model_token(model_id, model_token))

    async def acall_model(
        self,
        model_id: str,
        payload: Any,
        payload_type: str | None = None,
    ) -> Any:
        """Invoke an AI model via AI_CallModel.

        Mirrors the kernel signature ai_callmodel(model_id text, anyelement)
        which returns jsonb. The second argument is polymorphic (anyelement),
        so the payload may be any value the model accepts (a string prompt,
        a number, an array, etc.). The jsonb result is parsed into a Python
        object (dict, list, str, ...).

        Because a plain bind parameter arrives as the SQL ``unknown`` type,
        PostgreSQL cannot resolve the ``anyelement`` polymorphic argument and
        raises "could not determine polymorphic type". The payload is
        therefore cast to a concrete SQL type (mirroring the kernel example
        ``ai_callmodel('...', '老师表扬我了'::text)``). When ``payload_type``
        is not given, it is inferred from the Python value (str -> text,
        bool -> boolean, int -> bigint, float -> double precision).

        Args:
            model_id: The id of the model to call.
            payload: The input passed as the anyelement argument.
            payload_type: Optional SQL type name to cast the payload to
                (e.g. "text", "jsonb"). Inferred from the value when omitted.

        Returns:
            The model response, parsed from jsonb.
        """
        return await self._engine._run_as_async(
            self.__acall_model(model_id, payload, payload_type)
        )

    def call_model(
        self,
        model_id: str,
        payload: Any,
        payload_type: str | None = None,
    ) -> Any:
        """Invoke an AI model via AI_CallModel (sync).

        Returns:
            The model response, parsed from jsonb.
        """
        return self._engine._run_as_sync(
            self.__acall_model(model_id, payload, payload_type)
        )

    async def alist_models(
        self,
        model_id: str | None = None,
        model_provider: str | None = None,
        model_type: str | None = None,
    ) -> list[PolarDBPGModel]:
        """List registered AI models, optionally filtered.

        This is the single entry point for inspecting model metadata. With no
        arguments it returns every model; any provided argument is applied as
        an equality filter (combined with AND). Each returned PolarDBPGModel
        carries the full metadata, including model_config (where the token is
        stored under the 'token' key), so callers can reuse the result for
        token checks and other validation.

        Args:
            model_id: Filter by exact model_id.
            model_provider: Filter by exact model_provider.
            model_type: Filter by exact model_type.

        Returns:
            List of PolarDBPGModel instances matching the filters.
        """
        return await self._engine._run_as_async(
            self.__alist_models(model_id, model_provider, model_type)
        )

    def list_models(
        self,
        model_id: str | None = None,
        model_provider: str | None = None,
        model_type: str | None = None,
    ) -> list[PolarDBPGModel]:
        """List registered AI models, optionally filtered (sync).

        See alist_models for argument semantics.
        """
        return self._engine._run_as_sync(
            self.__alist_models(model_id, model_provider, model_type)
        )

    async def aget_model(self, model_id: str) -> PolarDBPGModel | None:
        """Get details of a specific model by its model_id.

        Convenience wrapper over alist_models(model_id=...) returning a single
        model or None.

        Args:
            model_id: The unique model_id.

        Returns:
            PolarDBPGModel if found, None otherwise.
        """
        models = await self.alist_models(model_id=model_id)
        return models[0] if models else None

    def get_model(self, model_id: str) -> PolarDBPGModel | None:
        """Get details of a specific model by its model_id (sync)."""
        models = self.list_models(model_id=model_id)
        return models[0] if models else None

    # ---- Private async implementations ----

    async def __avalidate(self, auto_create_extension: bool = False) -> None:
        """Ensure the polar_ai extension is available.

        Delegates to the shared helper so all AI components share one
        consistent check/create path.
        """
        await ensure_polar_ai_extension(
            self._engine._pool, auto_create=auto_create_extension
        )

    async def __query_db(
        self, query: str, params: dict[str, Any] | None = None
    ) -> Sequence[RowMapping]:
        """Execute a query and return result rows.

        Args:
            query: SQL text. May contain named bind parameters (e.g. :name).
            params: Optional mapping of bind parameter values.
        """
        async with self._engine._pool.connect() as conn:
            result = await conn.execute(text(query), params or {})
            return result.mappings().fetchall()

    async def __acreate_model(
        self,
        model_id: str,
        model_url: str,
        model_provider: str,
        model_type: str,
        **kwargs: Any,
    ) -> bool:
        """Create a model via AI_CreateModel SQL function.

        Binds every argument as a SQL named parameter that maps to the
        function's own named arguments (e.g. model_name => :model_name), so
        optional arguments can be supplied in any combination without relying
        on positional order. Optional kwargs are restricted to a whitelist.

        The function returns TABLE(model_seq, model_id, model_schema,
        model_name, created); this returns the 'created' boolean indicating
        whether a new model was actually registered.

        Args:
            model_id, model_url, model_provider, model_type: Required args.
            **kwargs: Optional args, limited to _CREATE_MODEL_OPTIONAL_ARGS.

        Returns:
            True if a new model was created, False if it already existed.

        Raises:
            ValueError: If an unknown optional argument name is provided.
        """
        # Required args, bound by name to the function's named parameters.
        named_args = [
            "model_id => :model_id",
            "model_url => :model_url",
            "model_provider => :model_provider",
            "model_type => :model_type",
        ]
        params: dict[str, Any] = {
            "model_id": model_id,
            "model_url": model_url,
            "model_provider": model_provider,
            "model_type": model_type,
        }

        for key, value in kwargs.items():
            if key not in self._CREATE_MODEL_OPTIONAL_ARGS:
                raise ValueError(
                    f"Unknown argument '{key}' for create_model. Allowed "
                    f"optional arguments: "
                    f"{', '.join(self._CREATE_MODEL_OPTIONAL_ARGS)}."
                )
            named_args.append(self.__build_named_arg(key))
            params[key] = self.__normalize_arg_value(key, value)

        args_str = ", ".join(named_args)
        query = text(f"SELECT created FROM polar_ai.ai_createmodel({args_str});")

        async with self._engine._pool.connect() as conn:
            result = await conn.execute(query, params)
            row = result.mappings().fetchone()
            await conn.commit()

        return bool(row["created"]) if row is not None else False

    async def __aalter_model(self, model_id: str, **kwargs: Any) -> bool:
        """Alter a model via AI_AlterModel SQL function.

        Binds arguments as SQL named parameters mapping to the function's own
        named arguments (e.g. model_url => :model_url), so any subset of the
        optional arguments can be supplied. Optional kwargs are whitelisted.

        The function returns a boolean indicating success.

        Args:
            model_id: The id of the model to alter (required).
            **kwargs: Optional args, limited to _ALTER_MODEL_OPTIONAL_ARGS.

        Returns:
            The boolean result of AI_AlterModel.

        Raises:
            ValueError: If an unknown optional argument name is provided.
        """
        named_args = ["model_id => :model_id"]
        params: dict[str, Any] = {"model_id": model_id}

        for key, value in kwargs.items():
            if key not in self._ALTER_MODEL_OPTIONAL_ARGS:
                raise ValueError(
                    f"Unknown argument '{key}' for alter_model. Allowed "
                    f"optional arguments: "
                    f"{', '.join(self._ALTER_MODEL_OPTIONAL_ARGS)}."
                )
            named_args.append(self.__build_named_arg(key))
            params[key] = self.__normalize_arg_value(key, value)

        args_str = ", ".join(named_args)
        query = text(f"SELECT polar_ai.ai_altermodel({args_str}) AS result;")

        async with self._engine._pool.connect() as conn:
            result = await conn.execute(query, params)
            row = result.mappings().fetchone()
            await conn.commit()

        return bool(row["result"]) if row is not None else False

    async def __adrop_model(self, model_id: str) -> bool:
        """Drop a model via AI_DropModel SQL function.

        The kernel signature is ai_dropmodel(text) returning boolean.

        Returns:
            The boolean result of AI_DropModel.
        """
        query = text("SELECT polar_ai.ai_dropmodel(:model_id) AS result;")

        async with self._engine._pool.connect() as conn:
            result = await conn.execute(query, {"model_id": model_id})
            row = result.mappings().fetchone()
            await conn.commit()

        return bool(row["result"]) if row is not None else False

    async def __aset_model_token(self, model_id: str, model_token: str) -> bool:
        """Set model token via AI_SetModelToken SQL function.

        Argument names match the kernel signature (model_id, model_token).
        The function returns a boolean indicating success.

        Returns:
            The boolean result of AI_SetModelToken.
        """
        query = text(
            "SELECT polar_ai.ai_setmodeltoken("
            "model_id => :model_id, model_token => :model_token) AS result;"
        )

        async with self._engine._pool.connect() as conn:
            result = await conn.execute(
                query, {"model_id": model_id, "model_token": model_token}
            )
            row = result.mappings().fetchone()
            await conn.commit()

        return bool(row["result"]) if row is not None else False

    async def __aget_model_token(self, model_id: str) -> str | None:
        """Read a model's token, reusing the unified list query.

        The token is not a dedicated column; it is stored inside the jsonb
        model_config under the 'token' key (e.g.
        {"author_type": "token", "token": "sk-..."}). Returns the token
        string ("" when the key is absent), or None if the model does not
        exist.
        """
        models = await self.__alist_models(model_id=model_id)
        if not models:
            return None
        return models[0].model_config.get("token", "")

    async def __acall_model(
        self,
        model_id: str,
        payload: Any,
        payload_type: str | None = None,
    ) -> Any:
        """Call a model via AI_CallModel SQL function.

        Before calling, the model's token (stored in model_config.token) is
        checked: if the model does not exist or its token is empty, a
        ValueError is raised, since the downstream AI service will reject an
        unauthorized request anyway.

        The kernel signature is ai_callmodel(model_id text, anyelement)
        returning jsonb. model_id is bound by name; the polymorphic
        anyelement payload is bound and cast to a concrete SQL type so
        PostgreSQL can resolve anyelement (a bare bind parameter is treated as
        the unknown type and fails polymorphic resolution).

        Returns:
            The model response parsed from the jsonb result, or None if the
            function returned NULL.

        Raises:
            ValueError: If the model is not found or its token is empty.
        """
        token = await self.__aget_model_token(model_id)
        if token is None:
            raise ValueError(
                f"Model '{model_id}' was not found in polar_ai._ai_models. "
                f"Create it first via create_model()."
            )
        if not token:
            raise ValueError(
                f"Model '{model_id}' has an empty token. Set it first via "
                f"set_model_token() before calling the model."
            )

        sql_type = payload_type or self.__infer_sql_type(payload)
        query = text(
            "SELECT polar_ai.ai_callmodel("
            f":model_id, CAST(:payload AS {sql_type})) AS result;"
        )

        async with self._engine._pool.connect() as conn:
            result = await conn.execute(
                query, {"model_id": model_id, "payload": payload}
            )
            row = result.mappings().fetchone()
            await conn.commit()

        if row is None:
            return None
        return self.__parse_jsonb(row["result"])

    # Columns of polar_ai._ai_models selected to populate a PolarDBPGModel.
    # model_seq is used only for ordering, not surfaced on the dataclass.
    _MODEL_COLUMNS = (
        "model_id",
        "model_url",
        "model_provider",
        "model_type",
        "model_name",
        "model_config",
        "model_headers_fn",
        "model_in_transform_fn",
        "model_out_transform_fn",
    )

    async def __alist_models(
        self,
        model_id: str | None = None,
        model_provider: str | None = None,
        model_type: str | None = None,
    ) -> list[PolarDBPGModel]:
        """List models from polar_ai._ai_models, optionally filtered.

        Builds a WHERE clause from whichever of model_id/model_provider/
        model_type are provided (combined with AND), all bound as named
        parameters. With no filters it returns every model.
        """
        columns = ", ".join(self._MODEL_COLUMNS)
        filters = {
            "model_id": model_id,
            "model_provider": model_provider,
            "model_type": model_type,
        }
        conditions = []
        params: dict[str, Any] = {}
        for column, value in filters.items():
            if value is not None:
                conditions.append(f"{column} = :{column}")
                params[column] = value

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            f"SELECT {columns} FROM polar_ai._ai_models"
            f"{where_clause} ORDER BY model_seq;"
        )
        results = await self.__query_db(query, params)
        return self.__convert_rows_to_models(results)

    def __convert_rows_to_models(
        self, rows: Sequence[RowMapping]
    ) -> list[PolarDBPGModel]:
        """Convert database rows to PolarDBPGModel instances.

        Maps each polar_ai._ai_models column 1:1 to a PolarDBPGModel field.
        The jsonb model_config is parsed into a dict; the *_fn regprocedure
        columns are surfaced as their string representation.
        """
        models = []
        for row in rows:
            row_dict = dict(row)
            model = PolarDBPGModel(
                model_id=row_dict.get("model_id", ""),
                model_url=row_dict.get("model_url", ""),
                model_name=row_dict.get("model_name", ""),
                model_config=self.__parse_model_config(row_dict.get("model_config")),
                model_provider=row_dict.get("model_provider"),
                model_type=row_dict.get("model_type"),
                model_headers_fn=self.__stringify_regprocedure(
                    row_dict.get("model_headers_fn")
                ),
                model_in_transform_fn=self.__stringify_regprocedure(
                    row_dict.get("model_in_transform_fn")
                ),
                model_out_transform_fn=self.__stringify_regprocedure(
                    row_dict.get("model_out_transform_fn")
                ),
            )
            models.append(model)
        return models

    @staticmethod
    def __parse_model_config(value: Any) -> dict[str, Any]:
        """Parse the jsonb model_config column into a dict.

        The driver may return a dict already or a JSON-encoded string,
        depending on type handling; normalize both to a dict.
        """
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def __build_named_arg(self, key: str) -> str:
        """Build the "name => :bind" fragment for a function argument.

        For regprocedure-typed arguments the bound value is explicitly cast
        with CAST(:bind AS regprocedure); a plain bind parameter is treated as
        text/unknown and will not implicitly coerce to regprocedure, so the
        function call would fail to match the regprocedure parameter.
        """
        if key in self._REGPROCEDURE_ARGS:
            return f"{key} => CAST(:{key} AS regprocedure)"
        return f"{key} => :{key}"

    @staticmethod
    def __normalize_arg_value(key: str, value: Any) -> Any:
        """Normalize a Python argument value for SQL binding.

        model_config is json in the kernel; a dict is serialized to a JSON
        string. All other values are passed through unchanged (regprocedure
        arguments are passed as their textual signature, e.g.
        "polar_ai._fn(text,jsonb)", and cast in the SQL fragment).
        """
        if key == "model_config" and isinstance(value, dict):
            return json.dumps(value)
        return value

    @staticmethod
    def __infer_sql_type(value: Any) -> str:
        """Infer the SQL type to cast an anyelement payload to.

        Maps common Python types to PostgreSQL types. bool is checked before
        int because bool is a subclass of int in Python. Anything unmapped
        falls back to text, matching the kernel's string-prompt example.
        """
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "bigint"
        if isinstance(value, float):
            return "double precision"
        return "text"

    @staticmethod
    def __parse_jsonb(value: Any) -> Any:
        """Parse a jsonb result value into a Python object.

        The driver may return an already-decoded object (dict/list/...) or a
        JSON-encoded string depending on type handling; normalize both. If the
        value is a string that is not valid JSON, it is returned as-is.
        """
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return value
        return value

    @staticmethod
    def __stringify_regprocedure(value: Any) -> str | None:
        """Render a regprocedure column value as a string, if present."""
        return None if value is None else str(value)
