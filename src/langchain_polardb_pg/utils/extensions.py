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

"""Generic helpers for managing PostgreSQL extensions.

These functions are component- and extension-agnostic: they operate on a
SQLAlchemy AsyncEngine (e.g. PolarDBPGEngine._pool) and a target extension
name, so any component (model manager, embeddings, vector store, ...) can
reuse the same logic to check, create, or require an extension such as
'polar_ai' or 'vector'.
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# PostgreSQL identifier rule used to validate extension names. CREATE EXTENSION
# does not accept bind parameters for the extension name, so the name is
# embedded into the statement; restricting it to a safe identifier pattern
# prevents SQL injection via an untrusted name.
_VALID_EXTENSION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _install_hint(extension_name: str) -> str:
    """Build actionable guidance for installing an extension manually."""
    return (
        f"The '{extension_name}' extension is not installed in the database. "
        f"Install it with a privileged account by running: "
        f"CREATE EXTENSION {extension_name} CASCADE;"
    )


def _validate_extension_name(extension_name: str) -> None:
    """Reject extension names that are not safe SQL identifiers."""
    if not _VALID_EXTENSION_NAME.match(extension_name):
        raise ValueError(
            f"Invalid extension name: {extension_name!r}. Extension names must "
            f"be valid PostgreSQL identifiers (letters, digits, underscore; "
            f"not starting with a digit)."
        )


async def extension_exists(engine: AsyncEngine, extension_name: str) -> bool:
    """Return whether the given extension is installed.

    Args:
        engine: SQLAlchemy AsyncEngine to query.
        extension_name: The extension to check (e.g. "polar_ai", "vector").

    Returns:
        True if the extension is present, False otherwise.
    """
    query = text("SELECT 1 FROM pg_extension WHERE extname = :name;")
    async with engine.connect() as conn:
        result = await conn.execute(query, {"name": extension_name})
        return result.first() is not None


async def create_extension(engine: AsyncEngine, extension_name: str) -> None:
    """Create the given extension if it does not already exist.

    Runs ``CREATE EXTENSION IF NOT EXISTS <name> CASCADE``. This requires a
    privileged database account; callers without sufficient privileges will
    receive the underlying database error wrapped with guidance.

    Args:
        engine: SQLAlchemy AsyncEngine to execute against.
        extension_name: The extension to create.

    Raises:
        ValueError: If extension_name is not a valid identifier.
        RuntimeError: If creation fails (e.g. insufficient privileges or the
            extension is unavailable on the server).
    """
    _validate_extension_name(extension_name)

    statement = text(
        f"CREATE EXTENSION IF NOT EXISTS {extension_name} CASCADE;"
    )
    try:
        async with engine.connect() as conn:
            await conn.execute(statement)
            await conn.commit()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create the '{extension_name}' extension: {exc}. "
            f"{_install_hint(extension_name)}"
        ) from exc


async def ensure_extension(
    engine: AsyncEngine,
    extension_name: str,
    auto_create: bool = False,
) -> None:
    """Ensure an extension is available, optionally creating it.

    This is the single entry point components should call before using an
    extension's SQL functions.

    Args:
        engine: SQLAlchemy AsyncEngine to operate on.
        extension_name: The extension to require (e.g. "polar_ai", "vector").
        auto_create: When True, attempt to create the extension if it is
            missing (requires a privileged account). When False (default), a
            missing extension raises with installation guidance.

    Raises:
        ValueError: If extension_name is not a valid identifier.
        RuntimeError: If the extension is missing and either auto_create is
            False, or auto_create is True but creation fails.
    """
    _validate_extension_name(extension_name)

    if await extension_exists(engine, extension_name):
        return

    if not auto_create:
        raise RuntimeError(_install_hint(extension_name))

    await create_extension(engine, extension_name)

    if not await extension_exists(engine, extension_name):
        raise RuntimeError(
            f"The '{extension_name}' extension is still not available after "
            f"attempting creation. {_install_hint(extension_name)}"
        )
