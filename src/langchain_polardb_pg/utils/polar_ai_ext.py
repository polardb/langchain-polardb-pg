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

"""polar_ai extension helpers.

Thin wrappers over the generic extension helpers in ``extensions.py``,
specialized for the PolarDB ``polar_ai`` extension. All AI features (model
management, in-database embeddings, etc.) depend on it being installed.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from .extensions import create_extension, ensure_extension, extension_exists

# The PolarDB AI extension name.
POLAR_AI_EXTENSION_NAME = "polar_ai"


async def polar_ai_extension_exists(engine: AsyncEngine) -> bool:
    """Return whether the polar_ai extension is installed."""
    return await extension_exists(engine, POLAR_AI_EXTENSION_NAME)


async def create_polar_ai_extension(engine: AsyncEngine) -> None:
    """Create the polar_ai extension if it does not already exist."""
    await create_extension(engine, POLAR_AI_EXTENSION_NAME)


async def ensure_polar_ai_extension(
    engine: AsyncEngine,
    auto_create: bool = False,
) -> None:
    """Ensure the polar_ai extension is available, optionally creating it."""
    await ensure_extension(
        engine, POLAR_AI_EXTENSION_NAME, auto_create=auto_create
    )
