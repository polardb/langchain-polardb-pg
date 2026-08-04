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

"""Shared utilities for the PolarDB for PostgreSQL LangChain integration."""

from .extensions import (
    create_extension,
    ensure_extension,
    extension_exists,
)
from .polar_ai_ext import (
    POLAR_AI_EXTENSION_NAME,
    create_polar_ai_extension,
    ensure_polar_ai_extension,
    polar_ai_extension_exists,
)

__all__ = [
    # Generic extension helpers (reusable for any extension, e.g. vector)
    "ensure_extension",
    "create_extension",
    "extension_exists",
    # polar_ai specializations
    "POLAR_AI_EXTENSION_NAME",
    "ensure_polar_ai_extension",
    "create_polar_ai_extension",
    "polar_ai_extension_exists",
]
