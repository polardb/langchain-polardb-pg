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

"""LangChain integration for PolarDB for PostgreSQL."""

from langchain_postgres import Column
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
    weighted_sum_ranking,
)

from .embeddings import EmbeddingMode, PolarDBPGEmbeddings
from .engine import PolarDBPGEngine
from .model_manager import PolarDBPGModel, PolarDBPGModelManager
from .utils.extensions import (
    create_extension,
    ensure_extension,
    extension_exists,
)
from .utils.polar_ai_ext import (
    POLAR_AI_EXTENSION_NAME,
    create_polar_ai_extension,
    ensure_polar_ai_extension,
    polar_ai_extension_exists,
)
from .vectorstores import PolarDBPGVector
from .version import __version__

__all__ = [
    "PolarDBPGEngine",
    "PolarDBPGModelManager",
    "PolarDBPGModel",
    "PolarDBPGEmbeddings",
    "EmbeddingMode",
    "PolarDBPGVector",
    "ensure_extension",
    "create_extension",
    "extension_exists",
    "ensure_polar_ai_extension",
    "create_polar_ai_extension",
    "polar_ai_extension_exists",
    "POLAR_AI_EXTENSION_NAME",
    "Column",
    "HybridSearchConfig",
    "reciprocal_rank_fusion",
    "weighted_sum_ranking",
    "__version__",
]
