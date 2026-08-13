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

"""Unit tests for PolarDBPGVector."""

from langchain_postgres.v2.vectorstores import PGVectorStore

from langchain_polardb_pg.vectorstores import PolarDBPGVector


class TestPolarDBPGVectorInheritance:
    """The vector store must inherit the full community capability set."""

    def test_subclasses_community_store(self):
        assert issubclass(PolarDBPGVector, PGVectorStore)

    def test_inherits_community_search_methods(self):
        for method in (
            "add_documents",
            "add_texts",
            "delete",
            "similarity_search_with_score",
            "max_marginal_relevance_search",
            "get_by_ids",
            "apply_vector_index",
        ):
            assert hasattr(PolarDBPGVector, method)

    def test_exposes_create_methods(self):
        assert hasattr(PolarDBPGVector, "create")
        assert hasattr(PolarDBPGVector, "create_sync")

    def test_no_from_instance(self):
        # Connection setup is the engine's responsibility, not the store's.
        assert not hasattr(PolarDBPGVector, "from_instance")
