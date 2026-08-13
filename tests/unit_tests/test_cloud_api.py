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

"""Unit tests for cloud_api utilities."""

import os
from unittest.mock import patch

import pytest

from langchain_polardb_pg.utils.cloud_api import (
    PolarDBEndpointInfo,
    _get_credentials,
)


class TestGetCredentials:
    """Tests for _get_credentials()."""

    def test_returns_explicit_credentials(self):
        key_id, key_secret = _get_credentials("my_id", "my_secret")
        assert key_id == "my_id"
        assert key_secret == "my_secret"

    def test_reads_from_env_vars(self):
        with patch.dict(
            os.environ,
            {
                "ALIBABA_CLOUD_ACCESS_KEY_ID": "env_id",
                "ALIBABA_CLOUD_ACCESS_KEY_SECRET": "env_secret",
            },
        ):
            key_id, key_secret = _get_credentials()
            assert key_id == "env_id"
            assert key_secret == "env_secret"

    def test_explicit_overrides_env(self):
        with patch.dict(
            os.environ,
            {
                "ALIBABA_CLOUD_ACCESS_KEY_ID": "env_id",
                "ALIBABA_CLOUD_ACCESS_KEY_SECRET": "env_secret",
            },
        ):
            key_id, key_secret = _get_credentials("explicit_id", "explicit_secret")
            assert key_id == "explicit_id"
            assert key_secret == "explicit_secret"

    def test_raises_without_credentials(self):
        env_backup_id = os.environ.pop("ALIBABA_CLOUD_ACCESS_KEY_ID", None)
        env_backup_secret = os.environ.pop("ALIBABA_CLOUD_ACCESS_KEY_SECRET", None)
        try:
            with pytest.raises(
                ValueError, match="Alibaba Cloud credentials are required"
            ):
                _get_credentials()
        finally:
            if env_backup_id:
                os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"] = env_backup_id
            if env_backup_secret:
                os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"] = env_backup_secret

    def test_raises_with_partial_credentials(self):
        env_backup_id = os.environ.pop("ALIBABA_CLOUD_ACCESS_KEY_ID", None)
        env_backup_secret = os.environ.pop("ALIBABA_CLOUD_ACCESS_KEY_SECRET", None)
        try:
            with pytest.raises(
                ValueError, match="Alibaba Cloud credentials are required"
            ):
                _get_credentials(access_key_id="only_id")
        finally:
            if env_backup_id:
                os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"] = env_backup_id
            if env_backup_secret:
                os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"] = env_backup_secret


class TestPolarDBEndpointInfo:
    """Tests for PolarDBEndpointInfo dataclass."""

    def test_create_endpoint_info(self):
        info = PolarDBEndpointInfo(
            host="pc-bp1234.rwlb.polardb-pg.rds.aliyuncs.com",
            port="5432",
            endpoint_type="Cluster",
            network_type="Private",
            endpoint_id="pe-bp1234",
        )
        assert info.host == "pc-bp1234.rwlb.polardb-pg.rds.aliyuncs.com"
        assert info.port == "5432"
        assert info.endpoint_type == "Cluster"
        assert info.network_type == "Private"
        assert info.endpoint_id == "pe-bp1234"
