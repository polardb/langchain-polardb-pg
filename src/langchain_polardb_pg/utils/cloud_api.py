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

"""Alibaba Cloud OpenAPI utilities for PolarDB endpoint discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PolarDBEndpointInfo:
    """Resolved PolarDB connection endpoint information.

    Captures the connection string plus additional address/endpoint metadata
    returned by DescribeDBClusterEndpoints.
    """

    host: str
    port: str
    endpoint_type: str
    network_type: str
    endpoint_id: str
    ip_address: str = ""
    vpc_id: str = ""
    vswitch_id: str = ""
    nodes: str = ""
    read_write_mode: str = ""
    service_name: str = ""


@dataclass
class PolarDBClusterAttribute:
    """PolarDB cluster attributes from DescribeDBClusterAttribute.

    Captures the subset of cluster metadata needed for region resolution
    and version-aware feature decisions (e.g. embedding endpoint selection),
    plus cluster-level AI info (fetched via DescribeType='AI').
    """

    region_id: str
    category: str
    dbversion: str
    dbtype: str
    engine: str
    ai_type: str = ""
    ai_creating_time: str = ""
    ai_free_mode: str = ""
    api_keys: list[str] = field(default_factory=list)

    @property
    def has_ai_node(self) -> bool:
        """Whether the cluster has an AI node provisioned (AiType is set)."""
        return bool(self.ai_type)

    @property
    def is_beijing_region(self) -> bool:
        """Whether the cluster is in the Beijing region (cn-beijing)."""
        return self.region_id == "cn-beijing"

    @property
    def is_standard_edition(self) -> bool:
        """Whether the cluster is the Standard edition.

        PolarDB uses 'SENormal' for the Standard edition and 'Normal' for
        the Enterprise edition.
        """
        return self.category == "SENormal"

    @property
    def is_enterprise_edition(self) -> bool:
        """Whether the cluster is the Enterprise edition (Category == 'Normal')."""
        return self.category == "Normal"


# Environment variable holding a PolarDB AI service API key. This is the
# AI model authorization key (used by polar_ai model calls), kept separate
# from the Alibaba Cloud AK/SK used for OpenAPI management calls.
AI_API_KEY_ENV_VAR = "POLARDB_AI_API_KEY"


def get_ai_api_key_from_env(env_var: str = AI_API_KEY_ENV_VAR) -> list[str]:
    """Read the AI service API key from an environment variable.

    Args:
        env_var: Name of the environment variable to read. Defaults to
            POLARDB_AI_API_KEY.

    Returns:
        A single-element list with the key value, or an empty list if the
        variable is unset or blank. A list is returned for symmetry with the
        cluster source (cluster_attribute.api_keys), so callers handle both
        sources uniformly.
    """
    value = os.environ.get(env_var)
    return [value] if value else []


def _get_credentials(
    access_key_id: Optional[str] = None,
    access_key_secret: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve Alibaba Cloud credentials from arguments or environment variables.

    Priority: explicit arguments > environment variables.

    Args:
        access_key_id: Alibaba Cloud Access Key ID.
        access_key_secret: Alibaba Cloud Access Key Secret.

    Returns:
        Tuple of (access_key_id, access_key_secret).

    Raises:
        ValueError: If credentials cannot be resolved.
    """
    resolved_key_id = access_key_id or os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
    resolved_key_secret = access_key_secret or os.environ.get(
        "ALIBABA_CLOUD_ACCESS_KEY_SECRET"
    )

    if not resolved_key_id or not resolved_key_secret:
        raise ValueError(
            "Alibaba Cloud credentials are required. Provide 'access_key_id' and "
            "'access_key_secret' arguments, or set the environment variables "
            "ALIBABA_CLOUD_ACCESS_KEY_ID and ALIBABA_CLOUD_ACCESS_KEY_SECRET."
        )
    return resolved_key_id, resolved_key_secret


def describe_polardb_cluster_attribute(
    cluster_id: str,
    access_key_id: Optional[str] = None,
    access_key_secret: Optional[str] = None,
    region: Optional[str] = None,
) -> PolarDBClusterAttribute:
    """Fetch PolarDB cluster attributes via DescribeDBClusterAttribute.

    Retrieves cluster metadata including region, edition (category), and
    database version. Uses the centralized OpenAPI endpoint
    (polardb.aliyuncs.com), which routes by cluster_id, so the region does
    not need to be known in advance.

    Args:
        cluster_id: PolarDB cluster ID (e.g. "pc-bp1234567890").
        access_key_id: Alibaba Cloud Access Key ID. Falls back to env var.
        access_key_secret: Alibaba Cloud Access Key Secret. Falls back to env var.
        region: Optional Alibaba Cloud region hint. Not required: the
            centralized endpoint resolves the cluster by ID regardless of
            region. The authoritative region is returned in region_id.

    Returns:
        PolarDBClusterAttribute with region_id, category, dbversion, etc.

    Raises:
        ImportError: If alibabacloud SDK is not installed.
    """
    try:
        from alibabacloud_polardb20170801.client import Client as PolarDBClient
        from alibabacloud_tea_openapi import models as open_api_models
        from alibabacloud_tea_openapi.models import Config as OpenAPIConfig
        from alibabacloud_tea_util import models as util_models
    except ImportError as exc:
        raise ImportError(
            "The 'alibabacloud-polardb20170801' and 'alibabacloud-tea-openapi' "
            "packages are required for PolarDBPGEngine.from_instance(). "
            "Install them with: pip install 'langchain-polardb-pg[cloud]'"
        ) from exc

    resolved_key_id, resolved_key_secret = _get_credentials(
        access_key_id, access_key_secret
    )

    config = OpenAPIConfig(
        access_key_id=resolved_key_id,
        access_key_secret=resolved_key_secret,
        region_id=region,
        endpoint="polardb.aliyuncs.com",
    )
    client = PolarDBClient(config)

    # Use the low-level call_api to read the raw response dict. The typed
    # SDK model (DescribeDBClusterAttributeResponseBody) does not define the
    # AI fields ApiKeys/AiFreeMode, so calling to_map() on it would silently
    # drop them. DescribeType='AI' returns the cluster-level AI info
    # (AiType, ApiKeys, AiCreatingTime, AiFreeMode), omitted by default.
    params = open_api_models.Params(
        action="DescribeDBClusterAttribute",
        version="2017-08-01",
        protocol="HTTPS",
        method="POST",
        auth_type="AK",
        style="RPC",
        pathname="/",
        req_body_type="formData",
        body_type="json",
    )
    api_request = open_api_models.OpenApiRequest(
        query={"DBClusterId": cluster_id, "DescribeType": "AI"}
    )
    response = client.call_api(params, api_request, util_models.RuntimeOptions())
    attr_map = response["body"]

    return PolarDBClusterAttribute(
        region_id=attr_map.get("RegionId", ""),
        category=attr_map.get("Category", ""),
        dbversion=str(attr_map.get("DBVersion", "")),
        dbtype=attr_map.get("DBType", ""),
        engine=attr_map.get("Engine", ""),
        ai_type=attr_map.get("AiType") or "",
        ai_creating_time=attr_map.get("AiCreatingTime") or "",
        ai_free_mode=attr_map.get("AiFreeMode") or "",
        api_keys=attr_map.get("ApiKeys") or [],
    )


def resolve_polardb_endpoint(
    cluster_id: str,
    region: Optional[str] = None,
    access_key_id: Optional[str] = None,
    access_key_secret: Optional[str] = None,
    endpoint_type: str = "Cluster",
    network_type: str = "Private",
) -> PolarDBEndpointInfo:
    """Resolve PolarDB cluster connection endpoint via Alibaba Cloud OpenAPI.

    Calls DescribeDBClusterEndpoints to discover the connection address
    for the specified cluster, filtering by endpoint_type and network_type.
    Uses the centralized OpenAPI endpoint (polardb.aliyuncs.com), which
    routes by cluster_id, so the region does not need to be known in advance.

    Args:
        cluster_id: PolarDB cluster ID (e.g. "pc-bp1234567890").
        region: Optional Alibaba Cloud region hint. Not required: the
            centralized endpoint resolves the cluster by ID regardless of
            region.
        access_key_id: Alibaba Cloud Access Key ID. Falls back to env var.
        access_key_secret: Alibaba Cloud Access Key Secret. Falls back to env var.
        endpoint_type: Endpoint type filter. One of "Cluster", "Primary", "Custom".
        network_type: Network type filter. One of "Private", "Public".

    Returns:
        PolarDBEndpointInfo with resolved host and port.

    Raises:
        ImportError: If alibabacloud SDK is not installed.
        ValueError: If no matching endpoint is found.
    """
    try:
        from alibabacloud_polardb20170801.client import Client as PolarDBClient
        from alibabacloud_polardb20170801.models import (
            DescribeDBClusterEndpointsRequest,
        )
        from alibabacloud_tea_openapi.models import Config as OpenAPIConfig
    except ImportError as exc:
        raise ImportError(
            "The 'alibabacloud-polardb20170801' and 'alibabacloud-tea-openapi' "
            "packages are required for PolarDBPGEngine.from_instance(). "
            "Install them with: pip install 'langchain-polardb-pg[cloud]'"
        ) from exc

    resolved_key_id, resolved_key_secret = _get_credentials(
        access_key_id, access_key_secret
    )

    config = OpenAPIConfig(
        access_key_id=resolved_key_id,
        access_key_secret=resolved_key_secret,
        region_id=region,
        endpoint="polardb.aliyuncs.com",
    )
    client = PolarDBClient(config)
    # DescribeType='AI' ensures AI nodes are included in the endpoint's Nodes
    # list (omitted from the default response).
    request = DescribeDBClusterEndpointsRequest(
        dbcluster_id=cluster_id, describe_type="AI"
    )
    response = client.describe_dbcluster_endpoints(request)

    # SDK returns snake_case attrs mapped from API's CamelCase JSON
    # e.g. EndpointType -> endpoint_type, AddressItems -> address_items
    # but the actual attr names depend on SDK version; use to_map() for safety
    response_map = response.body.to_map()
    items = response_map.get("Items", [])

    for item in items:
        if item.get("EndpointType") != endpoint_type:
            continue
        address_items = item.get("AddressItems", [])
        for addr in address_items:
            if addr.get("NetType") == network_type:
                return PolarDBEndpointInfo(
                    host=addr["ConnectionString"],
                    port=addr["Port"],
                    endpoint_type=item["EndpointType"],
                    network_type=addr["NetType"],
                    endpoint_id=item.get("DBEndpointId", ""),
                    ip_address=addr.get("IPAddress", ""),
                    vpc_id=addr.get("VPCId", ""),
                    vswitch_id=addr.get("VSwitchId", ""),
                    nodes=item.get("Nodes", ""),
                    read_write_mode=item.get("ReadWriteMode", ""),
                    service_name=item.get("ServiceName", ""),
                )

    available_endpoints = [
        f"  - endpoint_type={item.get('EndpointType')}, "
        f"networks=[{', '.join(a.get('NetType', '?') for a in item.get('AddressItems', []))}]"
        for item in items
    ]
    available_info = "\n".join(available_endpoints) if available_endpoints else "  (none)"
    raise ValueError(
        f"No matching endpoint found for cluster '{cluster_id}' with "
        f"endpoint_type='{endpoint_type}' and network_type='{network_type}'.\n"
        f"Available endpoints:\n{available_info}"
    )
