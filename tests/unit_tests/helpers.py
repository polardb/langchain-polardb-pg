"""Shared helpers for unit tests."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any
from unittest.mock import MagicMock


def _close_if_awaitable(value: Any) -> None:
    if inspect.iscoroutine(value):
        value.close()


def run_as_sync_returning(*values: Any) -> MagicMock:
    """Return a mocked _run_as_sync that consumes coroutine arguments."""
    remaining = list(values)
    fallback = values[-1] if values else None

    def run(coro: Awaitable[Any]) -> Any:
        _close_if_awaitable(coro)
        if remaining:
            return remaining.pop(0)
        return fallback

    return MagicMock(side_effect=run)


def run_as_sync_raising(exc: Exception) -> MagicMock:
    """Return a mocked _run_as_sync that closes its coroutine then raises."""

    def run(coro: Awaitable[Any]) -> Any:
        _close_if_awaitable(coro)
        raise exc

    return MagicMock(side_effect=run)
