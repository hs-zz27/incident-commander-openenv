"""
Client for the Incident Commander Environment.

Provides both sync and async interfaces for connecting to a running
Incident Commander server via HTTP/WebSocket.

Usage:
    # Async (recommended):
    async with IncidentCommanderEnv(base_url="http://localhost:8000") as client:
        obs = await client.reset(task_name="single_service_failure")
        obs = await client.step(IncidentAction(action_type="inspect_logs", service_name="cache"))

    # Sync:
    with IncidentCommanderEnv(base_url="http://localhost:8000").sync() as client:
        obs = client.reset(task_name="single_service_failure")
        obs = client.step(IncidentAction(action_type="inspect_logs", service_name="cache"))
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx

from server.models import IncidentAction, IncidentObservation


class IncidentCommanderEnv:
    """
    Client for the Incident Commander Environment.

    Connects to a running server via HTTP. Supports both async and sync usage.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "IncidentCommanderEnv":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Core API (async)
    # ------------------------------------------------------------------

    async def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> IncidentObservation:
        """Reset the environment and start a new episode."""
        payload: Dict[str, Any] = {}
        if task_name:
            payload["task_name"] = task_name
        if seed is not None:
            payload["seed"] = seed
        if episode_id:
            payload["episode_id"] = episode_id

        resp = await self._ensure_client().post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return IncidentObservation(**data["observation"])

    async def step(self, action: IncidentAction) -> IncidentObservation:
        """Execute an action in the environment."""
        resp = await self._ensure_client().post(
            "/step",
            json={"action": action.model_dump()},
        )
        resp.raise_for_status()
        data = resp.json()
        return IncidentObservation(**data["observation"])

    async def get_state(self) -> Dict[str, Any]:
        """Get the full environment state."""
        resp = await self._ensure_client().get("/state")
        resp.raise_for_status()
        return resp.json()["state"]

    async def grade(self) -> Dict[str, Any]:
        """Get the episode grade/score."""
        resp = await self._ensure_client().get("/grade")
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> Dict[str, Any]:
        """Check if the server is healthy."""
        resp = await self._ensure_client().get("/health")
        resp.raise_for_status()
        return resp.json()

    async def tasks(self) -> List[str]:
        """List available tasks."""
        resp = await self._ensure_client().get("/tasks")
        resp.raise_for_status()
        return resp.json()["tasks"]

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def sync(self) -> "SyncIncidentCommanderEnv":
        """Return a synchronous wrapper around this client."""
        return SyncIncidentCommanderEnv(self)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self._client


class SyncIncidentCommanderEnv:
    """Synchronous wrapper around the async IncidentCommanderEnv client."""

    def __init__(self, async_client: IncidentCommanderEnv) -> None:
        self._async = async_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "SyncIncidentCommanderEnv":
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async.__aenter__())
        return self

    def __exit__(self, *args: Any) -> None:
        if self._loop:
            self._loop.run_until_complete(self._async.__aexit__(*args))
            self._loop.close()
            self._loop = None

    def _run(self, coro: Any) -> Any:
        if self._loop is None:
            raise RuntimeError("Client not entered. Use 'with' statement.")
        return self._loop.run_until_complete(coro)

    def reset(self, **kwargs: Any) -> IncidentObservation:
        return self._run(self._async.reset(**kwargs))

    def step(self, action: IncidentAction) -> IncidentObservation:
        return self._run(self._async.step(action))

    def get_state(self) -> Dict[str, Any]:
        return self._run(self._async.get_state())

    def grade(self) -> Dict[str, Any]:
        return self._run(self._async.grade())

    def health(self) -> Dict[str, Any]:
        return self._run(self._async.health())

    def tasks(self) -> List[str]:
        return self._run(self._async.tasks())
