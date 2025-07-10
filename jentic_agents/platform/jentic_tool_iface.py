"""Concrete implementation of the ToolInterface for the Jentic platform."""
from __future__ import annotations

from typing import Any, Dict, List

from jentic_agents.tools.interface import ToolInterface
from jentic_agents.platform.jentic_client import JenticClient


class JenticToolInterface(ToolInterface):
    """Adapter that provides Jentic tools via the generic ToolInterface."""

    def __init__(self, client: JenticClient) -> None:
        """Initialise with a JenticClient instance."""
        self._client = client

    def search(self, query: str, *, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for tools using the Jentic platform."""
        return self._client.search(query, top_k=top_k)

    def load(self, tool_id: str) -> Dict[str, Any]:
        """Load a tool's specification from the Jentic platform."""
        return self._client.load(tool_id)

    def execute(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via the Jentic platform."""
        return self._client.execute(tool_id, parameters)
