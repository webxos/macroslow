# mcp_integration.py
# Description: Integration with Model Context Protocol (MCP) servers for BELUGA.
# Handles communication with CHIMERA 2048 and other MCP-compliant systems.
# Usage: Instantiate MCPIntegration and use send_mcp_message for API calls.

import requests
import json
from typing import Dict, Any
import asyncio
from websockets.sync.client import connect as websocket_connect

class MCPIntegration:
    """
    Manages communication with MCP servers for BELUGA workflows.
    Supports REST and WebSocket for real-time data exchange.
    """
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws"

    def send_mcp_message(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends data to an MCP server via REST.
        Input: Endpoint and data dictionary.
        Output: Server response as a dictionary.
        """
        response = requests.post(
            f"{self.base_url}/{endpoint}",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        return response.json()

    async def establish_websocket_connection(self):
        """
        Establishes a WebSocket connection for real-time data streaming.
        Handles incoming MCP messages asynchronously.
        """
        async with websocket_connect(self.ws_url) as websocket:
            while True:
                message = await websocket.recv()
                await self._handle_mcp_message(json.loads(message))

    async def _handle_mcp_message(self, message: Dict[str, Any]):
        """
        Processes incoming MCP messages (e.g., sensor data, navigation updates).
        """
        if message['type'] == 'sensor_data':
            print(f"Received sensor data: {message['data']}")
        elif message['type'] == 'navigation_update':
            print(f"Navigation update: {message['data']}")

# Example usage:
# mcp = MCPIntegration()
# response = mcp.send_mcp_message("sensor_data", {"data": [1, 2, 3]})
# print(response)