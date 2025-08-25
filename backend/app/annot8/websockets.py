from fastapi import WebSocket
from typing import Dict

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: int):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, websocket: WebSocket, client_id: int):
        self.active_connections.pop(client_id, None)

    async def broadcast_json(self, message: dict):
        for client_id, websocket in self.active_connections.items():
            await websocket.send_json(message)

    async def send_to_client(self, client_id: int, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()
