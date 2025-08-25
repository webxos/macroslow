from mcp.server import MCPServer
from pydantic import BaseModel
import os
import json

class FileMetadata(BaseModel):
    path: str
    size: int
    is_dir: bool
    created_at: str

class FilesystemServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.allowed_paths = ["/data", "/models"]

    async def list_files(self, path: str):
        if not any(path.startswith(p) for p in self.allowed_paths):
            return {"error": "Access denied"}
        files = []
        try:
            for entry in os.scandir(path):
                files.append(FileMetadata(
                    path=entry.path,
                    size=entry.stat().st_size,
                    is_dir=entry.is_dir(),
                    created_at=entry.stat().st_ctime
                ).dict())
            return {"files": files}
        except Exception as e:
            return {"error": str(e)}

    async def read_file(self, path: str):
        if not any(path.startswith(p) for p in self.allowed_paths):
            return {"error": "Access denied"}
        try:
            with open(path, 'r') as f:
                return {"content": f.read()}
        except Exception as e:
            return {"error": str(e)}

server = FilesystemServer()
server.run()
