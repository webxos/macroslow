from mcp.server import MCPServer
from pydantic import BaseModel
from logfire import Logfire
import os

class MonitoringEvent(BaseModel):
    event_type: str
    timestamp: str
    details: dict

class LogfireMonitoringServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.logfire = Logfire(api_key=os.getenv("LOGFIRE_API_KEY", "your_logfire_api_key"))

    async def log_event(self, event: MonitoringEvent):
        try:
            self.logfire.log(event.event_type, **event.details)
            return {"status": "logged", "timestamp": event.timestamp}
        except Exception as e:
            return {"error": str(e)}

server = LogfireMonitoringServer()
server.run()
