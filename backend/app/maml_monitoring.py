import logging
from datetime import datetime
from backend.app.database import MongoDBClient
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MAMLMonitoring:
    def __init__(self):
        self.db = MongoDBClient()

    def log_operation(self, maml_id: str, action: str, details: Dict[str, Any]) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "maml_id": maml_id,
            "action": action,
            "details": details
        }
        logger.info(f"MAML Operation: {log_entry}")
        self.db.update_maml_history(maml_id, log_entry)

    def get_metrics(self, maml_id: str) -> Dict[str, Any]:
        maml_data = self.db.get_maml(maml_id)
        if maml_data:
            history = maml_data.get("history", [])
            return {
                "execution_count": len(history),
                "last_execution": history[-1]["timestamp"] if history else None,
                "success_rate": sum(1 for h in history if h["status"] == "Success") / len(history) if history else 0
            }
        return {"error": "MAML not found"}
