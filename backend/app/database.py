from pymongo import MongoClient
from typing import Dict, Any
import os
from datetime import datetime

class MongoDBClient:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
        self.db = self.client["maml_database"]
        self.collection = self.db["maml_documents"]

    def save_maml(self, maml_id: str, maml_data: Dict[str, Any]) -> None:
        maml_data["last_updated"] = datetime.utcnow()
        self.collection.update_one({"id": maml_id}, {"$set": maml_data}, upsert=True)

    def get_maml(self, maml_id: str) -> Dict[str, Any]:
        return self.collection.find_one({"id": maml_id})

    def update_maml_history(self, maml_id: str, history_entry: Dict[str, Any]) -> None:
        self.collection.update_one(
            {"id": maml_id},
            {"$push": {"history": {**history_entry, "timestamp": datetime.utcnow()}}}
        )
