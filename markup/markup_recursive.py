```python
import torch
import torch.nn as nn
from markup_db import MarkupDatabase
from markup_parser import MarkupParser
from typing import List, Dict

class MarkupRecursive:
    def __init__(self, db_uri: str):
        """Initialize recursive training system for agentic recursion networks."""
        self.db = MarkupDatabase(db_uri)
        self.parser = MarkupParser()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predict recursive transformation accuracy
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    async def train_recursive(self, limit: int = 1000):
        """Train on mirrored receipts to create recursive networks."""
        session = self.db.Session()
        logs = session.query(TransformationLog).filter(TransformationLog.errors.contains("Receipt")).limit(limit).all()
        for log in logs:
            original = self.parser.parse_markdown(log.input_content)
            receipt = self.parser.parse_markdown(log.output_content)
            features = self._extract_recursive_features(original, receipt)
            target = torch.tensor(1.0 if not log.errors else 0.0, dtype=torch.float32).to(self.device)
            self.optimizer.zero_grad()
            output = self.model(features)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
        session.close()

    def _extract_recursive_features(self, original: Dict, receipt: Dict) -> torch.Tensor:
        """Extract features for recursive training from original and receipt."""
        features = []
        # Compare section counts
        features.append(len(original["sections"]) - len(receipt["sections"]))
        # Compare word counts in content
        orig_words = sum(len(" ".join(lines).split()) for lines in original["sections"].values())
        receipt_words = sum(len(" ".join(lines).split()) for lines in receipt["sections"].values())
        features.append(orig_words - receipt_words)
        return torch.tensor(features + [0.0] * (256 - len(features)), dtype=torch.float32).to(self.device)