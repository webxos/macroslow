import torch
import yaml
import re
from typing import Dict, List, Tuple
from markup_parser import MarkupParser
from markup_db import MarkupDatabase
from markup_visualizer import MarkupVisualizer

class MarkupAgent:
    def __init__(self, db_uri: str = "sqlite:///markup_logs.db"):
        """Initialize the MARKUP Agent with PyTorch, SQLAlchemy, and visualization."""
        self.parser = MarkupParser()
        self.db = MarkupDatabase(db_uri)
        self.visualizer = MarkupVisualizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.error_model = torch.nn.Linear(128, 64).to(self.device)  # Simple error detection model
        self.error_optimizer = torch.optim.Adam(self.error_model.parameters(), lr=0.001)

    async def convert_to_markup(self, markdown_content: str) -> Tuple[str, List[str]]:
        """Convert Markdown to Markup (.mu) syntax by reversing structure."""
        parsed = self.parser.parse_markdown(markdown_content)
        reversed_content = self.parser.reverse_to_markup(parsed)
        errors = self.detect_errors(parsed, reversed_content)
        self.db.log_transformation(markdown_content, reversed_content, errors)
        return reversed_content, errors

    async def convert_to_markdown(self, markup_content: str) -> Tuple[str, List[str]]:
        """Convert Markup (.mu) back to Markdown."""
        parsed = self.parser.parse_markup(markup_content)
        markdown_content = self.parser.reverse_to_markdown(parsed)
        errors = self.detect_errors(parsed, markdown_content)
        self.db.log_transformation(markup_content, markdown_content, errors)
        return markdown_content, errors

    def detect_errors(self, original: Dict, reversed: str) -> List[str]:
        """Detect structural and semantic errors using PyTorch-based analysis."""
        errors = []
        # Simple feature extraction for error detection
        features = torch.tensor(self.parser.extract_features(original), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            error_scores = self.error_model(features).cpu().numpy()
        if error_scores.max() > 0.5:  # Threshold for error detection
            errors.append(f"Structural mismatch detected in {original.get('id', 'unknown')}")
        # Additional rule-based checks
        if not self.parser.validate_structure(original, reversed):
            errors.append("Invalid reverse transformation: structure mismatch")
        return errors

    async def visualize_transformation(self, markdown_content: str, markup_content: str):
        """Generate a 3D ultra-graph of the transformation process."""
        graph_data = self.parser.generate_graph_data(markdown_content, markup_content)
        self.visualizer.render_3d_graph(graph_data)

    async def train_error_model(self, training_data: List[Dict]):
        """Train the PyTorch error detection model on transformation examples."""
        for data in training_data:
            features = torch.tensor(self.parser.extract_features(data["original"]), dtype=torch.float32).to(self.device)
            target = torch.tensor(data["error_score"], dtype=torch.float32).to(self.device)
            self.error_optimizer.zero_grad()
            output = self.error_model(features)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            self.error_optimizer.step()
