import yaml
import re
from typing import Dict, List

class MarkupParser:
    def parse_markdown(self, content: str) -> Dict:
        """Parse a Markdown/MAML file into a structured dictionary."""
        parts = content.split("---\n", 2)
        front_matter = yaml.safe_load(parts[1]) if len(parts) > 2 else {}
        body = parts[2] if len(parts) > 2 else parts[0]
        sections = {}
        current_section = None
        for line in body.splitlines():
            if line.startswith("## "):
                current_section = line[3:].strip()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)
        return {"front_matter": front_matter, "sections": sections}

    def reverse_to_markup(self, parsed: Dict) -> str:
        """Convert parsed Markdown to Markup (.mu) by reversing structure."""
        front_matter = parsed["front_matter"]
        sections = parsed["sections"]
        reversed_sections = []
        for section, content in reversed(sections.items()):
            reversed_content = [self._reverse_line(line) for line in content[::-1]]
            reversed_sections.append(f"## {self._reverse_text(section)}\n" + "\n".join(reversed_content))
        reversed_front_matter = self._reverse_yaml(front_matter)
        return f"---\n{yaml.dump(reversed_front_matter)}---\n" + "\n".join(reversed_sections)

    def parse_markup(self, content: str) -> Dict:
        """Parse a Markup (.mu) file into a structured dictionary."""
        return self.parse_markdown(content)  # Same structure, reversed logic

    def reverse_to_markdown(self, parsed: Dict) -> str:
        """Convert parsed Markup back to Markdown."""
        return self.reverse_to_markup(parsed)  # Reverse operation is symmetric

    def _reverse_line(self, line: str) -> str:
        """Reverse a single line of text, preserving code blocks."""
        if line.strip().startswith("```"):
            return line  # Preserve code blocks
        return line[::-1]

    def _reverse_text(self, text: str) -> str:
        """Reverse text while preserving meaning for headers."""
        return text[::-1].replace("(", ")").replace(")", "(").replace("[", "]").replace("]", "[")

    def _reverse_yaml(self, front_matter: Dict) -> Dict:
        """Reverse YAML front matter keys and values."""
        reversed_fm = {}
        for key, value in front_matter.items():
            reversed_key = self._reverse_text(key)
            if isinstance(value, list):
                reversed_value = [self._reverse_text(v) if isinstance(v, str) else v for v in value[::-1]]
            elif isinstance(value, dict):
                reversed_value = self._reverse_yaml(value)
            else:
                reversed_value = self._reverse_text(value) if isinstance(value, str) else value
            reversed_fm[reversed_key] = reversed_value
        return reversed_fm

    def validate_structure(self, original: Dict, reversed_content: str) -> bool:
        """Validate that the reversed content matches the original structure."""
        reversed_parsed = self.parse_markdown(reversed_content)
        return original["front_matter"].keys() == reversed_parsed["front_matter"].keys()

    def extract_features(self, parsed: Dict) -> List[float]:
        """Extract features for error detection model."""
        features = []
        features.append(len(parsed["sections"]))
        features.append(len(str(parsed["front_matter"])))
        return features

    def generate_graph_data(self, markdown: str, markup: str) -> Dict:
        """Generate data for 3D visualization of transformation process."""
        return {
            "nodes": [
                {"id": "markdown", "label": "Markdown Input", "group": "input"},
                {"id": "markup", "label": "Markup Output", "group": "output"},
            ],
            "edges": [{"from": "markdown", "to": "markup", "label": "Conversion"}]
        }