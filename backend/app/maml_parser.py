import yaml
import markdown
import re
from typing import Dict, Any

class MAMLParser:
    def __init__(self):
        self.required_sections = ["Intent", "Code_Blocks"]

    def parse(self, content: str) -> Dict[str, Any]:
        # Split YAML front matter and Markdown content
        match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
        if not match:
            raise ValueError("Invalid MAML format: Missing YAML front matter")

        # Parse metadata
        metadata = yaml.safe_load(match.group(1))
        if not metadata or "maml_version" not in metadata:
            raise ValueError("Invalid MAML metadata")

        # Parse Markdown content
        markdown_content = match.group(2)
        sections = self._extract_sections(markdown_content)

        # Validate required sections
        for section in self.required_sections:
            if section not in sections:
                raise ValueError(f"Missing required section: {section}")

        return {
            "metadata": metadata,
            "sections": sections,
            "raw_content": content
        }

    def _extract_sections(self, content: str) -> Dict[str, str]:
        sections = {}
        current_section = None
        current_content = []

        for line in content.split('\n'):
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections
