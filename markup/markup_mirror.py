```python
import re
from typing import List, Dict

class MarkupMirror:
    def mirror_text(self, text: str) -> str:
        """Reverse each word in the text (e.g., Hello -> olleH)."""
        return " ".join(word[::-1] for word in text.split())

    def mirror_line(self, line: str) -> str:
        """Reverse words in a line, preserving code blocks and Markdown syntax."""
        if line.strip().startswith("```") or line.strip().startswith("#"):
            return line  # Preserve code blocks and headers
        return self.mirror_text(line)

    def mirror_yaml(self, front_matter: Dict) -> Dict:
        """Reverse words in YAML front matter keys and values."""
        reversed_fm = {}
        for key, value in front_matter.items():
            reversed_key = self.mirror_text(key)
            if isinstance(value, list):
                reversed_value = [self.mirror_text(v) if isinstance(v, str) else v for v in value[::-1]]
            elif isinstance(value, dict):
                reversed_value = self.mirror_yaml(value)
            else:
                reversed_value = self.mirror_text(value) if isinstance(value, str) else value
            reversed_fm[reversed_key] = reversed_value
        return reversed_fm

    def mirror_code_block(self, code_lines: List[str]) -> List[str]:
        """Reverse code block lines while preserving syntax."""
        reversed_lines = []
        in_code_block = False
        for line in code_lines[::-1]:
            if line.strip().startswith("```"):
                reversed_lines.append(line)
                in_code_block = not in_code_block
            elif in_code_block:
                reversed_lines.append(line)  # Preserve code logic
            else:
                reversed_lines.append(self.mirror_line(line))
        return reversed_lines