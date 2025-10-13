# maml_processor.py: MAML file processing logic
# Purpose: Parses and validates MAML files with YAML front matter
# Instructions:
# 1. Ensure pyyaml is installed (in requirements.txt)
# 2. Input must have YAML front matter (see example.maml.md)
import yaml
import re

def process_maml_file(content: str) -> dict:
    """
    Process a MAML file, extracting YAML metadata and body.
    Args:
        content (str): Raw MAML file content
    Returns:
        dict: Parsed metadata and body
    Raises:
        ValueError: If YAML front matter or schema is missing
    """
    # Extract YAML front matter
    yaml_match = re.match(r'---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not yaml_match:
        raise ValueError("Invalid MAML file: Missing YAML front matter")
    
    metadata, body = yaml_match.groups()
    metadata = yaml.safe_load(metadata)
    
    # Validate schema
    if "schema" not in metadata:
        raise ValueError("MAML file must include schema in metadata")
    
    return {"metadata": metadata, "body": body.strip()}