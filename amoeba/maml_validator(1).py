# AMOEBA 2048AES MAML Validator
# Description: Validates MAML files for AMOEBA 2048AES SDK, ensuring compliance with MCP standards and schema requirements.

import yaml
import json
from pydantic import BaseModel, ValidationError
import click

class MAMLSchema(BaseModel):
    maml_version: str
    id: str
    type: str
    origin: str
    requires: dict
    permissions: dict
    created_at: str
    input_schema: dict
    output_schema: dict

def validate_maml(file_path: str) -> bool:
    """Validate a MAML file against the schema."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Extract YAML front matter
            if content.startswith('---'):
                yaml_content = content.split('---')[1]
                data = yaml.safe_load(yaml_content)
                MAMLSchema(**data)
                click.echo(f"MAML file {file_path} is valid.")
                return True
            else:
                click.echo(f"MAML file {file_path} missing YAML front matter.")
                return False
    except ValidationError as e:
        click.echo(f"MAML validation failed: {str(e)}")
        return False
    except FileNotFoundError:
        click.echo(f"MAML file {file_path} not found.")
        return False

@click.command()
@click.option('--maml', required=True, help='Path to MAML file')
def validate(maml):
    """CLI command to validate MAML files."""
    if validate_maml(maml):
        click.echo("MAML validation successful.")
    else:
        click.echo("MAML validation failed.")

if __name__ == "__main__":
    validate()