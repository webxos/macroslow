import yaml
from pydantic import BaseModel, ValidationError

# Team Instruction: Validate MAML files to ensure compatibility with Emeagwali’s structured workflows.
# Use Pydantic for rigorous data validation, inspired by his emphasis on data integrity.
class MAMLMetadata(BaseModel):
    maml_version: str
    id: str
    type: str
    origin: str
    requires: dict = {}
    permissions: dict
    created_at: str
    verification: dict = {}

def validate_maml_file(file_path: str) -> dict:
    """
    Validates a MAML file’s front matter against the schema.
    Team: Ensure all workflows are verifiable and secure before execution.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            front_matter = content.split('---')[1]
            metadata = yaml.safe_load(front_matter)
            MAMLMetadata(**metadata)
            return {"status": "valid", "metadata": metadata}
    except ValidationError as e:
        return {"status": "invalid", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Example usage
if __name__ == "__main__":
    result = validate_maml_file("example_matrix_multiply.maml.md")
    print(result)
