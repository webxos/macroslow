import yaml
from pydantic import BaseModel, ValidationError

# Team Instruction: Implement MAML validator for workflow parsing and validation.
# Ensure compatibility with Emeagwali’s structured dataflow principles.
class MAMLMetadata(BaseModel):
    maml_version: str
    id: str
    type: str
    encryption: str
    requires: list
    parameters: dict
    mu_validation_file: str

class MAMLValidator:
    def validate(self, file_path: str) -> dict:
        """Validates a MAML file against the schema."""
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
    validator = MAMLValidator()
    result = validator.validate("workflows/prime_sieve.maml.md")
    print(result)