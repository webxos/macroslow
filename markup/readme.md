# MARKUP Agent for Project Dunes

The MARKUP Agent is a modular, hybrid PyTorch-SQLAlchemy agent for converting Markdown/MAML files to a reverse "Markup" (.mu) syntax, detecting errors, and visualizing transformations. It acts as a "Chimera Head" agent for Project Dunes, with API access for standalone operation.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install torch sqlalchemy fastapi uvicorn pyyaml plotly
   ```

2. **Run the API**:
   ```bash
   uvicorn markup_api:app --host 0.0.0.0 --port 8000
   ```

3. **Usage Example**:
   ```python
   from markup_agent import MarkupAgent

   agent = MarkupAgent()
   markdown = """---
   maml_version: "1.0.0"
   id: "urn:uuid:123e4567-e89b-12d3-a456-426614174000"
   ---
   ## Intent
   Test Markdown file
   """
   markup, errors = await agent.convert_to_markup(markdown)
   print(f"Markup:\n{markup}\nErrors: {errors}")
   ```

## API Endpoints

- `POST /to_markup`: Convert Markdown to Markup.
- `POST /to_markdown`: Convert Markup to Markdown.
- `POST /visualize`: Generate a 3D graph of the transformation.

## Integration with Project Dunes

The agent can be deployed as a Docker container or integrated into the Dunes Gateway for quantum-parallel execution. Use the `/to_markup` endpoint to validate MAML files before submission.
