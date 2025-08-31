# MAML Syntax
## Structure and Components

MAML files are human-readable Markdown documents with YAML front matter and structured sections, designed for executable workflows in the Glastonbury 2048 SDK.

### Components
- **YAML Front Matter**:
  - `maml_version`: Version (e.g., 2.0.0).
  - `id`: Unique identifier (e.g., UUID).
  - `type`: Workflow type (e.g., quantum_workflow).
  - `requires`: Resources (e.g., CUDA, Qiskit).
  - `permissions`: Access controls (read, write, execute).
  - `verification`: Method (e.g., Ortac) and spec files.
- **Sections**:
  - `Intent`: Workflow purpose.
  - `Context`: Parameters (e.g., dataset, model path).
  - `Code_Blocks`: Executable code (Python, OCaml, etc.).
  - `Input_Schema`/`Output_Schema`: Data formats.
  - `History`: Audit trail of changes.

### Example
A MAML file might define a quantum workflow with a TORGO archive, specifying a Qiskit circuit and input/output schemas, executed via `curl -X POST http://localhost:8000/execute`.

### Benefits
- Human-readable for collaboration.
- Machine-parsable for automation.
- Integrates with TORGO for secure archiving.