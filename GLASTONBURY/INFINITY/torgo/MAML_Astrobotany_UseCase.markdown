# MAML Astrobotany Use Case
## Archiving Plant Data with TORGO

MAML workflows in the Glastonbury 2048 SDK support astrobotany research by processing and archiving plant growth data with TORGO.

### Use Case
A researcher uses a MAML workflow to analyze Arabidopsis thaliana growth in microgravity, archiving results in a TORGO file with GEA_MICROG.

### Example Workflow
- **MAML File**: Defines a Qiskit circuit to analyze gene expression, with input/output schemas.
- **TORGO Output**:
  ```
  [TORGO]
  version=1.0
  protocol=astrobotany
  encryption=AES-2048
  [Metadata]
  gea_type=GEA_MICROG
  experiment_id=ASTRO_MICROG_2025-08-31
  timestamp=2025-08-31T00:00:00Z
  source=NASA
  [Data]
  type=genomic
  content={"gene_expression": "RNA-Seq data"}
  quantum_linguistic_prompt=<xaiArtifact id="123e4567-e89b-12d3-a456-426614174000" title="Microgravity_Prompt" contentType="text/markdown"># Microgravity Analysis\nAnalyze gene expression for gravitropism.