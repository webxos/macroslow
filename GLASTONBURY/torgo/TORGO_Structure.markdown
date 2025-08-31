# TORGO Structure
## Syntax and Format

TORGO files are plaintext, structured documents designed for simplicity and security, compatible with the Glastonbury 2048 SDK. They consist of three sections: `[TORGO]`, `[Metadata]`, and `[Data]`.

### File Structure
- **Header ([TORGO])**:
  - `version`: Protocol version (e.g., 1.0).
  - `protocol`: Specifies domain (e.g., astrobotany).
  - `encryption`: Security standard (AES-2048).
- **Metadata**:
  - `gea_type`: Growth Environment Anchor (e.g., GEA_MICROG).
  - `experiment_id`: Unique identifier.
  - `timestamp`: ISO 8601 timestamp.
  - `source`: Data origin (NASA, SpaceX, Citizen).
- **Data**:
  - `type`: Data category (genomic, environmental, psychological).
  - `content`: Raw or processed data payload.
  - `quantum_linguistic_prompt`: MAML-based prompt for analysis.

### Example Usage
TORGO files are created using templates (e.g., `torgo_astro_template.torgo`) and populated via the SDK’s `data_ingestor.py`. Researchers use the CLI (`glas_astro_cli.py`) to ingest and verify data.

### Benefits
- Human-readable for accessibility.
- Machine-parsable for automation.
- Secure with AES-2048 encryption.