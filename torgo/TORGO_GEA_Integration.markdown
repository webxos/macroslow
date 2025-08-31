# TORGO and Growth Environment Anchors (GEAs)
## Structuring Astrobotany Data

TORGO integrates Growth Environment Anchors (GEAs) to categorize astrobotany data based on space-relevant conditions, enhancing the Glastonbury 2048 SDK’s research capabilities.

### GEA Types
- **GEA_MICROG**: Microgravity experiments (e.g., ISS).
- **GEA_PARTIALG**: Partial gravity simulations (Mars, Moon).
- **GEA_RADIATION**: Cosmic radiation effects.
- **GEA_HYDROPONIC**: Soilless growth systems.
- **GEA_REGOLITH**: Martian/Lunar regolith studies.
- **GEA_BLSS**: Bioregenerative Life Support Systems.
- **GEA_PSYCH**: Psychological benefits of plants.
- **GEA_TERRESTRIAL**: Earth-based controls.
- **GEA_CITIZEN**: Citizen science contributions.
- **GEA_GENOMIC**: Gene expression data.

### Integration
- GEAs are defined in `core_gea_standards.maml`.
- The `data_ingestor.py` module tags data with GEAs based on experimental conditions.
- Researchers query archives using GEAs for targeted analysis (e.g., all GEA_MICROG datasets).

### Use Case
A researcher studying plant growth in Martian regolith uses GEA_REGOLITH to archive and retrieve relevant data, ensuring precise, context-aware analysis.