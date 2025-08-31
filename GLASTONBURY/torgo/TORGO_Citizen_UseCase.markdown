# TORGO Template: Citizen Science
## Use Case for Glastonbury 2048 SDK

**Purpose**: Archive citizen science contributions to astrobotany, tagged with GEA_CITIZEN, for quantum linguistic analysis.

**Use Case**: Citizen scientists use the TOAST platform to submit hydroponic experiment data, archived for global research.

**TORGO Template**:
```
[TORGO]
version=1.0
protocol=astrobotany
encryption=AES-2048

[Metadata]
gea_type=GEA_CITIZEN
experiment_id=ASTRO_CITIZEN_2025-08-31
timestamp=2025-08-31T00:00:00Z
source=Citizen

[Data]
type=environmental
content={"light_intensity_lux": 5000, "growth_height_cm": 10}
quantum_linguistic_prompt=<xaiArtifact id="901m2345-6ghi-7890-1234-5abcdef67890" title="Citizen_Prompt" contentType="text/markdown"># Citizen Science Analysis\nAnalyze community-driven hydroponic data.