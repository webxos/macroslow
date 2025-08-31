import httpx
import json
import uuid
from datetime import datetime
from typing import Dict, List

class AstrobotanyIngestor:
    async def ingest_data(self, source: str, data: Dict) -> Dict:
        """Ingest and tag astrobotany data with GEAs."""
        gea_type = self._classify_gea(data)
        experiment_id = f"ASTRO_{datetime.utcnow().isoformat()}"
        
        if source == "NASA":
            data = await self._fetch_nasa_data(data)
        elif source == "SpaceX":
            data = await self._fetch_spacex_data(data)
            
        torgo_record = {
            "TORGO": {"version": "1.0", "protocol": "astrobotany", "encryption": "AES-2048"},
            "Metadata": {"gea_type": gea_type, "experiment_id": experiment_id, "timestamp": datetime.utcnow().isoformat(), "source": source},
            "Data": {"type": data.get("type"), "content": data, "quantum_linguistic_prompt": self._generate_prompt(data)}
        }
        return torgo_record

    async def _fetch_nasa_data(self, data: Dict) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.nasa.gov/planetary/apod?api_key={data['api_key']}")
            return response.json()

    async def _fetch_spacex_data(self, data: Dict) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.spacexdata.com/v4/launches/latest")
            return response.json()

    def _classify_gea(self, data: Dict) -> str:
        """Classify data with appropriate GEA."""
        conditions = data.get("conditions", "").lower()
        if "microgravity" in conditions:
            return "GEA_MICROG"
        elif "partial gravity" in conditions:
            return "GEA_PARTIALG"
        elif "radiation" in conditions:
            return "GEA_RADIATION"
        elif "hydroponic" in conditions:
            return "GEA_HYDROPONIC"
        elif "regolith" in conditions:
            return "GEA_REGOLITH"
        elif "bioregenerative" in conditions:
            return "GEA_BLSS"
        elif "psychological" in conditions:
            return "GEA_PSYCH"
        elif "citizen" in conditions:
            return "GEA_CITIZEN"
        elif "genomic" in conditions:
            return "GEA_GENOMIC"
        return "GEA_TERRESTRIAL"

    def _generate_prompt(self, data: Dict) -> str:
        """Generate MAML-based quantum linguistic prompt."""
        return f"<xaiArtifact id='{str(uuid.uuid4())}' title='Astrobotany_Prompt' contentType='text/markdown'># Analysis Prompt\nAnalyze {data.get('type')} data for semantic patterns.