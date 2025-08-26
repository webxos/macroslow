```python
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import svgwrite
import requests
from oqs import Signature
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib
import json
import yaml

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SVGVisualizerPayload(BaseModel):
    navigation_path: list
    oauth_token: str
    security_mode: str
    wallet_address: str
    reputation: int

class SVGVisualizerResponse(BaseModel):
    svg_content: str
    dunes_hash: str
    signature: str
    status: str

@app.post("/api/services/beluga_svg_visualizer", response_model=SVGVisualizerResponse)
async def beluga_svg_visualizer(payload: SVGVisualizerPayload):
    """
    Generate real-time SVG diagrams for BELUGA navigation paths with DUNES security.
    
    Args:
        payload (SVGVisualizerPayload): Navigation path, OAuth token, security mode, wallet address, and reputation.
    
    Returns:
        SVGVisualizerResponse: SVG content, DUNES hash, signature, and status.
    """
    try:
        # Validate OAuth token
        headers = {"Authorization": f"Bearer {payload.oauth_token}"}
        auth_response = requests.post(
            "https://webxos.auth.us-east-1.amazoncognito.com/oauth2/token",
            headers=headers
        )
        auth_response.raise_for_status()
        
        # Validate reputation
        if payload.reputation < 2000000000:
            raise HTTPException(status_code=403, detail="Insufficient reputation score")
        
        # Load configuration
        with open("config/beluga_mcp_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Generate SVG diagram
        dwg = svgwrite.Drawing(size=("800px", "600px"))
        dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="black"))
        for i, point in enumerate(payload.navigation_path):
            x = (i % 10) * 80 + 40
            y = (i // 10) * 60 + 30
            dwg.add(dwg.circle(center=(x, y), r=10, fill="neon green"))
            if i > 0:
                prev_x = ((i-1) % 10) * 80 + 40
                prev_y = ((i-1) // 10) * 60 + 30
                dwg.add(dwg.line(start=(prev_x, prev_y), end=(x, y), stroke="neon green"))
        svg_content = dwg.tostring()
        
        # DUNES encryption
        key_length = 512 if payload.security_mode == "advanced" else 256
        qrng_key = generate_quantum_key(key_length // 8)
        cipher = AES.new(qrng_key, AES.MODE_CBC)
        result_data = json.dumps({"svg_content": svg_content})
        encrypted_data = cipher.encrypt(pad(result_data.encode(), AES.block_size))
        dunes_hash = hashlib.sha3_512(encrypted_data).hexdigest()
        
        # Sign with CRYSTALS-Dilithium
        sig = Signature('Dilithium5')
        _, secret_key = sig.keypair()
        signature = sig.sign(encrypted_data, secret_key).hex()
        
        logger.info(f"BELUGA SVG visualization completed: {dunes_hash} 🐋🐪")
        return SVGVisualizerResponse(
            svg_content=svg_content,
            dunes_hash=dunes_hash,
            signature=signature,
            status="success"
        )
    except Exception as e:
        logger.error(f"BELUGA SVG visualization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

def generate_quantum_key(bits):
    from qiskit import QuantumCircuit, Aer
    from qiskit.utils import QuantumInstance
    qrng = QuantumInstance(backend=Aer.get_backend('qasm_simulator'))
    return qrng.random_bits(bits)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Deployment Instructions
# Path: webxos-vial-mcp/src/services/beluga_svg_visualizer.py
# Run: pip install fastapi pydantic uvicorn requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python svgwrite pyyaml
# Start: uvicorn src.services.beluga_svg_visualizer:app --host 0.0.0.0 --port 8000
```
