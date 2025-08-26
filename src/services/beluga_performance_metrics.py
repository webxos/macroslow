```python
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
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

# Prometheus metrics
threat_detection_counter = Counter('beluga_threat_detections_total', 'Total BELUGA threat detections')
navigation_latency = Histogram('beluga_navigation_latency_seconds', 'BELUGA navigation latency')
fusion_accuracy = Gauge('beluga_fusion_accuracy', 'BELUGA sensor fusion accuracy')

class PerformanceMetricsPayload(BaseModel):
    service_name: str
    oauth_token: str
    security_mode: str
    wallet_address: str
    reputation: int

class PerformanceMetricsResponse(BaseModel):
    metrics: dict
    dunes_hash: str
    signature: str
    status: str

@app.get("/api/services/beluga_performance_metrics", response_model=PerformanceMetricsResponse)
async def beluga_performance_metrics(payload: PerformanceMetricsPayload):
    """
    Collect BELUGA-specific performance metrics with DUNES security.
    
    Args:
        payload (PerformanceMetricsPayload): Service name, OAuth token, security mode, wallet address, and reputation.
    
    Returns:
        PerformanceMetricsResponse: Metrics, DUNES hash, signature, and status.
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
        
        # Collect metrics
        metrics = {
            "service_name": payload.service_name,
            "threat_detections": threat_detection_counter._value.get(),
            "navigation_latency": navigation_latency._sum.get(),
            "fusion_accuracy": fusion_accuracy._value.get()
        }
        
        # DUNES encryption
        key_length = 512 if payload.security_mode == "advanced" else 256
        qrng_key = generate_quantum_key(key_length // 8)
        cipher = AES.new(qrng_key, AES.MODE_CBC)
        result_data = json.dumps({"metrics": metrics})
        encrypted_data = cipher.encrypt(pad(result_data.encode(), AES.block_size))
        dunes_hash = hashlib.sha3_512(encrypted_data).hexdigest()
        
        # Sign with CRYSTALS-Dilithium
        sig = Signature('Dilithium5')
        _, secret_key = sig.keypair()
        signature = sig.sign(encrypted_data, secret_key).hex()
        
        logger.info(f"BELUGA performance metrics collected: {dunes_hash} 🐋🐪")
        return PerformanceMetricsResponse(
            metrics=metrics,
            dunes_hash=dunes_hash,
            signature=signature,
            status="success"
        )
    except Exception as e:
        logger.error(f"BELUGA performance metrics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

def generate_quantum_key(bits):
    from qiskit import QuantumCircuit, Aer
    from qiskit.utils import QuantumInstance
    qrng = QuantumInstance(backend=Aer.get_backend('qasm_simulator'))
    return qrng.random_bits(bits)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Deployment Instructions
# Path: webxos-vial-mcp/src/services/beluga_performance_metrics.py
# Run: pip install fastapi pydantic uvicorn requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python prometheus-client pyyaml
# Start: uvicorn src.services.beluga_performance_metrics:app --host 0.0.0.0 --port 8000
```
