```python
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import obswebsocket
from obswebsocket import requests as obs_requests
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

class OBSControllerPayload(BaseModel):
    sensor_data: dict
    oauth_token: str
    security_mode: str
    wallet_address: str
    reputation: int

class OBSControllerResponse(BaseModel):
    stream_status: str
    visualization_data: dict
    dunes_hash: str
    signature: str
    status: str

@app.post("/api/services/beluga_obs_controller", response_model=OBSControllerResponse)
async def beluga_obs_controller(payload: OBSControllerPayload):
    """
    Manage OBS WebSocket streaming for BELUGA sensor data visualization with DUNES security.
    
    Args:
        payload (OBSControllerPayload): Sensor data, OAuth token, security mode, wallet address, and reputation.
    
    Returns:
        OBSControllerResponse: Stream status, visualization data, DUNES hash, signature, and status.
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
        
        # Fuse sensor data
        fusion_response = requests.post(
            "http://localhost:8000/api/services/beluga_sensor_fusion",
            json={
                "sonar_data": payload.sensor_data.get("sonar", []),
                "lidar_data": payload.sensor_data.get("lidar", []),
                "oauth_token": payload.oauth_token,
                "security_mode": payload.security_mode,
                "wallet_address": payload.wallet_address,
                "reputation": payload.reputation
            }
        )
        fusion_response.raise_for_status()
        fused_features = fusion_response.json()['fused_features']
        
        # Connect to OBS WebSocket
        client = obswebsocket.obsws("localhost", 4455, "")
        client.connect()
        
        # Configure and start stream
        client.call(obs_requests.SetStreamSettings(
            type="rtmp_common",
            settings={"server": "rtmp://live.twitch.tv/app", "key": "live_xxx"}
        ))
        client.call(obs_requests.StartStreaming())
        stream_status = client.call(obs_requests.GetStreamingStatus()).getStreaming()
        
        # Prepare visualization data
        visualization_data = {"fused_features": fused_features}
        
        # Disconnect OBS
        client.disconnect()
        
        # DUNES encryption
        key_length = 512 if payload.security_mode == "advanced" else 256
        qrng_key = generate_quantum_key(key_length // 8)
        cipher = AES.new(qrng_key, AES.MODE_CBC)
        result_data = json.dumps({"stream_status": stream_status, "visualization_data": visualization_data})
        encrypted_data = cipher.encrypt(pad(result_data.encode(), AES.block_size))
        dunes_hash = hashlib.sha3_512(encrypted_data).hexdigest()
        
        # Sign with CRYSTALS-Dilithium
        sig = Signature('Dilithium5')
        _, secret_key = sig.keypair()
        signature = sig.sign(encrypted_data, secret_key).hex()
        
        logger.info(f"BELUGA OBS streaming completed: {dunes_hash} 🐋🐪")
        return OBSControllerResponse(
            stream_status="active" if stream_status else "inactive",
            visualization_data=visualization_data,
            dunes_hash=dunes_hash,
            signature=signature,
            status="success"
        )
    except Exception as e:
        logger.error(f"BELUGA OBS streaming failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

def generate_quantum_key(bits):
    from qiskit import QuantumCircuit, Aer
    from qiskit.utils import QuantumInstance
    qrng = QuantumInstance(backend=Aer.get_backend('qasm_simulator'))
    return qrng.random_bits(bits)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Deployment Instructions
# Path: webxos-vial-mcp/src/services/beluga_obs_controller.py
# Run: pip install fastapi pydantic uvicorn requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python obs-websocket-py pyyaml
# Start: uvicorn src.services.beluga_obs_controller:app --host 0.0.0.0 --port 8000
```
