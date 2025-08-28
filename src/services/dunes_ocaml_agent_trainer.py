import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.vial_wallet_service import vial_wallet_service
from oqs import Signature
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib
import json
import subprocess

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TrainerPayload(BaseModel):
    agent_name: str
    maml_content: str
    oauth_token: str
    security_mode: str
    wallet_address: str
    reputation: int

class TrainerResponse(BaseModel):
    status: str
    dunes_hash: str
    signature: str
    trained_agent: str

@app.post("/api/dunes/ocaml/train")
async def dunes_ocaml_agent_trainer(payload: TrainerPayload):
    """Train vial agents with OCaml and $WEBXOS wallets."""
    try:
        headers = {"Authorization": f"Bearer {payload.oauth_token}"}
        auth_response = requests.post(
            "https://webxos.auth.us-east-1.amazoncognito.com/oauth2/token",
            headers=headers
        )
        auth_response.raise_for_status()
        
        if payload.reputation < 2000000000:
            raise HTTPException(status_code=403, detail="Insufficient reputation score")
        
        wallet_result = await vial_wallet_service(
            {"wallet_address": payload.wallet_address, "oauth_token": payload.oauth_token,
             "security_mode": payload.security_mode, "reputation": payload.reputation}
        )
        
        maml_data = yaml.safe_load(payload.maml_content.split("---")[1])
        ocaml_code = maml_data.get("Code_Blocks", {}).get("ocaml", "")
        with open("train.ml", "w") as f:
            f.write(ocaml_code)
        subprocess.run(["ocamlc", "-c", "train.ml"], check=True)
        trained_agent = f"{payload.agent_name} trained with OCaml"
        
        qrng_key = generate_quantum_key(512 // 8)
        cipher = AES.new(qrng_key, AES.MODE_CBC)
        result_data = json.dumps({"trained_agent": trained_agent})
        encrypted_data = cipher.encrypt(pad(result_data.encode(), AES.block_size))
        dunes_hash = hashlib.sha3_512(encrypted_data).hexdigest()
        
        sig = Signature('Dilithium5')
        _, secret_key = sig.keypair()
        signature = sig.sign(encrypted_data, secret_key).hex()
        
        logger.info(f"Agent trained: {dunes_hash} 🐋🐪")
        return TrainerResponse(
            status="success",
            dunes_hash=dunes_hash,
            signature=signature,
            trained_agent=trained_agent
        )
    except Exception as e:
        logger.error(f"Agent training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

def generate_quantum_key(bits):
    from qiskit import QuantumCircuit, Aer
    from qiskit.utils import QuantumInstance
    qrng = QuantumInstance(backend=Aer.get_backend('qasm_simulator'))
    return qrng.random_bits(bits)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)

# Deployment Instructions
# Path: webxos-vial-mcp/src/services/dunes_ocaml_agent_trainer.py
# Run: pip install fastapi pydantic uvicorn requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python pyyaml
# Install OCaml: opam install ocaml
# Start: uvicorn src.services.dunes_ocaml_agent_trainer:app --host 0.0.0.0 --port 8009
