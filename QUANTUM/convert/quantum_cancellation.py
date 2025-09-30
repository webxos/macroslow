from fastapi import FastAPI
from collections import defaultdict

app = FastAPI()
quantum_ops = defaultdict(lambda: {"entangled": True})

@app.post("/notifications/quantum_cancelled")
async def handle_cancellation(quantum_request_id: str, decoherence_reason: str = None):
    quantum_ops[quantum_request_id]["entangled"] = False
    # Generate .mu receipt
    with open(f"receipt_{quantum_request_id}.mu", "w") as f:
        f.write(f"Decohered: {decoherence_reason[::-1]}")  # Reverse mirror
    return {"status": "decohered"}

@app.post("/quantum_tools/migrate")
async def quantum_migrate(data: list, quantum_request_id: str):
    quantum_ops[quantum_request_id]["entangled"] = True
    for item in data:
        if not quantum_ops[quantum_request_id]["entangled"]:
            return {"status": "decohered"}
        # Process item
    quantum_ops.pop(quantum_request_id)
    return {"status": "collapsed"}