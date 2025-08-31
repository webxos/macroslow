from mcp.server import Server
from mcp.server.models import InitializationOptions
import torch
from src.cm_2048.core.quadrilinear_engine import QuadrilinearEngine

# Team Instruction: Implement an MCP server for ultra-fast API access to the Quadrilinear Core.
# Ensure seamless synchronization across four nodes with low-latency responses.
app = Server("cm-2048-aes-server")

@app.listener()
async def on_initialize(session_id, params):
    print("Client connected to Connection Machine 2048-AES!")
    return InitializationOptions()

qe = QuadrilinearEngine()

@app.function()
async def simulate_connection_machine_operation(operation_name: str, input_data: list) -> dict:
    """
    Exposes the Quadrilinear Engine for parallel computations with 2048-bit AES encryption.
    Team: Optimize API speed for quantum synchronization (use async for low latency).
    """
    try:
        tensor_input = torch.tensor(input_data, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Invalid input data: {str(e)}"}

    if operation_name == "matrix_square":
        def op(x): return torch.mm(x, x) if x.dim() == 2 else x**2
    elif operation_name == "sigmoid":
        def op(x): return torch.sigmoid(x)
    else:
        return {"error": f"Unknown operation: {operation_name}"}

    result_tensor = qe.execute_parallel(op, tensor_input)
    result_list = result_tensor.tolist()

    return {
        "result": result_list,
        "original_shape": list(tensor_input.shape),
        "result_shape": list(result_tensor.shape),
        "message": "Computation executed across 4 parallel Connection Machines."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)