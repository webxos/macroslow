import torch
import numpy as np
import cudasieve
from src.legacy_2048.aes_2048 import AES2048Encryptor

# Team Instruction: Implement Connection Machine 2048-AES mode for final output and prime sieving.
# Use CUDASieve for CUDA-accelerated sieving, inspired by Emeagwali’s massive parallelism.
class ConnectionMachine2048AES:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encryptor = AES2048Encryptor()

    async def process(self, data: torch.Tensor) -> torch.Tensor:
        """Executes CUDA-accelerated prime sieve and 2048-bit AES encryption."""
        # Convert input to numpy for CUDASieve
        data_np = data.cpu().numpy().astype(np.uint32)
        primes = cudasieve.sieve(data_np)
        primes_tensor = torch.tensor(primes, dtype=torch.int64, device=self.device)

        # Encrypt output
        encrypted_data = self.encryptor.encrypt(primes_tensor.cpu().numpy().tobytes())
        return torch.tensor(np.frombuffer(encrypted_data, dtype=np.uint8), device=self.device)

# Example usage
if __name__ == "__main__":
    cm = ConnectionMachine2048AES()
    input_data = torch.arange(1, 1000001, device="cuda")
    result = asyncio.run(cm.process(input_data))
    print(f"Connection Machine 2048-AES output shape: {result.shape}")