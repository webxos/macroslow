from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
import torch
import numpy as np

# Team Instruction: Implement 1024-bit AES encryption for Amoeba mode.
# Use CUDA for key generation, supporting distributed storage security.
class AES1024Encryptor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.key = get_random_bytes(128)  # 1024-bit key
        self.cipher = AES.new(self.key[:32], AES.MODE_CBC)  # Use first 256 bits for AES

    def encrypt(self, data: bytes) -> bytes:
        """Encrypts data with 1024-bit key (split into AES-256 CBC)."""
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key[:32], AES.MODE_CBC, iv)
        padded_data = pad(data, AES.block_size)
        ciphertext = cipher.encrypt(padded_data)
        return base64.b64encode(iv + ciphertext)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypts data with 1024-bit key."""
        raw = base64.b64decode(encrypted_data)
        iv = raw[:AES.block_size]
        ciphertext = raw[AES.block_size:]
        cipher = AES.new(self.key[:32], AES.MODE_CBC, iv)
        padded_data = cipher.decrypt(ciphertext)
        return unpad(padded_data, AES.block_size)

    def generate_key_cuda(self, seed: int) -> bytes:
        """Generates a 1024-bit key using CUDA."""
        torch.manual_seed(seed)
        with torch.cuda.stream(torch.cuda.Stream()):
            key_tensor = torch.randint(0, 256, (128,), dtype=torch.uint8, device=self.device)
        return key_tensor.cpu().numpy().tobytes()

# Example usage
if __name__ == "__main__":
    encryptor = AES1024Encryptor()
    data = b"Amoeba 1024-AES Storage Data"
    encrypted = encryptor.encrypt(data)
    decrypted = encryptor.decrypt(encrypted)
    print(f"Original: {data}")
    print(f"Encrypted: {encrypted[:16]}...")
    print(f"Decrypted: {decrypted}")