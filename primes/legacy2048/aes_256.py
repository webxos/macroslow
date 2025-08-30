from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
import torch
import numpy as np

# Team Instruction: Implement 256-bit AES encryption for Fortran mode.
# Use CUDA for key generation, inspired by Emeagwali’s secure dataflow.
class AES256Encryptor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.key = get_random_bytes(32)  # 256-bit key
        self.cipher = AES.new(self.key, AES.MODE_CBC)

    def encrypt(self, data: bytes) -> bytes:
        """Encrypts data with 256-bit AES in CBC mode."""
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        padded_data = pad(data, AES.block_size)
        ciphertext = cipher.encrypt(padded_data)
        return base64.b64encode(iv + ciphertext)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypts data with 256-bit AES."""
        raw = base64.b64decode(encrypted_data)
        iv = raw[:AES.block_size]
        ciphertext = raw[AES.block_size:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        padded_data = cipher.decrypt(ciphertext)
        return unpad(padded_data, AES.block_size)

    def generate_key_cuda(self, seed: int) -> bytes:
        """Generates a 256-bit key using CUDA."""
        torch.manual_seed(seed)
        with torch.cuda.stream(torch.cuda.Stream()):
            key_tensor = torch.randint(0, 256, (32,), dtype=torch.uint8, device=self.device)
        return key_tensor.cpu().numpy().tobytes()

# Example usage
if __name__ == "__main__":
    encryptor = AES256Encryptor()
    data = b"Fortran 256-AES Input Data"
    encrypted = encryptor.encrypt(data)
    decrypted = encryptor.decrypt(encrypted)
    print(f"Original: {data}")
    print(f"Encrypted: {encrypted[:16]}...")
    print(f"Decrypted: {decrypted}")