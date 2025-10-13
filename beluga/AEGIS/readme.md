# 🔐 AES2048Encryptor

A high-security, quantum-resistant AES encryption module leveraging CUDA for key generation. This Python class provides 2048-bit AES encryption using PyCryptodome and PyTorch, ideal for secure data processing in high-performance environments.

## 🚀 Features

- **2048-bit AES encryption** (256 bytes) in CBC mode
- **CUDA-accelerated key generation** using PyTorch
- **Base64 encoding** for encrypted output
- **Padding/unpadding** via PKCS7 for block alignment
- **Quantum-resilient design** for future-proof security

## 📦 Dependencies

## 🧠 How It Works

Use this repo folder for all files:

### Key Concepts

- **AES-2048**: While AES officially supports 128, 192, and 256-bit keys, this implementation uses a 2048-bit key for experimental quantum-resistance.
- **CBC Mode**: Cipher Block Chaining ensures each block depends on the previous one, enhancing security.
- **CUDA Support**: Key generation uses GPU acceleration when available.

## 🧪 Example Usage

```python
from AES2048Encryptor import AES2048Encryptor

encryptor = AES2048Encryptor()
data = b"Connection Machine 2048-AES Data"

# Encrypt
encrypted = encryptor.encrypt(data)

# Decrypt
decrypted = encryptor.decrypt(encrypted)

# Generate CUDA-based key
cuda_key = encryptor.generate_key_cuda(seed=42)

print(f"Original: {data}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
print(f"CUDA-generated key: {cuda_key[:16]}...")
```

## ⚠️ Notes

- AES officially supports up to 256-bit keys. This 2048-bit implementation is experimental and not NIST-standard.
- CUDA acceleration requires a compatible GPU and PyTorch with CUDA support.

## 🛡️ License

MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Pull requests welcome! If you have ideas for improving quantum resilience or optimizing CUDA performance, feel free to contribute.

---

Would you like a logo or badge design for this repo? I can generate one to match your retro cyber aesthetic.
