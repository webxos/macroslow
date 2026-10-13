UNDER DEVELOPMENT "AUTH 3.0" RSA-2048


Because RSA-2048 is computationally slow and can only process limited amounts of data directly, real-world systems combine them (commonly called RSA-SHA256 or RS256):
1. Hashing: The original message is fed into SHA-256 to create a short, fixed-size 32-byte digest.
2. Signing/Encrypting: The RSA-2048 private key then encrypts that small hash (creating a signature) rather than the massive raw file
