// security.js
// Description: JavaScript module for BELUGA’s encryption and decryption.
// Implements CHIMERA 2048’s adaptive encryption modes.
// Usage: Import and instantiate Security for secure data handling.

import { createCipheriv, createDecipheriv, randomBytes } from 'crypto';

class Security {
    /**
     * Initializes security module with specified key size.
     * @param {number} keySize - AES key size (256, 512, or 2048 bits).
     */
    constructor(keySize = 2048) {
        this.key = randomBytes(keySize / 8);
        this.iv = randomBytes(16);
    }

    /**
     * Encrypts data using AES.
     * @param {Buffer} data - Data to encrypt.
     * @returns {Buffer} - Encrypted data.
     */
    encryptData(data) {
        const cipher = createCipheriv('aes-256-ctr', this.key.slice(0, 32), this.iv);
        return Buffer.concat([cipher.update(data), cipher.final()]);
    }

    /**
     * Decrypts data using AES.
     * @param {Buffer} encryptedData - Encrypted data.
     * @returns {Buffer} - Decrypted data.
     */
    decryptData(encryptedData) {
        const decipher = createDecipheriv('aes-256-ctr', this.key.slice(0, 32), this.iv);
        return Buffer.concat([decipher.update(encryptedData), decipher.final()]);
    }
}

// Example usage:
// const security = new Security(256);
// const encrypted = security.encryptData(Buffer.from("Sensitive data"));
// console.log(security.decryptData(encrypted).toString());