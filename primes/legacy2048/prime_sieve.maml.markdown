---
maml_version: "2.0"
id: "urn:uuid:123e4567-e89b-12d3-a456-426614174000"
type: "quantum-legacy-workflow"
encryption: "AES-2048"
requires: ["fortran-256aes", "c64-512aes", "amoeba-1024aes", "cm-2048aes"]
parameters:
  limit: 1000000
  output_pattern: "prime_sieve"
mu_validation_file: "prime_sieve_validation.mu.md"
---
# Prime Sieve Workflow
## Intent
Process a numerical range through the Legacy 2048 AES SDK for prime sieving, using CUDA and quantum logic.

## Context
input: range(1, ${parameters.limit})
output: /data/output/${parameters.output_pattern}.enc

## Execution Plan
1. mode: fortran-256aes
   operation: initialize_quantum_array
   input: ${input}
   output: /tmp/phase1.bin

2. mode: c64-512aes
   operation: analyze_pattern
   input: /tmp/phase1.bin
   output: /tmp/phase2.bin

3. mode: amoeba-1024aes
   operation: distribute_and_backup
   input: /tmp/phase2.bin
   backup_driver: dropbox_api://backups/
   output: /tmp/phase3.bin

4. mode: cm-2048aes
   operation: sieve_eratosthenes
   input: /tmp/phase3.bin
   output: ${output}