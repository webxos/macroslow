UNDER DEVELOPMENT "AUTH 3.0" RSA-2048 - For Testing

## What is this?

**SI Auth** is a reference implementation of an authentication scheme where the
browser generates an **RSA-2048 keypair**. The private key never leaves the
device. Login is a challenge signed client-side and verified server-side
against a stored public key.

No passwords. No shared secrets on the wire. No hashes in your database.
