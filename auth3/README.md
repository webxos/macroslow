# AUTH 3.0 (aka SI AUTH) v2.01 updated 10/13/2026 at 6:00 pm EST

Preface:
RSA-512 keys have been practically breakable since 1999 and factorable on ordinary hardware for years. The Claude story: in September 2026 Anthropic engineer Steve Weis used Claude to port CADO-NFS to GPUs, then ran it on up to 2048 idle GPUs for ~10 days (~30 GPU-years) to factor the RSA-896 challenge number. No algorithmic advance, just AI-assisted engineering. RSA-1024 is now reachable for large GPU fleets; RSA-2048 is not.

**SI Auth** is a reference implementation of an authentication scheme where the
browser generates an **RSA-2048 keypair**. The private key never leaves the
device. Login is a challenge signed client-side and verified server-side
against a stored public key.

No passwords. No shared secrets on the wire. No hashes in your database.

## Features

| | |
|---|---|
|  **RSA-2048 challenge–response** | RSASSA-PKCS1-v1_5 + SHA-256. |
|  **Rust accelerator** | Optional N-API module, ~4× faster verify. Auto-detected. |
|  **Non-extractable keys** | `extractable: false` in the browser. XSS can't exfiltrate. |
|  **Phishing resistant** | Signed payload binds server origin. |
|  **Replay proof** | Single-use, TTL-bound nonces. |
|  **HttpOnly sessions** | JWT in HttpOnly, SameSite=Strict cookie. |
|  **Multi-device** | Per-device credentials, individually revocable. |
|  **One-file launcher** | `python3 auth3.py` scaffolds + runs everything. |

## Quickstart

### Option A — the mega-launcher

```bash
python3 auth3.py si-auth/
```

Detects your OS, installs Node 20 + Rust (if missing), scaffolds the tree,
runs `npm install`, compiles the Rust accelerator, and launches the server.

### Option B — plain Node

```bash
cp .env.example .env
npm install
npm run dev
```

Open <http://localhost:3000>.

### Verify it works

```bash
npm test          # unit + E2E protocol tests
npm run typecheck # tsc --noEmit
npm run smoke     # against a running server
```

## How it works

```
Browser                                     Server
   │                                           │
   │  POST /api/auth/login/challenge           │
   │  {{ username }}                           │
   │──────────────────────────────────────────▶│
   │                                           │  generate 32-byte nonce
   │  {{ challengeId, challenge, origin, ...}} │  cache w/ TTL
   │◀──────────────────────────────────────────│
   │                                           │
   │  sign canonical payload with              │
   │  non-extractable CryptoKey                │
   │                                           │
   │  POST /api/auth/login/verify              │
   │  {{ challengeId, username, signature }}   │
   │──────────────────────────────────────────▶│
   │                                           │  Rust: verify fast
   │                                           │  Node: verify fallback
   │  200 {{ user, credential }} + Set-Cookie  │
   │◀──────────────────────────────────────────│
```

### Canonical signing payload

```
SI-AUTH-V1
origin:https://app.example.com
purpose:login
challenge:9f2c…e41a
username:me@example.com
issuedAt:2025-01-01T12:00:00.000Z
```

Both sides rebuild this from their own state. Tampering is impossible.

## Project structure

```
si-auth/
├── auth3.py                    #  one-file mega-launcher
├── src/                        # TypeScript server
│   ├── config.ts
│   ├── app.ts
│   ├── server.ts
│   ├── lib/{{crypto,cookies,logger}}.ts
│   ├── middleware/{{errorHandler,rateLimit,requestLogger,requireAuth}}.ts
│   ├── routes/{{auth,health}}.ts
│   └── services/{{auditLog,challengeStore,token,userStore}}.ts
├── rust/                       # N-API crypto accelerator
│   ├── Cargo.toml
│   ├── build.rs
│   └── src/lib.rs
├── public/                     # static frontend (zero build)
│   ├── css/styles.css
│   ├── js/{{api,dashboard,keystore,login,register,si-auth}}.js
│   └── {{index,register,dashboard}}.html
├── test/                       # node:test suites
├── scripts/smoke-test.ts
├── docs/
├── Dockerfile + docker-compose.yml
└── package.json
```

## Configuration

All via environment. See [`.env.example`](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `NODE_ENV` | `development` | `production` enables HSTS + strict secret validation |
| `PORT` | `3000` | |
| `APP_ORIGIN` | `http://localhost:3000` | **Must match browser origin exactly.** |
| `JWT_SECRET` | *(random in dev)* | **Required in production**, min 32 chars |
| `SESSION_TTL_SECONDS` | `86400` | |
| `CHALLENGE_TTL_SECONDS` | `60` | |
| `REGISTRATION_CHALLENGE_TTL_SECONDS` | `300` | |
| `DATA_DIR` | `./data` | |
| `COOKIE_SECURE` | `true` in prod | |
| `COOKIE_SAMESITE` | `strict` | |
| `TRUST_PROXY` | `false` | |
| `USER_ENUMERATION_SAFE` | `true` | |
| `RATE_LIMIT_MAX` | `30` | |
| `LOG_LEVEL` | `info` | |
| `SI_AUTH_USE_RUST` | `auto` | `auto`/`true`/`false` |

## Rust accelerator

`rust/` is a `napi-rs` crate that exposes:

```ts
verifySignature(spkiB64: string, payload: string, sigB64: string): boolean
publicKeyBits(spkiB64: string): number
generateChallenge(): string
rustVersion(): string
```

The TypeScript server imports it opportunistically:

```ts
try { rust = require("siauth-crypto"); } catch {{ /* fall back to node:crypto */ }}
```

Force the fallback with `SI_AUTH_USE_RUST=false`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`  | `/api/auth/config` | Public protocol parameters |
| `POST` | `/api/auth/register/challenge` | Get a registration nonce |
| `POST` | `/api/auth/register/complete` | Submit public key + signature |
| `POST` | `/api/auth/login/challenge` | Get a login nonce |
| `POST` | `/api/auth/login/verify` | Submit signature, receive session |
| `GET`  | `/api/auth/me` | Current user profile 🔒 |
| `GET`  | `/api/auth/credentials` | List devices 🔒 |
| `DELETE` | `/api/auth/credentials/:id` | Revoke a device 🔒 |
| `POST` | `/api/auth/logout` | Clear session |
| `GET`  | `/api/health` | Liveness (incl. Rust status) |

See [`docs/API.md`](docs/API.md).

## Security notes

See [`docs/SECURITY-MODEL.md`](docs/SECURITY-MODEL.md). Headlines:

* ✅ No secrets at rest on the server — only public keys.
* ✅ No secrets in transit — signatures are not reusable.
* ✅ Non-extractable private keys — XSS can't exfiltrate them.
* ✅ Origin-bound signatures — a relayed challenge is useless.
* ⚠️ TLS is still mandatory.
* ⚠️ This is a reference implementation. Get it audited.

### Why not WebAuthn?

Use WebAuthn if you can — it's the gold standard. SI Auth is for teams who
want a self-hosted, dependency-free, fully auditable protocol they can read in
an afternoon.

## Docker

```bash
docker compose up --build
```

## License

MIT
