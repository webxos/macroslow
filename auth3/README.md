## What is this?

**SI Auth** is a reference implementation of an authentication scheme where the
browser generates an **RSA-2048 keypair**. The private key never leaves the
device. Login is a challenge signed client-side and verified server-side
against a stored public key.

No passwords. No shared secrets on the wire. No hashes in your database.

## Features

| | |
|---|---|
| 🔐 **RSA-2048 challenge–response** | RSASSA-PKCS1-v1_5 + SHA-256. |
| ⚡ **Rust accelerator** | Optional N-API module, ~4× faster verify. Auto-detected. |
| 🧠 **Non-extractable keys** | `extractable: false` in the browser. XSS can't exfiltrate. |
| 🎣 **Phishing resistant** | Signed payload binds server origin. |
| ♻️ **Replay proof** | Single-use, TTL-bound nonces. |
| 🍪 **HttpOnly sessions** | JWT in HttpOnly, SameSite=Strict cookie. |
| 📱 **Multi-device** | Per-device credentials, individually revocable. |
| 🚀 **One-file launcher** | `python3 auth3.py` scaffolds + runs everything. |

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
├── auth3.py                    # ⭐ one-file mega-launcher
├── src/                        # TypeScript server
│   ├── config.ts
│   ├── app.ts
│   ├── server.ts
│   ├── lib/{{crypto,cookies,logger}}.ts
│   ├── middleware/{{errorHandler,rateLimit,requestLogger,requireAuth}}.ts
│   ├── routes/{{auth,health}}.ts
│   └── services/{{auditLog,challengeStore,token,userStore}}.ts
├── rust/                       # ⭐ N-API crypto accelerator
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
""", force=force)

    wfile(root, "docs/PROTOCOL.md", """# SI Auth Protocol Specification

**Version:** `SI-AUTH-V1`
**Signature algorithm:** RSASSA-PKCS1-v1_5 with SHA-256
**Key size:** RSA-2048 minimum

## 1. Canonical signing payload

UTF-8, joined with `\\n` (0x0A), no trailing newline:

```
SI-AUTH-V1
origin:<origin>
purpose:<purpose>
challenge:<challenge>
username:<username>
issuedAt:<issuedAt>
```

| Field | Notes |
|---|---|
| `origin` | Absolute origin, no trailing slash. |
| `purpose` | `login` or `register`. |
| `challenge` | 32-byte CSPRNG value, hex. |
| `username` | Lowercased, trimmed. |
| `issuedAt` | ISO 8601 UTC with milliseconds. |

## 2. Flow

### 2.1 Registration challenge
`POST /api/auth/register/challenge` → `{ username }`
→ `{ challengeId, challenge, origin, purpose, username, issuedAt, expiresInSeconds }`

### 2.2 Registration complete
`POST /api/auth/register/complete` → `{ challengeId, username, publicKeySpki, signature, deviceLabel? }`
→ `201` + `Set-Cookie: si_auth_session=…; HttpOnly; SameSite=Strict`

### 2.3 Login challenge
Same as 2.1 with `purpose: "login"` and a 60 s TTL.

### 2.4 Login verify
`POST /api/auth/login/verify` → `{ challengeId, username, signature }`
→ `200` + session cookie.

## 3. Server verification

1. Look up `challengeId`. If missing → 401.
2. **Immediately delete** the nonce (single-use). If expired → 401.
3. Check `purpose` + `username` match. Else 401.
4. Rebuild the canonical payload from the cached record.
5. For each credential: verify signature. First match wins.
6. No match → 401.

## 4. Nonces

- `crypto.randomBytes(32)` (or Rust `rand::thread_rng`), hex-encoded.
- Deleted on first use regardless of outcome.
- Default TTL: 60 s (login), 300 s (register).

## 5. Session token

HS256 JWT with claims:

```json
{ "sub": "user_…", "username": "me@example.com", "iat": 1700000000, "exp": 1700086400, "iss": "si-auth" }
```

Transported in an `HttpOnly`, `SameSite=Strict`, `Path=/` cookie. `Secure`
when `NODE_ENV=production`.
""", force=force)

    wfile(root, "docs/API.md", """# API Reference

Base: `{APP_ORIGIN}/api`. All bodies are JSON. Errors: `{ "error": "…" }`.

## `GET /api/health`

```json
{ "status": "ok", "uptime": 42, "env": "production", "protocol": "SI-AUTH-V1",
  "rust": { "enabled": true, "version": "siauth-crypto/0.1.0" } }
```

## `GET /api/auth/config`

```json
{ "protocol": "SI-AUTH-V1", "algorithm": "RSASSA-PKCS1-v1_5", "hash": "SHA-256",
  "modulusLength": 2048, "origin": "https://app.example.com",
  "challengeTtlSeconds": 60, "registrationChallengeTtlSeconds": 300 }
```

## `POST /api/auth/register/challenge`

Body: `{ "username": "me@example.com" }`
- `200` — challenge object
- `400` — invalid username
- `409` — already registered
- `429` — rate limited

## `POST /api/auth/register/complete`

Body: `{ challengeId, username, publicKeySpki, signature, deviceLabel? }`
- `201` — `{ user, credential }` + cookie
- `400` — validation failure
- `401` — bad signature
- `409` — race duplicate

## `POST /api/auth/login/challenge`

Body: `{ "username": "me@example.com" }`
- `200` — challenge (identical for known & unknown users)

## `POST /api/auth/login/verify`

Body: `{ challengeId, username, signature }`
- `200` — `{ user, credential }` + cookie
- `401` — `{ "error": "Authentication failed" }` (no detail leaks)

## `GET /api/auth/me` 🔒

```json
{ "user": { "id": "user_…", "username": "me@example.com", "createdAt": "…" },
  "session": { "iat": 1700000000, "exp": 1700086400 } }
```

## `GET /api/auth/credentials` 🔒

```json
{ "credentials": [ { "id": "cred_…", "label": "Chrome on macOS",
                     "fingerprint": "<64 hex>", "createdAt": "…", "lastUsedAt": "…" } ] }
```

## `DELETE /api/auth/credentials/:id` 🔒

- `200` — `{ ok: true }`
- `400` — cannot remove last credential
- `404` — not found

## `POST /api/auth/logout`

- `200` — `{ ok: true }` + cookie cleared
""", force=force)

    wfile(root, "docs/ARCHITECTURE.md", """# Architecture

## Module map

```
src/
├── config.ts              # env parsing → frozen config
├── server.ts              # HTTP listener + graceful shutdown
├── app.ts                 # express app factory (testable)
├── lib/
│   ├── crypto.ts          # ⭐ canonical payload + verify (Rust-first)
│   ├── cookies.ts         # parse/serialize Set-Cookie
│   └── logger.ts          # JSON-lines structured logging
├── middleware/
│   ├── errorHandler.ts
│   ├── rateLimit.ts
│   ├── requestLogger.ts
│   └── requireAuth.ts
├── routes/
│   ├── auth.ts            # all /api/auth endpoints
│   └── health.ts
└── services/
    ├── auditLog.ts
    ├── challengeStore.ts  # in-memory nonce store (TTL + single-use)
    ├── token.ts           # HS256 JWT (node:crypto only)
    └── userStore.ts       # JSON-file persistence (swappable)

rust/
└── src/lib.rs             # N-API accelerator (verify, generate, version)
```

## Crypto dispatch

`src/lib/crypto.ts` prefers the Rust N-API module and falls back to
`node:crypto`:

```ts
if (rust) return rust.verifySignature(spki, payload, sig);
return crypto.verify("sha256", Buffer.from(payload), { … }, Buffer.from(sig, "base64"));
```

Control via `SI_AUTH_USE_RUST`:
- `auto` (default) — use Rust if loadable, else fall back.
- `true` — require Rust; throw at boot if missing.
- `false` — always use `node:crypto`.

## Swapping storage

`userStore` (8 methods) and `challengeStore` (5 methods) are deliberately
small. Reimplement against Postgres/Mongo/Redis without touching routes.

### Redis challengeStore sketch

```ts
async issue(opts) {
  const id = generateId("ch");
  await redis.set(`ch:${id}`, JSON.stringify(record), "PX", opts.ttlSeconds * 1000, "NX");
  return record;
}
async take(id) {
  const raw = await redis.getdel(`ch:${id}`);
  return raw ? JSON.parse(raw) : null;
}
```

## Client crypto core

`public/js/si-auth.js` is dependency-free Web Crypto. Drop it into React,
Vue, Svelte, or a native webview. It exposes:

- `generateKeyPair()`
- `exportPublicKeySpki(publicKey)`
- `importPublicKeySpki(spkiBase64)`
- `signPayload(privateKey, payloadString)`
- `verifyPayload(publicKey, payloadString, signatureBase64)`
- `buildSigningPayload(fields)`
- `fingerprintPublicKey(spkiBase64)`
""", force=force)

    wfile(root, "docs/SECURITY-MODEL.md", """# Security Model

## Threat model

| Threat | Mitigation |
|---|---|
| Credential stuffing | No shared credentials exist. |
| Server DB breach | Only public keys stored; not replayable. |
| Password phishing | Signed payload binds `origin`. A phishing site cannot produce a valid signature. |
| Replay | Single-use, TTL-bound nonces. |
| XSS on auth origin | Keys are non-extractable `CryptoKey`s. XSS can *use* them while session is live, but cannot export them. |
| Username enumeration | Identical challenges and generic 401s. |
| Token theft via JS | Session JWT is `HttpOnly`. |
| CSRF | `SameSite=Strict`; state-changing endpoints require JSON body. |
| MITM | TLS mandatory in production. |
| Weak keys | Server rejects < 2048-bit RSA. |

## Residual risks

1. XSS while session is active — mitigated by CSP `script-src 'self'`.
2. Device compromise — filesystem access can *use* the key but not export it.
3. No key recovery — pre-register a second device or ship a recovery flow.
4. Stateless JWT — keep TTLs short; add a revocation list if needed.

## Comparison to WebAuthn

WebAuthn is the industry standard: hardware-backed keys, browser-enforced
origin binding, attestation. Use it if you can.

SI Auth provides: self-hosted, dependency-free, ~400 lines audit-able in an
afternoon. Teaching tool / stepping stone.

**Do not roll your own crypto for high-value systems without professional audit.**

## Deployment checklist

- [ ] `JWT_SECRET` ≥ 32 random bytes, rotated periodically.
- [ ] `APP_ORIGIN` matches exactly.
- [ ] TLS + `COOKIE_SECURE=true`.
- [ ] `TRUST_PROXY=true` behind a proxy.
- [ ] `NODE_ENV=production`.
- [ ] Rate limits tuned.
- [ ] Audit log shipped to durable storage.
- [ ] Backup/restore tested.
- [ ] Recovery flow documented.
""", force=force)

    wfile(root, "docs/DEPLOYMENT.md", """# Deployment

## Docker

```bash
git clone <repo> && cd si-auth
JWT_SECRET="$(node -e 'console.log(require("crypto").randomBytes(48).toString("hex"))')" \\
  docker compose up -d --build
```

## Bare VPS (systemd)

```bash
sudo useradd -r -m -d /opt/si-auth si-auth
sudo -u si-auth git clone <repo> /opt/si-auth
cd /opt/si-auth
sudo -u si-auth npm install --omit=dev
sudo -u si-auth cp .env.example .env
# edit .env — set JWT_SECRET, APP_ORIGIN, NODE_ENV=production
```

`/etc/systemd/system/si-auth.service`:

```ini
[Unit]
Description=SI Auth
After=network.target

[Service]
Type=simple
User=si-auth
WorkingDirectory=/opt/si-auth
EnvironmentFile=/opt/si-auth/.env
ExecStart=/usr/bin/npx tsx src/server.ts
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

## Caddy reverse proxy

```
auth.example.com {
  reverse_proxy 127.0.0.1:3000
  encode zstd gzip
}
```

Set `TRUST_PROXY=true`.

## Scaling

Replace `challengeStore` with Redis and `userStore` with a real database.
Share `JWT_SECRET` across replicas.

## Health

`GET /api/health` for liveness + readiness. Includes Rust status.
""", force=force)

    wfile(root, "docs/FAQ.md", """# FAQ

### Why not WebAuthn?

Use it if you can. SI Auth is for teams wanting a self-hosted,
dependency-free, fully auditable implementation — or a teaching tool.

### I lost my device — can I recover?

Not by default. The server has no secret it could use to impersonate you.
Pre-register a second device or ship a recovery flow.

### Can I export my private key?

No. Keys are generated with `extractable: false`. Register a second device
to have a backup.

### Does this work in Safari / Firefox?

Yes. Web Crypto, IndexedDB, and CryptoKey structured-clone are supported in
every modern browser. **HTTPS or localhost is required.**

### Why not `localStorage` for the token?

Because it's readable by any script. An XSS can steal it. HttpOnly cookies
cannot be read by JavaScript.

### Native mobile?

iOS: `SecKeyCreateRandomKey` + Keychain, sign with `SecKeyCreateSignature(…, kSecKeyAlgorithmRSASignatureMessagePKCS1v15SHA256, …)`.
Android: `KeyPairGenerator("RSA")` + Android Keystore, `Signature.getInstance("SHA256withRSA")`.
Post the same payload to the same endpoints.

### What happens if two users pick the same username?

The second registration receives `409 Conflict`.

### How do I rotate `JWT_SECRET`?

Roll it — active sessions invalidate. Users sign in again with their existing
keys. No re-registration needed.

### What does the Rust accelerator do?

Verifies RSA-2048 signatures in the hot path. It's optional — the server works
fine without it (`SI_AUTH_USE_RUST=false`).
""", force=force)
