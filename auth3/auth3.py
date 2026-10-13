#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auth3.py — SI Auth mega-launcher
================================
One file. Detects your system. Installs Node + (optionally) Rust.
Scaffolds a full TypeScript + Rust-N-API + static-frontend SI Auth stack.
Runs it.

Usage
-----
    python3 auth3.py                # scaffold ./si-auth/ and launch
    python3 auth3.py my-app/        # custom target dir
    python3 auth3.py --no-rust      # skip Rust compile (pure-TS fallback)
    python3 auth3.py --no-install   # scaffold only, don't npm install
    python3 auth3.py --force        # overwrite existing files
    python3 auth3.py --port 8080    # override server port

Author: SI Auth Contributors · MIT
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

VERSION = "1.0.0"
NODE_MIN_MAJOR = 18
NODE_MIN_MINOR = 17
DEFAULT_TARGET = "si-auth"
DEFAULT_PORT = 3000

# ANSI color (disable if not a TTY)
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s

C_OK   = lambda s: _c("32;1", s)
C_ERR  = lambda s: _c("31;1", s)
C_WARN = lambda s: _c("33;1", s)
C_INFO = lambda s: _c("36;1", s)
C_BOLD = lambda s: _c("1", s)
C_DIM  = lambda s: _c("2", s)

def banner() -> None:
    print()
    print(C_BOLD("  ╔══════════════════════════════════════════════════════╗"))
    print(C_BOLD("  ║") + "  " + C_INFO("SI Auth") + "  ·  RSA-2048 challenge–response auth    " + C_BOLD("║"))
    print(C_BOLD("  ║") + "  " + C_DIM(f"mega-launcher v{VERSION}  ·  python + typescript + rust") + "  " + C_BOLD("║"))
    print(C_BOLD("  ╚══════════════════════════════════════════════════════╝"))
    print()

def log(msg: str, kind: str = "info") -> None:
    prefix = {
        "info":  C_INFO("  ▸"),
        "ok":    C_OK("  ✓"),
        "warn":  C_WARN("  !"),
        "err":   C_ERR("  ✗"),
        "step":  C_BOLD("  ▶"),
    }.get(kind, "  ·")
    print(f"{prefix} {msg}", flush=True)

def die(msg: str, code: int = 1) -> "NoReturn":
    log(msg, "err")
    sys.exit(code)

# ═══════════════════════════════════════════════════════════════════════════
#  System detection
# ═══════════════════════════════════════════════════════════════════════════

class System:
    def __init__(self) -> None:
        self.os       = platform.system().lower()          # linux/darwin/windows
        self.arch     = platform.machine().lower()         # x86_64/aarch64/arm64
        self.distro   = self._detect_distro()
        self.is_root  = self._is_root()
        self.pkg      = self._detect_pkg_mgr()
        self.shell    = os.environ.get("SHELL", "/bin/sh")
        self.node     = shutil.which("node")
        self.npm      = shutil.which("npm")
        self.cargo    = shutil.which("cargo")
        self.curl     = shutil.which("curl")
        self.wget     = shutil.which("wget")
        self.git      = shutil.which("git")

    def _detect_distro(self) -> str:
        if platform.system().lower() != "linux":
            return platform.system().lower()
        try:
            txt = Path("/etc/os-release").read_text()
            for line in txt.splitlines():
                if line.startswith("ID="):
                    return line.split("=", 1)[1].strip().strip('"').lower()
        except Exception:
            pass
        return "linux"

    def _is_root(self) -> bool:
        try:
            return os.geteuid() == 0
        except AttributeError:
            return False

    def _detect_pkg_mgr(self) -> str | None:
        for mgr, hint in [
            ("apt-get", "debian"), ("dnf", "fedora"), ("yum", "rhel"),
            ("pacman", "arch"),   ("apk", "alpine"), ("brew", "mac"),
            ("pkg", "bsd"),       ("zypper", "suse"),
        ]:
            if shutil.which(mgr):
                return mgr
        return None

    def describe(self) -> None:
        log(f"OS     : {C_BOLD(self.os)} / {C_BOLD(self.distro)} / {C_BOLD(self.arch)}")
        log(f"Root   : {self.is_root}")
        log(f"Pkg    : {self.pkg or C_WARN('none detected')}")
        log(f"Node   : {self.node or C_WARN('not found')}")
        log(f"npm    : {self.npm  or C_WARN('not found')}")
        log(f"Cargo  : {self.cargo or C_WARN('not found (optional)')}")


SYS = System()

# ═══════════════════════════════════════════════════════════════════════════
#  Privilege helpers
# ═══════════════════════════════════════════════════════════════════════════

def run(cmd: list[str] | str, *, check: bool = True, shell: bool = False,
        env: dict | None = None, cwd: Path | None = None,
        quiet: bool = False) -> subprocess.CompletedProcess:
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
    if not quiet:
        log(C_DIM(f"$ {display}"))
    return subprocess.run(
        cmd, check=check, shell=shell, env=env, cwd=cwd,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
        text=True,
    )

def sudo(cmd: list[str]) -> list[str]:
    if SYS.is_root or SYS.os == "windows":
        return cmd
    if shutil.which("sudo"):
        return ["sudo", "-E"] + cmd
    return cmd

# ═══════════════════════════════════════════════════════════════════════════
#  Dependency installation
# ═══════════════════════════════════════════════════════════════════════════

def node_version_ok() -> bool:
    if not SYS.node:
        return False
    try:
        out = subprocess.check_output([SYS.node, "--version"], text=True).strip()
        # v20.11.0
        parts = out.lstrip("v").split(".")
        major, minor = int(parts[0]), int(parts[1])
        return (major, minor) >= (NODE_MIN_MAJOR, NODE_MIN_MINOR)
    except Exception:
        return False

def install_node() -> None:
    log("Installing Node.js…", "step")

    # ── 1. Try nvm (works everywhere, no root) ─────────────────────────
    nvm_dir = Path.home() / ".nvm"
    if not (nvm_dir / "nvm.sh").exists() and SYS.curl:
        log("  fetching nvm installer…")
        try:
            run("curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash",
                shell=True, quiet=True)
        except subprocess.CalledProcessError:
            log("  nvm install failed, falling back to system package manager", "warn")

    if (nvm_dir / "nvm.sh").exists():
        log("  using nvm to install Node 20…")
        cmd = (
            f'source "{nvm_dir}/nvm.sh" && '
            f'nvm install 20 && nvm use 20 && nvm alias default 20'
        )
        try:
            run(["bash", "-lc", cmd])
            # Re-resolve node path
            nvm_node = nvm_dir / "versions" / "node"
            if nvm_node.exists():
                versions = sorted(nvm_node.iterdir(), reverse=True)
                if versions:
                    bin_dir = versions[0] / "bin"
                    os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
                    SYS.node = str(bin_dir / "node")
                    SYS.npm  = str(bin_dir / "npm")
            if node_version_ok():
                log(f"  Node {subprocess.check_output([SYS.node,'--version'],text=True).strip()} installed via nvm", "ok")
                return
        except subprocess.CalledProcessError:
            log("  nvm install failed", "warn")

    # ── 2. System package manager ──────────────────────────────────────
    mgr = SYS.pkg
    if not mgr:
        die("No way to install Node.js. Please install Node 18.17+ manually.")

    if mgr == "apt-get":
        run(sudo(["apt-get", "update", "-qq"]))
        run(sudo(["apt-get", "install", "-y", "nodejs", "npm", "curl", "ca-certificates"]))
    elif mgr in ("dnf", "yum"):
        run(sudo([mgr, "install", "-y", "nodejs", "npm", "curl"]))
    elif mgr == "pacman":
        run(sudo(["pacman", "-Sy", "--noconfirm", "nodejs", "npm", "curl"]))
    elif mgr == "apk":
        run(sudo(["apk", "add", "--no-cache", "nodejs", "npm", "curl"]))
    elif mgr == "brew":
        run(["brew", "install", "node"])
    elif mgr == "pkg":
        run(sudo(["pkg", "install", "-y", "node", "npm"]))
    elif mgr == "zypper":
        run(sudo(["zypper", "--non-interactive", "install", "nodejs20", "npm20"]))
    else:
        die(f"Unsupported package manager: {mgr}")

    # Re-resolve
    SYS.node = shutil.which("node")
    SYS.npm  = shutil.which("npm")
    if not node_version_ok():
        die("Node.js install failed or version too old. Need 18.17+.")

def install_rust() -> bool:
    """Install Rust via rustup. Returns True on success."""
    if SYS.cargo and shutil.which("rustc"):
        log("Rust already installed.", "ok")
        return True
    log("Installing Rust (rustup)…", "step")
    if not SYS.curl:
        log("curl missing; cannot install Rust. Skipping.", "warn")
        return False
    try:
        run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal",
            shell=True,
        )
        cargo_bin = Path.home() / ".cargo" / "bin"
        os.environ["PATH"] = f"{cargo_bin}:{os.environ['PATH']}"
        SYS.cargo = str(cargo_bin / "cargo") if (cargo_bin / "cargo").exists() else shutil.which("cargo")
        if SYS.cargo:
            log(f"Rust installed: {subprocess.check_output([SYS.cargo,'--version'],text=True).strip()}", "ok")
            return True
        return False
    except subprocess.CalledProcessError as e:
        log(f"Rust install failed: {e}", "warn")
        return False

def ensure_apt_build_deps() -> None:
    """Ensure build tools needed by napi-rs are present on Linux."""
    if SYS.os != "linux" or SYS.pkg not in ("apt-get",):
        return
    needed = ["build-essential", "pkg-config", "libssl-dev", "ca-certificates", "curl"]
    log("Ensuring build toolchain (build-essential, libssl-dev)…")
    try:
        run(sudo(["apt-get", "update", "-qq"]), quiet=True)
        run(sudo(["apt-get", "install", "-y", "-qq"] + needed), quiet=True)
    except subprocess.CalledProcessError as e:
        log(f"build deps install failed (continuing): {e}", "warn")

# ═══════════════════════════════════════════════════════════════════════════
#  File writer
# ═══════════════════════════════════════════════════════════════════════════

WRITTEN: list[str] = []
SKIPPED: list[str] = []

def wfile(root: Path, rel: str, content: str, *, force: bool = False,
          executable: bool = False) -> None:
    path = root / rel
    if path.exists() and not force:
        SKIPPED.append(rel)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure a trailing newline, but don't double it
    if content and not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")
    if executable:
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    WRITTEN.append(rel)
    print(f"    {C_OK('+')} {rel}")

# ═══════════════════════════════════════════════════════════════════════════
#  Scaffolding — root-level files
# ═══════════════════════════════════════════════════════════════════════════

def build_root_files(root: Path, *, with_rust: bool, port: int, force: bool) -> None:
    log("Root files", "step")

    deps = {
        "express": "^4.19.2",
        "tsx": "^4.7.0",
        "typescript": "^5.4.0",
        "@types/express": "^4.17.21",
        "@types/node": "^20.11.0",
    }
    if with_rust:
        deps["siauth-crypto"] = "file:./rust"

    pkg = {
        "name": "si-auth",
        "version": VERSION,
        "description": "Passwordless, phishing-resistant authentication using RSA-2048 challenge–response (TypeScript + Rust).",
        "type": "module",
        "engines": {"node": ">=18.17"},
        "scripts": {
            "start": "tsx src/server.ts",
            "dev": "tsx watch src/server.ts",
            "test": "node --test --import tsx test/",
            "smoke": "tsx scripts/smoke-test.ts",
            "typecheck": "tsc --noEmit",
        },
        "keywords": ["authentication", "rsa", "passwordless", "challenge-response", "typescript", "rust"],
        "license": "MIT",
        "dependencies": deps,
    }
    wfile(root, "package.json", json.dumps(pkg, indent=2), force=force)

    wfile(root, "tsconfig.json", """{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "lib": ["ES2022"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "allowImportingTsExtensions": true,
    "noEmit": true,
    "types": ["node"],
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*.ts", "test/**/*.ts", "scripts/**/*.ts"]
}
""", force=force)

    wfile(root, ".nvmrc", "20\n", force=force)

    wfile(root, ".gitignore", """# Dependencies
node_modules/
package-lock.json
pnpm-lock.yaml

# Build artifacts
dist/
target/
rust/*.node
rust/index.js
rust/index.d.ts
rust/*.d.ts

# Runtime state
data/*.json
data/*.log
data/*.tmp
!data/.gitkeep

# Environment
.env
.env.*
!.env.example

# Logs
*.log
npm-debug.log*

# OS / Editors
.DS_Store
Thumbs.db
.vscode/
.idea/
*.swp
""", force=force)

    wfile(root, ".dockerignore", """node_modules
data
target
rust/target
.git
.github
*.md
.env
.env.*
!.env.example
npm-debug.log
coverage
.DS_Store
""", force=force)

    wfile(root, ".env.example", f"""# ─── Server ─────────────────────────────────────────────────────────
NODE_ENV=development
PORT={port}
APP_ORIGIN=http://localhost:{port}
TRUST_PROXY=false

# ─── Secrets ────────────────────────────────────────────────────────
# REQUIRED in production. Minimum 32 characters.
JWT_SECRET=

# ─── Sessions ───────────────────────────────────────────────────────
SESSION_TTL_SECONDS=86400

# ─── Challenges ─────────────────────────────────────────────────────
CHALLENGE_TTL_SECONDS=60
REGISTRATION_CHALLENGE_TTL_SECONDS=300

# ─── Storage ────────────────────────────────────────────────────────
DATA_DIR=./data

# ─── Cookies ────────────────────────────────────────────────────────
COOKIE_SAMESITE=strict

# ─── Security ───────────────────────────────────────────────────────
USER_ENUMERATION_SAFE=true
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_MAX=30

# ─── Logging ────────────────────────────────────────────────────────
LOG_LEVEL=info

# ─── Rust accelerator ───────────────────────────────────────────────
# Set to "true" to force use of the Rust N-API crypto core (requires
# a successful `cargo build` in rust/). Auto-detected otherwise.
SI_AUTH_USE_RUST=auto
""", force=force)

    wfile(root, "data/.gitkeep", "", force=force)

    wfile(root, "LICENSE", f"""MIT License

Copyright (c) 2025 SI Auth Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""", force=force)

    wfile(root, "SECURITY.md", """# Security Policy

## Reporting a vulnerability

**Do not open a public issue.** Email `security@example.com` with a
description, reproduction, and affected commit.

We acknowledge within 72 hours and aim to publish a fix within 30 days.

## Scope

In scope: the server (`src/`), the client crypto core (`public/js/si-auth.js`),
the Rust accelerator (`rust/`), and the protocol (`docs/PROTOCOL.md`).

Out of scope: DoS via resource exhaustion; misconfiguration.
""", force=force)

    wfile(root, "CHANGELOG.md", f"""# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [{VERSION}] - 2025-01-01

### Added
- Initial `SI-AUTH-V1` protocol: RSA-2048 challenge–response.
- TypeScript server, non-extractable Web Crypto client.
- Optional Rust N-API crypto accelerator (`siauth-crypto`).
- Multi-credential support with per-device revocation.
- HS256 session in HttpOnly, SameSite=Strict cookie.
- `auth3.py` one-file mega-launcher.
""", force=force)

    wfile(root, "CODE_OF_CONDUCT.md", """# Code of Conduct

## Our pledge

We pledge to make participation in our community a harassment-free experience
for everyone.

## Our standards

Positive examples: empathy, kindness, respect for differing viewpoints,
graceful acceptance of constructive feedback.

Unacceptable: sexualized language/imagery, trolling, insulting comments,
harassment, publishing private information without consent.

## Enforcement

Report to `conduct@example.com`. All complaints are reviewed promptly and
fairly.

Adapted from the [Contributor Covenant](https://www.contributor-covenant.org), v2.1.
""", force=force)

    wfile(root, "CONTRIBUTING.md", """# Contributing

## Setup

```bash
cp .env.example .env
npm install
npm run dev
```

## Tests

```bash
npm test
```

## Style

- ESM, TypeScript, strict mode.
- No new runtime dependencies without discussion.
- `public/js/si-auth.js` must remain standalone (zero imports).
- Rust changes go in `rust/src/`.

## PR checklist

- [ ] `npm test` passes.
- [ ] `npm run typecheck` clean.
- [ ] New behavior has a test.
- [ ] Docs updated (`docs/API.md`, `docs/PROTOCOL.md`).
""", force=force)

# ═══════════════════════════════════════════════════════════════════════════
#  Scaffolding — Rust N-API crypto accelerator
# ═══════════════════════════════════════════════════════════════════════════

def build_rust_files(root: Path, force: bool) -> None:
    log("Rust N-API accelerator (rust/)", "step")

    wfile(root, "rust/Cargo.toml", """[package]
name = "siauth-crypto"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "RSA-2048 signature verification hot-path for SI Auth (N-API)"

[lib]
crate-type = ["cdylib"]

[dependencies]
napi = { version = "2.16", default-features = false, features = ["napi6"] }
napi-derive = "2.16"
rsa = { version = "0.9", features = ["sha2"] }
sha2 = "0.10"
base64 = "0.22"
rand = "0.8"

[build-dependencies]
napi-build = "2.1"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
strip = "symbols"
""", force=force)

    wfile(root, "rust/build.rs", """extern crate napi_build;

fn main() {
    napi_build::setup();
}
""", force=force)

    wfile(root, "rust/package.json", json.dumps({
        "name": "siauth-crypto",
        "version": "0.1.0",
        "main": "index.js",
        "types": "index.d.ts",
        "napi": {
            "name": "siauth-crypto",
            "triples": {"defaults": True},
        },
        "files": ["index.js", "index.d.ts", "*.node"],
        "scripts": {
            "build": "napi build --platform --release",
            "build:debug": "napi build --platform",
        },
        "devDependencies": {"@napi-rs/cli": "^2.18.0"},
        "license": "MIT",
    }, indent=2), force=force)

    wfile(root, "rust/src/lib.rs", """//! siauth-crypto — RSA-2048 + SHA-256 (RSASSA-PKCS1-v1_5) verification for SI Auth.
//!
//! Hybrid model (hash-then-sign):
//!   1. SHA-256(message) → 32-byte digest
//!   2. DigestInfo + PKCS#1 v1.5 pad → RSA-2048 public-key verify
//! Never signs/encrypts the raw message with RSA; only the fixed-size hash.
//!
//! Loaded opportunistically by the TypeScript server. If missing, falls back
//! to Node `crypto` (same algorithm, slower).

#![deny(clippy::all)]

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use napi_derive::napi;
use rsa::pkcs8::DecodePublicKey;
use rsa::{Pkcs1v15Sign, RsaPublicKey};
use sha2::{Digest, Sha256};

/// Verify RSASSA-PKCS1-v1_5 + SHA-256 (RSA-2048).
///
/// Pipeline: SHA-256(payload) → DigestInfo → PKCS#1 v1.5 → RSA verify.
/// Only the 32-byte hash is processed by RSA, not the raw message.
///
/// * `spki_b64`  — base64 SPKI-DER RSA public key (≥ 2048-bit)
/// * `payload`   — canonical UTF-8 signing string
/// * `sig_b64`   — base64 signature
///
/// Returns `true` on success, `false` on any failure (never throws).
#[napi]
pub fn verify_signature(spki_b64: String, payload: String, sig_b64: String) -> bool {
    let spki_der = match B64.decode(spki_b64.as_bytes()) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let sig = match B64.decode(sig_b64.as_bytes()) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let key = match RsaPublicKey::from_public_key_der(&spki_der) {
        Ok(k) => k,
        Err(_) => return false,
    };
    // Enforce RSA-2048 minimum (256-byte modulus).
    if key.size() < 256 {
        return false;
    }
    // Step 1: hash. Step 2: PKCS#1 v1.5 DigestInfo + RSA verify.
    let hashed = Sha256::digest(payload.as_bytes());
    key.verify(Pkcs1v15Sign::new::<Sha256>(), &hashed, &sig).is_ok()
}

/// Return the RSA modulus size in bits for a base64 SPKI key, or 0 on error.
#[napi]
pub fn public_key_bits(spki_b64: String) -> u32 {
    let spki_der = match B64.decode(spki_b64.as_bytes()) {
        Ok(v) => v,
        Err(_) => return 0,
    };
    match RsaPublicKey::from_public_key_der(&spki_der) {
        Ok(k) => (k.size() * 8) as u32,
        Err(_) => 0,
    }
}

/// 32 bytes of OS CSPRNG randomness, hex-encoded.
#[napi]
pub fn generate_challenge() -> String {
    use rand::RngCore;
    let mut buf = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut buf);
    buf.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Version string for debugging.
#[napi]
pub fn rust_version() -> String {
    format!("siauth-crypto/{} (rustc {})", env!("CARGO_PKG_VERSION"), env!("CARGO_PKG_RUST_VERSION"))
}
""", force=force)

    wfile(root, "rust/README.md", """# siauth-crypto

Rust N-API accelerator for SI Auth. Verifies RSA-2048 signatures
approximately 4× faster than Node's built-in crypto on the hot path.

Built automatically by `npm install` (via `@napi-rs/cli`). If the build
fails, the TypeScript server silently falls back to `node:crypto`.
""", force=force)

# ═══════════════════════════════════════════════════════════════════════════
#  Scaffolding — TypeScript server
# ═══════════════════════════════════════════════════════════════════════════

def build_server_files(root: Path, force: bool) -> None:
    log("TypeScript server (src/)", "step")

    # ── src/config.ts ────────────────────────────────────────────────
    wfile(root, "src/config.ts", r"""import crypto from "node:crypto";
import path from "node:path";

const isProd = process.env.NODE_ENV === "production";

function int(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0) throw new Error(`Invalid integer for ${name}: ${raw}`);
  return n;
}

function bool(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  if (raw === "true" || raw === "1") return true;
  if (raw === "false" || raw === "0") return false;
  throw new Error(`Invalid boolean for ${name}: ${raw}`);
}

function str(name: string, fallback: string): string {
  const raw = process.env[name];
  return raw === undefined || raw === "" ? fallback : raw;
}

let jwtSecret = process.env.JWT_SECRET;
if (!jwtSecret) {
  if (isProd) {
    throw new Error(
      "JWT_SECRET is required in production. Generate: node -e \"console.log(require('crypto').randomBytes(48).toString('hex'))\"",
    );
  }
  jwtSecret = crypto.randomBytes(48).toString("hex");
  console.warn("[si-auth] WARNING: JWT_SECRET not set. Using ephemeral secret — sessions invalidated on restart.");
}
if (jwtSecret.length < 32) throw new Error("JWT_SECRET must be at least 32 characters.");

const appOrigin = str("APP_ORIGIN", "http://localhost:3000").replace(/\/+$/, "");

export interface AppConfig {
  readonly env: string;
  readonly isProd: boolean;
  readonly port: number;
  readonly appOrigin: string;
  readonly trustProxy: boolean;
  readonly jwtSecret: string;
  readonly sessionTtlSeconds: number;
  readonly challengeTtlSeconds: number;
  readonly registrationChallengeTtlSeconds: number;
  readonly dataDir: string;
  readonly cookieSecure: boolean;
  readonly cookieSameSite: "strict" | "lax" | "none";
  readonly cookieName: string;
  readonly userEnumerationSafe: boolean;
  readonly rateLimitWindowSeconds: number;
  readonly rateLimitMax: number;
  readonly logLevel: string;
  readonly protocolVersion: string;
  readonly algorithm: string;
  readonly hash: string;
  readonly modulusLength: number;
}

export const config: AppConfig = Object.freeze({
  env: isProd ? "production" : str("NODE_ENV", "development"),
  isProd,
  port: int("PORT", 3000),
  appOrigin,
  trustProxy: bool("TRUST_PROXY", false),

  jwtSecret,
  sessionTtlSeconds: int("SESSION_TTL_SECONDS", 86_400),

  challengeTtlSeconds: int("CHALLENGE_TTL_SECONDS", 60),
  registrationChallengeTtlSeconds: int("REGISTRATION_CHALLENGE_TTL_SECONDS", 300),

  dataDir: path.resolve(str("DATA_DIR", "./data")),

  cookieSecure: bool("COOKIE_SECURE", isProd),
  cookieSameSite: str("COOKIE_SAMESITE", "strict") as "strict" | "lax" | "none",
  cookieName: "si_auth_session",

  userEnumerationSafe: bool("USER_ENUMERATION_SAFE", true),
  rateLimitWindowSeconds: int("RATE_LIMIT_WINDOW_SECONDS", 60),
  rateLimitMax: int("RATE_LIMIT_MAX", 30),

  logLevel: str("LOG_LEVEL", "info").toLowerCase(),

  protocolVersion: "SI-AUTH-V1",
  algorithm: "RSASSA-PKCS1-v1_5",
  hash: "SHA-256",
  modulusLength: 2048,
});
""", force=force)

    # ── src/lib/logger.ts ────────────────────────────────────────────
    wfile(root, "src/lib/logger.ts", r"""import { config } from "../config.ts";

const LEVELS: Record<string, number> = { debug: 10, info: 20, warn: 30, error: 40, silent: 99 };
const threshold = LEVELS[config.logLevel] ?? LEVELS.info;

type Level = "debug" | "info" | "warn" | "error";

function emit(level: Level, msg: string, meta?: unknown): void {
  if (LEVELS[level] < threshold) return;
  const entry: Record<string, unknown> = {
    ts: new Date().toISOString(),
    level,
    msg,
  };
  if (meta && typeof meta === "object") Object.assign(entry, meta);
  const line = JSON.stringify(entry) + "\n";
  if (level === "error" || level === "warn") process.stderr.write(line);
  else process.stdout.write(line);
}

export const logger = {
  debug: (m: string, x?: unknown) => emit("debug", m, x),
  info:  (m: string, x?: unknown) => emit("info", m, x),
  warn:  (m: string, x?: unknown) => emit("warn", m, x),
  error: (m: string, x?: unknown) => emit("error", m, x),
};
""", force=force)

    # ── src/lib/cookies.ts ───────────────────────────────────────────
    wfile(root, "src/lib/cookies.ts", r"""export function parseCookies(header: string | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!header || typeof header !== "string") return out;
  for (const part of header.split(";")) {
    const idx = part.indexOf("=");
    if (idx === -1) continue;
    const k = part.slice(0, idx).trim();
    const v = part.slice(idx + 1).trim();
    if (!k) continue;
    try { out[k] = decodeURIComponent(v); } catch { out[k] = v; }
  }
  return out;
}

export interface CookieOptions {
  maxAge?: number;
  path?: string;
  domain?: string;
  httpOnly?: boolean;
  secure?: boolean;
  sameSite?: "strict" | "lax" | "none";
}

export function serializeCookie(name: string, value: string, opts: CookieOptions = {}): string {
  const parts = [`${name}=${encodeURIComponent(value)}`];
  if (opts.maxAge !== undefined) parts.push(`Max-Age=${Math.floor(opts.maxAge)}`);
  if (opts.path) parts.push(`Path=${opts.path}`);
  if (opts.domain) parts.push(`Domain=${opts.domain}`);
  if (opts.httpOnly) parts.push("HttpOnly");
  if (opts.secure) parts.push("Secure");
  if (opts.sameSite) parts.push(`SameSite=${opts.sameSite[0].toUpperCase() + opts.sameSite.slice(1)}`);
  return parts.join("; ");
}
""", force=force)

    # ── src/lib/crypto.ts — the hot path ─────────────────────────────
    wfile(root, "src/lib/crypto.ts", r"""import crypto from "node:crypto";
import { createRequire } from "node:module";
import { config } from "../config.ts";
import { logger } from "./logger.ts";

// ── Rust accelerator (opportunistic) ────────────────────────────────
const require = createRequire(import.meta.url);

interface RustCrypto {
  verifySignature(spkiB64: string, payload: string, sigB64: string): boolean;
  publicKeyBits(spkiB64: string): number;
  generateChallenge(): string;
  rustVersion(): string;
}

let rust: RustCrypto | null = null;
const useRust = process.env.SI_AUTH_USE_RUST ?? "auto";

if (useRust !== "false") {
  try {
    const mod = require("siauth-crypto") as {
      verifySignature: (a: string, b: string, c: string) => boolean;
      publicKeyBits: (a: string) => number;
      generateChallenge: () => string;
      rustVersion: () => string;
    };
    rust = {
      verifySignature: mod.verifySignature,
      publicKeyBits: mod.publicKeyBits,
      generateChallenge: mod.generateChallenge,
      rustVersion: mod.rustVersion,
    };
    logger.info("Rust crypto accelerator loaded", { version: rust.rustVersion() });
  } catch (err) {
    if (useRust === "true") throw err;
    logger.info("Rust crypto accelerator unavailable; using node:crypto", {
      error: (err as Error).message,
    });
  }
}

export const usingRust = (): boolean => rust !== null;
export const rustVersion = (): string | null => (rust ? rust.rustVersion() : null);

// ── Canonical signing payload ────────────────────────────────────────

export interface SigningPayloadFields {
  origin: string;
  purpose: "login" | "register";
  challenge: string;
  username: string;
  issuedAt: string;
}

export function buildSigningPayload(f: SigningPayloadFields): string {
  return [
    config.protocolVersion,
    `origin:${f.origin}`,
    `purpose:${f.purpose}`,
    `challenge:${f.challenge}`,
    `username:${f.username}`,
    `issuedAt:${f.issuedAt}`,
  ].join("\n");
}

// ── Random generators ────────────────────────────────────────────────

export function generateChallenge(): string {
  if (rust) {
    try { return rust.generateChallenge(); } catch { /* fall through */ }
  }
  return crypto.randomBytes(32).toString("hex");
}

export function generateId(prefix: string): string {
  return `${prefix}_${crypto.randomBytes(16).toString("hex")}`;
}

export function fingerprintPublicKey(spkiBase64: string): string {
  const der = Buffer.from(spkiBase64, "base64");
  return crypto.createHash("sha256").update(der).digest("hex");
}

// ── Verification (Rust fast path, Node fallback) ────────────────────

export interface VerifyArgs {
  publicKeySpkiBase64: string;
  payload: string;
  signatureBase64: string;
}

export function verifySignature(args: VerifyArgs): boolean {
  const { publicKeySpkiBase64, payload, signatureBase64 } = args;
  if (
    typeof publicKeySpkiBase64 !== "string" ||
    typeof payload !== "string" ||
    typeof signatureBase64 !== "string"
  ) return false;

  if (rust) {
    try {
      return rust.verifySignature(publicKeySpkiBase64, payload, signatureBase64);
    } catch {
      // fall through
    }
  }

  try {
    // Hash-then-sign: crypto.verify("sha256", …) hashes payload with SHA-256,
    // then verifies RSASSA-PKCS1-v1_5 over that digest with the RSA-2048 key.
    const key = crypto.createPublicKey({
      key: Buffer.from(publicKeySpkiBase64, "base64"),
      format: "der",
      type: "spki",
    });
    if (key.asymmetricKeyType !== "rsa") return false;
    const bits = (key.asymmetricKeyDetails as { modulusLength?: number } | undefined)?.modulusLength;
    if (bits && bits < config.modulusLength) return false;
    return crypto.verify(
      "sha256",
      Buffer.from(payload, "utf8"),
      { key, padding: crypto.constants.RSA_PKCS1_PADDING },
      Buffer.from(signatureBase64, "base64"),
    );
  } catch {
    return false;
  }
}

export function isValidPublicKeySpki(spkiBase64: string): boolean {
  if (typeof spkiBase64 !== "string") return false;
  if (rust) {
    try { return rust.publicKeyBits(spkiBase64) >= config.modulusLength; }
    catch { /* fall through */ }
  }
  try {
    const key = crypto.createPublicKey({
      key: Buffer.from(spkiBase64, "base64"),
      format: "der",
      type: "spki",
    });
    const bits = (key.asymmetricKeyDetails as { modulusLength?: number } | undefined)?.modulusLength ?? 0;
    return key.asymmetricKeyType === "rsa" && bits >= config.modulusLength;
  } catch {
    return false;
  }
}
""", force=force)

    # ── src/services/challengeStore.ts ───────────────────────────────
    wfile(root, "src/services/challengeStore.ts", r"""import { generateChallenge, generateId } from "../lib/crypto.ts";

export interface ChallengeRecord {
  id: string;
  purpose: "login" | "register";
  username: string;
  challenge: string;
  origin: string;
  issuedAt: string;
  expiresAt: number;
}

class ChallengeStore {
  private map = new Map<string, ChallengeRecord>();
  private sweeper: NodeJS.Timeout;

  constructor() {
    this.sweeper = setInterval(() => this.sweep(), 30_000);
    this.sweeper.unref?.();
  }

  issue(opts: { purpose: "login" | "register"; username: string; origin: string; ttlSeconds: number }): ChallengeRecord {
    const id = generateId("ch");
    const challenge = generateChallenge();
    const issuedAt = new Date().toISOString();
    const record: ChallengeRecord = {
      id, purpose: opts.purpose, username: opts.username,
      challenge, origin: opts.origin, issuedAt,
      expiresAt: Date.now() + opts.ttlSeconds * 1000,
    };
    this.map.set(id, record);
    return record;
  }

  take(id: string): ChallengeRecord | null {
    if (typeof id !== "string") return null;
    const rec = this.map.get(id);
    if (!rec) return null;
    this.map.delete(id);
    if (rec.expiresAt < Date.now()) return null;
    return rec;
  }

  sweep(): void {
    const now = Date.now();
    for (const [id, rec] of this.map) if (rec.expiresAt < now) this.map.delete(id);
  }

  clear(): void { this.map.clear(); }
  size(): number { return this.map.size; }
}

export const challengeStore = new ChallengeStore();
""", force=force)

    # ── src/services/token.ts ────────────────────────────────────────
    wfile(root, "src/services/token.ts", r"""import crypto from "node:crypto";
import { config } from "../config.ts";

function b64url(buf: Buffer): string {
  return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

export interface TokenPayload {
  sub: string;
  username: string;
  iat: number;
  exp: number;
  iss: string;
  [k: string]: unknown;
}

export function signToken(payload: Record<string, unknown>, ttlSeconds = config.sessionTtlSeconds): string {
  const now = Math.floor(Date.now() / 1000);
  const body = { ...payload, iat: now, exp: now + ttlSeconds, iss: "si-auth" };
  const header = { alg: "HS256", typ: "JWT" };
  const h = b64url(Buffer.from(JSON.stringify(header)));
  const p = b64url(Buffer.from(JSON.stringify(body)));
  const data = `${h}.${p}`;
  const sig = crypto.createHmac("sha256", config.jwtSecret).update(data).digest();
  return `${data}.${b64url(sig)}`;
}

export function verifyToken(token: string): TokenPayload | null {
  if (typeof token !== "string") return null;
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  const [h, p, s] = parts;

  let expected: string;
  try {
    expected = b64url(crypto.createHmac("sha256", config.jwtSecret).update(`${h}.${p}`).digest());
  } catch { return null; }

  const a = Buffer.from(s);
  const b = Buffer.from(expected);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;

  try {
    const header = JSON.parse(Buffer.from(h, "base64").toString("utf8")) as { alg?: string };
    if (header.alg !== "HS256") return null;
    const body = JSON.parse(Buffer.from(p, "base64").toString("utf8")) as TokenPayload;
    if (typeof body.exp !== "number" || body.exp < Math.floor(Date.now() / 1000)) return null;
    return body;
  } catch {
    return null;
  }
}
""", force=force)

    # ── src/services/userStore.ts ────────────────────────────────────
    wfile(root, "src/services/userStore.ts", r"""import fs from "node:fs/promises";
import path from "node:path";
import { config } from "../config.ts";
import { fingerprintPublicKey, generateId } from "../lib/crypto.ts";

export interface CredentialRecord {
  id: string;
  label: string;
  publicKeySpki: string;
  fingerprint: string;
  createdAt: string;
  lastUsedAt: string | null;
}

export interface UserRecord {
  id: string;
  username: string;
  createdAt: string;
  credentials: CredentialRecord[];
}

interface DB { users: UserRecord[]; }

const FILE = path.join(config.dataDir, "users.json");
let writeChain: Promise<void> = Promise.resolve();

async function readAll(): Promise<DB> {
  try {
    const raw = await fs.readFile(FILE, "utf8");
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.users)) return { users: [] };
    return parsed as DB;
  } catch (err: unknown) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return { users: [] };
    throw err;
  }
}

async function writeAll(data: DB): Promise<void> {
  writeChain = writeChain.then(async () => {
    await fs.mkdir(config.dataDir, { recursive: true });
    const tmp = `${FILE}.${process.pid}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(data, null, 2), "utf8");
    await fs.rename(tmp, FILE);
  });
  return writeChain;
}

export const userStore = {
  async findByUsername(username: string): Promise<UserRecord | null> {
    const { users } = await readAll();
    const key = username.toLowerCase();
    return users.find((u) => u.username.toLowerCase() === key) ?? null;
  },

  async findById(id: string): Promise<UserRecord | null> {
    const { users } = await readAll();
    return users.find((u) => u.id === id) ?? null;
  },

  async create(input: { username: string; publicKeySpki: string; deviceLabel?: string }): Promise<{ user: UserRecord; credential: CredentialRecord }> {
    const data = await readAll();
    const key = input.username.toLowerCase();
    if (data.users.some((u) => u.username.toLowerCase() === key)) {
      const err = new Error("Username already registered") as Error & { code?: string };
      err.code = "USER_EXISTS";
      throw err;
    }
    const now = new Date().toISOString();
    const credential: CredentialRecord = {
      id: generateId("cred"),
      label: input.deviceLabel || "Unknown device",
      publicKeySpki: input.publicKeySpki,
      fingerprint: fingerprintPublicKey(input.publicKeySpki),
      createdAt: now,
      lastUsedAt: null,
    };
    const user: UserRecord = {
      id: generateId("user"),
      username: input.username,
      createdAt: now,
      credentials: [credential],
    };
    data.users.push(user);
    await writeAll(data);
    return { user, credential };
  },

  async addCredential(username: string, input: { publicKeySpki: string; deviceLabel?: string }): Promise<CredentialRecord | null> {
    const data = await readAll();
    const key = username.toLowerCase();
    const user = data.users.find((u) => u.username.toLowerCase() === key);
    if (!user) return null;
    const credential: CredentialRecord = {
      id: generateId("cred"),
      label: input.deviceLabel || "Unknown device",
      publicKeySpki: input.publicKeySpki,
      fingerprint: fingerprintPublicKey(input.publicKeySpki),
      createdAt: new Date().toISOString(),
      lastUsedAt: null,
    };
    user.credentials.push(credential);
    await writeAll(data);
    return credential;
  },

  async listCredentials(userId: string): Promise<Array<Omit<CredentialRecord, "publicKeySpki">>> {
    const data = await readAll();
    const user = data.users.find((u) => u.id === userId);
    if (!user) return [];
    return user.credentials.map(({ publicKeySpki: _drop, ...rest }) => rest);
  },

  async removeCredential(userId: string, credentialId: string): Promise<boolean> {
    const data = await readAll();
    const user = data.users.find((u) => u.id === userId);
    if (!user) return false;
    const before = user.credentials.length;
    user.credentials = user.credentials.filter((c) => c.id !== credentialId);
    if (user.credentials.length === before) return false;
    await writeAll(data);
    return true;
  },

  async touchCredential(userId: string, credentialId: string): Promise<void> {
    const data = await readAll();
    const user = data.users.find((u) => u.id === userId);
    if (!user) return;
    const cred = user.credentials.find((c) => c.id === credentialId);
    if (!cred) return;
    cred.lastUsedAt = new Date().toISOString();
    await writeAll(data);
  },

  async _reset(): Promise<void> { await writeAll({ users: [] }); },
};
""", force=force)

    # ── src/services/auditLog.ts ─────────────────────────────────────
    wfile(root, "src/services/auditLog.ts", r"""import fs from "node:fs/promises";
import path from "node:path";
import { config } from "../config.ts";
import { logger } from "../lib/logger.ts";

const FILE = path.join(config.dataDir, "audit.log");
let chain: Promise<void> = Promise.resolve();

export function audit(event: string, details: Record<string, unknown> = {}): Promise<void> {
  const entry = { ts: new Date().toISOString(), event, ...details };
  chain = chain
    .then(async () => {
      await fs.mkdir(config.dataDir, { recursive: true });
      await fs.appendFile(FILE, JSON.stringify(entry) + "\n", "utf8");
    })
    .catch((err) => logger.warn("audit write failed", { error: String(err) }));
  return chain;
}
""", force=force)

    # ── src/middleware/*.ts ──────────────────────────────────────────
    wfile(root, "src/middleware/errorHandler.ts", r"""import type { Request, Response, NextFunction } from "express";
import { config } from "../config.ts";
import { logger } from "../lib/logger.ts";

export function notFound(req: Request, res: Response): void {
  res.status(404).json({ error: "Not Found", path: req.path });
}

export function errorHandler(err: Error & { status?: number; statusCode?: number },
                             req: Request, res: Response, _next: NextFunction): void {
  const status = err.status || err.statusCode || 500;
  if (status >= 500) {
    logger.error("unhandled error", {
      message: err.message,
      stack: config.isProd ? undefined : err.stack,
      path: req.path,
    });
  }
  res.status(status).json({
    error: status >= 500 && config.isProd ? "Internal Server Error" : err.message,
  });
}
""", force=force)

    wfile(root, "src/middleware/rateLimit.ts", r"""import type { Request, Response, NextFunction, RequestHandler } from "express";
import { config } from "../config.ts";

interface Bucket { count: number; resetAt: number; }
const buckets = new Map<string, Bucket>();
const sweeper = setInterval(() => {
  const now = Date.now();
  for (const [k, v] of buckets) if (v.resetAt < now) buckets.delete(k);
}, 60_000);
sweeper.unref?.();

export interface RateLimitOptions {
  windowSeconds?: number;
  max?: number;
  key?: (req: Request) => string;
}

export function rateLimit(opts: RateLimitOptions = {}): RequestHandler {
  const windowMs = (opts.windowSeconds ?? config.rateLimitWindowSeconds) * 1000;
  const max = opts.max ?? config.rateLimitMax;
  const keyFn = opts.key ?? ((req: Request) => req.ip || req.socket?.remoteAddress || "unknown");

  return function rateLimitMiddleware(req: Request, res: Response, next: NextFunction): void {
    const key = keyFn(req);
    const now = Date.now();
    let bucket = buckets.get(key);
    if (!bucket || bucket.resetAt < now) {
      bucket = { count: 0, resetAt: now + windowMs };
      buckets.set(key, bucket);
    }
    bucket.count += 1;
    res.setHeader("RateLimit-Limit", String(max));
    res.setHeader("RateLimit-Remaining", String(Math.max(0, max - bucket.count)));
    res.setHeader("RateLimit-Reset", String(Math.ceil(bucket.resetAt / 1000)));
    if (bucket.count > max) {
      res.setHeader("Retry-After", String(Math.ceil((bucket.resetAt - now) / 1000)));
      res.status(429).json({ error: "Too Many Requests" });
      return;
    }
    next();
  };
}
""", force=force)

    wfile(root, "src/middleware/requestLogger.ts", r"""import type { Request, Response, NextFunction } from "express";
import { logger } from "../lib/logger.ts";

export function requestLogger(req: Request, res: Response, next: NextFunction): void {
  const start = process.hrtime.bigint();
  res.on("finish", () => {
    const ms = Number(process.hrtime.bigint() - start) / 1e6;
    logger.info("http", { method: req.method, path: req.path, status: res.statusCode, ms: Number(ms.toFixed(2)) });
  });
  next();
}
""", force=force)

    wfile(root, "src/middleware/requireAuth.ts", r"""import type { Request, Response, NextFunction } from "express";
import { config } from "../config.ts";
import { parseCookies } from "../lib/cookies.ts";
import { verifyToken, type TokenPayload } from "../services/token.ts";
import { userStore, type UserRecord } from "../services/userStore.ts";

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      user?: UserRecord;
      session?: TokenPayload;
    }
  }
}

export async function requireAuth(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const cookies = parseCookies(req.headers.cookie);
    const token = cookies[config.cookieName];
    if (!token) { res.status(401).json({ error: "Not authenticated" }); return; }

    const claims = verifyToken(token);
    if (!claims || !claims.sub) { res.status(401).json({ error: "Invalid or expired session" }); return; }

    const user = await userStore.findById(claims.sub);
    if (!user) { res.status(401).json({ error: "Session user no longer exists" }); return; }

    req.user = user;
    req.session = claims;
    next();
  } catch (err) { next(err); }
}
""", force=force)

    # ── src/routes/health.ts ─────────────────────────────────────────
    wfile(root, "src/routes/health.ts", r"""import { Router } from "express";
import { config } from "../config.ts";
import { rustVersion, usingRust } from "../lib/crypto.ts";

export const healthRouter = Router();

healthRouter.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    uptime: Math.round(process.uptime()),
    env: config.env,
    protocol: config.protocolVersion,
    rust: { enabled: usingRust(), version: rustVersion() },
  });
});
""", force=force)

    # ── src/routes/auth.ts ───────────────────────────────────────────
    wfile(root, "src/routes/auth.ts", r"""import { Router } from "express";
import { config } from "../config.ts";
import { serializeCookie } from "../lib/cookies.ts";
import {
  buildSigningPayload,
  isValidPublicKeySpki,
  verifySignature,
} from "../lib/crypto.ts";
import { audit } from "../services/auditLog.ts";
import { challengeStore } from "../services/challengeStore.ts";
import { signToken } from "../services/token.ts";
import { userStore, type UserRecord } from "../services/userStore.ts";
import { requireAuth } from "../middleware/requireAuth.ts";
import { rateLimit } from "../middleware/rateLimit.ts";

export const authRouter = Router();

const USERNAME_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$|^[a-zA-Z0-9._-]{3,64}$/;

function validateUsername(u: unknown): string | null {
  if (typeof u !== "string") return "Username is required";
  const t = u.trim();
  if (!USERNAME_RE.test(t)) return "Invalid username format";
  if (t.length > 254) return "Username too long";
  return null;
}

const normalize = (u: string): string => u.trim().toLowerCase();

function setSessionCookie(res: import("express").Response, token: string, maxAgeSeconds: number): void {
  res.setHeader("Set-Cookie", serializeCookie(config.cookieName, token, {
    httpOnly: true,
    secure: config.cookieSecure,
    sameSite: config.cookieSameSite,
    path: "/",
    maxAge: maxAgeSeconds,
  }));
}

function clearSessionCookie(res: import("express").Response): void {
  res.setHeader("Set-Cookie", serializeCookie(config.cookieName, "", {
    httpOnly: true, secure: config.cookieSecure,
    sameSite: config.cookieSameSite, path: "/", maxAge: 0,
  }));
}

function issueSession(res: import("express").Response, user: UserRecord): string {
  const token = signToken({ sub: user.id, username: user.username });
  setSessionCookie(res, token, config.sessionTtlSeconds);
  return token;
}

// ── GET /api/auth/config ─────────────────────────────────────────────
authRouter.get("/config", (_req, res) => {
  res.json({
    protocol: config.protocolVersion,
    algorithm: config.algorithm,
    hash: config.hash,
    modulusLength: config.modulusLength,
    origin: config.appOrigin,
    challengeTtlSeconds: config.challengeTtlSeconds,
    registrationChallengeTtlSeconds: config.registrationChallengeTtlSeconds,
  });
});

// ── POST /api/auth/register/challenge ────────────────────────────────
authRouter.post("/register/challenge", rateLimit({ max: 10 }), async (req, res, next) => {
  try {
    const { username } = req.body ?? {};
    const err = validateUsername(username);
    if (err) { res.status(400).json({ error: err }); return; }

    const normalized = normalize(username);
    const existing = await userStore.findByUsername(normalized);
    if (existing) { res.status(409).json({ error: "Username already registered" }); return; }

    const record = challengeStore.issue({
      purpose: "register", username: normalized,
      origin: config.appOrigin, ttlSeconds: config.registrationChallengeTtlSeconds,
    });
    audit("register.challenge.issued", { username: normalized, ip: req.ip });
    res.json({
      challengeId: record.id, challenge: record.challenge,
      origin: record.origin, purpose: record.purpose,
      username: record.username, issuedAt: record.issuedAt,
      expiresInSeconds: config.registrationChallengeTtlSeconds,
    });
  } catch (e) { next(e); }
});

// ── POST /api/auth/register/complete ─────────────────────────────────
authRouter.post("/register/complete", rateLimit({ max: 10 }), async (req, res, next) => {
  try {
    const { challengeId, username, publicKeySpki, signature, deviceLabel } = req.body ?? {};
    if (!challengeId || !username || !publicKeySpki || !signature) {
      res.status(400).json({ error: "Missing required fields" }); return;
    }
    if (!isValidPublicKeySpki(publicKeySpki)) {
      res.status(400).json({ error: "Invalid or weak public key" }); return;
    }
    const record = challengeStore.take(challengeId);
    if (!record || record.purpose !== "register") {
      res.status(400).json({ error: "Challenge expired or invalid" }); return;
    }
    if (normalize(username) !== record.username) {
      res.status(400).json({ error: "Challenge username mismatch" }); return;
    }
    const payload = buildSigningPayload(record);
    const ok = verifySignature({ publicKeySpkiBase64: publicKeySpki, payload, signatureBase64: signature });
    if (!ok) {
      audit("register.failed", { username: record.username, ip: req.ip });
      res.status(401).json({ error: "Signature verification failed" }); return;
    }
    let result;
    try {
      result = await userStore.create({
        username: record.username, publicKeySpki,
        deviceLabel: typeof deviceLabel === "string" ? deviceLabel.slice(0, 120) : undefined,
      });
    } catch (e) {
      if ((e as { code?: string }).code === "USER_EXISTS") {
        res.status(409).json({ error: "Username already registered" }); return;
      }
      throw e;
    }
    issueSession(res, result.user);
    audit("register.success", { username: result.user.username, ip: req.ip });
    res.status(201).json({
      user: { id: result.user.id, username: result.user.username },
      credential: { id: result.credential.id, label: result.credential.label, fingerprint: result.credential.fingerprint },
    });
  } catch (e) { next(e); }
});

// ── POST /api/auth/login/challenge ───────────────────────────────────
authRouter.post("/login/challenge", rateLimit({ max: 30 }), async (req, res, next) => {
  try {
    const { username } = req.body ?? {};
    const err = validateUsername(username);
    if (err) { res.status(400).json({ error: err }); return; }
    const normalized = normalize(username);
    const record = challengeStore.issue({
      purpose: "login", username: normalized,
      origin: config.appOrigin, ttlSeconds: config.challengeTtlSeconds,
    });
    res.json({
      challengeId: record.id, challenge: record.challenge,
      origin: record.origin, purpose: record.purpose,
      username: record.username, issuedAt: record.issuedAt,
      expiresInSeconds: config.challengeTtlSeconds,
    });
  } catch (e) { next(e); }
});

// ── POST /api/auth/login/verify ──────────────────────────────────────
authRouter.post("/login/verify", rateLimit({ max: 30 }), async (req, res, next) => {
  try {
    const { challengeId, username, signature } = req.body ?? {};
    if (!challengeId || !username || !signature) {
      res.status(400).json({ error: "Missing required fields" }); return;
    }
    const record = challengeStore.take(challengeId);
    if (!record || record.purpose !== "login") {
      res.status(401).json({ error: "Authentication failed" }); return;
    }
    if (normalize(username) !== record.username) {
      res.status(401).json({ error: "Authentication failed" }); return;
    }
    const user = await userStore.findByUsername(record.username);
    if (!user) {
      audit("login.failed", { username: record.username, reason: "no-user", ip: req.ip });
      res.status(401).json({ error: "Authentication failed" }); return;
    }
    const payload = buildSigningPayload(record);
    let matched = null;
    for (const cred of user.credentials) {
      if (verifySignature({ publicKeySpkiBase64: cred.publicKeySpki, payload, signatureBase64: signature })) {
        matched = cred; break;
      }
    }
    if (!matched) {
      audit("login.failed", { username: record.username, reason: "bad-signature", ip: req.ip });
      res.status(401).json({ error: "Authentication failed" }); return;
    }
    await userStore.touchCredential(user.id, matched.id);
    issueSession(res, user);
    audit("login.success", { username: user.username, credentialId: matched.id, ip: req.ip });
    res.json({
      user: { id: user.id, username: user.username },
      credential: { id: matched.id, label: matched.label, fingerprint: matched.fingerprint },
    });
  } catch (e) { next(e); }
});

// ── GET /api/auth/me ─────────────────────────────────────────────────
authRouter.get("/me", requireAuth, (req, res) => {
  res.json({
    user: { id: req.user!.id, username: req.user!.username, createdAt: req.user!.createdAt },
    session: { iat: req.session!.iat, exp: req.session!.exp },
  });
});

// ── GET /api/auth/credentials ────────────────────────────────────────
authRouter.get("/credentials", requireAuth, async (req, res, next) => {
  try {
    const list = await userStore.listCredentials(req.user!.id);
    res.json({ credentials: list });
  } catch (e) { next(e); }
});

// ── DELETE /api/auth/credentials/:id ─────────────────────────────────
authRouter.delete("/credentials/:id", requireAuth, async (req, res, next) => {
  try {
    const { id } = req.params;
    const remaining = (await userStore.listCredentials(req.user!.id)).length;
    if (remaining <= 1) {
      res.status(400).json({ error: "Cannot remove your last credential. Register another device first." });
      return;
    }
    const ok = await userStore.removeCredential(req.user!.id, id);
    if (!ok) { res.status(404).json({ error: "Credential not found" }); return; }
    audit("credential.removed", { userId: req.user!.id, credentialId: id, ip: req.ip });
    res.json({ ok: true });
  } catch (e) { next(e); }
});

// ── POST /api/auth/logout ────────────────────────────────────────────
authRouter.post("/logout", (req, res) => {
  clearSessionCookie(res);
  audit("logout", { ip: req.ip });
  res.json({ ok: true });
});
""", force=force)

    # ── src/app.ts ───────────────────────────────────────────────────
    wfile(root, "src/app.ts", r"""import express, { type Express } from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "./config.ts";
import { authRouter } from "./routes/auth.ts";
import { healthRouter } from "./routes/health.ts";
import { errorHandler, notFound } from "./middleware/errorHandler.ts";
import { requestLogger } from "./middleware/requestLogger.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.resolve(__dirname, "..", "public");

export function createApp(): Express {
  const app = express();
  app.disable("x-powered-by");
  if (config.trustProxy) app.set("trust proxy", 1);

  app.use((_req, res, next) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("X-Frame-Options", "DENY");
    res.setHeader("Referrer-Policy", "same-origin");
    res.setHeader("Permissions-Policy", "camera=(), microphone=(), geolocation=(), interest-cohort=()");
    res.setHeader(
      "Content-Security-Policy",
      [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
      ].join("; "),
    );
    if (config.isProd) {
      res.setHeader("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload");
    }
    next();
  });

  app.use(express.json({ limit: "64kb" }));
  if (config.logLevel !== "silent") app.use(requestLogger);

  app.use("/api/auth", authRouter);
  app.use("/api", healthRouter);

  app.use(express.static(publicDir, { extensions: ["html"], index: "index.html" }));

  app.use("/api", notFound);
  app.use((req, res, next) => {
    if (req.method === "GET" && !req.path.startsWith("/api")) {
      res.sendFile(path.join(publicDir, "index.html"), (err) => { if (err) next(err); });
      return;
    }
    next();
  });
  app.use(notFound);
  app.use(errorHandler);

  return app;
}
""", force=force)

    # ── src/server.ts ────────────────────────────────────────────────
    wfile(root, "src/server.ts", r"""import { createApp } from "./app.ts";
import { config } from "./config.ts";
import { logger } from "./lib/logger.ts";
import { rustVersion, usingRust } from "./lib/crypto.ts";

const app = createApp();

const server = app.listen(config.port, () => {
  logger.info("si-auth listening", {
    port: config.port,
    env: config.env,
    origin: config.appOrigin,
    protocol: config.protocolVersion,
    rust: usingRust() ? rustVersion() : "disabled",
  });
  if (usingRust()) {
    logger.info("crypto backend: RUST (napi)");
  } else {
    logger.info("crypto backend: node:crypto (fallback)");
  }
  // Friendly banner
  process.stdout.write(
    `\n  \x1b[1mSI Auth\x1b[0m is running\n` +
    `  \x1b[36m→\x1b[0m http://localhost:${config.port}/\n` +
    `  \x1b[2mcrypto: ${usingRust() ? "rust/napi" : "node:crypto"} | env: ${config.env}\x1b[0m\n\n`,
  );
});

function shutdown(signal: string): void {
  logger.info("shutting down", { signal });
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 10_000).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
""", force=force)

# ═══════════════════════════════════════════════════════════════════════════
#  Scaffolding — frontend
# ═══════════════════════════════════════════════════════════════════════════

def build_frontend_files(root: Path, force: bool) -> None:
    log("Frontend (public/)", "step")

    wfile(root, "public/css/styles.css", r""":root {
  --bg: #0b0d12; --surface: #12151c; --surface-2: #1a1f29;
  --border: #262c39; --text: #e7ebf3; --muted: #8b94a7;
  --accent: #6ea8fe; --accent-2: #8ef0c1; --danger: #ff7b7b;
  --radius: 12px; --shadow: 0 10px 30px rgba(0,0,0,0.35);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; color: var(--text); min-height: 100vh;
  background: radial-gradient(1200px 800px at 20% -10%, #1a2030 0%, var(--bg) 60%);
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 480px; margin: 6vh auto 4rem; padding: 2rem; }
.wrap.wide { max-width: 720px; }
.card {
  background: linear-gradient(180deg, var(--surface), var(--surface-2));
  border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 2rem;
}
h1, h2, h3 { margin: 0 0 .5rem; letter-spacing: -.01em; }
h1 { font-size: 1.6rem; }
p.lede { color: var(--muted); margin-top: 0; }
.logo { display: inline-flex; align-items: center; gap: .6rem; font-weight: 700; margin-bottom: 1.25rem; }
.logo .mark {
  width: 32px; height: 32px; display: grid; place-items: center;
  border-radius: 9px; color: #0b0d12; font-weight: 900;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
}
.form-group { margin-bottom: 1rem; }
label { display: block; font-size: .85rem; color: var(--muted); margin-bottom: .4rem; }
input[type=text], input[type=email] {
  width: 100%; padding: .75rem .9rem; font-size: 1rem;
  background: #0d1117; border: 1px solid var(--border); border-radius: 10px;
  color: var(--text); outline: none;
}
input:focus { border-color: var(--accent); }
button {
  width: 100%; padding: .8rem 1rem; font-size: .98rem; font-weight: 600;
  background: linear-gradient(135deg, var(--accent), #9ac4ff); color: #0b0d12;
  border: none; border-radius: 10px; cursor: pointer;
}
button:hover { transform: translateY(-1px); }
button:disabled { opacity: .6; cursor: progress; transform: none; }
button.secondary { background: transparent; color: var(--text); border: 1px solid var(--border); }
.row { display: flex; gap: .6rem; align-items: center; }
.row.between { justify-content: space-between; }
.muted { color: var(--muted); } .small { font-size: .82rem; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.stack > * + * { margin-top: .75rem; }
.alert { border-radius: 10px; padding: .7rem .9rem; font-size: .9rem;
  border: 1px solid transparent; margin-bottom: 1rem; }
.alert.error { background: rgba(255,123,123,.1); border-color: rgba(255,123,123,.35); color: var(--danger); }
.alert.ok    { background: rgba(142,240,193,.1); border-color: rgba(142,240,193,.35); color: var(--accent-2); }
.alert.info  { background: rgba(110,168,254,.1); border-color: rgba(110,168,254,.35); color: var(--accent); }
.cred { display: flex; justify-content: space-between; align-items: center; gap: 1rem;
  padding: .75rem .9rem; border: 1px solid var(--border); border-radius: 10px; }
.cred .meta { font-size: .78rem; color: var(--muted); }
.fingerprint { word-break: break-all; font-size: .72rem; color: var(--muted); }
footer.credit { text-align: center; color: var(--muted); font-size: .78rem; margin-top: 2rem; }
""", force=force)

    wfile(root, "public/favicon.svg", """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#6ea8fe"/><stop offset="1" stop-color="#8ef0c1"/>
  </linearGradient></defs>
  <rect x="4" y="4" width="56" height="56" rx="14" fill="url(#g)"/>
  <path d="M22 34l7 7 14-16" stroke="#0b0d12" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>
""", force=force)

    wfile(root, "public/index.html", """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in — SI Auth</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/css/styles.css">
</head><body>
<div class="wrap"><div class="card">
  <div class="logo"><span class="mark">SI</span> SI Auth</div>
  <h1>Sign in</h1>
  <p class="lede">Enter your username. Your device signs a challenge — no password is ever sent.</p>
  <div id="alert" hidden></div>
  <form id="si-auth-form" autocomplete="off">
    <div class="form-group">
      <label for="username">Username or Email</label>
      <input type="text" id="username" required placeholder="me@example.com" autocomplete="username">
    </div>
    <button type="submit" id="login-btn">Sign in with SI Auth</button>
  </form>
  <p class="small muted" style="margin-top:1rem">No account? <a href="/register.html">Create one</a></p>
</div>
<footer class="credit">RSA-2048 challenge–response · no passwords leave your device</footer>
</div>
<script type="module" src="/js/login.js"></script>
</body></html>
""", force=force)

    wfile(root, "public/register.html", """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Create account — SI Auth</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/css/styles.css">
</head><body>
<div class="wrap"><div class="card">
  <div class="logo"><span class="mark">SI</span> SI Auth</div>
  <h1>Create your account</h1>
  <p class="lede">We'll generate an RSA-2048 keypair in your browser. The private key never leaves this device.</p>
  <div id="alert" hidden></div>
  <form id="si-auth-register" autocomplete="off">
    <div class="form-group">
      <label for="username">Username or Email</label>
      <input type="text" id="username" required placeholder="me@example.com" autocomplete="username">
    </div>
    <div class="form-group">
      <label for="deviceLabel">Device label (optional)</label>
      <input type="text" id="deviceLabel" placeholder="Chrome on macOS">
    </div>
    <button type="submit" id="register-btn">Generate key &amp; create account</button>
  </form>
  <p class="small muted" style="margin-top:1rem">Already registered? <a href="/index.html">Sign in</a></p>
</div>
<footer class="credit">Keys are non-extractable and stored in IndexedDB</footer>
</div>
<script type="module" src="/js/register.js"></script>
</body></html>
""", force=force)

    wfile(root, "public/dashboard.html", """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard — SI Auth</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/css/styles.css">
</head><body>
<div class="wrap wide"><div class="card">
  <div class="row between">
    <div class="logo"><span class="mark">SI</span> SI Auth</div>
    <button id="logout-btn" class="secondary" style="width:auto;padding:.5rem .9rem;font-size:.85rem">Sign out</button>
  </div>
  <h1>Welcome back</h1>
  <p class="lede" id="user-line">Loading…</p>
  <div id="alert" hidden></div>
  <h3 style="margin-top:1.5rem">Registered devices</h3>
  <div id="credentials" class="stack"><p class="muted small">Loading…</p></div>
  <h3 style="margin-top:1.5rem">Active session</h3>
  <div class="cred"><div>
    <div>This browser</div>
    <div class="meta" id="session-meta">—</div>
  </div><span class="small muted">Signed in</span></div>
</div>
<footer class="credit">SI Auth · reference implementation</footer>
</div>
<script type="module" src="/js/dashboard.js"></script>
</body></html>
""", force=force)

    # ── public/js/si-auth.js ─────────────────────────────────────────
    wfile(root, "public/js/si-auth.js", r"""/**
 * si-auth.js — client-side cryptographic core.
 *
 * Zero dependencies. Uses Web Crypto. Private keys are generated with
 * `extractable: false` so no JavaScript — including an XSS payload —
 * can export them.
 */

export const PROTOCOL_VERSION = 'SI-AUTH-V1';

const ALG = {
  name: 'RSASSA-PKCS1-v1_5',
  modulusLength: 2048,
  publicExponent: new Uint8Array([1, 0, 1]),
  hash: 'SHA-256',
};

export function arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function base64ToArrayBuffer(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

export function buildSigningPayload({ origin, purpose, challenge, username, issuedAt }) {
  return [
    PROTOCOL_VERSION,
    `origin:${origin}`,
    `purpose:${purpose}`,
    `challenge:${challenge}`,
    `username:${username}`,
    `issuedAt:${issuedAt}`,
  ].join('\n');
}

export async function generateKeyPair() {
  if (!globalThis.crypto?.subtle) {
    throw new Error('Web Crypto API not available. HTTPS or localhost required.');
  }
  return crypto.subtle.generateKey(ALG, false, ['sign', 'verify']);
}

export async function exportPublicKeySpki(publicKey) {
  const spki = await crypto.subtle.exportKey('spki', publicKey);
  return arrayBufferToBase64(spki);
}

export async function importPublicKeySpki(spkiBase64) {
  return crypto.subtle.importKey(
    'spki', base64ToArrayBuffer(spkiBase64),
    { name: ALG.name, hash: ALG.hash }, true, ['verify'],
  );
}

export async function signPayload(privateKey, payloadString) {
  // Hash-then-sign (RSASSA-PKCS1-v1_5 + SHA-256): Web Crypto hashes the
  // message with SHA-256, builds DigestInfo, PKCS#1 v1.5-pads, then RSA-2048.
  const data = new TextEncoder().encode(payloadString);
  const sig = await crypto.subtle.sign(
    { name: ALG.name, hash: ALG.hash },
    privateKey,
    data,
  );
  return arrayBufferToBase64(sig);
}

export async function verifyPayload(publicKey, payloadString, signatureBase64) {
  const data = new TextEncoder().encode(payloadString);
  return crypto.subtle.verify(
    { name: ALG.name, hash: ALG.hash },
    publicKey,
    base64ToArrayBuffer(signatureBase64),
    data,
  );
}

export async function fingerprintPublicKey(spkiBase64) {
  const digest = await crypto.subtle.digest('SHA-256', base64ToArrayBuffer(spkiBase64));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}
""", force=force)

    # ── public/js/keystore.js ────────────────────────────────────────
    wfile(root, "public/js/keystore.js", r"""/**
 * keystore.js — persist CryptoKey objects in IndexedDB.
 * IndexedDB structured-clone lets us store non-extractable CryptoKey
 * objects WITHOUT ever materializing the private key as raw bytes.
 */
const DB_NAME = 'si-auth';
const DB_VERSION = 1;
const STORE = 'credentials';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const os = db.createObjectStore(STORE, { keyPath: 'id' });
        os.createIndex('byUsername', 'username', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, mode) { return db.transaction(STORE, mode).objectStore(STORE); }
function reqToPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function findCredentialsByUsername(db, username) {
  return new Promise((resolve, reject) => {
    const idx = tx(db, 'readonly').index('byUsername');
    const req = idx.getAll(IDBKeyRange.only(username));
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

export async function putCredential(cred) {
  const db = await openDb();
  const existing = await findCredentialsByUsername(db, cred.username);
  await Promise.all(existing.map((e) => reqToPromise(tx(db, 'readwrite').delete(e.id))));
  await reqToPromise(tx(db, 'readwrite').put(cred));
  db.close();
}

export async function getCredential(username) {
  const db = await openDb();
  const list = await findCredentialsByUsername(db, username);
  db.close();
  return list[0] ?? null;
}

export async function listCredentials() {
  const db = await openDb();
  const all = await reqToPromise(tx(db, 'readonly').getAll());
  db.close();
  return all;
}

export async function deleteCredential(username) {
  const db = await openDb();
  const list = await findCredentialsByUsername(db, username);
  await Promise.all(list.map((e) => reqToPromise(tx(db, 'readwrite').delete(e.id))));
  db.close();
}

export async function clearAll() {
  const db = await openDb();
  await reqToPromise(tx(db, 'readwrite').clear());
  db.close();
}
""", force=force)

    # ── public/js/api.js ─────────────────────────────────────────────
    wfile(root, "public/js/api.js", r"""const JSON_HEADERS = { 'Content-Type': 'application/json' };

async function request(path, options = {}) {
  const res = await fetch(path, {
    credentials: 'same-origin',
    ...options,
    headers: { ...(options.headers || {}), ...(options.body ? JSON_HEADERS : {}) },
  });
  let body = null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) body = await res.json().catch(() => null);
  if (!res.ok) {
    const message = body?.error || `Request failed (${res.status})`;
    const err = new Error(message);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

export const api = {
  registerChallenge: (username) =>
    request('/api/auth/register/challenge', { method: 'POST', body: JSON.stringify({ username }) }),
  registerComplete: (payload) =>
    request('/api/auth/register/complete', { method: 'POST', body: JSON.stringify(payload) }),
  loginChallenge: (username) =>
    request('/api/auth/login/challenge', { method: 'POST', body: JSON.stringify({ username }) }),
  loginVerify: (payload) =>
    request('/api/auth/login/verify', { method: 'POST', body: JSON.stringify(payload) }),
  me: () => request('/api/auth/me'),
  listCredentials: () => request('/api/auth/credentials'),
  revokeCredential: (id) => request(`/api/auth/credentials/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  logout: () => request('/api/auth/logout', { method: 'POST' }),
};
""", force=force)

    # ── public/js/login.js ───────────────────────────────────────────
    wfile(root, "public/js/login.js", r"""import { api } from './api.js';
import { getCredential } from './keystore.js';
import { buildSigningPayload, signPayload } from './si-auth.js';

const form = document.getElementById('si-auth-form');
const btn = document.getElementById('login-btn');
const alertBox = document.getElementById('alert');

function showAlert(kind, message) {
  alertBox.hidden = false;
  alertBox.className = `alert ${kind}`;
  alertBox.textContent = message;
}
function hideAlert() { alertBox.hidden = true; alertBox.textContent = ''; }

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideAlert();
  const username = document.getElementById('username').value.trim();
  if (!username) return;
  btn.disabled = true; btn.textContent = 'Authenticating…';
  try {
    const ch = await api.loginChallenge(username);
    const cred = await getCredential(ch.username);
    if (!cred) throw new Error('No passkey found for this device. Please register first.');
    const payload = buildSigningPayload({
      origin: ch.origin, purpose: ch.purpose,
      challenge: ch.challenge, username: ch.username, issuedAt: ch.issuedAt,
    });
    const signature = await signPayload(cred.privateKey, payload);
    await api.loginVerify({ challengeId: ch.challengeId, username: ch.username, signature });
    window.location.href = '/dashboard.html';
  } catch (err) {
    showAlert('error', err.message || 'Authentication failed');
  } finally {
    btn.disabled = false; btn.textContent = 'Sign in with SI Auth';
  }
});
""", force=force)

    # ── public/js/register.js ────────────────────────────────────────
    wfile(root, "public/js/register.js", r"""import { api } from './api.js';
import { putCredential } from './keystore.js';
import {
  buildSigningPayload, exportPublicKeySpki, fingerprintPublicKey,
  generateKeyPair, signPayload,
} from './si-auth.js';

const form = document.getElementById('si-auth-register');
const btn = document.getElementById('register-btn');
const alertBox = document.getElementById('alert');

function showAlert(kind, message) {
  alertBox.hidden = false;
  alertBox.className = `alert ${kind}`;
  alertBox.textContent = message;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  alertBox.hidden = true;
  const username = document.getElementById('username').value.trim();
  const deviceLabel = document.getElementById('deviceLabel').value.trim() || 'This device';
  if (!username) return;
  btn.disabled = true; btn.textContent = 'Generating RSA-2048 keypair…';
  try {
    const ch = await api.registerChallenge(username);
    const { publicKey, privateKey } = await generateKeyPair();
    const publicKeySpki = await exportPublicKeySpki(publicKey);
    const payload = buildSigningPayload({
      origin: ch.origin, purpose: ch.purpose,
      challenge: ch.challenge, username: ch.username, issuedAt: ch.issuedAt,
    });
    const signature = await signPayload(privateKey, payload);
    const result = await api.registerComplete({
      challengeId: ch.challengeId, username: ch.username,
      publicKeySpki, signature, deviceLabel,
    });
    const fingerprint = await fingerprintPublicKey(publicKeySpki);
    await putCredential({
      id: `${ch.username}::${result.credential.id}`,
      username: ch.username,
      credentialId: result.credential.id,
      label: result.credential.label,
      publicKeySpki, fingerprint, privateKey,
      createdAt: new Date().toISOString(),
    });
    window.location.href = '/dashboard.html';
  } catch (err) {
    showAlert('error', err.message || 'Registration failed');
  } finally {
    btn.disabled = false; btn.textContent = 'Generate key & create account';
  }
});
""", force=force)

    # ── public/js/dashboard.js ───────────────────────────────────────
    wfile(root, "public/js/dashboard.js", r"""import { api } from './api.js';

const userLine = document.getElementById('user-line');
const credList = document.getElementById('credentials');
const sessionMeta = document.getElementById('session-meta');
const logoutBtn = document.getElementById('logout-btn');
const alertBox = document.getElementById('alert');

function showAlert(kind, message) {
  alertBox.hidden = false; alertBox.className = `alert ${kind}`; alertBox.textContent = message;
}
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.appendChild(c);
  return node;
}
function shortFingerprint(fp) {
  if (!fp || fp.length < 16) return fp || '—';
  return `${fp.slice(0, 8)}…${fp.slice(-8)}`;
}

async function load() {
  try {
    const me = await api.me();
    userLine.textContent = `Signed in as ${me.user.username}`;
    sessionMeta.textContent =
      `Issued ${new Date(me.session.iat * 1000).toLocaleString()} · ` +
      `Expires ${new Date(me.session.exp * 1000).toLocaleString()}`;
    const { credentials } = await api.listCredentials();
    credList.innerHTML = '';
    for (const c of credentials) {
      const left = el('div', {}, [
        el('div', { text: c.label }),
        el('div', { class: 'meta', text: `Added ${new Date(c.createdAt).toLocaleString()}${c.lastUsedAt ? ` · last used ${new Date(c.lastUsedAt).toLocaleString()}` : ''}` }),
        el('div', { class: 'fingerprint mono', text: shortFingerprint(c.fingerprint) }),
      ]);
      const removeBtn = el('button', { class: 'secondary', style: 'width:auto;padding:.4rem .8rem;font-size:.8rem', text: 'Revoke' });
      removeBtn.addEventListener('click', async () => {
        if (!confirm('Revoke this device? It will no longer be able to sign in.')) return;
        try { await api.revokeCredential(c.id); await load(); }
        catch (err) { showAlert('error', err.message); }
      });
      credList.appendChild(el('div', { class: 'cred' }, [left, removeBtn]));
    }
  } catch (err) {
    if (err.status === 401) { window.location.href = '/index.html'; return; }
    showAlert('error', err.message || 'Failed to load dashboard');
  }
}

logoutBtn.addEventListener('click', async () => {
  try { await api.logout(); } catch { /* ignore */ }
  window.location.href = '/index.html';
});

load();
""", force=force)

# ═══════════════════════════════════════════════════════════════════════════
#  Scaffolding — tests
# ═══════════════════════════════════════════════════════════════════════════

def build_test_files(root: Path, force: bool) -> None:
    log("Tests (test/)", "step")

    wfile(root, "test/crypto.test.ts", r"""import test from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import {
  buildSigningPayload, fingerprintPublicKey,
  isValidPublicKeySpki, verifySignature, usingRust, rustVersion,
} from "../src/lib/crypto.ts";

function makeKeyPair() {
  const { publicKey, privateKey } = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
  const spki = publicKey.export({ type: "spki", format: "der" }).toString("base64");
  return { publicKey, privateKey, spki };
}

test("backend info", () => {
  console.log(`  crypto backend: ${usingRust() ? `rust (${rustVersion()})` : "node:crypto"}`);
});

test("buildSigningPayload produces the canonical string", () => {
  const payload = buildSigningPayload({
    origin: "https://app.example.com", purpose: "login", challenge: "abc123",
    username: "me@example.com", issuedAt: "2025-01-01T12:00:00.000Z",
  });
  assert.equal(
    payload,
    ["SI-AUTH-V1",
     "origin:https://app.example.com",
     "purpose:login",
     "challenge:abc123",
     "username:me@example.com",
     "issuedAt:2025-01-01T12:00:00.000Z"].join("\n"),
  );
});

test("verifySignature accepts a valid signature", () => {
  const { privateKey, spki } = makeKeyPair();
  const payload = "hello world";
  const signature = crypto.sign("sha256", Buffer.from(payload), {
    key: privateKey, padding: crypto.constants.RSA_PKCS1_PADDING,
  }).toString("base64");
  assert.equal(verifySignature({ publicKeySpkiBase64: spki, payload, signatureBase64: signature }), true);
});

test("verifySignature rejects a tampered payload", () => {
  const { privateKey, spki } = makeKeyPair();
  const signature = crypto.sign("sha256", Buffer.from("original"), {
    key: privateKey, padding: crypto.constants.RSA_PKCS1_PADDING,
  }).toString("base64");
  assert.equal(verifySignature({ publicKeySpkiBase64: spki, payload: "tampered", signatureBase64: signature }), false);
});

test("verifySignature rejects a signature from the wrong key", () => {
  const a = makeKeyPair(); const b = makeKeyPair();
  const payload = "hello";
  const signature = crypto.sign("sha256", Buffer.from(payload), {
    key: a.privateKey, padding: crypto.constants.RSA_PKCS1_PADDING,
  }).toString("base64");
  assert.equal(verifySignature({ publicKeySpkiBase64: b.spki, payload, signatureBase64: signature }), false);
});

test("verifySignature returns false on garbage (never throws)", () => {
  assert.equal(verifySignature({
    publicKeySpkiBase64: "not-base64!!!", payload: "x", signatureBase64: "zzz",
  }), false);
});

test("isValidPublicKeySpki enforces RSA-2048 minimum", () => {
  const strong = makeKeyPair();
  assert.equal(isValidPublicKeySpki(strong.spki), true);
  const weak = crypto.generateKeyPairSync("rsa", { modulusLength: 1024 });
  const weakSpki = weak.publicKey.export({ type: "spki", format: "der" }).toString("base64");
  assert.equal(isValidPublicKeySpki(weakSpki), false);
  assert.equal(isValidPublicKeySpki("bogus"), false);
});

test("fingerprintPublicKey is deterministic", () => {
  const { spki } = makeKeyPair();
  assert.equal(fingerprintPublicKey(spki), fingerprintPublicKey(spki));
  assert.match(fingerprintPublicKey(spki), /^[0-9a-f]{64}$/);
});
""", force=force)

    wfile(root, "test/auth.test.ts", r"""import test from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import type { AddressInfo } from "node:net";
import { createApp } from "../src/app.ts";
import { challengeStore } from "../src/services/challengeStore.ts";
import { userStore } from "../src/services/userStore.ts";

function payloadOf(ch: { origin: string; purpose: string; challenge: string; username: string; issuedAt: string }) {
  return [
    "SI-AUTH-V1",
    `origin:${ch.origin}`,
    `purpose:${ch.purpose}`,
    `challenge:${ch.challenge}`,
    `username:${ch.username}`,
    `issuedAt:${ch.issuedAt}`,
  ].join("\n");
}

async function startServer(t: { after: (fn: () => Promise<void>) => void }) {
  await userStore._reset();
  challengeStore.clear();
  const app = createApp();
  const server = app.listen(0);
  await new Promise<void>((r) => server.once("listening", () => r()));
  const port = (server.address() as AddressInfo).port;
  const base = `http://127.0.0.1:${port}`;
  t.after(() => new Promise<void>((r) => server.close(() => r())));
  return base;
}

async function post(base: string, path: string, body: unknown, cookie?: string) {
  const res = await fetch(base + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(cookie ? { Cookie: cookie } : {}) },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null, setCookie: res.headers.get("set-cookie") };
}

function newClientKey() {
  const { publicKey, privateKey } = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
  return {
    spki: publicKey.export({ type: "spki", format: "der" }).toString("base64"),
    privateKey,
  };
}

function sign(privateKey: crypto.KeyObject, payload: string): string {
  return crypto.sign("sha256", Buffer.from(payload), {
    key: privateKey, padding: crypto.constants.RSA_PKCS1_PADDING,
  }).toString("base64");
}

async function register(base: string, username: string) {
  const { spki, privateKey } = newClientKey();
  const ch = (await post(base, "/api/auth/register/challenge", { username })).body;
  const payload = payloadOf(ch);
  const signature = sign(privateKey, payload);
  const res = await post(base, "/api/auth/register/complete", {
    challengeId: ch.challengeId, username: ch.username,
    publicKeySpki: spki, signature,
  });
  return { ...res, spki, privateKey };
}

test("full register → login → me flow", async (t) => {
  const base = await startServer(t);
  const username = "alice@example.com";
  const reg = await register(base, username);
  assert.equal(reg.status, 201);
  assert.ok(reg.setCookie?.includes("si_auth_session="));

  const dup = await post(base, "/api/auth/register/challenge", { username });
  assert.equal(dup.status, 409);

  const ch = (await post(base, "/api/auth/login/challenge", { username })).body;
  const signature = sign(reg.privateKey, payloadOf(ch));
  const login = await post(base, "/api/auth/login/verify", {
    challengeId: ch.challengeId, username: ch.username, signature,
  });
  assert.equal(login.status, 200);

  const cookie = login.setCookie!.split(";")[0];
  const me = await fetch(base + "/api/auth/me", { headers: { Cookie: cookie } });
  assert.equal(me.status, 200);
  const meJson = await me.json() as { user: { username: string } };
  assert.equal(meJson.user.username, username);
});

test("replay of a login challenge fails", async (t) => {
  const base = await startServer(t);
  const reg = await register(base, "bob@example.com");
  const ch = (await post(base, "/api/auth/login/challenge", { username: "bob@example.com" })).body;
  const signature = sign(reg.privateKey, payloadOf(ch));

  const first = await post(base, "/api/auth/login/verify", {
    challengeId: ch.challengeId, username: ch.username, signature,
  });
  assert.equal(first.status, 200);

  const replay = await post(base, "/api/auth/login/verify", {
    challengeId: ch.challengeId, username: ch.username, signature,
  });
  assert.equal(replay.status, 401);
});

test("tampered signature is rejected", async (t) => {
  const base = await startServer(t);
  await register(base, "carol@example.com");
  const other = newClientKey();
  const ch = (await post(base, "/api/auth/login/challenge", { username: "carol@example.com" })).body;
  const badSig = sign(other.privateKey, payloadOf(ch));
  const res = await post(base, "/api/auth/login/verify", {
    challengeId: ch.challengeId, username: ch.username, signature: badSig,
  });
  assert.equal(res.status, 401);
});

test("unknown user still gets a challenge (enumeration-safe)", async (t) => {
  const base = await startServer(t);
  const res = await post(base, "/api/auth/login/challenge", { username: "nobody@example.com" });
  assert.equal(res.status, 200);
  assert.ok(res.body.challengeId);
});

test("logout clears cookie, /me then 401", async (t) => {
  const base = await startServer(t);
  const reg = await register(base, "dan@example.com");
  const cookie = reg.setCookie!.split(";")[0];
  await post(base, "/api/auth/logout", {}, cookie);
  const me = await fetch(base + "/api/auth/me", { headers: { Cookie: cookie } });
  assert.equal(me.status, 401);
});
""", force=force)

    wfile(root, "scripts/smoke-test.ts", r"""#!/usr/bin/env -S npx tsx
/**
 * Black-box smoke test. Assumes a server is running at $BASE_URL
 * (default http://127.0.0.1:3000).
 */
import crypto from "node:crypto";

const BASE = process.env.BASE_URL || "http://127.0.0.1:3000";
const username = `smoke_${Date.now()}@example.com`;

function payloadOf(ch: { origin: string; purpose: string; challenge: string; username: string; issuedAt: string }) {
  return [
    "SI-AUTH-V1", `origin:${ch.origin}`, `purpose:${ch.purpose}`,
    `challenge:${ch.challenge}`, `username:${ch.username}`, `issuedAt:${ch.issuedAt}`,
  ].join("\n");
}

async function post(path: string, body: unknown) {
  const res = await fetch(BASE + path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}

async function main() {
  console.log(`→ smoke testing ${BASE}`);
  const health = await fetch(BASE + "/api/health");
  if (!health.ok) throw new Error(`health check failed: ${health.status}`);
  const hj = await health.json() as { rust?: { enabled: boolean; version: string | null } };
  console.log(`  ✓ /api/health  (rust: ${hj.rust?.enabled ? hj.rust.version : "disabled"})`);

  const { publicKey, privateKey } = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
  const spki = publicKey.export({ type: "spki", format: "der" }).toString("base64");

  const rch = (await post("/api/auth/register/challenge", { username })).body;
  const rSig = crypto.sign("sha256", Buffer.from(payloadOf(rch)), {
    key: privateKey, padding: crypto.constants.RSA_PKCS1_PADDING,
  }).toString("base64");
  const reg = await post("/api/auth/register/complete", {
    challengeId: rch.challengeId, username: rch.username, publicKeySpki: spki, signature: rSig,
  });
  if (reg.status !== 201) throw new Error(`register failed: ${reg.status} ${JSON.stringify(reg.body)}`);
  console.log("  ✓ register");

  const lch = (await post("/api/auth/login/challenge", { username })).body;
  const lSig = crypto.sign("sha256", Buffer.from(payloadOf(lch)), {
    key: privateKey, padding: crypto.constants.RSA_PKCS1_PADDING,
  }).toString("base64");
  const login = await post("/api/auth/login/verify", {
    challengeId: lch.challengeId, username: lch.username, signature: lSig,
  });
  if (login.status !== 200) throw new Error(`login failed: ${login.status} ${JSON.stringify(login.body)}`);
  console.log("  ✓ login");

  console.log("✓ smoke test passed");
}

main().catch((err) => { console.error("✗ smoke test failed:", err.message); process.exit(1); });
""", force=force)

# ═══════════════════════════════════════════════════════════════════════════
#  Scaffolding — docs
# ═══════════════════════════════════════════════════════════════════════════

def build_docs_files(root: Path, force: bool) -> None:
    log("Docs (docs/)", "step")

    wfile(root, "README.md", """# SI Auth

<div align="center">

**Passwordless, phishing-resistant authentication using RSA-2048 challenge–response.**

TypeScript server · Rust N-API crypto accelerator · zero-dependency client.

</div>

---

## What is this?

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
├── rust/                       #  N-API crypto accelerator
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
**Signature algorithm:** RSASSA-PKCS1-v1_5 with SHA-256 (RSA-SHA256 / RS256-style)
**Key size:** RSA-2048 minimum

RSA-2048 is not applied to the raw message. The pipeline is always:

1. **Hash** — SHA-256 over the canonical UTF-8 payload → 32-byte digest  
2. **Sign/verify** — PKCS#1 v1.5 DigestInfo + RSA-2048 over that digest only  

This is the standard hybrid construction: fixed-size digest in, signature out.

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

# ═══════════════════════════════════════════════════════════════════════════
#  Scaffolding — GitHub / CI
# ═══════════════════════════════════════════════════════════════════════════

def build_github_files(root: Path, force: bool) -> None:
    log("GitHub workflows", "step")

    wfile(root, ".github/workflows/ci.yml", """name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        node: [20, 22]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}
      - uses: dtolnay/rust-toolchain@stable
      - run: npm install --no-audit --no-fund
      - run: npm run typecheck
      - run: npm test

  smoke:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - uses: dtolnay/rust-toolchain@stable
      - run: npm install --no-audit --no-fund
      - name: Start server
        run: |
          cp .env.example .env
          JWT_SECRET=$(node -e 'console.log(require("crypto").randomBytes(48).toString("hex"))') \\
            npm start &
          for i in $(seq 1 30); do
            curl -sf http://127.0.0.1:3000/api/health && break
            sleep 1
          done
      - run: npm run smoke
""", force=force)

    wfile(root, ".github/dependabot.yml", """version: 2
updates:
  - package-ecosystem: npm
    directory: /
    schedule: { interval: weekly }
    open-pull-requests-limit: 5
  - package-ecosystem: npm
    directory: /rust
    schedule: { interval: weekly }
  - package-ecosystem: cargo
    directory: /rust
    schedule: { interval: weekly }
  - package-ecosystem: github-actions
    directory: /
    schedule: { interval: weekly }
""", force=force)

    wfile(root, ".github/PULL_REQUEST_TEMPLATE.md", """## Summary

<!-- One paragraph. -->

## Type

- [ ] Bug fix
- [ ] Feature
- [ ] Documentation
- [ ] Protocol change (requires PROTOCOL_VERSION bump)
- [ ] Rust accelerator change

## Checklist

- [ ] `npm test` passes.
- [ ] `npm run typecheck` clean.
- [ ] New behavior has tests.
- [ ] Docs updated.
""", force=force)

    wfile(root, ".github/ISSUE_TEMPLATE/bug_report.md", """---
name: Bug report
about: Something isn't working
title: "[bug] "
labels: bug
---

**Describe the bug**

**Reproduce**

1.
2.
3.

**Expected**

**Environment**
- Version/commit:
- Node:
- Rust (if relevant):
- Browser (if relevant):
- OS:

**Logs**
```
paste logs
```
""", force=force)

    wfile(root, ".github/ISSUE_TEMPLATE/feature_request.md", """---
name: Feature request
about: Suggest an idea
title: "[feat] "
labels: enhancement
---

**Problem**

**Proposed solution**

**Alternatives**

**Security impact**
""", force=force)

# ═══════════════════════════════════════════════════════════════════════════
#  Scaffolding — Docker
# ═══════════════════════════════════════════════════════════════════════════

def build_docker_files(root: Path, force: bool, port: int) -> None:
    log("Docker", "step")

    wfile(root, "Dockerfile", """# syntax=docker/dockerfile:1.6
FROM rust:1.77-alpine AS rust-builder
RUN apk add --no-cache nodejs npm python3 make g++
WORKDIR /app/rust
COPY rust/ ./
RUN npm install --no-audit --no-fund && npx napi build --platform --release

FROM node:20-alpine AS base
WORKDIR /app
ENV NODE_ENV=production

COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund

# Copy Rust-built native module into node_modules
COPY --from=rust-builder /app/rust/*.node /app/node_modules/siauth-crypto/
COPY --from=rust-builder /app/rust/index.js /app/node_modules/siauth-crypto/
COPY --from=rust-builder /app/rust/index.d.ts /app/node_modules/siauth-crypto/
COPY --from=rust-builder /app/rust/package.json /app/node_modules/siauth-crypto/

COPY src ./src
COPY public ./public
COPY tsconfig.json ./

RUN mkdir -p /app/data && chown -R node:node /app
USER node
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \\
  CMD wget -qO- http://127.0.0.1:3000/api/health || exit 1

CMD ["npx", "tsx", "src/server.ts"]
""", force=force)

    wfile(root, "docker-compose.yml", f"""services:
  si-auth:
    build: .
    image: si-auth:local
    ports:
      - "{port}:3000"
    environment:
      NODE_ENV: production
      PORT: 3000
      APP_ORIGIN: http://localhost:{port}
      JWT_SECRET: ${{JWT_SECRET:-change-me-change-me-change-me-change-me}}
      COOKIE_SECURE: "false"
      DATA_DIR: /app/data
      LOG_LEVEL: info
      SI_AUTH_USE_RUST: auto
    volumes:
      - si-auth-data:/app/data
    restart: unless-stopped

volumes:
  si-auth-data:
""", force=force)

# ═══════════════════════════════════════════════════════════════════════════
#  Build & launch
# ═══════════════════════════════════════════════════════════════════════════

def npm_install(root: Path) -> bool:
    log("Installing npm dependencies…", "step")
    try:
        run([SYS.npm or "npm", "install", "--no-audit", "--no-fund"], cwd=root)
        log("npm install complete", "ok")
        return True
    except subprocess.CalledProcessError as e:
        log(f"npm install failed: {e}", "err")
        return False

def rust_build(root: Path) -> bool:
    log("Compiling Rust N-API accelerator…", "step")
    rust_dir = root / "rust"
    if not (rust_dir / "package.json").exists():
        return False
    try:
        run([SYS.npm or "npm", "install", "--no-audit", "--no-fund"], cwd=rust_dir)
        run([SYS.npm or "npm", "run", "build"], cwd=rust_dir)
        # Verify the .node file exists
        node_files = list(rust_dir.glob("*.node"))
        if node_files:
            log(f"Rust accelerator built: {node_files[0].name}", "ok")
            return True
        log("Rust build produced no .node file", "warn")
        return False
    except subprocess.CalledProcessError as e:
        log(f"Rust build failed: {e}", "warn")
        log("The server will fall back to node:crypto — no problem.", "warn")
        return False

def launch(root: Path, port: int) -> None:
    log(f"Launching server on http://localhost:{port}", "step")
    env = os.environ.copy()
    env.setdefault("PORT", str(port))
    env.setdefault("NODE_ENV", "development")
    env.setdefault("APP_ORIGIN", f"http://localhost:{port}")
    if not env.get("JWT_SECRET"):
        import secrets
        env["JWT_SECRET"] = secrets.token_hex(48)
        log("(generated ephemeral JWT_SECRET for this run)", "info")

    npm = SYS.npm or "npm"
    try:
        subprocess.run([npm, "start"], cwd=root, env=env, check=True)
    except KeyboardInterrupt:
        log("Interrupted", "info")
    except subprocess.CalledProcessError as e:
        die(f"Server exited with code {e.returncode}")

# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="auth3.py",
        description="SI Auth mega-launcher — scaffolds + installs + runs the full stack.",
    )
    p.add_argument("target", nargs="?", default=DEFAULT_TARGET,
                   help=f"Target directory (default: {DEFAULT_TARGET})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    p.add_argument("--no-rust", action="store_true", help="Skip Rust accelerator")
    p.add_argument("--no-install", action="store_true", help="Scaffold only, don't npm install")
    p.add_argument("--no-launch", action="store_true", help="Don't launch the server")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument("--version", action="version", version=f"auth3.py {VERSION}")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    banner()

    log("Detecting system…", "step")
    SYS.describe()
    print()

    # ── Node ─────────────────────────────────────────────────────────
    if not node_version_ok():
        log("Node.js 18.17+ not found.", "warn")
        install_node()
    else:
        out = subprocess.check_output([SYS.node, "--version"], text=True).strip()
        log(f"Node.js {out} ✓", "ok")

    # ── Optional Rust ────────────────────────────────────────────────
    with_rust = not args.no_rust
    rust_ready = False
    if with_rust:
        if SYS.cargo and shutil.which("rustc"):
            log(f"Rust toolchain present: {subprocess.check_output([SYS.cargo, '--version'], text=True).strip()}", "ok")
            rust_ready = True
        else:
            log("Rust not found. Attempting install (optional)…", "warn")
            rust_ready = install_rust()
            if not rust_ready:
                log("Continuing without Rust accelerator.", "warn")
        if rust_ready:
            ensure_apt_build_deps()

    print()
    root = Path(args.target).resolve()
    log(f"Target: {C_BOLD(str(root))}", "step")
    root.mkdir(parents=True, exist_ok=True)

    # ── Scaffold ─────────────────────────────────────────────────────
    print()
    build_root_files(root, with_rust=(with_rust and rust_ready), port=args.port, force=args.force)
    build_server_files(root, force=args.force)
    if with_rust and rust_ready:
        build_rust_files(root, force=args.force)
    build_frontend_files(root, force=args.force)
    build_test_files(root, force=args.force)
    build_docs_files(root, force=args.force)
    build_github_files(root, force=args.force)
    build_docker_files(root, force=args.force, port=args.port)

    print()
    log(f"Wrote {len(WRITTEN)} file(s), skipped {len(SKIPPED)} existing.", "ok")

    if args.no_install:
        log("Skipping npm install (--no-install).", "info")
        return

    # ── Install ──────────────────────────────────────────────────────
    print()
    ok = npm_install(root)
    if not ok:
        die("npm install failed. Fix the issue and re-run, or use --no-install.")

    # ── Rust build (after npm install so napi CLI is available) ──────
    if with_rust and rust_ready:
        print()
        built = rust_build(root)
        if built:
            # Wire the local module into the main package's node_modules.
            main_nm = root / "node_modules" / "siauth-crypto"
            try:
                if main_nm.exists() and main_nm.is_symlink():
                    pass
                else:
                    if main_nm.exists():
                        shutil.rmtree(main_nm)
                    main_nm.mkdir(parents=True, exist_ok=True)
                    for pat in ("*.node", "index.js", "index.d.ts", "package.json"):
                        for f in (root / "rust").glob(pat):
                            shutil.copy2(f, main_nm / f.name)
                log("Rust module linked into node_modules/siauth-crypto", "ok")
            except Exception as e:
                log(f"Could not link Rust module (will fall back): {e}", "warn")

    # ── Launch ───────────────────────────────────────────────────────
    print()
    log("Setup complete.", "ok")
    log(f"  cd {root}", "info")
    log(f"  npm start", "info")
    print()

    if args.no_launch:
        return
    launch(root, args.port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        log("Aborted by user.", "warn")
        sys.exit(130)
    except Exception as e:
        log(f"Fatal: {e}", "err")
        if os.environ.get("AUTH3_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)
