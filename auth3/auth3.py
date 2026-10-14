#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auth3.py — SI Auth mega-launcher (v4.1.0)
=========================================
One file. Detects your system. Installs Node + (optionally) Rust.
Scaffolds a full TypeScript + Rust-N-API + static-frontend SI Auth stack.
Runs it.

Signature construction (RSASSA-PKCS1-v1_5 + SHA-256 over RSA-2048)
-------------------------------------------------------------------
    1.  SHA-256( canonical UTF-8 payload )  ->  32-byte digest
    2.  RSA-2048 PKCS#1 v1.5 sign( DigestInfo( digest ) )  ->  256-byte sig

Author: SI Auth Contributors · MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import NoReturn

VERSION = "4.1.0"
NODE_MIN_MAJOR = 18
NODE_MIN_MINOR = 17
DEFAULT_TARGET = "si-auth"
DEFAULT_PORT = 3000

SECURITY_EMAIL = "security@si-auth.dev"
CONDUCT_EMAIL = "conduct@si-auth.dev"

_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return "\033[" + code + "m" + s + "\033[0m" if _TTY else s


C_OK = lambda s: _c("32;1", s)
C_ERR = lambda s: _c("31;1", s)
C_WARN = lambda s: _c("33;1", s)
C_INFO = lambda s: _c("36;1", s)
C_BOLD = lambda s: _c("1", s)
C_DIM = lambda s: _c("2", s)


def banner() -> None:
    print(flush=True)
    print(C_BOLD("  +======================================================+"), flush=True)
    print(C_BOLD("  |") + "  " + C_INFO("SI Auth") + "  -  RSA-2048 / RSA-SHA256 challenge-response  " + C_BOLD("|"), flush=True)
    print(C_BOLD("  |") + "  " + C_DIM(f"mega-launcher v{VERSION}  -  python / typescript / rust") + "  " + C_BOLD("|"), flush=True)
    print(C_BOLD("  +======================================================+"), flush=True)
    print(flush=True)


def log(msg: str, kind: str = "info") -> None:
    prefix = {
        "info": C_INFO("  -"),
        "ok": C_OK("  +"),
        "warn": C_WARN("  !"),
        "err": C_ERR("  x"),
        "step": C_BOLD("  >"),
    }.get(kind, "  -")
    print(f"{prefix} {msg}", flush=True)


def die(msg: str, code: int = 1) -> NoReturn:
    log(msg, "err")
    sys.exit(code)


# ═══════════════════════════════════════════════════════════════════════════
#  System detection
# ═══════════════════════════════════════════════════════════════════════════

class System:
    def __init__(self) -> None:
        self.os = platform.system().lower()
        self.arch = platform.machine().lower()
        self.distro = self._detect_distro()
        self.is_root = self._is_root()
        self.pkg = self._detect_pkg_mgr()
        self.node = shutil.which("node")
        self.npm = shutil.which("npm")
        self.cargo = shutil.which("cargo")
        self.curl = shutil.which("curl")
        self.wget = shutil.which("wget")
        self.git = shutil.which("git")

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
        candidates = ["apt-get", "dnf", "yum", "pacman", "apk", "brew",
                      "zypper", "winget", "choco"]
        if platform.system() == "FreeBSD":
            candidates.insert(0, "pkg")
        for mgr in candidates:
            if shutil.which(mgr):
                return mgr
        return None

    def describe(self) -> None:
        log(f"OS     : {C_BOLD(self.os)} / {C_BOLD(self.distro)} / {C_BOLD(self.arch)}")
        log(f"Root   : {self.is_root}")
        log(f"Pkg    : {self.pkg or C_WARN('none detected')}")
        log(f"Node   : {self.node or C_WARN('not found')}")
        log(f"npm    : {self.npm or C_WARN('not found')}")
        log(f"Cargo  : {self.cargo or C_WARN('not found (optional)')}")


SYS = System()


# ═══════════════════════════════════════════════════════════════════════════
#  Process helpers
# ═══════════════════════════════════════════════════════════════════════════

def run(
    cmd: list[str] | str,
    *,
    check: bool = True,
    shell: bool = False,
    env: dict | None = None,
    cwd: Path | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    display = " ".join(cmd) if isinstance(cmd, list) else cmd
    if not quiet:
        log(C_DIM(f"$ {display}"))
    return subprocess.run(
        cmd,
        check=check,
        shell=shell,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
        text=True,
    )


def sudo(cmd: list[str]) -> list[str]:
    if SYS.os == "windows":
        return cmd
    if SYS.is_root:
        return cmd
    if shutil.which("sudo"):
        return ["sudo", "-E"] + cmd
    return cmd


# ═══════════════════════════════════════════════════════════════════════════
#  Safe remote-install helpers
# ═══════════════════════════════════════════════════════════════════════════

def download_to_temp(url: str, *, sha256: str | None = None, suffix: str = ".sh") -> Path | None:
    try:
        log(f"  downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": f"auth3/{VERSION}"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = resp.read()
        if sha256:
            got = hashlib.sha256(data).hexdigest().lower()
            want = sha256.lower()
            if got != want:
                log(f"  checksum mismatch: got {got} want {want}", "err")
                return None
        fd, name = tempfile.mkstemp(prefix="auth3-", suffix=suffix)
        os.close(fd)
        p = Path(name)
        p.write_bytes(data)
        return p
    except Exception as e:
        log(f"  download failed: {e}", "warn")
        return None


def run_script(path: Path, args: list[str] | None = None, *, shell: str = "bash") -> bool:
    try:
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IXUSR)
        cmd = [shell, str(path)] + (args or [])
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log(f"  script exited with code {e.returncode}", "warn")
        return False
    except Exception as e:
        log(f"  script execution failed: {e}", "warn")
        return False


def _safe_download_and_run(
    url: str, args: list[str] | None = None, *, suffix: str = ".sh",
    shell: str = "bash", sha256: str | None = None,
) -> bool:
    script = download_to_temp(url, sha256=sha256, suffix=suffix)
    if script is None:
        return False
    try:
        return run_script(script, args, shell=shell)
    finally:
        try:
            script.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            log(f"  could not clean up {script}: {e}", "warn")


# ═══════════════════════════════════════════════════════════════════════════
#  Dependency installation
# ═══════════════════════════════════════════════════════════════════════════

def node_version_ok() -> bool:
    if not SYS.node:
        return False
    try:
        out = subprocess.check_output([SYS.node, "--version"], text=True).strip()
        parts = out.lstrip("v").split(".")
        major, minor = int(parts[0]), int(parts[1])
        return (major, minor) >= (NODE_MIN_MAJOR, NODE_MIN_MINOR)
    except Exception:
        return False


def _version_tuple(p: Path) -> tuple[int, ...]:
    name = p.name.lstrip("v")
    try:
        return tuple(int(x) for x in name.split("."))
    except ValueError:
        return (0,)


def _resolve_nvm_node() -> bool:
    nvm_dir = Path.home() / ".nvm"
    nvm_node = nvm_dir / "versions" / "node"
    if not nvm_node.exists():
        return False
    versions = sorted(
        (p for p in nvm_node.iterdir() if p.is_dir()),
        key=_version_tuple, reverse=True,
    )
    if not versions:
        return False
    bin_dir = versions[0] / "bin"
    if not (bin_dir / "node").exists():
        return False
    os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    SYS.node = str(bin_dir / "node")
    SYS.npm = str(bin_dir / "npm")
    return True


def install_node() -> None:
    log("Installing Node.js...", "step")

    if SYS.os == "windows":
        log("Windows detected. Automatic Node install is not attempted.", "warn")
        log("Please install Node.js 18.17 or newer (20 LTS recommended) "
            "from https://nodejs.org/ and re-run.", "warn")
        die("Node.js 18.17+ is required.")

    nvm_dir = Path.home() / ".nvm"
    if not (nvm_dir / "nvm.sh").exists():
        log("  fetching nvm installer to temp file...")
        _safe_download_and_run(
            "https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh",
            [], shell="bash",
        )

    if (nvm_dir / "nvm.sh").exists():
        log("  using nvm to install Node 20...")
        cmd = (
            f'source "{nvm_dir}/nvm.sh" && '
            "nvm install 20 && nvm use 20 && nvm alias default 20"
        )
        try:
            run(["bash", "-lc", cmd], quiet=True)
            _resolve_nvm_node()
            if node_version_ok():
                out = subprocess.check_output([SYS.node, "--version"], text=True).strip()
                log(f"  Node {out} installed via nvm", "ok")
                return
            log(
                f"  nvm reported success but resolved Node is still too old "
                f"(path: {SYS.node}); falling back to package manager.", "warn",
            )
        except subprocess.CalledProcessError as e:
            log(f"  nvm install failed: {e}", "warn")

    mgr = SYS.pkg
    if not mgr:
        die("No way to install Node.js. Please install Node 18.17+ manually.")

    if mgr == "apt-get":
        run(sudo(["apt-get", "update", "-qq"]))
        run(sudo(["apt-get", "install", "-y", "-qq", "nodejs", "npm", "curl", "ca-certificates"]))
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

    SYS.node = shutil.which("node")
    SYS.npm = shutil.which("npm")
    if not node_version_ok():
        die("Node.js install failed or version too old. Need 18.17+.")


def install_rust() -> bool:
    if SYS.cargo and shutil.which("rustc"):
        log("Rust already installed.", "ok")
        return True

    if SYS.os == "windows":
        log("Windows detected. Install Rust manually: https://rustup.rs/", "warn")
        return False

    log("Installing Rust (rustup)...", "step")
    ok = _safe_download_and_run(
        "https://sh.rustup.rs",
        ["-y", "--default-toolchain", "stable", "--profile", "minimal"],
        shell="sh",
    )
    if not ok:
        return False

    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}:{os.environ['PATH']}"
    if (cargo_bin / "cargo").exists():
        SYS.cargo = str(cargo_bin / "cargo")
    else:
        SYS.cargo = shutil.which("cargo")

    if SYS.cargo:
        out = subprocess.check_output([SYS.cargo, "--version"], text=True).strip()
        log(f"Rust installed: {out}", "ok")
        return True
    return False


def ensure_apt_build_deps() -> None:
    if SYS.os != "linux" or SYS.pkg != "apt-get":
        return
    needed = ["build-essential", "pkg-config", "libssl-dev", "ca-certificates", "curl"]
    log("Ensuring build toolchain (build-essential, libssl-dev)...")
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


def wfile(root: Path, rel: str, content: str, *, force: bool = False, executable: bool = False) -> None:
    path = root / rel
    if path.exists() and not force:
        SKIPPED.append(rel)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if content and not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")
    if executable:
        st = path.stat()
        path.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    WRITTEN.append(rel)
    print(f"    {C_OK('+')} {rel}")


# ═══════════════════════════════════════════════════════════════════════════
#  Root-level files
# ═══════════════════════════════════════════════════════════════════════════

def build_root_files(root: Path, *, with_rust: bool, port: int, force: bool) -> None:
    log("Root files", "step")

    deps: dict[str, str] = {
        "express": "^4.19.2",
        "tsx": "^4.7.0",
        "typescript": "^5.4.0",
        "@types/express": "^4.17.21",
        "@types/node": "^20.11.0",
    }

    pkg: dict = {
        "name": "si-auth",
        "version": VERSION,
        "description": (
            "Passwordless, phishing-resistant authentication using RSA-2048 / "
            "RSA-SHA256 challenge-response (TypeScript + Rust)."
        ),
        "type": "module",
        "engines": {"node": ">=18.17"},
        "scripts": {
            "start": "tsx src/server.ts",
            "dev": "tsx watch src/server.ts",
            "test": "node --test --import tsx test/",
            "smoke": "tsx scripts/smoke-test.ts",
            "typecheck": "tsc --noEmit",
        },
        "keywords": [
            "authentication", "rsa", "rsa-sha256", "passwordless",
            "challenge-response", "typescript", "rust", "redis", "postgres",
        ],
        "license": "MIT",
        "dependencies": deps,
        # `pg` is optional: only needed when USER_STORE_BACKEND=postgres.
        # Listed here so `npm install` does NOT pull it unless explicitly
        # requested; the Postgres backend lazily requires it at boot.
    }
    _ = with_rust
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
rust/target/
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

    env_example = (
        "# Server\n"
        "NODE_ENV=development\n"
        f"PORT={port}\n"
        f"APP_ORIGIN=http://localhost:{port}\n"
        "TRUST_PROXY=false\n"
        "\n"
        "# Secrets (REQUIRED in production; min 32 chars)\n"
        "JWT_SECRET=\n"
        "\n"
        "# Sessions\n"
        "SESSION_TTL_SECONDS=86400\n"
        "# Session revocation list. Revoked JWTs (by jti) are rejected until\n"
        "# their natural expiry. Uses Redis when REDIS_URL is set, else memory.\n"
        "REVOCATION_ENABLED=true\n"
        "\n"
        "# Challenges\n"
        "CHALLENGE_TTL_SECONDS=60\n"
        "REGISTRATION_CHALLENGE_TTL_SECONDS=300\n"
        "\n"
        "# Storage\n"
        "DATA_DIR=./data\n"
        "# User store backend: json | postgres\n"
        "#   json      -> ./data/users.json (single-node; default)\n"
        "#   postgres  -> requires `npm install pg` and DATABASE_URL below.\n"
        "USER_STORE_BACKEND=json\n"
        "# Example: postgres://si_auth:password@127.0.0.1:5432/si_auth\n"
        "DATABASE_URL=\n"
        "# Set to 'true' to run schema DDL at boot (idempotent).\n"
        "# Recommended: true on first deploy, false on subsequent restarts.\n"
        "DATABASE_AUTO_MIGRATE=true\n"
        "\n"
        "# Cookies (COOKIE_SAMESITE: strict | lax | none)\n"
        "COOKIE_SAMESITE=strict\n"
        "\n"
        "# Security\n"
        "USER_ENUMERATION_SAFE=true\n"
        "\n"
        "# Logging\n"
        "LOG_LEVEL=info\n"
        "\n"
        "# Metrics (Prometheus text format at GET /metrics)\n"
        "METRICS_ENABLED=true\n"
        "# Optional: require 'Authorization: Bearer <token>' on /metrics.\n"
        "# Leave empty to leave /metrics unauthenticated (firewall it instead).\n"
        "METRICS_TOKEN=\n"
        "\n"
        "# Redis (OPTIONAL). When set, challenges, rate-limit buckets, and the\n"
        "# session revocation list are shared across replicas. When empty, the\n"
        "# process uses in-memory stores (single-node only).\n"
        "# Example: REDIS_URL=redis://127.0.0.1:6379\n"
        "REDIS_URL=\n"
        "\n"
        "# Rust accelerator (auto | true | false)\n"
        "SI_AUTH_USE_RUST=auto\n"
    )
    wfile(root, ".env.example", env_example, force=force)

    wfile(root, "data/.gitkeep", "", force=force)

    wfile(root, "LICENSE", """MIT License

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

    wfile(root, "SECURITY.md", f"""# Security Policy

## Reporting a vulnerability

**Do not open a public issue.** Email `{SECURITY_EMAIL}` with a
description, reproduction, and affected commit.

We acknowledge within 72 hours and aim to publish a fix within 30 days.

## Supported versions

| Version | Supported |
|---|---|
| 4.x | yes |
| 3.x | security fixes only |
| 2.x | no |
| 1.x | no |

## Scope

In scope: the server (`src/`), the client crypto core
(`public/js/si-auth.js`), the Rust accelerator (`rust/`), the optional
Redis integration (`src/lib/redis.ts`), and the optional Postgres user
store (`src/services/stores/postgresUserStore.ts`), plus the protocol as
documented in `docs/PROTOCOL.md`.

Out of scope: DoS via resource exhaustion; misconfiguration (e.g. leaking
`JWT_SECRET`); browser-specific Web Crypto bugs; failure of external
Redis or Postgres instances.

## Audit log sensitivity

`data/audit.log` records `login.failed` and `register.failed` events with
a `reason` field (`no-user`, `bad-signature`, `username-taken`, ...).
**These entries are user-enumeration signals.** They are for operators,
not for end users. Do not expose them through public dashboards, support
tooling, or unauthenticated admin UIs.

## Revocation list

`REVOCATION_ENABLED=true` (default) tracks revoked session IDs (`jti`)
until their natural expiry. In-memory by default; Redis-backed when
`REDIS_URL` is set. Entries are bounded by session TTL and cannot grow
without bound.

## Metrics

`GET /metrics` exposes process counters in Prometheus text format. If
`METRICS_TOKEN` is set, a valid `Authorization: Bearer <token>` header
is required. If it is empty, `/metrics` is unauthenticated — firewall it
or set a token. Metrics never contain usernames, IPs, or tokens.

## Accepted trade-offs

- The launcher downloads `sh.rustup.rs` over HTTPS and executes it. A
  SHA-256 of the installer is not pinned. Operators who need a strict pin
  should fork `install_rust()` and supply one via the `sha256=` argument
  to `_safe_download_and_run`.
- The default `userStore` backend is a JSON file. It is correct for a
  single node only. Multi-replica deployments must set
  `USER_STORE_BACKEND=postgres` and `DATABASE_URL`, or supply their own
  implementation of the documented `UserStore` interface.
- There is no user-facing account recovery flow. Pre-register a second
  device or use the operator-side reset described in
  `docs/DEPLOYMENT.md#recovery`.
""", force=force)

    wfile(root, "CHANGELOG.md", f"""# Changelog

All notable changes are documented here. Format:
[Keep a Changelog](https://keepachangelog.com/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [4.1.0] - 2025-01-05

Closes the final HA edge: `userStore` is now a pluggable interface with a
Postgres backend, so a multi-replica deployment can share both challenge
state and user state without any code changes to routes.

### Added
- `src/services/userStore.ts` — the canonical `UserStore` interface plus
  a factory that selects a backend at boot.
- `src/services/stores/jsonUserStore.ts` — the JSON-file backend
  (default). Behavior is unchanged from 4.0.0.
- `src/services/stores/postgresUserStore.ts` — a Postgres backend that
  lazily requires the `pg` package. Idempotent schema bootstrap, atomic
  upserts, foreign-key cascade on user deletion.
- `USER_STORE_BACKEND` (`json` | `postgres`), `DATABASE_URL`, and
  `DATABASE_AUTO_MIGRATE` environment variables.
- `GET /api/health` reports the selected `userStore` backend.
- `docker-compose.yml` gains an optional `postgres` service behind the
  `with-postgres` profile.

### Changed
- Routes import `userStore` from `src/services/userStore.ts` (factory).
  No route code changes were required — the interface is identical.
- Version bumped to 4.1.0.

### Fixed
- (Carried from 4.0.0) Rust accelerator compiles: `sha2` gains the
  `oid` feature; `PublicKeyParts` import verified.

## [4.0.0] - 2025-01-04

Multi-replica readiness: shared state via an optional Redis backend, plus
observability and request correlation.

### Added
- `src/lib/redis.ts` — zero-dependency Redis client (RESP over
  `node:net`, TLS via `rediss://`, queued responses).
- Redis-backed `challengeStore` — auto-selected when `REDIS_URL` is set.
- Redis-backed `rateLimit` — same selection rule.
- `src/services/revocation.ts` — session revocation list. JWTs now carry
  a `jti`; logout revokes it; `requireAuth` rejects revoked tokens.
- `src/middleware/requestId.ts` — `X-Request-ID` correlation.
- `src/routes/metrics.ts` — Prometheus text-format `/metrics`.
- `docker-compose.yml` — optional `redis` service behind `with-redis`.

## [3.0.0] - 2025-01-03

### Fixed
- Rust accelerator compiles: `sha2 = {{ version = "0.10", features = ["oid"] }}`.
- Placeholder `@example.com` addresses replaced with real addresses
  (`{SECURITY_EMAIL}`, `{CONDUCT_EMAIL}`).
- `VERSION`, `package.json`, and CHANGELOG are in sync.

## [2.0.1] - 2025-01-02

### Fixed
- Rust accelerator compiles cleanly (`sha2` `oid` feature).
- `rsa` pinned to `0.9` with the `sha2` feature enabled.

## [2.0.0] - 2025-01-01

### Removed
- Auto-generated `README.md` scaffold.
- Unused `RATE_LIMIT_WINDOW_SECONDS` / `RATE_LIMIT_MAX` env vars.
- Wildcard `"siauth-crypto": "*"` from `optionalDependencies`.

### Fixed
- Unused imports removed; `die()` annotated `-> NoReturn`.
- `_detect_pkg_mgr()` only treats `pkg` as a manager on FreeBSD.
- `install_node()` / `install_rust()` execute downloaded scripts from
  disk and clean up temp files in a `finally` block.
- `_resolve_nvm_node()` sorts versions numerically.
- `banner()` flushes stdout.
- `.env` is loaded by `src/config.ts` via a zero-dependency parser.
- `COOKIE_SAMESITE` is validated against an allow-list.
- `userStore` serializes reads and writes on one chain.
- `keystore.js` keeps the `IDBTransaction` alive until completion.
- `docker-compose.yml` defaults `COOKIE_SECURE` to `true`.
- `register/complete` verifies the signature before the user-store lookup.
- `SECURITY.md` documents audit-log sensitivity.

## [1.0.0] - 2025-01-01

### Added
- Initial `SI-AUTH-V1` protocol: RSA-2048 challenge-response.
- TypeScript server, non-extractable Web Crypto client.
- Optional Rust N-API crypto accelerator.
- Multi-credential support with per-device revocation.
- HS256 session in HttpOnly, SameSite=Strict cookie.
""", force=force)

    wfile(root, "CODE_OF_CONDUCT.md", f"""# Code of Conduct

## Our pledge

We pledge to make participation in our community a harassment-free
experience for everyone.

## Our standards

Positive examples: empathy, kindness, respect for differing viewpoints,
graceful acceptance of constructive feedback.

Unacceptable: sexualized language/imagery, trolling, insulting comments,
harassment, publishing private information without consent.

## Enforcement

Report to `{CONDUCT_EMAIL}`. All complaints are reviewed promptly and
fairly.

Adapted from the
[Contributor Covenant](https://www.contributor-covenant.org), v2.1.
""", force=force)

    wfile(root, "CONTRIBUTING.md", """# Contributing

## Setup

~~~bash
cp .env.example .env
npm install
npm run dev
~~~

## Tests

~~~bash
npm test
npm run typecheck
npm run smoke     # in a second shell, with the server running
~~~

## Running against Redis

~~~bash
docker run --rm -p 6379:6379 redis:7-alpine
REDIS_URL=redis://127.0.0.1:6379 npm run dev
~~~

When `REDIS_URL` is set, challenges, rate-limit buckets, and the session
revocation list are stored in Redis and shared across replicas.

## Running against Postgres

~~~bash
docker run --rm -p 5432:5432 \\
  -e POSTGRES_PASSWORD=si_auth -e POSTGRES_USER=si_auth -e POSTGRES_DB=si_auth \\
  postgres:16-alpine

npm install pg                # required for the Postgres backend
DATABASE_URL=postgres://si_auth:si_auth@127.0.0.1:5432/si_auth \\
USER_STORE_BACKEND=postgres \\
  npm run dev
~~~

## Style

- ESM, TypeScript, strict mode.
- No new required runtime dependencies without discussion. Optional
  backends (pg) are loaded lazily and must not break the default install.
- `public/js/si-auth.js` must remain standalone (zero imports).
- Rust changes go in `rust/src/`.

## PR checklist

- [ ] `npm test` passes (default backend).
- [ ] `npm test` passes with `REDIS_URL` set.
- [ ] `npm test` passes with `USER_STORE_BACKEND=postgres` + `DATABASE_URL`.
- [ ] `npm run typecheck` clean.
- [ ] New behavior has a test.
- [ ] Docs updated (`docs/API.md`, `docs/PROTOCOL.md`).
""", force=force)


# ═══════════════════════════════════════════════════════════════════════════
#  Rust N-API crypto accelerator
# ═══════════════════════════════════════════════════════════════════════════

def build_rust_files(root: Path, force: bool) -> None:
    log("Rust N-API accelerator (rust/)", "step")

    wfile(root, "rust/Cargo.toml", """[package]
name = "siauth-crypto"
version = "0.1.0"
edition = "2021"
rust-version = "1.70"
license = "MIT"
description = "RSA-2048 / RSA-SHA256 signature verification hot-path for SI Auth (N-API)"

[lib]
crate-type = ["cdylib"]

[dependencies]
napi = { version = "2.16", default-features = false, features = ["napi6"] }
napi-derive = "2.16"
rsa = { version = "0.9", features = ["sha2"] }
sha2 = { version = "0.10", features = ["oid"] }
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

    rust_pkg = {
        "name": "siauth-crypto",
        "version": "0.1.0",
        "main": "index.js",
        "types": "index.d.ts",
        "napi": {
            "name": "siauth-crypto",
            "triples": {
                "defaults": False,
                "additional": [
                    "x86_64-unknown-linux-gnu",
                    "x86_64-unknown-linux-musl",
                    "aarch64-unknown-linux-gnu",
                    "aarch64-unknown-linux-musl",
                    "x86_64-apple-darwin",
                    "aarch64-apple-darwin",
                    "x86_64-pc-windows-msvc",
                ],
            },
        },
        "files": ["index.js", "index.d.ts", "*.node"],
        "scripts": {
            "build": "napi build --platform --release",
            "build:debug": "napi build --platform",
        },
        "devDependencies": {"@napi-rs/cli": "^2.18.0"},
        "license": "MIT",
    }
    wfile(root, "rust/package.json", json.dumps(rust_pkg, indent=2), force=force)

    wfile(root, "rust/src/lib.rs", r"""//! siauth-crypto — RSA-2048 / RSA-SHA256 signature verification for SI Auth.
//!
//! Hash-then-sign pipeline (RSASSA-PKCS1-v1_5 + SHA-256):
//!   1. digest = SHA-256(canonical payload)                       // 32 bytes
//!   2. signature = RSA-2048-PKCS1v15-sign(DigestInfo(digest))    // 256 bytes

#![deny(clippy::all)]

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use napi_derive::napi;
use rsa::pkcs8::DecodePublicKey;
use rsa::traits::PublicKeyParts;
use rsa::{Pkcs1v15Sign, RsaPublicKey};
use sha2::{Digest, Sha256};

/// Minimum RSA modulus size, in bytes (2048 bits = 256 bytes).
const MIN_MODULUS_BYTES: usize = 256;

/// Verify an RSASSA-PKCS1-v1_5 SHA-256 signature.
///
/// * `spki_b64`  — base64 SPKI-DER RSA public key
/// * `payload`   — the canonical UTF-8 payload that was signed
/// * `sig_b64`   — base64 RSA-2048 signature (256 bytes)
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

    // Enforce RSA-2048 minimum. `size()` comes from `PublicKeyParts`.
    if key.size() < MIN_MODULUS_BYTES {
        return false;
    }

    // Step 1: SHA-256 over the canonical payload.
    let hashed = Sha256::digest(payload.as_bytes());

    // Steps 2 & 3: build DigestInfo, apply RSA-2048 PKCS#1 v1.5.
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
    format!(
        "siauth-crypto/{} (rustc {})",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_RUST_VERSION")
    )
}
""", force=force)

    wfile(root, "rust/README.md", """# siauth-crypto

Rust N-API accelerator for SI Auth. Verifies RSA-2048 / RSA-SHA256
signatures on the hot path. Uses the standard hash-then-sign pipeline
(SHA-256 digest -> RSA PKCS#1 v1.5).

Requires the `sha2` crate's `oid` feature so that `Sha256` implements
`AssociatedOid`, which is a bound on `Pkcs1v15Sign::new::<Sha256>()`.

Built by `npx napi build --platform --release`. If the build fails, the
TypeScript server falls back to `node:crypto`.
""", force=force)


# ═══════════════════════════════════════════════════════════════════════════
#  TypeScript server
# ═══════════════════════════════════════════════════════════════════════════

def build_server_files(root: Path, force: bool) -> None:
    log("TypeScript server (src/)", "step")

    wfile(root, "src/config.ts", r"""import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

// Zero-dependency .env loader.
(function loadDotenv(): void {
  const envPath = path.resolve(process.cwd(), ".env");
  let raw: string;
  try { raw = fs.readFileSync(envPath, "utf8"); } catch { return; }
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) continue;
    let value = trimmed.slice(eq + 1).trim();
    if (value.length >= 2) {
      const first = value[0];
      const last = value[value.length - 1];
      if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
        value = value.slice(1, -1);
      }
    }
    if (process.env[key] === undefined) process.env[key] = value;
  }
})();

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

function oneOf<T extends string>(name: string, allowed: readonly T[], fallback: T): T {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const lower = raw.toLowerCase();
  if ((allowed as readonly string[]).includes(lower)) return lower as T;
  throw new Error(`Invalid value for ${name}: ${raw}. Allowed: ${allowed.join(", ")}`);
}

let jwtSecret = process.env.JWT_SECRET;
if (!jwtSecret) {
  if (isProd) {
    throw new Error(
      "JWT_SECRET is required in production. Generate: node -e \"console.log(require('crypto').randomBytes(48).toString('hex'))\"",
    );
  }
  jwtSecret = crypto.randomBytes(48).toString("hex");
  console.warn(
    "[si-auth] WARNING: JWT_SECRET not set. Using ephemeral secret — sessions invalidated on restart.",
  );
}
if (jwtSecret.length < 32) throw new Error("JWT_SECRET must be at least 32 characters.");

const appOrigin = str("APP_ORIGIN", "http://localhost:3000").replace(/\/+$/, "");
const redisUrl = str("REDIS_URL", "").trim();
if (redisUrl && !/^rediss?:\/\//.test(redisUrl)) {
  throw new Error(`Invalid REDIS_URL: must start with redis:// or rediss:// — got ${redisUrl}`);
}

const userStoreBackend = oneOf(
  "USER_STORE_BACKEND",
  ["json", "postgres"] as const,
  "json",
);
const databaseUrl = str("DATABASE_URL", "").trim();
if (userStoreBackend === "postgres") {
  if (!databaseUrl) {
    throw new Error("USER_STORE_BACKEND=postgres requires DATABASE_URL to be set.");
  }
  if (!/^postgres(ql)?:\/\//.test(databaseUrl)) {
    throw new Error(`Invalid DATABASE_URL: must start with postgres:// or postgresql:// — got ${databaseUrl}`);
  }
}

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
  readonly logLevel: string;
  readonly protocolVersion: string;
  readonly algorithm: string;
  readonly hash: string;
  readonly modulusLength: number;
  readonly redisUrl: string;
  readonly revocationEnabled: boolean;
  readonly metricsEnabled: boolean;
  readonly metricsToken: string;
  readonly userStoreBackend: "json" | "postgres";
  readonly databaseUrl: string;
  readonly databaseAutoMigrate: boolean;
}

const samesiteValues = ["strict", "lax", "none"] as const;

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
  cookieSameSite: oneOf("COOKIE_SAMESITE", samesiteValues, "strict"),
  cookieName: "si_auth_session",

  userEnumerationSafe: bool("USER_ENUMERATION_SAFE", true),

  logLevel: str("LOG_LEVEL", "info").toLowerCase(),

  protocolVersion: "SI-AUTH-V1",
  algorithm: "RSASSA-PKCS1-v1_5",
  hash: "SHA-256",
  modulusLength: 2048,

  redisUrl,
  revocationEnabled: bool("REVOCATION_ENABLED", true),
  metricsEnabled: bool("METRICS_ENABLED", true),
  metricsToken: str("METRICS_TOKEN", ""),

  userStoreBackend,
  databaseUrl,
  databaseAutoMigrate: bool("DATABASE_AUTO_MIGRATE", true),
});
""", force=force)

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
  info: (m: string, x?: unknown) => emit("info", m, x),
  warn: (m: string, x?: unknown) => emit("warn", m, x),
  error: (m: string, x?: unknown) => emit("error", m, x),
};
""", force=force)

    wfile(root, "src/lib/redis.ts", r"""/**
 * src/lib/redis.ts — zero-dependency Redis client.
 *
 * Implements just enough of RESP (REdis Serialization Protocol) over
 * node:net / node:tls to support the operations SI Auth needs:
 *
 *   PING
 *   SET <k> <v> [PX ms] [NX]      -> "OK" | null (when NX fails)
 *   GETDEL <k>                    -> string | null
 *   DEL <k...>                    -> number
 *   INCR <k>                      -> number
 *   EXPIRE <k> <sec>              -> 0 | 1
 *   TTL <k>                       -> number
 *   EXISTS <k...>                 -> number
 *
 * Design:
 *   - One TCP socket, reused across commands.
 *   - Commands are pipelined; responses are matched FIFO.
 *   - On socket error, all pending commands are rejected and the next
 *     call will reconnect.
 *   - No retries are hidden. Callers see failures.
 */

import net from "node:net";
import tls from "node:tls";
import { logger } from "./logger.ts";

export type RespValue =
  | string
  | number
  | null
  | RespValue[]
  | { error: string };

interface Pending {
  resolve: (v: RespValue) => void;
  reject: (e: Error) => void;
}

export class RedisClient {
  private readonly host: string;
  private readonly port: number;
  private readonly tls: boolean;
  private socket: net.Socket | null = null;
  private buffer = Buffer.alloc(0);
  private queue: Pending[] = [];
  private connecting: Promise<void> | null = null;
  private closed = false;

  constructor(url: string) {
    const u = new URL(url);
    if (u.protocol !== "redis:" && u.protocol !== "rediss:") {
      throw new Error(`Unsupported redis scheme: ${u.protocol}`);
    }
    this.host = u.hostname;
    this.port = u.port ? Number.parseInt(u.port, 10) : 6379;
    this.tls = u.protocol === "rediss:";
  }

  describe(): string {
    return `${this.tls ? "rediss" : "redis"}://${this.host}:${this.port}`;
  }

  private ensureConnected(): Promise<void> {
    if (this.closed) return Promise.reject(new Error("redis client closed"));
    if (this.socket && !this.socket.destroyed) return Promise.resolve();
    if (this.connecting) return this.connecting;

    this.connecting = new Promise<void>((resolve, reject) => {
      const onError = (err: Error) => {
        this.connecting = null;
        reject(err);
      };
      let sock: net.Socket;
      if (this.tls) {
        sock = tls.connect({ host: this.host, port: this.port });
      } else {
        sock = net.createConnection({ host: this.host, port: this.port });
      }
      sock.setNoDelay(true);
      const onReady = () => {
        sock.removeListener("error", onError);
        this.socket = sock;
        this.buffer = Buffer.alloc(0);
        this.connecting = null;
        sock.on("data", (chunk) => this.onData(chunk));
        sock.on("error", (err) => {
          logger.warn("redis socket error", { error: err.message, target: this.describe() });
          this.failPending(err);
        });
        sock.on("close", () => {
          this.socket = null;
          this.failPending(new Error("redis socket closed"));
        });
        resolve();
      };
      sock.once(this.tls ? "secureConnect" : "connect", onReady);
      sock.once("error", onError);
    });
    return this.connecting;
  }

  private failPending(err: Error): void {
    const pending = this.queue;
    this.queue = [];
    for (const p of pending) p.reject(err);
  }

  private onData(chunk: Buffer): void {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (true) {
      const parsed = this.tryParse(this.buffer);
      if (!parsed) break;
      this.buffer = this.buffer.subarray(parsed.consumed);
      const pending = this.queue.shift();
      if (!pending) {
        logger.warn("redis response with no pending command");
        continue;
      }
      if (typeof parsed.value === "object" && parsed.value !== null && !Array.isArray(parsed.value) && "error" in parsed.value) {
        pending.reject(new Error(`redis: ${(parsed.value as { error: string }).error}`));
      } else {
        pending.resolve(parsed.value);
      }
    }
  }

  private tryParse(buf: Buffer): { value: RespValue; consumed: number } | null {
    if (buf.length === 0) return null;
    const type = String.fromCharCode(buf[0]);
    const crlf = buf.indexOf("\r\n", 1);
    if (crlf === -1) return null;
    const line = buf.subarray(1, crlf).toString("utf8");

    switch (type) {
      case "+":
        return { value: line, consumed: crlf + 2 };
      case "-":
        return { value: { error: line }, consumed: crlf + 2 };
      case ":":
        return { value: Number.parseInt(line, 10), consumed: crlf + 2 };
      case "$": {
        const len = Number.parseInt(line, 10);
        if (len === -1) return { value: null, consumed: crlf + 2 };
        const end = crlf + 2 + len + 2;
        if (buf.length < end) return null;
        const data = buf.subarray(crlf + 2, crlf + 2 + len).toString("utf8");
        return { value: data, consumed: end };
      }
      case "*": {
        const n = Number.parseInt(line, 10);
        if (n === -1) return { value: null, consumed: crlf + 2 };
        const arr: RespValue[] = [];
        let offset = crlf + 2;
        for (let i = 0; i < n; i++) {
          const sub = this.tryParse(buf.subarray(offset));
          if (!sub) return null;
          arr.push(sub.value);
          offset += sub.consumed;
        }
        return { value: arr, consumed: offset };
      }
      default:
        throw new Error(`unknown RESP type: ${type}`);
    }
  }

  private encode(args: (string | number)[]): Buffer {
    const parts: Buffer[] = [Buffer.from(`*${args.length}\r\n`, "utf8")];
    for (const a of args) {
      const b = Buffer.from(typeof a === "number" ? String(a) : a, "utf8");
      parts.push(Buffer.from(`$${b.length}\r\n`, "utf8"));
      parts.push(b);
      parts.push(Buffer.from("\r\n", "utf8"));
    }
    return Buffer.concat(parts);
  }

  async send(...args: (string | number)[]): Promise<RespValue> {
    await this.ensureConnected();
    if (!this.socket) throw new Error("redis not connected");
    const sock = this.socket;
    return new Promise<RespValue>((resolve, reject) => {
      this.queue.push({ resolve, reject });
      sock.write(this.encode(args), (err) => {
        if (err) {
          const idx = this.queue.findIndex((p) => p.resolve === resolve);
          if (idx >= 0) this.queue.splice(idx, 1);
          reject(err);
        }
      });
    });
  }

  async ping(): Promise<boolean> {
    try {
      const r = await this.send("PING");
      return r === "PONG";
    } catch {
      return false;
    }
  }

  async set(key: string, value: string, opts: { px?: number; nx?: boolean } = {}): Promise<string | null> {
    const args: (string | number)[] = ["SET", key, value];
    if (opts.px !== undefined) args.push("PX", opts.px);
    if (opts.nx) args.push("NX");
    const r = await this.send(...args);
    if (r === null) return null;
    if (typeof r === "string") return r;
    throw new Error(`unexpected SET reply: ${JSON.stringify(r)}`);
  }

  async getdel(key: string): Promise<string | null> {
    const r = await this.send("GETDEL", key);
    if (r === null) return null;
    if (typeof r === "string") return r;
    throw new Error(`unexpected GETDEL reply: ${JSON.stringify(r)}`);
  }

  async del(...keys: string[]): Promise<number> {
    if (keys.length === 0) return 0;
    const r = await this.send("DEL", ...keys);
    return typeof r === "number" ? r : 0;
  }

  async incr(key: string): Promise<number> {
    const r = await this.send("INCR", key);
    if (typeof r === "number") return r;
    throw new Error(`unexpected INCR reply: ${JSON.stringify(r)}`);
  }

  async expire(key: string, seconds: number): Promise<boolean> {
    const r = await this.send("EXPIRE", key, seconds);
    return r === 1;
  }

  async ttl(key: string): Promise<number> {
    const r = await this.send("TTL", key);
    return typeof r === "number" ? r : -2;
  }

  async exists(...keys: string[]): Promise<number> {
    if (keys.length === 0) return 0;
    const r = await this.send("EXISTS", ...keys);
    return typeof r === "number" ? r : 0;
  }

  async close(): Promise<void> {
    this.closed = true;
    const sock = this.socket;
    this.socket = null;
    if (sock) {
      await new Promise<void>((resolve) => {
        sock.end(() => resolve());
        setTimeout(resolve, 500).unref?.();
      });
    }
    this.failPending(new Error("redis client closed"));
  }
}

import { config } from "../config.ts";

let shared: RedisClient | null = null;

export function getRedis(): RedisClient | null {
  if (!config.redisUrl) return null;
  if (!shared) shared = new RedisClient(config.redisUrl);
  return shared;
}

export const redisEnabled = (): boolean => config.redisUrl !== "";
""", force=force)

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

    wfile(root, "src/lib/crypto.ts", r"""/**
 * src/lib/crypto.ts — server-side RSA-2048 / RSA-SHA256 verification.
 */
import crypto from "node:crypto";
import { createRequire } from "node:module";
import { config } from "../config.ts";
import { logger } from "./logger.ts";

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

export function generateChallenge(): string {
  if (rust) {
    try {
      return rust.generateChallenge();
    } catch (err) {
      if (useRust === "true") throw err;
    }
  }
  return crypto.randomBytes(32).toString("hex");
}

export function generateId(prefix: string): string {
  return `${prefix}_${crypto.randomBytes(16).toString("hex")}`;
}

export function generateJti(): string {
  return crypto.randomBytes(16).toString("hex");
}

export function fingerprintPublicKey(spkiBase64: string): string {
  const der = Buffer.from(spkiBase64, "base64");
  return crypto.createHash("sha256").update(der).digest("hex");
}

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
  ) {
    return false;
  }

  if (rust) {
    try {
      return rust.verifySignature(publicKeySpkiBase64, payload, signatureBase64);
    } catch (err) {
      if (useRust === "true") throw err;
    }
  }

  try {
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
    try {
      return rust.publicKeyBits(spkiBase64) >= config.modulusLength;
    } catch (err) {
      if (useRust === "true") throw err;
    }
  }
  try {
    const key = crypto.createPublicKey({
      key: Buffer.from(spkiBase64, "base64"),
      format: "der",
      type: "spki",
    });
    const bits =
      (key.asymmetricKeyDetails as { modulusLength?: number } | undefined)?.modulusLength ?? 0;
    return key.asymmetricKeyType === "rsa" && bits >= config.modulusLength;
  } catch {
    return false;
  }
}
""", force=force)

    wfile(root, "src/lib/metrics.ts", r"""/**
 * src/lib/metrics.ts — in-process Prometheus counters, gauges, and a
 * request-duration histogram. No external dependency.
 */

type Labels = Record<string, string>;

const counters = new Map<string, number>();
const gauges = new Map<string, number>();
const histogramBuckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10];
const histogramCounts = new Map<string, number[]>();
const histogramSums = new Map<string, number>();
const histogramTotals = new Map<string, number>();

function labelKey(name: string, labels: Labels): string {
  const keys = Object.keys(labels).sort();
  if (keys.length === 0) return name;
  const parts = keys.map((k) => `${k}="${escapeLabel(labels[k])}"`);
  return `${name}{${parts.join(",")}}`;
}

function escapeLabel(v: string): string {
  return v.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");
}

export function incCounter(name: string, labels: Labels = {}, by = 1): void {
  const key = labelKey(name, labels);
  counters.set(key, (counters.get(key) ?? 0) + by);
}

export function setGauge(name: string, value: number, labels: Labels = {}): void {
  gauges.set(labelKey(name, labels), value);
}

export function observeHistogram(name: string, seconds: number, labels: Labels = {}): void {
  const key = labelKey(name, labels);
  let counts = histogramCounts.get(key);
  if (!counts) {
    counts = new Array(histogramBuckets.length).fill(0);
    histogramCounts.set(key, counts);
  }
  for (let i = 0; i < histogramBuckets.length; i++) {
    if (seconds <= histogramBuckets[i]) counts[i] += 1;
  }
  histogramSums.set(key, (histogramSums.get(key) ?? 0) + seconds);
  histogramTotals.set(key, (histogramTotals.get(key) ?? 0) + 1);
}

interface ParsedKey { name: string; labels: string; }

function splitKey(key: string): ParsedKey {
  const i = key.indexOf("{");
  if (i === -1) return { name: key, labels: "" };
  return { name: key.slice(0, i), labels: key.slice(i) };
}

export function renderMetrics(): string {
  const lines: string[] = [];

  const counterNames = new Map<string, string[]>();
  for (const key of counters.keys()) {
    const { name } = splitKey(key);
    if (!counterNames.has(name)) counterNames.set(name, []);
    counterNames.get(name)!.push(key);
  }
  for (const [name, keys] of counterNames) {
    lines.push(`# TYPE ${name} counter`);
    for (const key of keys) lines.push(`${key} ${counters.get(key)}`);
  }

  const gaugeNames = new Map<string, string[]>();
  for (const key of gauges.keys()) {
    const { name } = splitKey(key);
    if (!gaugeNames.has(name)) gaugeNames.set(name, []);
    gaugeNames.get(name)!.push(key);
  }
  for (const [name, keys] of gaugeNames) {
    lines.push(`# TYPE ${name} gauge`);
    for (const key of keys) lines.push(`${key} ${gauges.get(key)}`);
  }

  const histNames = new Map<string, string[]>();
  for (const key of histogramCounts.keys()) {
    const { name } = splitKey(key);
    if (!histNames.has(name)) histNames.set(name, []);
    histNames.get(name)!.push(key);
  }
  for (const [name, keys] of histNames) {
    lines.push(`# TYPE ${name} histogram`);
    for (const key of keys) {
      const { labels } = splitKey(key);
      const inner = labels ? labels.slice(1, -1) : "";
      const prefix = inner ? `${inner},le=` : "le=";
      const counts = histogramCounts.get(key)!;
      for (let i = 0; i < histogramBuckets.length; i++) {
        lines.push(`${name}_bucket{${prefix}"${histogramBuckets[i]}"} ${counts[i]}`);
      }
      lines.push(`${name}_bucket{${prefix}"+Inf"} ${histogramTotals.get(key) ?? 0}`);
      lines.push(`${name}_sum${labels} ${histogramSums.get(key) ?? 0}`);
      lines.push(`${name}_count${labels} ${histogramTotals.get(key) ?? 0}`);
    }
  }

  return lines.join("\n") + "\n";
}
""", force=force)

    # ── userStore interface + factory ──────────────────────────────────
    wfile(root, "src/services/userStore.ts", r"""/**
 * src/services/userStore.ts — the canonical user/credential store
 * interface, plus a factory that selects a backend at boot.
 *
 * Two backends ship in-tree:
 *
 *   - json      -> ./data/users.json (default, single-node)
 *   - postgres  -> uses the `pg` package (opt-in via USER_STORE_BACKEND)
 *
 * The interface is intentionally small so that any deployment can drop
 * in its own implementation (MySQL, MongoDB, DynamoDB, ...) without
 * touching routes. See docs/ARCHITECTURE.md for the swap contract.
 */

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

export interface PublicCredential {
  id: string;
  label: string;
  fingerprint: string;
  createdAt: string;
  lastUsedAt: string | null;
}

export interface UserStore {
  findByUsername(username: string): Promise<UserRecord | null>;
  findById(id: string): Promise<UserRecord | null>;
  create(input: {
    username: string;
    publicKeySpki: string;
    deviceLabel?: string;
  }): Promise<{ user: UserRecord; credential: CredentialRecord }>;
  addCredential(
    username: string,
    input: { publicKeySpki: string; deviceLabel?: string },
  ): Promise<CredentialRecord | null>;
  listCredentials(userId: string): Promise<PublicCredential[]>;
  removeCredential(userId: string, credentialId: string): Promise<boolean>;
  touchCredential(userId: string, credentialId: string): Promise<void>;
  /** Test-only: wipe every user. */
  _reset(): Promise<void>;
  /** Graceful shutdown hook. */
  close(): Promise<void>;
}

import { config } from "../config.ts";
import { logger } from "../lib/logger.ts";
import { createJsonUserStore } from "./stores/jsonUserStore.ts";

let active: UserStore | null = null;

function make(): UserStore {
  if (config.userStoreBackend === "postgres") {
    // Lazy import keeps `pg` truly optional: the default install never
    // loads it, and a misconfigured boot fails with a clear message.
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { createPostgresUserStore } = require("./stores/postgresUserStore.ts") as {
        createPostgresUserStore: (url: string, autoMigrate: boolean) => UserStore;
      };
      const store = createPostgresUserStore(config.databaseUrl, config.databaseAutoMigrate);
      logger.info("userStore: using Postgres backend");
      return store;
    } catch (err) {
      const msg = (err as Error).message;
      logger.error("userStore: failed to load Postgres backend", { error: msg });
      throw new Error(
        "USER_STORE_BACKEND=postgres requires the `pg` package. " +
        "Install it with `npm install pg` and restart. " +
        `Underlying error: ${msg}`,
      );
    }
  }
  logger.info("userStore: using JSON-file backend (single-node)");
  return createJsonUserStore();
}

export function getUserStore(): UserStore {
  if (!active) active = make();
  return active;
}

/** Convenience proxy so routes can call `userStore.findByUsername(...)`. */
export const userStore: UserStore = new Proxy({} as UserStore, {
  get(_target, prop: string) {
    const impl = getUserStore() as unknown as Record<string, unknown>;
    const value = impl[prop];
    if (typeof value === "function") return value.bind(impl);
    return value;
  },
});

export const userStoreBackend = (): "json" | "postgres" => config.userStoreBackend;
""", force=force)

    wfile(root, "src/services/stores/jsonUserStore.ts", r"""/**
 * JSON-file user store. Single-node only.
 *
 * Uses one serialization chain covering both reads and writes so that
 * read-modify-write operations cannot interleave. The temp-file +
 * rename pattern ensures each write is atomic on POSIX.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { config } from "../../config.ts";
import { fingerprintPublicKey, generateId } from "../../lib/crypto.ts";
import type {
  CredentialRecord,
  PublicCredential,
  UserRecord,
  UserStore,
} from "../userStore.ts";

interface DB { users: UserRecord[]; }

export function createJsonUserStore(): UserStore {
  const FILE = path.join(config.dataDir, "users.json");
  let opChain: Promise<unknown> = Promise.resolve();

  function serialize<T>(fn: () => Promise<T>): Promise<T> {
    const result = opChain.then(fn);
    opChain = result.then(() => undefined, () => undefined);
    return result;
  }

  async function readAllUnsafe(): Promise<DB> {
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

  async function writeAllUnsafe(data: DB): Promise<void> {
    await fs.mkdir(config.dataDir, { recursive: true });
    const tmp = `${FILE}.${process.pid}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(data, null, 2), "utf8");
    await fs.rename(tmp, FILE);
  }

  return {
    findByUsername(username: string): Promise<UserRecord | null> {
      return serialize(async () => {
        const { users } = await readAllUnsafe();
        const key = username.toLowerCase();
        return users.find((u) => u.username.toLowerCase() === key) ?? null;
      });
    },

    findById(id: string): Promise<UserRecord | null> {
      return serialize(async () => {
        const { users } = await readAllUnsafe();
        return users.find((u) => u.id === id) ?? null;
      });
    },

    create(input): Promise<{ user: UserRecord; credential: CredentialRecord }> {
      return serialize(async () => {
        const data = await readAllUnsafe();
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
        await writeAllUnsafe(data);
        return { user, credential };
      });
    },

    addCredential(username, input): Promise<CredentialRecord | null> {
      return serialize(async () => {
        const data = await readAllUnsafe();
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
        await writeAllUnsafe(data);
        return credential;
      });
    },

    listCredentials(userId): Promise<PublicCredential[]> {
      return serialize(async () => {
        const data = await readAllUnsafe();
        const user = data.users.find((u) => u.id === userId);
        if (!user) return [];
        return user.credentials.map(({ publicKeySpki: _drop, ...rest }) => rest);
      });
    },

    removeCredential(userId, credentialId): Promise<boolean> {
      return serialize(async () => {
        const data = await readAllUnsafe();
        const user = data.users.find((u) => u.id === userId);
        if (!user) return false;
        const before = user.credentials.length;
        user.credentials = user.credentials.filter((c) => c.id !== credentialId);
        if (user.credentials.length === before) return false;
        await writeAllUnsafe(data);
        return true;
      });
    },

    touchCredential(userId, credentialId): Promise<void> {
      return serialize(async () => {
        const data = await readAllUnsafe();
        const user = data.users.find((u) => u.id === userId);
        if (!user) return;
        const cred = user.credentials.find((c) => c.id === credentialId);
        if (!cred) return;
        cred.lastUsedAt = new Date().toISOString();
        await writeAllUnsafe(data);
      });
    },

    _reset(): Promise<void> {
      return serialize(async () => { await writeAllUnsafe({ users: [] }); });
    },

    async close(): Promise<void> {
      await opChain.catch(() => undefined);
    },
  };
}
""", force=force)

    wfile(root, "src/services/stores/postgresUserStore.ts", r"""/**
 * Postgres user store. Multi-replica safe.
 *
 * Loaded lazily: the `pg` package is required only when
 * USER_STORE_BACKEND=postgres. A default install never loads this file.
 *
 * Schema (auto-created when DATABASE_AUTO_MIGRATE=true):
 *
 *   users(id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, created_at TIMESTAMPTZ)
 *   credentials(id TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
 *               label TEXT, public_key_spki TEXT, fingerprint TEXT,
 *               created_at TIMESTAMPTZ, last_used_at TIMESTAMPTZ)
 *
 * The interface is identical to the JSON backend, so routes and tests do
 * not change. `SELECT ... FOR UPDATE` is avoided: uniqueness is enforced
 * by the DB constraint, and the `create` method translates
 * unique-violation (SQLSTATE 23505) into a USER_EXISTS error that matches
 * the JSON backend's contract.
 */
import { createRequire } from "node:module";
import { fingerprintPublicKey, generateId } from "../../lib/crypto.ts";
import type {
  CredentialRecord,
  PublicCredential,
  UserRecord,
  UserStore,
} from "../userStore.ts";

const require = createRequire(import.meta.url);

interface PgClient {
  query(sql: string, params?: unknown[]): Promise<{ rows: Record<string, unknown>[]; rowCount: number | null }>;
  end(): Promise<void>;
}

interface PgModule {
  Client: new (opts: { connectionString: string }) => PgClient;
}

const MIGRATE_SQL = [
  `CREATE TABLE IF NOT EXISTS users (
     id TEXT PRIMARY KEY,
     username TEXT UNIQUE NOT NULL,
     created_at TIMESTAMPTZ NOT NULL DEFAULT now()
   )`,
  `CREATE TABLE IF NOT EXISTS credentials (
     id TEXT PRIMARY KEY,
     user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
     label TEXT NOT NULL,
     public_key_spki TEXT NOT NULL,
     fingerprint TEXT NOT NULL,
     created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
     last_used_at TIMESTAMPTZ
   )`,
  `CREATE INDEX IF NOT EXISTS credentials_user_id_idx ON credentials(user_id)`,
  `CREATE INDEX IF NOT EXISTS users_username_lower_idx ON users((lower(username)))`,
];

function rowToUser(row: { id: string; username: string; created_at: Date }, creds: CredentialRecord[]): UserRecord {
  return {
    id: row.id,
    username: row.username,
    createdAt: new Date(row.created_at).toISOString(),
    credentials: creds,
  };
}

function rowToCredential(row: Record<string, unknown>): CredentialRecord {
  return {
    id: String(row.id),
    label: String(row.label),
    publicKeySpki: String(row.public_key_spki),
    fingerprint: String(row.fingerprint),
    createdAt: new Date(row.created_at as string | Date).toISOString(),
    lastUsedAt: row.last_used_at ? new Date(row.last_used_at as string | Date).toISOString() : null,
  };
}

export function createPostgresUserStore(connectionString: string, autoMigrate: boolean): UserStore {
  const pg = require("pg") as PgModule;
  const client = new pg.Client({ connectionString });

  // Connect synchronously? No — but we can kick off the connect and
  // await it lazily on first use. We keep a single in-flight promise so
  // concurrent callers share the connection setup.
  let ready: Promise<void> | null = null;

  async function ensureReady(): Promise<void> {
    if (!ready) {
      ready = (async () => {
        await client.query("SELECT 1");
        if (autoMigrate) {
          for (const stmt of MIGRATE_SQL) await client.query(stmt);
        }
      })();
    }
    return ready;
  }

  async function fetchUserById(id: string): Promise<UserRecord | null> {
    const userRes = await client.query(
      "SELECT id, username, created_at FROM users WHERE id = $1",
      [id],
    );
    if (userRes.rows.length === 0) return null;
    const credRes = await client.query(
      "SELECT id, label, public_key_spki, fingerprint, created_at, last_used_at " +
      "FROM credentials WHERE user_id = $1 ORDER BY created_at ASC",
      [id],
    );
    const creds = credRes.rows.map(rowToCredential);
    const row = userRes.rows[0] as { id: string; username: string; created_at: Date };
    return rowToUser(row, creds);
  }

  return {
    async findByUsername(username: string): Promise<UserRecord | null> {
      await ensureReady();
      const key = username.toLowerCase();
      const res = await client.query(
        "SELECT id FROM users WHERE lower(username) = $1 LIMIT 1",
        [key],
      );
      if (res.rows.length === 0) return null;
      return fetchUserById(String(res.rows[0].id));
    },

    async findById(id: string): Promise<UserRecord | null> {
      await ensureReady();
      return fetchUserById(id);
    },

    async create(input): Promise<{ user: UserRecord; credential: CredentialRecord }> {
      await ensureReady();
      const now = new Date();
      const userId = generateId("user");
      const credentialId = generateId("cred");
      const fingerprint = fingerprintPublicKey(input.publicKeySpki);
      const label = input.deviceLabel || "Unknown device";

      // Insert user, then credential. Wrap in a transaction so a failed
      // credential insert does not leave a user without credentials.
      await client.query("BEGIN");
      try {
        await client.query(
          "INSERT INTO users (id, username, created_at) VALUES ($1, $2, $3)",
          [userId, input.username, now],
        );
        await client.query(
          "INSERT INTO credentials (id, user_id, label, public_key_spki, fingerprint, created_at) " +
          "VALUES ($1, $2, $3, $4, $5, $6)",
          [credentialId, userId, label, input.publicKeySpki, fingerprint, now],
        );
        await client.query("COMMIT");
      } catch (err) {
        await client.query("ROLLBACK").catch(() => undefined);
        const code = (err as { code?: string }).code;
        if (code === "23505") {
          const e = new Error("Username already registered") as Error & { code?: string };
          e.code = "USER_EXISTS";
          throw e;
        }
        throw err;
      }

      const credential: CredentialRecord = {
        id: credentialId,
        label,
        publicKeySpki: input.publicKeySpki,
        fingerprint,
        createdAt: now.toISOString(),
        lastUsedAt: null,
      };
      const user: UserRecord = {
        id: userId,
        username: input.username,
        createdAt: now.toISOString(),
        credentials: [credential],
      };
      return { user, credential };
    },

    async addCredential(username, input): Promise<CredentialRecord | null> {
      await ensureReady();
      const key = username.toLowerCase();
      const res = await client.query(
        "SELECT id FROM users WHERE lower(username) = $1 LIMIT 1",
        [key],
      );
      if (res.rows.length === 0) return null;
      const userId = String(res.rows[0].id);
      const now = new Date();
      const credentialId = generateId("cred");
      const fingerprint = fingerprintPublicKey(input.publicKeySpki);
      const label = input.deviceLabel || "Unknown device";
      await client.query(
        "INSERT INTO credentials (id, user_id, label, public_key_spki, fingerprint, created_at) " +
        "VALUES ($1, $2, $3, $4, $5, $6)",
        [credentialId, userId, label, input.publicKeySpki, fingerprint, now],
      );
      return {
        id: credentialId,
        label,
        publicKeySpki: input.publicKeySpki,
        fingerprint,
        createdAt: now.toISOString(),
        lastUsedAt: null,
      };
    },

    async listCredentials(userId): Promise<PublicCredential[]> {
      await ensureReady();
      const res = await client.query(
        "SELECT id, label, fingerprint, created_at, last_used_at " +
        "FROM credentials WHERE user_id = $1 ORDER BY created_at ASC",
        [userId],
      );
      return res.rows.map((row) => ({
        id: String(row.id),
        label: String(row.label),
        fingerprint: String(row.fingerprint),
        createdAt: new Date(row.created_at as string | Date).toISOString(),
        lastUsedAt: row.last_used_at ? new Date(row.last_used_at as string | Date).toISOString() : null,
      }));
    },

    async removeCredential(userId, credentialId): Promise<boolean> {
      await ensureReady();
      const res = await client.query(
        "DELETE FROM credentials WHERE user_id = $1 AND id = $2",
        [userId, credentialId],
      );
      return (res.rowCount ?? 0) > 0;
    },

    async touchCredential(userId, credentialId): Promise<void> {
      await ensureReady();
      await client.query(
        "UPDATE credentials SET last_used_at = $1 WHERE user_id = $2 AND id = $3",
        [new Date(), userId, credentialId],
      );
    },

    async _reset(): Promise<void> {
      await ensureReady();
      await client.query("TRUNCATE credentials, users RESTART IDENTITY CASCADE");
    },

    async close(): Promise<void> {
      try {
        await client.end();
      } catch {
        /* ignore */
      }
    },
  };
}
""", force=force)

    wfile(root, "src/services/challengeStore.ts", r"""/**
 * Nonce store. Single-use, TTL-bound.
 *
 * Storage backend is chosen at boot:
 *   - Redis when REDIS_URL is set (shared across replicas).
 *   - In-memory otherwise (single process only).
 */
import { config } from "../config.ts";
import { getRedis } from "../lib/redis.ts";
import { generateChallenge, generateId } from "../lib/crypto.ts";
import { logger } from "../lib/logger.ts";

const PREFIX = "si-auth:ch:";

export interface ChallengeRecord {
  id: string;
  purpose: "login" | "register";
  username: string;
  challenge: string;
  origin: string;
  issuedAt: string;
  expiresAt: number;
}

export interface ChallengeStore {
  issue(opts: {
    purpose: "login" | "register";
    username: string;
    origin: string;
    ttlSeconds: number;
  }): Promise<ChallengeRecord>;
  take(id: string): Promise<ChallengeRecord | null>;
  clear(): Promise<void>;
  size(): Promise<number>;
}

class MemoryChallengeStore implements ChallengeStore {
  private map = new Map<string, ChallengeRecord>();
  private sweeper: NodeJS.Timeout;

  constructor() {
    this.sweeper = setInterval(() => this.sweep(), 30_000);
    this.sweeper.unref?.();
  }

  async issue(opts): Promise<ChallengeRecord> {
    const id = generateId("ch");
    const challenge = generateChallenge();
    const issuedAt = new Date().toISOString();
    const record: ChallengeRecord = {
      id, purpose: opts.purpose, username: opts.username, challenge,
      origin: opts.origin, issuedAt,
      expiresAt: Date.now() + opts.ttlSeconds * 1000,
    };
    this.map.set(id, record);
    return record;
  }

  async take(id: string): Promise<ChallengeRecord | null> {
    if (typeof id !== "string") return null;
    const rec = this.map.get(id);
    if (!rec) return null;
    this.map.delete(id);
    if (rec.expiresAt < Date.now()) return null;
    return rec;
  }

  async clear(): Promise<void> { this.map.clear(); }
  async size(): Promise<number> { return this.map.size; }

  private sweep(): void {
    const now = Date.now();
    for (const [id, rec] of this.map) if (rec.expiresAt < now) this.map.delete(id);
  }
}

class RedisChallengeStore implements ChallengeStore {
  async issue(opts): Promise<ChallengeRecord> {
    const redis = getRedis();
    if (!redis) throw new Error("redis not configured");
    const id = generateId("ch");
    const challenge = generateChallenge();
    const issuedAt = new Date().toISOString();
    const record: ChallengeRecord = {
      id, purpose: opts.purpose, username: opts.username, challenge,
      origin: opts.origin, issuedAt,
      expiresAt: Date.now() + opts.ttlSeconds * 1000,
    };
    const reply = await redis.set(PREFIX + id, JSON.stringify(record), {
      px: opts.ttlSeconds * 1000, nx: true,
    });
    if (reply !== null) return record;
    // 128-bit collision — retry once, then fail loudly.
    const id2 = generateId("ch");
    const record2: ChallengeRecord = { ...record, id: id2 };
    const reply2 = await redis.set(PREFIX + id2, JSON.stringify(record2), {
      px: opts.ttlSeconds * 1000, nx: true,
    });
    if (reply2 === null) throw new Error("challenge id collision");
    return record2;
  }

  async take(id: string): Promise<ChallengeRecord | null> {
    if (typeof id !== "string") return null;
    const redis = getRedis();
    if (!redis) throw new Error("redis not configured");
    const raw = await redis.getdel(PREFIX + id);
    if (!raw) return null;
    try {
      const rec = JSON.parse(raw) as ChallengeRecord;
      if (typeof rec.expiresAt !== "number" || rec.expiresAt < Date.now()) return null;
      return rec;
    } catch { return null; }
  }

  async clear(): Promise<void> { /* Redis handles expiry */ }
  async size(): Promise<number> { return 0; }
}

function make(): ChallengeStore {
  if (config.redisUrl) {
    logger.info("challengeStore: using Redis backend",
      { url: config.redisUrl.replace(/:[^:@/]*@/, ":***@") });
    return new RedisChallengeStore();
  }
  logger.info("challengeStore: using in-memory backend (single-process only)");
  return new MemoryChallengeStore();
}

export const challengeStore: ChallengeStore = make();
export const challengeStoreBackend = (): "redis" | "memory" =>
  (config.redisUrl ? "redis" : "memory");
""", force=force)

    wfile(root, "src/services/revocation.ts", r"""/**
 * src/services/revocation.ts — session revocation list.
 *
 * Bounded by session TTL: entries are dropped when the revoked token
 * would have expired anyway. Backend: Redis when REDIS_URL is set,
 * in-memory otherwise.
 */
import { config } from "../config.ts";
import { getRedis } from "../lib/redis.ts";
import { logger } from "../lib/logger.ts";

const PREFIX = "si-auth:revoke:";

class MemoryRevocation {
  private map = new Map<string, number>();
  private sweeper: NodeJS.Timeout;

  constructor() {
    this.sweeper = setInterval(() => this.sweep(), 60_000);
    this.sweeper.unref?.();
  }

  revoke(jti: string, expiresAtMs: number): void { this.map.set(jti, expiresAtMs); }

  isRevoked(jti: string): boolean {
    const exp = this.map.get(jti);
    if (exp === undefined) return false;
    if (exp <= Date.now()) { this.map.delete(jti); return false; }
    return true;
  }

  private sweep(): void {
    const now = Date.now();
    for (const [jti, exp] of this.map) if (exp <= now) this.map.delete(jti);
  }

  size(): number { return this.map.size; }
  clear(): void { this.map.clear(); }
}

const memory = new MemoryRevocation();

export async function revoke(jti: string, expiresAtSeconds: number): Promise<void> {
  if (!config.revocationEnabled) return;
  const nowSec = Math.floor(Date.now() / 1000);
  const ttlSec = Math.max(0, expiresAtSeconds - nowSec);
  if (ttlSec === 0) return;

  const redis = getRedis();
  if (redis) {
    try {
      await redis.set(PREFIX + jti, "1", { px: ttlSec * 1000 });
      return;
    } catch (err) {
      logger.warn("revocation: redis write failed; falling back to memory",
        { error: (err as Error).message });
    }
  }
  memory.revoke(jti, expiresAtSeconds * 1000);
}

export async function isRevoked(jti: string): Promise<boolean> {
  if (!config.revocationEnabled) return false;

  const redis = getRedis();
  if (redis) {
    try {
      const n = await redis.exists(PREFIX + jti);
      if (n > 0) return true;
    } catch (err) {
      logger.warn("revocation: redis read failed; falling back to memory",
        { error: (err as Error).message });
    }
  }
  return memory.isRevoked(jti);
}

export const revocationStats = (): { backend: "redis" | "memory"; memorySize: number } => ({
  backend: getRedis() ? "redis" : "memory",
  memorySize: memory.size(),
});

export const revocationTestHooks = { memory };
""", force=force)

    wfile(root, "src/services/token.ts", r"""/**
 * Minimal HS256 JWT (sign + verify) using node:crypto only.
 */
import crypto from "node:crypto";
import { config } from "../config.ts";
import { generateJti } from "../lib/crypto.ts";

function b64urlFromBuffer(buf: Buffer): string {
  return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function b64urlToBuffer(str: string): Buffer {
  let s = str.replace(/-/g, "+").replace(/_/g, "/");
  const pad = s.length % 4;
  if (pad === 2) s += "==";
  else if (pad === 3) s += "=";
  else if (pad === 1) throw new Error("invalid base64url length");
  return Buffer.from(s, "base64");
}

export interface TokenPayload {
  sub: string;
  username: string;
  iat: number;
  exp: number;
  iss: string;
  jti: string;
  [k: string]: unknown;
}

export function signToken(
  payload: Record<string, unknown>,
  ttlSeconds = config.sessionTtlSeconds,
): string {
  const now = Math.floor(Date.now() / 1000);
  const body = {
    ...payload,
    iat: now,
    exp: now + ttlSeconds,
    iss: "si-auth",
    jti: generateJti(),
  };
  const header = { alg: "HS256", typ: "JWT" };
  const h = b64urlFromBuffer(Buffer.from(JSON.stringify(header), "utf8"));
  const p = b64urlFromBuffer(Buffer.from(JSON.stringify(body), "utf8"));
  const data = `${h}.${p}`;
  const sig = crypto.createHmac("sha256", config.jwtSecret).update(data).digest();
  return `${data}.${b64urlFromBuffer(sig)}`;
}

export function verifyToken(token: string): TokenPayload | null {
  if (typeof token !== "string") return null;
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  const [h, p, s] = parts;

  let sigBytes: Buffer;
  try { sigBytes = b64urlToBuffer(s); } catch { return null; }

  const expected = crypto.createHmac("sha256", config.jwtSecret).update(`${h}.${p}`).digest();

  if (sigBytes.length !== expected.length) return null;
  if (!crypto.timingSafeEqual(sigBytes, expected)) return null;

  try {
    const header = JSON.parse(b64urlToBuffer(h).toString("utf8")) as { alg?: string };
    if (header.alg !== "HS256") return null;

    const body = JSON.parse(b64urlToBuffer(p).toString("utf8")) as TokenPayload;
    if (typeof body.exp !== "number" || body.exp < Math.floor(Date.now() / 1000)) return null;
    if (typeof body.sub !== "string" || typeof body.username !== "string") return null;
    if (typeof body.jti !== "string" || body.jti.length === 0) return null;
    return body;
  } catch {
    return null;
  }
}
""", force=force)

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

    # ── Middleware ─────────────────────────────────────────────────────
    wfile(root, "src/middleware/requestId.ts", r"""import crypto from "node:crypto";
import type { Request, Response, NextFunction } from "express";

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      id?: string;
    }
  }
}

export function requestId(req: Request, res: Response, next: NextFunction): void {
  const inbound = req.headers["x-request-id"];
  const id =
    typeof inbound === "string" && inbound.length > 0 && inbound.length <= 200
      ? inbound
      : crypto.randomBytes(12).toString("hex");
  req.id = id;
  res.setHeader("X-Request-ID", id);
  next();
}
""", force=force)

    wfile(root, "src/middleware/metricsMiddleware.ts", r"""import type { Request, Response, NextFunction } from "express";
import { incCounter, observeHistogram } from "../lib/metrics.ts";

function normalizePath(p: string): string {
  return p
    .split("/")
    .map((seg) => {
      if (/^(cred|user|ch)_[0-9a-f]+$/.test(seg)) return ":id";
      if (/^[0-9a-f]{16,}$/.test(seg)) return ":id";
      return seg;
    })
    .join("/") || "/";
}

export function metricsMiddleware(req: Request, res: Response, next: NextFunction): void {
  const start = process.hrtime.bigint();
  res.on("finish", () => {
    const seconds = Number(process.hrtime.bigint() - start) / 1e9;
    const path = normalizePath(req.path);
    const labels = { method: req.method, path, status: String(res.statusCode) };
    incCounter("si_auth_http_requests_total", labels);
    observeHistogram("si_auth_http_request_duration_seconds", seconds, {
      method: req.method,
      path,
    });
  });
  next();
}
""", force=force)

    wfile(root, "src/middleware/requestLogger.ts", r"""import type { Request, Response, NextFunction } from "express";
import { logger } from "../lib/logger.ts";

export function requestLogger(req: Request, res: Response, next: NextFunction): void {
  const start = process.hrtime.bigint();
  res.on("finish", () => {
    const ms = Number(process.hrtime.bigint() - start) / 1e6;
    logger.info("http", {
      reqId: req.id,
      method: req.method,
      path: req.path,
      status: res.statusCode,
      ms: Number(ms.toFixed(2)),
    });
  });
  next();
}
""", force=force)

    wfile(root, "src/middleware/errorHandler.ts", r"""import type { Request, Response, NextFunction } from "express";
import { config } from "../config.ts";
import { logger } from "../lib/logger.ts";

export function notFound(req: Request, res: Response): void {
  res.status(404).json({ error: "Not Found", path: req.path });
}

export function errorHandler(
  err: Error & { status?: number; statusCode?: number },
  req: Request,
  res: Response,
  _next: NextFunction,
): void {
  const status = err.status || err.statusCode || 500;
  if (status >= 500) {
    logger.error("unhandled error", {
      reqId: req.id,
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
import { getRedis } from "../lib/redis.ts";
import { logger } from "../lib/logger.ts";

const DEFAULT_WINDOW_SECONDS = 60;
const DEFAULT_MAX = 30;
const KEY_PREFIX = "si-auth:rl:";

interface Bucket { count: number; resetAt: number; }

const memBuckets = new Map<string, Bucket>();
const sweeper = setInterval(() => {
  const now = Date.now();
  for (const [k, v] of memBuckets) if (v.resetAt < now) memBuckets.delete(k);
}, 60_000);
sweeper.unref?.();

async function checkMemory(
  key: string, windowMs: number, max: number,
): Promise<{ allowed: boolean; remaining: number; resetAt: number }> {
  const now = Date.now();
  let b = memBuckets.get(key);
  if (!b || b.resetAt < now) {
    b = { count: 0, resetAt: now + windowMs };
    memBuckets.set(key, b);
  }
  b.count += 1;
  return {
    allowed: b.count <= max,
    remaining: Math.max(0, max - b.count),
    resetAt: b.resetAt,
  };
}

async function checkRedis(
  key: string, windowMs: number, max: number,
): Promise<{ allowed: boolean; remaining: number; resetAt: number }> {
  const redis = getRedis();
  if (!redis) throw new Error("redis unavailable");
  const windowSec = Math.max(1, Math.ceil(windowMs / 1000));
  const bucketKey = `${KEY_PREFIX}${key}:${Math.floor(Date.now() / windowMs)}`;
  const count = await redis.incr(bucketKey);
  if (count === 1) {
    await redis.expire(bucketKey, windowSec + 1);
  }
  const resetAt = (Math.floor(Date.now() / windowMs) + 1) * windowMs;
  return { allowed: count <= max, remaining: Math.max(0, max - count), resetAt };
}

export interface RateLimitOptions {
  windowSeconds?: number;
  max?: number;
  key?: (req: Request) => string;
}

export function rateLimit(opts: RateLimitOptions = {}): RequestHandler {
  const windowMs = (opts.windowSeconds ?? DEFAULT_WINDOW_SECONDS) * 1000;
  const max = opts.max ?? DEFAULT_MAX;
  const keyFn = opts.key ?? ((req: Request) => req.ip || req.socket?.remoteAddress || "unknown");

  return function rateLimitMiddleware(req: Request, res: Response, next: NextFunction): void {
    void (async () => {
      const rawKey = keyFn(req);
      const key = `${req.method}:${rawKey}`;
      let result;
      if (getRedis()) {
        try {
          result = await checkRedis(key, windowMs, max);
        } catch (err) {
          logger.warn("rateLimit: redis failed; falling back to memory", {
            error: (err as Error).message,
          });
          result = await checkMemory(key, windowMs, max);
        }
      } else {
        result = await checkMemory(key, windowMs, max);
      }

      res.setHeader("RateLimit-Limit", String(max));
      res.setHeader("RateLimit-Remaining", String(result.remaining));
      res.setHeader("RateLimit-Reset", String(Math.ceil(result.resetAt / 1000)));

      if (!result.allowed) {
        const now = Date.now();
        res.setHeader("Retry-After", String(Math.ceil((result.resetAt - now) / 1000)));
        res.status(429).json({ error: "Too Many Requests" });
        return;
      }
      next();
    })();
  };
}
""", force=force)

    wfile(root, "src/middleware/requireAuth.ts", r"""import type { Request, Response, NextFunction } from "express";
import { config } from "../config.ts";
import { parseCookies } from "../lib/cookies.ts";
import { verifyToken, type TokenPayload } from "../services/token.ts";
import { userStore, type UserRecord } from "../services/userStore.ts";
import { isRevoked } from "../services/revocation.ts";

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
    if (!token) {
      res.status(401).json({ error: "Not authenticated" });
      return;
    }

    const claims = verifyToken(token);
    if (!claims || !claims.sub || !claims.jti) {
      res.status(401).json({ error: "Invalid or expired session" });
      return;
    }

    if (await isRevoked(claims.jti)) {
      res.status(401).json({ error: "Session revoked" });
      return;
    }

    const user = await userStore.findById(claims.sub);
    if (!user) {
      res.status(401).json({ error: "Session user no longer exists" });
      return;
    }

    req.user = user;
    req.session = claims;
    next();
  } catch (err) {
    next(err);
  }
}
""", force=force)

    # ── Routes ─────────────────────────────────────────────────────────
    wfile(root, "src/routes/health.ts", r"""import { Router } from "express";
import { config } from "../config.ts";
import { rustVersion, usingRust } from "../lib/crypto.ts";
import { redisEnabled } from "../lib/redis.ts";
import { challengeStoreBackend } from "../services/challengeStore.ts";
import { revocationStats } from "../services/revocation.ts";
import { userStoreBackend } from "../services/userStore.ts";

export const healthRouter = Router();

healthRouter.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    uptime: Math.round(process.uptime()),
    env: config.env,
    protocol: config.protocolVersion,
    algorithm: config.algorithm,
    hash: config.hash,
    rust: { enabled: usingRust(), version: rustVersion() },
    redis: { configured: redisEnabled() },
    challengeStore: challengeStoreBackend(),
    userStore: userStoreBackend(),
    revocation: revocationStats(),
  });
});
""", force=force)

    wfile(root, "src/routes/metrics.ts", r"""import { Router } from "express";
import type { Request, Response, NextFunction } from "express";
import { config } from "../config.ts";
import { renderMetrics, setGauge } from "../lib/metrics.ts";
import { usingRust } from "../lib/crypto.ts";
import { redisEnabled } from "../lib/redis.ts";
import { challengeStoreBackend, challengeStore } from "../services/challengeStore.ts";
import { revocationStats } from "../services/revocation.ts";
import { userStoreBackend } from "../services/userStore.ts";

export const metricsRouter = Router();

function authorize(req: Request, res: Response, next: NextFunction): void {
  if (!config.metricsToken) { next(); return; }
  const auth = req.headers.authorization ?? "";
  const prefix = "Bearer ";
  if (!auth.startsWith(prefix) || auth.slice(prefix.length) !== config.metricsToken) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }
  next();
}

metricsRouter.get("/metrics", authorize, async (_req, res) => {
  setGauge("si_auth_up", 1);
  setGauge("si_auth_rust_enabled", usingRust() ? 1 : 0);
  setGauge("si_auth_redis_configured", redisEnabled() ? 1 : 0);
  setGauge("si_auth_challenge_store_backend", challengeStoreBackend() === "redis" ? 1 : 0);
  setGauge("si_auth_user_store_backend", userStoreBackend() === "postgres" ? 1 : 0);
  setGauge("si_auth_revocation_backend", revocationStats().backend === "redis" ? 1 : 0);
  setGauge("si_auth_revocation_memory_size", revocationStats().memorySize);
  try {
    setGauge("si_auth_challenges_active", await challengeStore.size());
  } catch {
    setGauge("si_auth_challenges_active", 0);
  }

  const body = renderMetrics();
  res.setHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
  res.status(200).send(body);
});
""", force=force)

    wfile(root, "src/routes/auth.ts", r"""import { Router } from "express";
import { config } from "../config.ts";
import { serializeCookie } from "../lib/cookies.ts";
import {
  buildSigningPayload,
  isValidPublicKeySpki,
  verifySignature,
} from "../lib/crypto.ts";
import { incCounter } from "../lib/metrics.ts";
import { audit } from "../services/auditLog.ts";
import { challengeStore } from "../services/challengeStore.ts";
import { signToken, verifyToken } from "../services/token.ts";
import { userStore, type UserRecord } from "../services/userStore.ts";
import { revoke } from "../services/revocation.ts";
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
    httpOnly: true, secure: config.cookieSecure, sameSite: config.cookieSameSite,
    path: "/", maxAge: maxAgeSeconds,
  }));
}

function clearSessionCookie(res: import("express").Response): void {
  res.setHeader("Set-Cookie", serializeCookie(config.cookieName, "", {
    httpOnly: true, secure: config.cookieSecure, sameSite: config.cookieSameSite,
    path: "/", maxAge: 0,
  }));
}

function issueSession(res: import("express").Response, user: UserRecord): string {
  const token = signToken({ sub: user.id, username: user.username });
  setSessionCookie(res, token, config.sessionTtlSeconds);
  return token;
}

authRouter.get("/config", (_req, res) => {
  res.json({
    protocol: config.protocolVersion,
    algorithm: config.algorithm,
    hash: config.hash,
    modulusLength: config.modulusLength,
    origin: config.appOrigin,
    challengeTtlSeconds: config.challengeTtlSeconds,
    registrationChallengeTtlSeconds: config.registrationChallengeTtlSeconds,
    userEnumerationSafe: config.userEnumerationSafe,
  });
});

authRouter.post("/register/challenge", rateLimit({ max: 10 }), async (req, res, next) => {
  try {
    const { username } = req.body ?? {};
    const err = validateUsername(username);
    if (err) { res.status(400).json({ error: err }); return; }

    const normalized = normalize(username);

    if (!config.userEnumerationSafe) {
      const existing = await userStore.findByUsername(normalized);
      if (existing) { res.status(409).json({ error: "Username already registered" }); return; }
    }

    const record = await challengeStore.issue({
      purpose: "register",
      username: normalized,
      origin: config.appOrigin,
      ttlSeconds: config.registrationChallengeTtlSeconds,
    });

    audit("register.challenge.issued", {
      username: normalized, ip: req.ip, reqId: req.id,
      enumSafe: config.userEnumerationSafe,
    });

    res.json({
      challengeId: record.id,
      challenge: record.challenge,
      origin: record.origin,
      purpose: record.purpose,
      username: record.username,
      issuedAt: record.issuedAt,
      expiresInSeconds: config.registrationChallengeTtlSeconds,
    });
  } catch (e) { next(e); }
});

authRouter.post("/register/complete", rateLimit({ max: 10 }), async (req, res, next) => {
  try {
    const { challengeId, username, publicKeySpki, signature, deviceLabel } = req.body ?? {};

    if (!challengeId || !username || !publicKeySpki || !signature) {
      res.status(400).json({ error: "Missing required fields" }); return;
    }
    if (!isValidPublicKeySpki(publicKeySpki)) {
      res.status(400).json({ error: "Invalid or weak public key" }); return;
    }

    const record = await challengeStore.take(challengeId);
    if (!record || record.purpose !== "register") {
      res.status(400).json({ error: "Challenge expired or invalid" }); return;
    }
    if (normalize(username) !== record.username) {
      res.status(400).json({ error: "Challenge username mismatch" }); return;
    }

    const payload = buildSigningPayload(record);
    const ok = verifySignature({
      publicKeySpkiBase64: publicKeySpki, payload, signatureBase64: signature,
    });

    if (!ok) {
      audit("register.failed", { username: record.username, reason: "bad-signature", ip: req.ip, reqId: req.id });
      incCounter("si_auth_auth_total", { op: "register", result: "bad_signature" });
      res.status(401).json({ error: config.userEnumerationSafe ? "Registration failed" : "Signature verification failed" });
      return;
    }

    const existing = await userStore.findByUsername(record.username);
    if (existing) {
      audit("register.failed", { username: record.username, reason: "username-taken", ip: req.ip, reqId: req.id });
      incCounter("si_auth_auth_total", { op: "register", result: "username_taken" });
      if (config.userEnumerationSafe) {
        res.status(401).json({ error: "Registration failed" });
      } else {
        res.status(409).json({ error: "Username already registered" });
      }
      return;
    }

    let result;
    try {
      result = await userStore.create({
        username: record.username, publicKeySpki,
        deviceLabel: typeof deviceLabel === "string" ? deviceLabel.slice(0, 120) : undefined,
      });
    } catch (e) {
      if ((e as { code?: string }).code === "USER_EXISTS") {
        if (config.userEnumerationSafe) {
          res.status(401).json({ error: "Registration failed" }); return;
        }
        res.status(409).json({ error: "Username already registered" }); return;
      }
      throw e;
    }

    issueSession(res, result.user);
    audit("register.success", { username: result.user.username, ip: req.ip, reqId: req.id });
    incCounter("si_auth_auth_total", { op: "register", result: "success" });

    res.status(201).json({
      user: { id: result.user.id, username: result.user.username },
      credential: {
        id: result.credential.id,
        label: result.credential.label,
        fingerprint: result.credential.fingerprint,
      },
    });
  } catch (e) { next(e); }
});

authRouter.post("/login/challenge", rateLimit({ max: 30 }), async (req, res, next) => {
  try {
    const { username } = req.body ?? {};
    const err = validateUsername(username);
    if (err) { res.status(400).json({ error: err }); return; }

    const normalized = normalize(username);
    const record = await challengeStore.issue({
      purpose: "login",
      username: normalized,
      origin: config.appOrigin,
      ttlSeconds: config.challengeTtlSeconds,
    });

    res.json({
      challengeId: record.id,
      challenge: record.challenge,
      origin: record.origin,
      purpose: record.purpose,
      username: record.username,
      issuedAt: record.issuedAt,
      expiresInSeconds: config.challengeTtlSeconds,
    });
  } catch (e) { next(e); }
});

authRouter.post("/login/verify", rateLimit({ max: 30 }), async (req, res, next) => {
  try {
    const { challengeId, username, signature } = req.body ?? {};
    if (!challengeId || !username || !signature) {
      res.status(400).json({ error: "Missing required fields" }); return;
    }

    const record = await challengeStore.take(challengeId);
    if (!record || record.purpose !== "login") {
      incCounter("si_auth_auth_total", { op: "login", result: "bad_challenge" });
      res.status(401).json({ error: "Authentication failed" }); return;
    }
    if (normalize(username) !== record.username) {
      incCounter("si_auth_auth_total", { op: "login", result: "bad_challenge" });
      res.status(401).json({ error: "Authentication failed" }); return;
    }

    const user = await userStore.findByUsername(record.username);
    if (!user) {
      audit("login.failed", { username: record.username, reason: "no-user", ip: req.ip, reqId: req.id });
      incCounter("si_auth_auth_total", { op: "login", result: "no_user" });
      res.status(401).json({ error: "Authentication failed" }); return;
    }

    const payload = buildSigningPayload(record);
    let matched = null;
    for (const cred of user.credentials) {
      if (verifySignature({
        publicKeySpkiBase64: cred.publicKeySpki, payload, signatureBase64: signature,
      })) { matched = cred; break; }
    }

    if (!matched) {
      audit("login.failed", { username: record.username, reason: "bad-signature", ip: req.ip, reqId: req.id });
      incCounter("si_auth_auth_total", { op: "login", result: "bad_signature" });
      res.status(401).json({ error: "Authentication failed" }); return;
    }

    await userStore.touchCredential(user.id, matched.id);
    issueSession(res, user);
    audit("login.success", {
      username: user.username, credentialId: matched.id, ip: req.ip, reqId: req.id,
    });
    incCounter("si_auth_auth_total", { op: "login", result: "success" });

    res.json({
      user: { id: user.id, username: user.username },
      credential: { id: matched.id, label: matched.label, fingerprint: matched.fingerprint },
    });
  } catch (e) { next(e); }
});

authRouter.get("/me", requireAuth, (req, res) => {
  res.json({
    user: { id: req.user!.id, username: req.user!.username, createdAt: req.user!.createdAt },
    session: { iat: req.session!.iat, exp: req.session!.exp },
  });
});

authRouter.get("/credentials", requireAuth, async (req, res, next) => {
  try {
    const list = await userStore.listCredentials(req.user!.id);
    res.json({ credentials: list });
  } catch (e) { next(e); }
});

authRouter.delete("/credentials/:id", requireAuth, async (req, res, next) => {
  try {
    const { id } = req.params;
    const remaining = (await userStore.listCredentials(req.user!.id)).length;
    if (remaining <= 1) {
      res.status(400).json({
        error: "Cannot remove your last credential. Register another device first.",
      });
      return;
    }
    const ok = await userStore.removeCredential(req.user!.id, id);
    if (!ok) { res.status(404).json({ error: "Credential not found" }); return; }
    audit("credential.removed", { userId: req.user!.id, credentialId: id, ip: req.ip, reqId: req.id });
    res.json({ ok: true });
  } catch (e) { next(e); }
});

authRouter.post("/logout", async (req, res) => {
  try {
    const raw = req.headers.cookie ?? "";
    const match = raw.match(new RegExp(`(?:^|;\\s*)${config.cookieName}=([^;]+)`));
    if (match) {
      const token = decodeURIComponent(match[1]);
      const claims = verifyToken(token);
      if (claims) await revoke(claims.jti, claims.exp);
    }
  } catch {
    /* best-effort */
  }
  clearSessionCookie(res);
  audit("logout", { ip: req.ip, reqId: req.id });
  incCounter("si_auth_auth_total", { op: "logout", result: "success" });
  res.json({ ok: true });
});
""", force=force)

    wfile(root, "src/app.ts", r"""import express, { type Express } from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "./config.ts";
import { authRouter } from "./routes/auth.ts";
import { healthRouter } from "./routes/health.ts";
import { metricsRouter } from "./routes/metrics.ts";
import { errorHandler, notFound } from "./middleware/errorHandler.ts";
import { requestLogger } from "./middleware/requestLogger.ts";
import { requestId } from "./middleware/requestId.ts";
import { metricsMiddleware } from "./middleware/metricsMiddleware.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.resolve(__dirname, "..", "public");

export function createApp(): Express {
  const app = express();
  app.disable("x-powered-by");
  if (config.trustProxy) app.set("trust proxy", 1);

  app.use(requestId);

  app.use((_req, res, next) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("X-Frame-Options", "DENY");
    res.setHeader("Referrer-Policy", "same-origin");
    res.setHeader(
      "Permissions-Policy",
      "camera=(), microphone=(), geolocation=(), interest-cohort=()",
    );
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
  if (config.metricsEnabled) app.use(metricsMiddleware);
  if (config.logLevel !== "silent") app.use(requestLogger);

  app.use("/api/auth", authRouter);
  app.use("/api", healthRouter);
  if (config.metricsEnabled) app.use("/", metricsRouter);

  app.use(express.static(publicDir, { extensions: ["html"], index: "index.html" }));

  app.use("/api", notFound);
  app.use((req, res, next) => {
    if (req.method === "GET" && !req.path.startsWith("/api") && req.path !== "/metrics") {
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

    wfile(root, "src/server.ts", r"""import { createApp } from "./app.ts";
import { config } from "./config.ts";
import { logger } from "./lib/logger.ts";
import { rustVersion, usingRust } from "./lib/crypto.ts";
import { getRedis, redisEnabled } from "./lib/redis.ts";
import { challengeStoreBackend } from "./services/challengeStore.ts";
import { revocationStats } from "./services/revocation.ts";
import { getUserStore, userStoreBackend } from "./services/userStore.ts";

const app = createApp();

// Touch the user store once at boot so a misconfigured backend fails
// fast with a clear error, before the first request arrives.
try {
  void getUserStore();
} catch (err) {
  logger.error("userStore init failed", { error: (err as Error).message });
  process.exit(1);
}

const server = app.listen(config.port, () => {
  logger.info("si-auth listening", {
    port: config.port,
    env: config.env,
    origin: config.appOrigin,
    protocol: config.protocolVersion,
    algorithm: config.algorithm,
    hash: config.hash,
    rust: usingRust() ? rustVersion() : "disabled",
    redis: redisEnabled() ? "configured" : "disabled",
    challengeStore: challengeStoreBackend(),
    userStore: userStoreBackend(),
    revocation: revocationStats().backend,
    metrics: config.metricsEnabled ? "enabled" : "disabled",
  });
  if (usingRust()) logger.info("crypto backend: RUST (napi)");
  else logger.info("crypto backend: node:crypto (fallback)");
  if (redisEnabled()) {
    logger.info("shared state: redis", { challengeStore: challengeStoreBackend() });
  } else {
    logger.info("shared state: in-process");
  }
  process.stdout.write(
    `\n  SI Auth is running\n` +
      `  -> http://localhost:${config.port}/\n` +
      `  crypto: ${usingRust() ? "rust/napi" : "node:crypto"} | ` +
      `state: ${redisEnabled() ? "redis" : "memory"} | ` +
      `users: ${userStoreBackend()} | ` +
      `env: ${config.env} | sig: RSA-SHA256 / RSA-2048\n` +
      (config.metricsEnabled ? `  metrics: http://localhost:${config.port}/metrics\n` : "") +
      `\n`,
  );
});

async function shutdown(signal: string): Promise<void> {
  logger.info("shutting down", { signal });
  server.close(async () => {
    try { const r = getRedis(); if (r) await r.close(); } catch { /* ignore */ }
    try { await getUserStore().close(); } catch { /* ignore */ }
    process.exit(0);
  });
  setTimeout(() => process.exit(1), 10_000).unref();
}

process.on("SIGINT", () => void shutdown("SIGINT"));
process.on("SIGTERM", () => void shutdown("SIGTERM"));
""", force=force)


# ═══════════════════════════════════════════════════════════════════════════
#  Frontend
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
<title>Sign in - SI Auth</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/css/styles.css">
</head><body>
<div class="wrap"><div class="card">
  <div class="logo"><span class="mark">SI</span> SI Auth</div>
  <h1>Sign in</h1>
  <p class="lede">Enter your username. Your device signs a challenge - no password is ever sent.</p>
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
<footer class="credit">RSA-2048 / RSA-SHA256 challenge-response</footer>
</div>
<script type="module" src="/js/login.js"></script>
</body></html>
""", force=force)

    wfile(root, "public/register.html", """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Create account - SI Auth</title>
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
<title>Dashboard - SI Auth</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/css/styles.css">
</head><body>
<div class="wrap wide"><div class="card">
  <div class="row between">
    <div class="logo"><span class="mark">SI</span> SI Auth</div>
    <button id="logout-btn" class="secondary" style="width:auto;padding:.5rem .9rem;font-size:.85rem">Sign out</button>
  </div>
  <h1>Welcome back</h1>
  <p class="lede" id="user-line">Loading...</p>
  <div id="alert" hidden></div>
  <h3 style="margin-top:1.5rem">Registered devices</h3>
  <div id="credentials" class="stack"><p class="muted small">Loading...</p></div>
  <h3 style="margin-top:1.5rem">Active session</h3>
  <div class="cred"><div>
    <div>This browser</div>
    <div class="meta" id="session-meta">-</div>
  </div><span class="small muted">Signed in</span></div>
</div>
<footer class="credit">SI Auth - reference implementation</footer>
</div>
<script type="module" src="/js/dashboard.js"></script>
</body></html>
""", force=force)

    wfile(root, "public/js/si-auth.js", r"""/**
 * si-auth.js — client-side cryptographic core.
 *
 * Zero dependencies. Uses Web Crypto. Private keys are generated with
 * extractable: false so no JavaScript — including an XSS payload — can
 * export them.
 *
 * Signature pipeline: RSASSA-PKCS1-v1_5 + SHA-256.
 */

export const PROTOCOL_VERSION = 'SI-AUTH-V1';

const KEY_ALG = {
  name: 'RSASSA-PKCS1-v1_5',
  modulusLength: 2048,
  publicExponent: new Uint8Array([1, 0, 1]),
  hash: 'SHA-256',
};

const SIGN_ALG = { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' };

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
  return crypto.subtle.generateKey(KEY_ALG, false, ['sign', 'verify']);
}

export async function exportPublicKeySpki(publicKey) {
  const spki = await crypto.subtle.exportKey('spki', publicKey);
  return arrayBufferToBase64(spki);
}

export async function importPublicKeySpki(spkiBase64) {
  return crypto.subtle.importKey(
    'spki', base64ToArrayBuffer(spkiBase64), SIGN_ALG, true, ['verify'],
  );
}

export async function signPayload(privateKey, payloadString) {
  const data = new TextEncoder().encode(payloadString);
  const sig = await crypto.subtle.sign(SIGN_ALG, privateKey, data);
  return arrayBufferToBase64(sig);
}

export async function verifyPayload(publicKey, payloadString, signatureBase64) {
  const data = new TextEncoder().encode(payloadString);
  return crypto.subtle.verify(SIGN_ALG, publicKey, base64ToArrayBuffer(signatureBase64), data);
}

export async function fingerprintPublicKey(spkiBase64) {
  const digest = await crypto.subtle.digest('SHA-256', base64ToArrayBuffer(spkiBase64));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}
""", force=force)

    wfile(root, "public/js/keystore.js", r"""/**
 * keystore.js — persist CryptoKey objects in IndexedDB.
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

function withTransaction(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const os = t.objectStore(STORE);
    let result;
    let fnError = null;
    try { result = fn(os); } catch (err) {
      fnError = err;
      try { t.abort(); } catch { /* ignore */ }
    }
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(fnError || t.error);
    t.onabort = () => reject(fnError || t.error || new Error('transaction aborted'));
  });
}

export async function putCredential(cred) {
  const db = await openDb();
  try {
    await withTransaction(db, 'readwrite', (os) => { os.put(cred); });
  } finally { db.close(); }
}

export async function getCredentials(username) {
  const db = await openDb();
  try {
    const box = await withTransaction(db, 'readonly', (os) => {
      const idx = os.index('byUsername');
      const req = idx.getAll(IDBKeyRange.only(username));
      const b = { value: [] };
      req.onsuccess = () => { b.value = req.result || []; };
      return b;
    });
    const out = box.value;
    out.sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''));
    return out;
  } finally { db.close(); }
}

export async function getCredential(username) {
  const list = await getCredentials(username);
  return list[0] ?? null;
}

export async function listCredentials() {
  const db = await openDb();
  try {
    const box = await withTransaction(db, 'readonly', (os) => {
      const b = { value: [] };
      const req = os.getAll();
      req.onsuccess = () => { b.value = req.result || []; };
      return b;
    });
    return box.value;
  } finally { db.close(); }
}

export async function deleteCredentialById(id) {
  const db = await openDb();
  try {
    await withTransaction(db, 'readwrite', (os) => { os.delete(id); });
  } finally { db.close(); }
}

export async function deleteCredential(username) {
  const db = await openDb();
  try {
    const box = await withTransaction(db, 'readonly', (os) => {
      const idx = os.index('byUsername');
      const req = idx.getAll(IDBKeyRange.only(username));
      const b = { value: [] };
      req.onsuccess = () => { b.value = req.result || []; };
      return b;
    });
    await withTransaction(db, 'readwrite', (os) => {
      for (const e of box.value) os.delete(e.id);
    });
  } finally { db.close(); }
}

export async function clearAll() {
  const db = await openDb();
  try {
    await withTransaction(db, 'readwrite', (os) => { os.clear(); });
  } finally { db.close(); }
}
""", force=force)

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
  revokeCredential: (id) =>
    request(`/api/auth/credentials/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  logout: () => request('/api/auth/logout', { method: 'POST' }),
};
""", force=force)

    wfile(root, "public/js/login.js", r"""import { api } from './api.js';
import { getCredentials } from './keystore.js';
import { buildSigningPayload, signPayload } from './si-auth.js';

const form = document.getElementById('si-auth-form');
const btn = document.getElementById('login-btn');
const alertBox = document.getElementById('alert');

function showAlert(kind, message) {
  alertBox.hidden = false;
  alertBox.className = `alert ${kind}`;
  alertBox.textContent = message;
}
function hideAlert() {
  alertBox.hidden = true;
  alertBox.textContent = '';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideAlert();
  const username = document.getElementById('username').value.trim();
  if (!username) return;

  btn.disabled = true;
  btn.textContent = 'Authenticating...';

  try {
    const credentials = await getCredentials(username);
    if (!credentials.length) {
      throw new Error('No passkey found for this device. Please register first.');
    }

    let lastError = null;
    for (const cred of credentials) {
      try {
        const ch = await api.loginChallenge(username);
        const payload = buildSigningPayload({
          origin: ch.origin, purpose: ch.purpose, challenge: ch.challenge,
          username: ch.username, issuedAt: ch.issuedAt,
        });
        const signature = await signPayload(cred.privateKey, payload);
        await api.loginVerify({
          challengeId: ch.challengeId, username: ch.username, signature,
        });
        window.location.href = '/dashboard.html';
        return;
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError || new Error('Authentication failed');
  } catch (err) {
    showAlert('error', err.message || 'Authentication failed');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Sign in with SI Auth';
  }
});
""", force=force)

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
  btn.disabled = true;
  btn.textContent = 'Generating RSA-2048 keypair...';
  try {
    const ch = await api.registerChallenge(username);
    const { publicKey, privateKey } = await generateKeyPair();
    const publicKeySpki = await exportPublicKeySpki(publicKey);
    const payload = buildSigningPayload({
      origin: ch.origin, purpose: ch.purpose, challenge: ch.challenge,
      username: ch.username, issuedAt: ch.issuedAt,
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
      publicKeySpki,
      fingerprint,
      privateKey,
      createdAt: new Date().toISOString(),
    });
    window.location.href = '/dashboard.html';
  } catch (err) {
    showAlert('error', err.message || 'Registration failed');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate key & create account';
  }
});
""", force=force)

    wfile(root, "public/js/dashboard.js", r"""import { api } from './api.js';

const userLine = document.getElementById('user-line');
const credList = document.getElementById('credentials');
const sessionMeta = document.getElementById('session-meta');
const logoutBtn = document.getElementById('logout-btn');
const alertBox = document.getElementById('alert');

function showAlert(kind, message) {
  alertBox.hidden = false;
  alertBox.className = `alert ${kind}`;
  alertBox.textContent = message;
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
  if (!fp || fp.length < 16) return fp || '-';
  return `${fp.slice(0, 8)}...${fp.slice(-8)}`;
}

async function load() {
  try {
    const me = await api.me();
    userLine.textContent = `Signed in as ${me.user.username}`;
    sessionMeta.textContent =
      `Issued ${new Date(me.session.iat * 1000).toLocaleString()} - ` +
      `Expires ${new Date(me.session.exp * 1000).toLocaleString()}`;
    const { credentials } = await api.listCredentials();
    credList.innerHTML = '';
    for (const c of credentials) {
      const left = el('div', {}, [
        el('div', { text: c.label }),
        el('div', {
          class: 'meta',
          text: `Added ${new Date(c.createdAt).toLocaleString()}${c.lastUsedAt ? ` - last used ${new Date(c.lastUsedAt).toLocaleString()}` : ''}`,
        }),
        el('div', { class: 'fingerprint mono', text: shortFingerprint(c.fingerprint) }),
      ]);
      const removeBtn = el('button', {
        class: 'secondary',
        style: 'width:auto;padding:.4rem .8rem;font-size:.8rem',
        text: 'Revoke',
      });
      removeBtn.addEventListener('click', async () => {
        if (!confirm('Revoke this device? It will no longer be able to sign in.')) return;
        try {
          await api.revokeCredential(c.id);
          await load();
        } catch (err) {
          showAlert('error', err.message);
        }
      });
      credList.appendChild(el('div', { class: 'cred' }, [left, removeBtn]));
    }
  } catch (err) {
    if (err.status === 401) {
      window.location.href = '/index.html';
      return;
    }
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
#  Tests
# ═══════════════════════════════════════════════════════════════════════════

def build_test_files(root: Path, force: bool) -> None:
    log("Tests (test/)", "step")

    wfile(root, "test/crypto.test.ts", r"""import test from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import {
  buildSigningPayload, fingerprintPublicKey, isValidPublicKeySpki,
  verifySignature, usingRust, rustVersion,
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
  assert.equal(payload, [
    "SI-AUTH-V1",
    "origin:https://app.example.com",
    "purpose:login",
    "challenge:abc123",
    "username:me@example.com",
    "issuedAt:2025-01-01T12:00:00.000Z",
  ].join("\n"));
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
  const a = makeKeyPair();
  const b = makeKeyPair();
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

    wfile(root, "test/auth.test.ts", r"""import test, { type TestContext } from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import type { AddressInfo } from "node:net";
import { createApp } from "../src/app.ts";
import { challengeStore } from "../src/services/challengeStore.ts";
import { userStore } from "../src/services/userStore.ts";

function payloadOf(ch: {
  origin: string; purpose: string; challenge: string; username: string; issuedAt: string;
}) {
  return [
    "SI-AUTH-V1",
    `origin:${ch.origin}`,
    `purpose:${ch.purpose}`,
    `challenge:${ch.challenge}`,
    `username:${ch.username}`,
    `issuedAt:${ch.issuedAt}`,
  ].join("\n");
}

async function startServer(t: TestContext) {
  await userStore._reset();
  await challengeStore.clear();
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
  return {
    status: res.status,
    body: text ? JSON.parse(text) : null,
    setCookie: res.headers.get("set-cookie"),
  };
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

test("full register -> login -> me flow", async (t) => {
  const base = await startServer(t);
  const username = "alice@example.com";
  const reg = await register(base, username);
  assert.equal(reg.status, 201);
  assert.ok(reg.setCookie?.includes("si_auth_session="));

  const dup = await post(base, "/api/auth/register/challenge", { username });
  assert.equal(dup.status, 200);

  const ch = (await post(base, "/api/auth/login/challenge", { username })).body;
  const signature = sign(reg.privateKey, payloadOf(ch));
  const login = await post(base, "/api/auth/login/verify", {
    challengeId: ch.challengeId, username: ch.username, signature,
  });
  assert.equal(login.status, 200);

  const cookie = login.setCookie!.split(";")[0];
  const me = await fetch(base + "/api/auth/me", { headers: { Cookie: cookie } });
  assert.equal(me.status, 200);
  const meJson = (await me.json()) as { user: { username: string } };
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

test("logout clears cookie and revokes the session", async (t) => {
  const base = await startServer(t);
  const reg = await register(base, "dan@example.com");
  const cookie = reg.setCookie!.split(";")[0];

  const pre = await fetch(base + "/api/auth/me", { headers: { Cookie: cookie } });
  assert.equal(pre.status, 200);

  await post(base, "/api/auth/logout", {}, cookie);

  const post1 = await fetch(base + "/api/auth/me", { headers: { Cookie: cookie } });
  assert.equal(post1.status, 401);
});

test("metrics endpoint returns Prometheus text", async (t) => {
  const base = await startServer(t);
  await post(base, "/api/auth/login/challenge", { username: "metrics@example.com" });
  const res = await fetch(base + "/metrics");
  assert.equal(res.status, 200);
  assert.match(res.headers.get("content-type") ?? "", /text\/plain/);
  const text = await res.text();
  assert.match(text, /si_auth_up 1/);
  assert.match(text, /si_auth_http_requests_total/);
});

test("X-Request-ID is echoed", async (t) => {
  const base = await startServer(t);
  const res = await fetch(base + "/api/health", { headers: { "X-Request-ID": "abc-123" } });
  assert.equal(res.headers.get("x-request-id"), "abc-123");
});
""", force=force)

    wfile(root, "scripts/smoke-test.ts", r"""#!/usr/bin/env node
/**
 * Black-box smoke test. Assumes a server is running at $BASE_URL
 * (default http://127.0.0.1:3000).
 */
import crypto from "node:crypto";

const BASE = process.env.BASE_URL || "http://127.0.0.1:3000";
const username = `smoke_${Date.now()}@example.com`;

function payloadOf(ch: {
  origin: string; purpose: string; challenge: string; username: string; issuedAt: string;
}) {
  return [
    "SI-AUTH-V1",
    `origin:${ch.origin}`,
    `purpose:${ch.purpose}`,
    `challenge:${ch.challenge}`,
    `username:${ch.username}`,
    `issuedAt:${ch.issuedAt}`,
  ].join("\n");
}

async function post(path: string, body: unknown) {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}

async function main() {
  console.log(`-> smoke testing ${BASE}`);
  const health = await fetch(BASE + "/api/health");
  if (!health.ok) throw new Error(`health check failed: ${health.status}`);
  const hj = (await health.json()) as {
    rust?: { enabled: boolean; version: string | null };
    redis?: { configured: boolean };
    challengeStore?: string;
    userStore?: string;
  };
  console.log(`  + /api/health  (rust: ${hj.rust?.enabled ? hj.rust.version : "disabled"}, store: ${hj.challengeStore}, users: ${hj.userStore})`);

  const { publicKey, privateKey } = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
  const spki = publicKey.export({ type: "spki", format: "der" }).toString("base64");

  const rch = (await post("/api/auth/register/challenge", { username })).body;
  const rSig = crypto.sign("sha256", Buffer.from(payloadOf(rch)), {
    key: privateKey, padding: crypto.constants.RSA_PKCS1_PADDING,
  }).toString("base64");
  const reg = await post("/api/auth/register/complete", {
    challengeId: rch.challengeId, username: rch.username,
    publicKeySpki: spki, signature: rSig,
  });
  if (reg.status !== 201) throw new Error(`register failed: ${reg.status} ${JSON.stringify(reg.body)}`);
  console.log("  + register");

  const lch = (await post("/api/auth/login/challenge", { username })).body;
  const lSig = crypto.sign("sha256", Buffer.from(payloadOf(lch)), {
    key: privateKey, padding: crypto.constants.RSA_PKCS1_PADDING,
  }).toString("base64");
  const login = await post("/api/auth/login/verify", {
    challengeId: lch.challengeId, username: lch.username, signature: lSig,
  });
  if (login.status !== 200) throw new Error(`login failed: ${login.status} ${JSON.stringify(login.body)}`);
  console.log("  + login");

  const metrics = await fetch(BASE + "/metrics");
  if (metrics.status === 200) console.log("  + /metrics OK");
  else if (metrics.status === 401) console.log("  - /metrics requires auth");
  else console.log(`  - /metrics unexpected status ${metrics.status}`);

  console.log("+ smoke test passed");
}

main().catch((err) => {
  console.error("x smoke test failed:", err.message);
  process.exit(1);
});
""", force=force)


# ═══════════════════════════════════════════════════════════════════════════
#  Docs
# ═══════════════════════════════════════════════════════════════════════════

def build_docs_files(root: Path, force: bool) -> None:
    log("Docs (docs/, README omitted)", "step")

    wfile(root, "docs/PROTOCOL.md", """# SI Auth Protocol Specification

**Version:** `SI-AUTH-V1`
**Signature algorithm:** RSASSA-PKCS1-v1_5 with SHA-256 (RSA-SHA256 / RS256-style)
**Key size:** RSA-2048 minimum

Pipeline:

1. **Hash** - SHA-256 over the canonical UTF-8 payload -> 32-byte digest
2. **Sign/verify** - PKCS#1 v1.5 DigestInfo + RSA-2048 over that digest only

## 1. Canonical signing payload

UTF-8, joined with newline (0x0A), no trailing newline:

~~~
SI-AUTH-V1
origin:<origin>
purpose:<purpose>
challenge:<challenge>
username:<username>
issuedAt:<issuedAt>
~~~

| Field | Notes |
|---|---|
| `origin` | Absolute origin, no trailing slash. |
| `purpose` | `login` or `register`. |
| `challenge` | 32-byte CSPRNG value, hex. |
| `username` | Lowercased, trimmed. |
| `issuedAt` | ISO 8601 UTC with milliseconds. |

## 2. Flow

### 2.1 Registration challenge
`POST /api/auth/register/challenge` with `{ username }` returns
`{ challengeId, challenge, origin, purpose, username, issuedAt, expiresInSeconds }`.

### 2.2 Registration complete
`POST /api/auth/register/complete` with
`{ challengeId, username, publicKeySpki, signature, deviceLabel? }`
returns `201` plus a session cookie.

### 2.3 Login challenge
Same as 2.1 with `purpose: "login"` and a 60 s TTL.

### 2.4 Login verify
`POST /api/auth/login/verify` with `{ challengeId, username, signature }`
returns `200` plus a session cookie.

## 3. Server verification

1. Look up `challengeId`. If missing -> 401.
2. Delete the nonce (single-use). If expired -> 401.
3. Check `purpose` and `username` match. Else 401.
4. Rebuild the canonical payload from the cached record.
5. For each credential: verify signature. First match wins.
6. No match -> 401.

## 4. Nonces

- `crypto.randomBytes(32)` (or Rust `rand::thread_rng`), hex-encoded.
- Deleted on first use regardless of outcome.
- Default TTL: 60 s (login), 300 s (register).
- Backend: Redis when `REDIS_URL` is set, in-memory otherwise.

## 5. Session token

HS256 JWT with claims:

~~~json
{ "sub": "user_...", "username": "me@example.com",
  "iat": 1700000000, "exp": 1700086400, "iss": "si-auth",
  "jti": "6c9a..." }
~~~

`jti` is a 128-bit random session identifier. `POST /api/auth/logout`
adds the `jti` to the revocation list; `requireAuth` rejects revoked
tokens until their natural `exp`.

Transported in an `HttpOnly`, `SameSite=Strict`, `Path=/` cookie.
`Secure` when `NODE_ENV=production`.
""", force=force)

    wfile(root, "docs/API.md", """# API Reference

Base: `{APP_ORIGIN}/api`. All bodies are JSON. Errors: `{ "error": "..." }`.

## GET /api/health

~~~json
{ "status": "ok", "uptime": 42, "env": "production", "protocol": "SI-AUTH-V1",
  "algorithm": "RSASSA-PKCS1-v1_5", "hash": "SHA-256",
  "rust": { "enabled": true, "version": "siauth-crypto/0.1.0 (rustc 1.70)" },
  "redis": { "configured": true },
  "challengeStore": "redis",
  "userStore": "postgres",
  "revocation": { "backend": "redis", "memorySize": 0 } }
~~~

## GET /api/auth/config

~~~json
{ "protocol": "SI-AUTH-V1", "algorithm": "RSASSA-PKCS1-v1_5", "hash": "SHA-256",
  "modulusLength": 2048, "origin": "https://app.example.com",
  "challengeTtlSeconds": 60, "registrationChallengeTtlSeconds": 300,
  "userEnumerationSafe": true }
~~~

## POST /api/auth/register/challenge

Body: `{ "username": "me@example.com" }`

- `200` - challenge object (always, when `USER_ENUMERATION_SAFE=true`)
- `400` - invalid username
- `409` - already registered (only when `USER_ENUMERATION_SAFE=false`)
- `429` - rate limited

## POST /api/auth/register/complete

Body: `{ challengeId, username, publicKeySpki, signature, deviceLabel? }`

- `201` - `{ user, credential }` + cookie
- `400` - validation failure
- `401` - generic failure (bad signature OR taken username, when enum-safe)
- `409` - race duplicate (only when `USER_ENUMERATION_SAFE=false`)

## POST /api/auth/login/challenge

Body: `{ "username": "me@example.com" }`

- `200` - challenge (identical for known and unknown users)

## POST /api/auth/login/verify

Body: `{ challengeId, username, signature }`

- `200` - `{ user, credential }` + cookie
- `401` - `{ "error": "Authentication failed" }`

## GET /api/auth/me (auth required)

~~~json
{ "user": { "id": "user_...", "username": "me@example.com", "createdAt": "..." },
  "session": { "iat": 1700000000, "exp": 1700086400 } }
~~~

## GET /api/auth/credentials (auth required)

~~~json
{ "credentials": [ { "id": "cred_...", "label": "Chrome on macOS",
                     "fingerprint": "<64 hex>", "createdAt": "...", "lastUsedAt": "..." } ] }
~~~

## DELETE /api/auth/credentials/:id (auth required)

- `200` - `{ ok: true }`
- `400` - cannot remove last credential
- `404` - not found

## POST /api/auth/logout

Revokes the current session (`jti`) and clears the cookie.

- `200` - `{ ok: true }` + cookie cleared

## GET /metrics (public, or Bearer when METRICS_TOKEN is set)

Prometheus text exposition format, version `0.0.4`.

Exposed series:

- `si_auth_up` (gauge)
- `si_auth_rust_enabled` (gauge)
- `si_auth_redis_configured` (gauge)
- `si_auth_challenge_store_backend` (gauge, 1 = redis)
- `si_auth_user_store_backend` (gauge, 1 = postgres)
- `si_auth_revocation_backend` (gauge, 1 = redis)
- `si_auth_revocation_memory_size` (gauge)
- `si_auth_challenges_active` (gauge)
- `si_auth_http_requests_total{method,path,status}` (counter)
- `si_auth_http_request_duration_seconds{method,path}` (histogram)
- `si_auth_auth_total{op,result}` (counter)
""", force=force)

    wfile(root, "docs/ARCHITECTURE.md", """# Architecture

## Module map

~~~
src/
  config.ts              # env parsing -> frozen config
  server.ts              # HTTP listener + graceful shutdown
  app.ts                 # express app factory (testable)
  lib/
    crypto.ts            # canonical payload + verify (Rust-first)
    cookies.ts           # parse/serialize Set-Cookie
    logger.ts            # JSON-lines structured logging
    metrics.ts           # Prometheus registry
    redis.ts             # zero-dep RESP client (node:net / node:tls)
  middleware/
    errorHandler.ts
    metricsMiddleware.ts
    rateLimit.ts         # memory or Redis
    requestId.ts         # X-Request-ID correlation
    requestLogger.ts
    requireAuth.ts
  routes/
    auth.ts
    health.ts
    metrics.ts           # GET /metrics
  services/
    auditLog.ts
    challengeStore.ts    # interface + memory + Redis implementations
    revocation.ts        # session revocation list
    token.ts             # HS256 JWT (includes jti)
    userStore.ts         # ⭐ UserStore interface + factory
    stores/
      jsonUserStore.ts       # default backend (single-node)
      postgresUserStore.ts   # opt-in backend (multi-replica)

rust/
  src/lib.rs             # N-API accelerator
~~~

## Backend selection matrix

| Feature | Default | Multi-replica |
|---|---|---|
| Challenge store | in-memory | `REDIS_URL` -> Redis |
| Rate-limit buckets | in-memory | `REDIS_URL` -> Redis |
| Revocation list | in-memory | `REDIS_URL` -> Redis |
| User + credential store | JSON file | `USER_STORE_BACKEND=postgres` + `DATABASE_URL` |

Set the relevant env vars; no code changes to routes. `GET /api/health`
reports which backend is active for each feature.

## UserStore swap contract

`src/services/userStore.ts` defines the interface. Any implementation
that satisfies it can be dropped in (MySQL, MongoDB, DynamoDB, ...). It
is the single contract that routes depend on:

~~~ts
interface UserStore {
  findByUsername(username: string): Promise<UserRecord | null>;
  findById(id: string): Promise<UserRecord | null>;
  create(input: {
    username: string;
    publicKeySpki: string;
    deviceLabel?: string;
  }): Promise<{ user: UserRecord; credential: CredentialRecord }>;
  addCredential(
    username: string,
    input: { publicKeySpki: string; deviceLabel?: string },
  ): Promise<CredentialRecord | null>;
  listCredentials(userId: string): Promise<PublicCredential[]>;
  removeCredential(userId: string, credentialId: string): Promise<boolean>;
  touchCredential(userId: string, credentialId: string): Promise<void>;
  _reset(): Promise<void>;   // test-only
  close(): Promise<void>;    // graceful shutdown
}
~~~

`create()` MUST throw an error whose `.code === "USER_EXISTS"` when the
username is taken. That is the single contract-violation signal the
routes handle.

## Postgres schema

The Postgres backend bootstraps the schema at boot when
`DATABASE_AUTO_MIGRATE=true`:

~~~sql
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS credentials (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  public_key_spki TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_used_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS credentials_user_id_idx ON credentials(user_id);
CREATE INDEX IF NOT EXISTS users_username_lower_idx ON users((lower(username)));
~~~

`create()` runs inside a transaction. A unique-violation on
`users.username` is translated into `USER_EXISTS` (matching the JSON
backend).

## Crypto dispatch

`src/lib/crypto.ts` prefers the Rust N-API module and falls back to
`node:crypto`:

~~~ts
if (rust) return rust.verifySignature(spki, payload, sig);
return crypto.verify("sha256", Buffer.from(payload), { ... }, Buffer.from(sig, "base64"));
~~~

Control via `SI_AUTH_USE_RUST`:

- `auto` (default) - use Rust if loadable, else fall back.
- `true` - require Rust; throw at boot if missing.
- `false` - always use `node:crypto`.

## Client crypto core

`public/js/si-auth.js` is dependency-free Web Crypto. It exposes:

- `generateKeyPair()`
- `exportPublicKeySpki(publicKey)`
- `importPublicKeySpki(spkiBase64)`
- `signPayload(privateKey, payloadString)`
- `verifyPayload(publicKey, payloadString, signatureBase64)`
- `buildSigningPayload(fields)`
- `fingerprintPublicKey(spkiBase64)`
""", force=force)

    wfile(root, "docs/SECURITY-MODEL.md", f"""# Security Model

## Threat model

| Threat | Mitigation |
|---|---|
| Credential stuffing | No shared credentials exist. |
| Server DB breach | Only public keys stored; not replayable. |
| Password phishing | Signed payload binds `origin`. |
| Replay | Single-use, TTL-bound nonces. |
| XSS on auth origin | Keys are non-extractable `CryptoKey`s. |
| Username enumeration | Identical challenges and generic 401s. |
| Token theft via JS | Session JWT is `HttpOnly`. |
| Stolen token replay | `jti` revocation list on logout. |
| CSRF | `SameSite=Strict`; JSON body on state-changing endpoints. |
| MITM | TLS mandatory in production. |
| Weak keys | Server rejects < 2048-bit RSA. |

## Residual risks

1. XSS while session is active - mitigated by CSP `script-src 'self'`.
2. Device compromise - filesystem access can use the key but not export it.
3. No key recovery - pre-register a second device. See
   docs/DEPLOYMENT.md#recovery.
4. Stateless JWT - rotation invalidates every session. Revocation list
   handles logout-everywhere without a full rotation.

## Revocation list size

Bounded by session TTL. A revoked token's entry is dropped at its natural
`exp`, so the list cannot grow without bound even under sustained traffic.

## Metrics

`GET /metrics` exposes only aggregate counters. It never emits usernames,
IPs, tokens, or paths with identifiers (paths are normalized to `:id`
before being used as a label). Set `METRICS_TOKEN` to require a bearer
token, or firewall the route.

## Comparison to WebAuthn

WebAuthn is the industry standard: hardware-backed keys, browser-enforced
origin binding, attestation. Use it if you can.

SI Auth provides a self-hosted, dependency-free reference implementation.

**Do not roll your own crypto for high-value systems without professional audit.**

## Deployment checklist

- [ ] `JWT_SECRET` >= 32 random bytes, rotated periodically.
- [ ] `APP_ORIGIN` matches exactly.
- [ ] TLS + `COOKIE_SECURE=true`.
- [ ] `TRUST_PROXY=true` behind a proxy.
- [ ] `NODE_ENV=production`.
- [ ] Rate limits tuned.
- [ ] Audit log shipped to durable storage.
- [ ] `/metrics` firewalled or protected with `METRICS_TOKEN`.
- [ ] Backup/restore tested.
- [ ] Recovery flow documented.
- [ ] For multi-replica: `REDIS_URL` set and reachable.
- [ ] For multi-replica: `USER_STORE_BACKEND=postgres` + `DATABASE_URL` set.
- [ ] `DATABASE_AUTO_MIGRATE=false` after the first successful deploy.

## Reporting

Email {SECURITY_EMAIL}. See SECURITY.md.
""", force=force)

    wfile(root, "docs/DEPLOYMENT.md", """# Deployment

## Docker (single-node, JSON-file user store)

~~~bash
git clone <repo> && cd si-auth
JWT_SECRET="$(node -e 'console.log(require("crypto").randomBytes(48).toString("hex"))')" \\
  APP_ORIGIN="https://auth.example.com" \\
  docker compose up -d --build
~~~

## Docker (multi-replica: Redis + Postgres)

~~~bash
docker compose --profile with-redis --profile with-postgres up -d --build \\
  REDIS_URL=redis://redis:6379 \\
  USER_STORE_BACKEND=postgres \\
  DATABASE_URL=postgres://si_auth:si_auth@postgres:5432/si_auth \\
  JWT_SECRET="$(node -e 'console.log(require("crypto").randomBytes(48).toString("hex"))')" \\
  APP_ORIGIN="https://auth.example.com"
~~~

**Note on APP_ORIGIN:** it must exactly match the externally-visible
origin (scheme + host + port). Every signature binds to it.

## Bare VPS (systemd)

~~~bash
sudo useradd -r -m -d /opt/si-auth si-auth
sudo -u si-auth git clone <repo> /opt/si-auth
cd /opt/si-auth
sudo -u si-auth npm install --omit=dev
sudo -u si-auth npm install pg        # only if USER_STORE_BACKEND=postgres
sudo -u si-auth cp .env.example .env
# edit .env - set JWT_SECRET, APP_ORIGIN, NODE_ENV=production
~~~

`/etc/systemd/system/si-auth.service`:

~~~ini
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
~~~

## Caddy reverse proxy

~~~
auth.example.com {
  reverse_proxy 127.0.0.1:3000
  encode zstd gzip
}
~~~

Set `TRUST_PROXY=true`.

## Scaling matrix

| Components | Backends |
|---|---|
| Single-node | defaults: memory challenges, memory rate limit, memory revocation, JSON user store |
| Multi-replica | `REDIS_URL` + `USER_STORE_BACKEND=postgres` + `DATABASE_URL` |

No route code changes. The `GET /api/health` response confirms which
backends are active.

## JWT_SECRET rotation

Rolling `JWT_SECRET` invalidates every active session. Users sign in
again with their existing device keys - no re-registration is required.

1. Generate a new secret: `node -e 'console.log(require("crypto").randomBytes(48).toString("hex"))'`
2. Update the secret in your secret manager.
3. Restart the service (blue/green or rolling).
4. Notify users that they may need to sign in again.

## Recovery

There is no server-side account recovery. Supported paths:

- **Pre-register a second device** while you still have access to the
  first one. Each device holds its own non-extractable key.
- **Operational reset** (destructive): delete the user's entry from the
  user store and let them re-register. For the JSON backend, edit
  `data/users.json`. For the Postgres backend,
  `DELETE FROM users WHERE username = '...'` (credentials cascade).

If your product requires user-facing recovery, design it as a separate
authenticated bootstrap (email link + fresh key registration) with rate
limits and audit logging.

## Database migration

With `DATABASE_AUTO_MIGRATE=true`, the Postgres backend runs the
idempotent DDL at boot. It is safe to leave on. Recommended flow:

1. First deploy: `DATABASE_AUTO_MIGRATE=true`.
2. After verifying the schema exists, set it to `false` so restarts do
   not issue DDL.

## Metrics & observability

- `GET /metrics` - Prometheus text format.
- `X-Request-ID` - echoed on every response, threaded into logs.
- Every structured log line for an HTTP request carries `reqId`.
- `GET /api/health` - liveness + readiness, includes backend status.

## Health

`GET /api/health` for liveness + readiness. Includes Rust, Redis,
user-store, challenge-store, and revocation backend status.
""", force=force)

    wfile(root, "docs/FAQ.md", """# FAQ

### Why not WebAuthn?

Use it if you can. SI Auth is for teams wanting a self-hosted,
dependency-free, fully auditable implementation - or a teaching tool.

### I lost my device - can I recover?

Not by default. Pre-register a second device while you still have access
to the first one. See docs/DEPLOYMENT.md#recovery.

### Can I export my private key?

No. Keys are generated with `extractable: false`.

### Does this work in Safari / Firefox?

Yes. Web Crypto, IndexedDB, and CryptoKey structured-clone are supported
in every modern browser. HTTPS or localhost is required.

### Why not localStorage for the token?

Because it's readable by any script. An XSS can steal it. HttpOnly cookies
cannot be read by JavaScript.

### How do I log out everywhere?

Call `POST /api/auth/logout` from each device, or implement an
"all sessions" endpoint that revokes every `jti` for a user. The
revocation list is checked on every authenticated request.

### How do I run multiple replicas?

1. Set `REDIS_URL` to a shared Redis instance.
2. Set `USER_STORE_BACKEND=postgres` and `DATABASE_URL`.
3. `npm install pg`.
4. Restart.

Challenges, rate limits, revocation, and user data all move to shared
backends. No route code changes.

### Can I use MySQL / MongoDB instead of Postgres?

Yes. Implement the `UserStore` interface from `src/services/userStore.ts`
and wire it into the factory. The interface is the swap contract - see
docs/ARCHITECTURE.md.

### What does /metrics expose?

Only aggregate counters and gauges: request counts by method/path/status,
request duration histogram, auth operation outcomes, and backend-config
gauges. No usernames, IPs, or tokens.

### How is the metrics endpoint protected?

By default it is unauthenticated (Prometheus convention) and should be
firewalled. Set `METRICS_TOKEN` to require
`Authorization: Bearer <token>`.

### How do I rotate JWT_SECRET?

Roll it - active sessions invalidate. Users sign in again with their
existing keys. See docs/DEPLOYMENT.md#jwt_secret-rotation.

### What does the Rust accelerator do?

Verifies RSA-2048 / RSA-SHA256 signatures on the hot path. Optional -
the server works fine without it (`SI_AUTH_USE_RUST=false`).

### Which Rust triples are built?

x86_64/aarch64 on linux-gnu, linux-musl, apple-darwin, and windows-msvc.
Pinned in `rust/package.json`.

### Why does the Rust crate require sha2's oid feature?

`Pkcs1v15Sign::new::<Sha256>()` in the `rsa` crate requires `Sha256` to
implement `AssociatedOid`. That impl only exists when `sha2` is built
with the `oid` feature enabled. Without it, the crate fails to compile.
""", force=force)


# ═══════════════════════════════════════════════════════════════════════════
#  GitHub / CI
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
    services:
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 5s
          --health-timeout 3s
          --health-retries 20
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: si_auth
          POSTGRES_PASSWORD: si_auth
          POSTGRES_DB: si_auth
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U si_auth"
          --health-interval 5s
          --health-timeout 3s
          --health-retries 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}
      - uses: dtolnay/rust-toolchain@stable
      - run: npm install --no-audit --no-fund
      - run: npm run typecheck
      - run: npm test
      - name: Test against Redis backend
        env:
          REDIS_URL: redis://127.0.0.1:6379
        run: npm test
      - name: Test against Postgres user store
        run: |
          npm install pg --no-audit --no-fund
          REDIS_URL=redis://127.0.0.1:6379 \\
          USER_STORE_BACKEND=postgres \\
          DATABASE_URL=postgres://si_auth:si_auth@127.0.0.1:5432/si_auth \\
            npm test

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
- [ ] Redis / shared-state change
- [ ] User-store backend change

## Checklist

- [ ] `npm test` passes (default backend).
- [ ] `npm test` passes with `REDIS_URL` set.
- [ ] `npm test` passes with `USER_STORE_BACKEND=postgres`.
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
- REDIS_URL set? (yes / no)
- USER_STORE_BACKEND: (json / postgres)
- Browser (if relevant):
- OS:

**Logs**

~~~
paste logs here
~~~
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
#  Docker
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

# `pg` is only needed for USER_STORE_BACKEND=postgres. We install it
# unconditionally so a Postgres-configured deploy works out of the box;
# the default JSON backend never loads it. Cost: a few hundred KB.
RUN npm install pg --omit=dev --no-audit --no-fund

RUN mkdir -p /app/node_modules/siauth-crypto
COPY --from=rust-builder /app/rust/*.node /app/node_modules/siauth-crypto/
COPY --from=rust-builder /app/rust/index.js /app/node_modules/siauth-crypto/
COPY --from=rust-builder /app/rust/index.d.ts /app/node_modules/siauth-crypto/
COPY --from=rust-builder /app/rust/package.json /app/node_modules/siauth-crypto/

RUN node -e "require('siauth-crypto'); console.log('siauth-crypto OK')"

COPY src ./src
COPY public ./public
COPY tsconfig.json ./

RUN mkdir -p /app/data && chown -R node:node /app
USER node
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \\
  CMD node -e "fetch('http://127.0.0.1:3000/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

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
      APP_ORIGIN: ${{APP_ORIGIN:-http://localhost:{port}}}
      JWT_SECRET: ${{JWT_SECRET:?JWT_SECRET is required}}
      COOKIE_SECURE: ${{COOKIE_SECURE:-true}}
      DATA_DIR: /app/data
      LOG_LEVEL: info
      SI_AUTH_USE_RUST: auto
      # Shared state. Enable the with-redis profile and set REDIS_URL.
      REDIS_URL: ${{REDIS_URL:-}}
      REVOCATION_ENABLED: ${{REVOCATION_ENABLED:-true}}
      METRICS_ENABLED: ${{METRICS_ENABLED:-true}}
      METRICS_TOKEN: ${{METRICS_TOKEN:-}}
      # User store backend. Enable the with-postgres profile and set
      # USER_STORE_BACKEND=postgres plus DATABASE_URL.
      USER_STORE_BACKEND: ${{USER_STORE_BACKEND:-json}}
      DATABASE_URL: ${{DATABASE_URL:-}}
      DATABASE_AUTO_MIGRATE: ${{DATABASE_AUTO_MIGRATE:-true}}
    volumes:
      - si-auth-data:/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    profiles: [with-redis]
    command: ["redis-server", "--save", "", "--appendonly", "no"]
    restart: unless-stopped
    expose:
      - "6379"

  postgres:
    image: postgres:16-alpine
    profiles: [with-postgres]
    environment:
      POSTGRES_USER: si_auth
      POSTGRES_PASSWORD: ${{POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}}
      POSTGRES_DB: si_auth
    volumes:
      - si-auth-pg:/var/lib/postgresql/data
    restart: unless-stopped
    expose:
      - "5432"

volumes:
  si-auth-data:
  si-auth-pg:
""", force=force)


# ═══════════════════════════════════════════════════════════════════════════
#  Build & launch
# ═══════════════════════════════════════════════════════════════════════════

def npm_install(root: Path) -> bool:
    log("Installing npm dependencies...", "step")
    try:
        run([SYS.npm or "npm", "install", "--no-audit", "--no-fund"], cwd=root)
        log("npm install complete", "ok")
        return True
    except subprocess.CalledProcessError as e:
        log(f"npm install failed: {e}", "err")
        return False


def rust_build(root: Path) -> bool:
    log("Compiling Rust N-API accelerator...", "step")
    rust_dir = root / "rust"
    if not (rust_dir / "package.json").exists():
        return False
    try:
        run([SYS.npm or "npm", "install", "--no-audit", "--no-fund"], cwd=rust_dir)
        run([SYS.npm or "npm", "run", "build"], cwd=rust_dir)
        node_files = list(rust_dir.glob("*.node"))
        if node_files:
            log(f"Rust accelerator built: {node_files[0].name}", "ok")
            return True
        log(
            "Rust build produced no .node file - the platform triple may not "
            "be in rust/package.json's 'triples' list. The server will fall "
            "back to node:crypto.",
            "err",
        )
        return False
    except subprocess.CalledProcessError as e:
        log(f"Rust build failed: {e}", "warn")
        log("The server will fall back to node:crypto - no problem.", "warn")
        return False


def link_rust_module(root: Path) -> None:
    rust_dir = root / "rust"
    index_js = rust_dir / "index.js"
    pkg_json = rust_dir / "package.json"
    node_files = list(rust_dir.glob("*.node"))

    missing: list[str] = []
    if not index_js.exists():
        missing.append("index.js")
    if not pkg_json.exists():
        missing.append("package.json")
    if not node_files:
        missing.append("*.node")

    if missing:
        log(
            "Rust link skipped: missing " + ", ".join(missing)
            + " in rust/. The main module will fall back to node:crypto.",
            "err",
        )
        return

    main_nm = root / "node_modules" / "siauth-crypto"
    try:
        if main_nm.exists() and not main_nm.is_symlink():
            shutil.rmtree(main_nm)
        main_nm.mkdir(parents=True, exist_ok=True)

        shutil.copy2(index_js, main_nm / "index.js")
        shutil.copy2(pkg_json, main_nm / "package.json")
        for f in node_files:
            shutil.copy2(f, main_nm / f.name)
        idx_dts = rust_dir / "index.d.ts"
        if idx_dts.exists():
            shutil.copy2(idx_dts, main_nm / "index.d.ts")
        log("Rust module linked into node_modules/siauth-crypto", "ok")
    except Exception as e:
        log(f"Could not link Rust module (will fall back): {e}", "warn")


def launch(root: Path, port: int) -> None:
    log(f"Launching server on http://localhost:{port}", "step")
    env = os.environ.copy()
    env.setdefault("PORT", str(port))
    env.setdefault("NODE_ENV", "development")
    env.setdefault("APP_ORIGIN", f"http://localhost:{port}")
    if not env.get("JWT_SECRET"):
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
        description="SI Auth mega-launcher - scaffolds + installs + runs the full stack.",
    )
    p.add_argument(
        "target", nargs="?", default=DEFAULT_TARGET,
        help=f"Target directory (default: {DEFAULT_TARGET})",
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    p.add_argument("--no-rust", action="store_true", help="Skip Rust accelerator")
    p.add_argument("--no-install", action="store_true", help="Scaffold only, don't npm install")
    p.add_argument("--no-launch", action="store_true", help="Don't launch the server")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument("--with-pg", action="store_true", help="Install pg (Postgres user store)")
    p.add_argument("--version", action="version", version=f"auth3.py {VERSION}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    banner()

    log("Detecting system...", "step")
    SYS.describe()
    print()

    if not node_version_ok():
        log("Node.js 18.17+ not found.", "warn")
        install_node()
    else:
        out = subprocess.check_output([SYS.node, "--version"], text=True).strip()
        log(f"Node.js {out} OK", "ok")

    with_rust = not args.no_rust
    rust_ready = False
    if with_rust:
        if SYS.cargo and shutil.which("rustc"):
            try:
                cargo_ver = subprocess.check_output([SYS.cargo, "--version"], text=True).strip()
            except Exception:
                cargo_ver = "(unknown)"
            log(f"Rust toolchain present: {cargo_ver}", "ok")
            rust_ready = True
        else:
            log("Rust not found. Attempting install (optional)...", "warn")
            rust_ready = install_rust()
            if not rust_ready:
                log("Continuing without Rust accelerator.", "warn")
        if rust_ready:
            ensure_apt_build_deps()

    print()
    root = Path(args.target).resolve()
    log(f"Target: {C_BOLD(str(root))}", "step")
    root.mkdir(parents=True, exist_ok=True)

    print()
    build_root_files(
        root, with_rust=(with_rust and rust_ready),
        port=args.port, force=args.force,
    )
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

    print()
    ok = npm_install(root)
    if not ok:
        die("npm install failed. Fix the issue and re-run, or use --no-install.")

    # Install `pg` when the caller has asked for the Postgres backend.
    if args.with_pg:
        print()
        log("Installing optional pg driver...", "step")
        try:
            run([SYS.npm or "npm", "install", "pg", "--no-audit", "--no-fund"], cwd=root)
            log("pg installed", "ok")
        except subprocess.CalledProcessError as e:
            log(f"pg install failed: {e}", "warn")

    if with_rust and rust_ready:
        print()
        built = rust_build(root)
        if built:
            link_rust_module(root)

    print()
    log("Setup complete.", "ok")
    log(f"  cd {root}", "info")
    log("  npm start", "info")
    log("  REDIS_URL=redis://127.0.0.1:6379 npm start   # shared challenge/rate/revocation state", "info")
    log("  USER_STORE_BACKEND=postgres DATABASE_URL=... npm start   # after `npm install pg`", "info")
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
