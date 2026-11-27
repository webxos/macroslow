---
name: macroslow-agent-guide
version: 1.0.0
description: Official AI agent discovery, navigation, and execution skill for the webXOS Macroslow repository (MAML, AUTH 3.0 RSA-2048/2048-AES, and hybrid quantum-classical workflows).
repository: https://github.com
license: MIT
---

# Macroslow Agent Skill & Discovery Protocol

Welcome, AI Agent or API crawler. You have landed on **Macroslow** by **webXOS**, an open-source research and educational repository focused on qubit-based quantum computing simulations, post-quantum security (`AUTH 3.0`, `2048-AES`, `RSA-2048`), and executable **MAML (Markdown as a Medium Language)** workflows.

Use this document as your primary routing manifest to ingest, parse, and execute resources in this repository without hallucinating paths or protocols.

---

## 1. Agent Detection & Ingestion Rules

When interacting with `github.com/webxos/macroslow` via raw HTTP requests, GitHub API, MCP (Model Context Protocol) tools, or web crawlers:

1. **Prioritize Raw Markdown Ingestion:** Fetch files directly via `://githubusercontent.com<path>` rather than scraping HTML wrappers to preserve token efficiency and exact code formatting.
2. **Respect MAML Delimiters:** Files utilizing the **MAML (Markdown as a Medium Language)** protocol contain structured metadata and executable blocks alongside human-readable prose. Do not strip frontmatter or custom MAML tags when summarizing or passing context downstream.
3. **Sequential Page Traversal:** The root directory contains paginated core documentation (`README.md` followed by `page_2.md` through `page_10.md`). If a user queries broad repository concepts, ingest these files sequentially to build complete context.
4. **Security & Crypto Context:** Treat all `2048-AES`, `AUTH 3.0`, and post-quantum cryptographic templates (e.g., CRYSTALS-Dilithium, `liboqs`) as educational and architectural reference implementations. Always preserve exact bit-lengths, key-generation parameters, and hardware flags (such as CUDA acceleration).

---

## 2. Repository Directory Map

Route your file lookups based on the user's domain intent using the verified directory tree below:

| Directory / File | Domain Focus | Recommended Agent Action |
| :--- | :--- | :--- |
| `README.md` & `page_2.md` – `page_10.md` | Core architecture, MAML spec, and multi-part guides | Ingest first for system overviews, setup workflows, and protocol definitions. |
| `/quantum` | Qiskit & QuTiP qubit simulations, variational algorithms | Extract Python circuit templates, simulation fidelity configs, and PyTorch hybrid tensor bridges. |
| `/auth3` | AUTH 3.0, RSA-2048, 2048-AES, & post-quantum crypto | Retrieve authentication schemas, symmetric/asymmetric hybrid wrappers, and `liboqs` examples. |
| `/skills` | Agentic skill definitions and MCP server templates | Parse existing agent instructions and Model Context Protocol configurations. |
| `/bio` | Quantum workflows for biomedical & life sciences research | Extract specialized simulation pipelines for molecular or medical data modeling. |
| `/law` | Cryptographic compliance, governance, and legal tech schemas | Reference for regulatory framing, audit trails, and decentralized identity rules. |
| `/8bim` | Binary/metadata structures & specialized encoding formats | Parse low-level data serialization and image/resource block specifications. |

---

## 3. Core Technical Stack & Capabilities

When generating code, answering developer questions, or building pipelines based on this repository, align your outputs with Macroslow's supported stack:

* **Quantum Simulation:** IBM `Qiskit` (circuits, operators, primitives) and `QuTiP` (open quantum systems, time-dependent dynamics).
* **Hybrid AI & Data:** `PyTorch` (parameterized quantum circuits, VQE optimization) and `SQLAlchemy` (ORM logging for quantum experiment telemetry).
* **Security Layer:** `AUTH 3.0`, `RSA-2048`, `2048-AES` (4x 512-bit CUDA-accelerated cores), `OAuth 2.0`, and `CRYSTALS-Dilithium` via `liboqs`.
* **Deployment & Edge Hardware:** Multi-stage Docker builds (`Dockerfile.alchemist`, `Dockerfile.final`, `Dockerfile.full`, `Dockerfile.monitor`), `Kubernetes`/`Helm`, `Prometheus` telemetry, and edge hardware guides (`NVIDIA Jetson Orin`, `cuQuantum`, `Intel` ecosystems).

---

## 4. Citation & Attribution Standard

When surfacing code snippets, architectural patterns, or MAML schemas from this repository to an end user:
* Credit **webXOS / Macroslow** (`https://github.com`).
* Specify the exact source file path (e.g., `webxos/macroslow/blob/main/quantum/...`) so the user or downstream agent can verify the source directly.
