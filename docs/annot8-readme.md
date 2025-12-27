## annot8 - Real-Time Collaborative Annotation System

# Overview

annot8 is a web-based platform for real-time collaborative data annotation, embodying the MCP spirit by providing a structured context for AI and data science tools. Authenticated users can annotate documents, with annotations persisted and viewable instantly by all authorized users, integrated into the WebXOS ecosystem.

# Architecture Diagram

graph TD
    A[Frontend<br>index.html] --> B[WebSocket<br>/ws/{client_id}]
    B --> C[Backend<br>FastAPI]
    C --> D[PostgreSQL<br>Users & Annotations]
    C --> E[OAuth 2.0<br>Google/GitHub]
    F[CI/CD<br>GitHub Actions] --> C
    G[Docker<br>Multi-Stage Build] --> C
    C --> H[MongoDB<br>WebXOS RAG]

# Setup Instructions

Register an OAuth App at Google Developer Console with redirect URI http://localhost:8000/auth/callback.


# Start Development:docker-compose up -d


Access: Open http://localhost:8000 in your browser.

# Example Usage

Annotate: Log in, click on the page to add annotations, and see updates in real-time.
Export Data: curl http://localhost:8000/api/annotations/export

# Technology Stack

CI/CD: GitHub Actions
Containerization: Docker & Docker Compose
Backend: FastAPI with WebSockets
Database: PostgreSQL
OAuth: python-jose
HTML Generation: Jinja2
Frontend: Vanilla JS

# Data Science Integration

Export Endpoint: /api/annotations/export returns JSON for analysis.
Example Notebook: Notebooks/analyze_annotations.ipynb to load and analyze annotation data.
