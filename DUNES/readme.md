# Core Files for PROJECT DUNES 2048-AES SDK

This set of 10 core files provides a minimalist implementation of the **PROJECT DUNES 2048-AES SDK**, enabling users to build a hybrid Model Context Protocol (MCP) server with MAML (Markdown as Medium Language) processing and MARKUP Agent functionality. Each file includes embedded instructions to guide new users through setup, configuration, and deployment using Netlify for hosting and GitHub for version control. The architecture follows the hybrid gateway pattern described in *The Minimalist's Guide to Building Hybrid MCP Servers* (Claude Artifact: https://claude.ai/public/artifacts/992a72b5-bfe3-47e8-ace7-409ebc280f87).

## 1. README.md
<xaiArtifact artifact_id="6cf8ce39-8028-4c5e-a4da-01b881e5fe5e" artifact_version_id="d0c3a848-df65-4d67-be20-4397dac08d75" title="README.md" contentType="text/markdown"> 

project_dunes@outlook.com for more info

# 🐪 PROJECT DUNES 2048-AES SDK: Minimalist MCP Server Setup

Welcome to the **PROJECT DUNES 2048-AES SDK**, an open-source toolkit for building quantum-resistant, AI-orchestrated applications using the MAML protocol. This guide provides step-by-step instructions for new users to set up a hybrid MCP server, integrate third-party services, and deploy with Netlify and GitHub.

## Overview
This SDK implements a hybrid MCP server that acts as a gateway for routing requests to third-party APIs, custom logic, and database services. It includes the MAML processor for secure data handling and the MARKUP Agent for reverse Markdown (.mu) generation.

## Prerequisites
- **Python**: 3.8+ (for backend)
- **Node.js**: 14+ (for Netlify CLI)
- **Docker**: For containerized deployment
- **Git**: For version control
- **Netlify CLI**: For hosting (`npm install -g netlify-cli`)
- **GitHub Account**: For repository management
- **AWS Cognito**: For OAuth2.0 authentication (optional, for production)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/webxos/project-dunes-2048-aes.git
   cd project-dunes-2048-aes
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   - Copy `.env.example` to `.env`.
   - Update `.env` with your credentials (e.g., AWS Cognito, database URL).
   ```bash
   cp .env.example .env
   ```

5. **Run Locally**
   ```bash
   uvicorn app.main:app --reload
   ```
   Access the FastAPI server at `http://localhost:8000`.

6. **Test MAML Processing**
   - Use `example.maml.md` to test MAML processing via the `/maml/process` endpoint.
   - Use `curl` or Postman to send a POST request with the file content.

7. **Docker Deployment**
   ```bash
   docker build -t dunes-2048-aes .
   docker run -p 8000:8000 dunes-2048-aes
   ```

## Deploying to Netlify
1. **Install Netlify CLI**
   ```bash
   npm install -g netlify-cli
   ```

2. **Configure Netlify**
   - Log in: `netlify login`
   - Initialize site: `netlify init`
   - Set build command to `npm run build` and publish directory to `dist`.

3. **Deploy**
   ```bash
   netlify deploy --prod
   ```

## GitHub Integration
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial MCP server setup"
   git push origin main
   ```

2. **Set Up GitHub Actions**
   - The `.github/workflows/deploy.yml` file automates deployment to Netlify.
   - Add `NETLIFY_AUTH_TOKEN` and `NETLIFY_SITE_ID` to GitHub Secrets.

## Key Files
- `app/main.py`: Core FastAPI application.
- `app/services/mcp_server.py`: Hybrid MCP server implementation.
- `app/services/maml_processor.py`: Processes `.maml.md` files.
- `app/services/markup_agent.py`: Generates `.mu` files for reverse Markdown.
- `example.maml.md`: Sample MAML file for testing.

## Next Steps
- Extend `app/services/custom_logic.py` for your business logic.
- Integrate third-party APIs in `app/services/third_party.py`.
- Explore `docker-compose.yml` for multi-service setups.
- Visit `webxos.netlify.app` for documentation and community support.

**Contact**: project_dunes@outlook.com  
**License**: MIT with attribution to WebXOS Research Group.
