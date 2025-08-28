#!/bin/bash

# DUNES Deployment Script

# Exit on error
set -e

# Variables
REPO_DIR="webxos-vial-mcp"
GITHUB_REPO="https://github.com/webxos/webxos-vial-mcp.git"
PAGES_REPO="https://github.com/vial/vial.github.io.git"
BRANCH="main"

# Install dependencies
echo "Installing dependencies..."
opam init -y
opam install ocaml ortac -y
pip install fastapi pydantic uvicorn requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python pyyaml

# Clone or update repositories
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git pull origin $BRANCH
else
    git clone $GITHUB_REPO $REPO_DIR
    cd "$REPO_DIR"
fi

# Build and deploy
echo "Building and deploying DUNES SDK..."
mkdir -p src/services scripts
cp ../dunes_ocaml_agent_trainer.py src/services/
cp ../dunes_maml_optimizer.py src/services/
cp ../dunes_workflow_automator.py src/services/
cp ../dunes_ocaml_security_module.py src/services/
cp ../dunes_deployment_script.sh scripts/

# Update GitHub Pages
if [ -d "../vial.github.io" ]; then
    cd ../vial.github.io
    git pull origin $BRANCH
else
    git clone $PAGES_REPO ../vial.github.io
    cd ../vial.github.io
fi
cp ../webxos-vial-mcp/docs/* .
git add .
git commit -m "Deploy DUNES SDK updates $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push origin $BRANCH

# Start services
echo "Starting services..."
cd ../webxos-vial-mcp
uvicorn src.services.dunes_ocaml_agent_trainer:app --host 0.0.0.0 --port 8009 &
uvicorn src.services.dunes_maml_optimizer:app --host 0.0.0.0 --port 8010 &
uvicorn src.services.dunes_workflow_automator:app --host 0.0.0.0 --port 8011 &
uvicorn src.services.dunes_ocaml_security_module:app --host 0.0.0.0 --port 8012 &

echo "Deployment completed at $(date -u +%Y-%m-%dT%H:%M:%SZ) 🐋🐪"

# Deployment Instructions
# Path: webxos-vial-mcp/scripts/dunes_deployment_script.sh
# Run: chmod +x scripts/dunes_deployment_script.sh && ./scripts/dunes_deployment_script.sh
