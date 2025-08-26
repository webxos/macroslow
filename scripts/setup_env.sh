#!/bin/bash

# Environment Bootstrapping
install_dependencies() {
    echo "Installing dependencies..."
    pip install -r backend/requirements.txt
    npm install
    apt-get update && apt-get install -y python3.11 curl
}

# Health Check
health_check() {
    echo "Checking service health..."
    if curl -s http://localhost:8000/api/maml/health > /dev/null; then
        echo "Service is healthy"
    else
        echo "Service is unhealthy" >&2
        exit 1
    fi
}

# Cleanup
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf /tmp/maml_sandbox/*
}

# Main execution
install_dependencies
health_check
cleanup
