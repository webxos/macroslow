## Annot8 Export User Guide

# Overview

The Annot8 Export feature allows users to download annotated data from the MCP SDKs in various formats, enhancing data analysis and sharing capabilities.

# Getting Started


Configure Environment:Copy .env.example to .env and update with your credentials.
Run the Application:docker-compose up -d
helm install webxos ./deploy/helm/mcp-stack -f deploy/helm/mcp-stack/annot8-export.yaml

# Features

Data Export: Export annotations in CSV format via /api/annot8/export.
Client-Side Interface: Use the export button in index.html to download data.
Security: Requires a valid JWT token for access.

# Usage

Exporting Annotations: Click the "Export" button to download a CSV file containing all annotations.
Format Support: Currently supports CSV; additional formats may be added in future updates.
Authentication: Ensure your token is stored in localStorage before exporting.

# Troubleshooting

Export Fails: Check token validity and network connection.
File Not Downloading: Verify browser settings allow file downloads.
