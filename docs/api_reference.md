MAML Gateway API Reference
Overview
This document provides an auto-generated reference for the MAML Gateway API endpoints.
Endpoints
1. POST /api/maml/upload

Description: Upload a new MAML file.
Request Body: multipart/form-data with file
Response: {"message": "MAML file uploaded", "id": "<maml_id>"}
Authentication: JWT required

2. GET /api/maml/{maml_id}

Description: Retrieve a MAML file by ID.
Parameters: maml_id (string)
Response: MAML data or {"error": "MAML not found"}
Authentication: JWT required

3. POST /api/maml/execute/{maml_id}

Description: Execute a MAML file by ID.
Parameters: maml_id (string)
Response: Execution result or {"error": "MAML not found"}
Authentication: JWT required

4. POST /api/maml/validate

Description: Validate a MAML file.
Request Body: multipart/form-data with file
Response: {"status": "success", "result": {"valid": true}} or error
Authentication: JWT required

5. GET /api/maml/search

Description: Search MAML files by query.
Parameters: query (string)
Response: List of matching MAMLs or {"status": "error"}
Authentication: JWT required

WebSocket

Endpoint: /ws/maml
Description: Real-time MAML execution updates.
Authentication: JWT via WebSocket handshake

Notes

All endpoints require a valid JWT token in the Authorization header.
Responses are JSON-formatted with standard HTTP status codes.
