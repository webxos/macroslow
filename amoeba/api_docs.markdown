# AMOEBA 2048AES SDK API Documentation

## Description: API documentation for the AMOEBA 2048AES SDK, detailing endpoints for the orchestrator and Dropbox integration.

## Base URL
`https://your-vercel-url`

## Endpoints

### POST /execute-task
Execute a MAML-based task with Dropbox integration.

#### Request Body
```json
{
  "maml_file": "string",
  "task_id": "string"
}
```

#### Response
```json
{
  "status": "success",
  "task_result": {
    "result": [float]
  },
  "upload_result": {
    "file_path": "string",
    "signature": "string"
  }
}
```

#### Example
```bash
curl -X POST https://your-vercel-url/execute-task \
  -H "Content-Type: application/json" \
  -d '{"maml_file": "workflow.maml.md", "task_id": "sample_task"}'
```

## Authentication
- Requires Dropbox API tokens configured in `dropbox_config.yml`.
- MAML files must be signed with quantum-safe signatures (Dilithium2).

## Error Responses
- `400 Bad Request`: Invalid MAML file or signature.
- `500 Internal Server Error`: Server-side issues (e.g., Dropbox API failure).

## Notes
- Ensure `DROPBOX_ACCESS_TOKEN`, `DROPBOX_APP_KEY`, and `DROPBOX_APP_SECRET` are set in environment variables for Vercel deployments.
- Monitor API performance via Prometheus at `http://localhost:9090`.