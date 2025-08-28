Implementation Guide: PROJECT DUNES - The Dockerized Chimera Hub

This guide will walk you through building a custom build ground up, minimal yet powerful version of the Chimera Hub, focusing on the critical path: OAuth token management and a standardized gateway.
Step 1: Define the Architecture

Your gateway will be a separate service. The data flow is key:

    AI Client (Claude) ↔ MCP Client (in your Laravel App): Standard MCP communication (SSE/HTTP).

    MCP Client ↔ Chimera Hub (Your Gateway): Your Laravel app forwards requests to the gateway via an internal API call.

    Chimera Hub ↔ Firestore/KMS: The gateway fetches and decrypts tokens.

    Chimera Hub ↔ External APIs (Gmail, Outlook, etc.): The gateway makes the final, authenticated call to the provider.

Step 2: The Dockerized Chimera Hub Server (Python Example)

This is your custom MCP server, containerized.

Project Structure:
text

chimera-hub/
├── Dockerfile
├── requirements.txt
└── src/
    ├── server.py          # Main MCP server
    ├── auth.py            # Firestore & KMS logic
    └── providers/         # Adapters for each service
        ├── gmail.py
        └── outlook.py

1. Dockerfile
dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

# Use the MCP standard port (optional but good practice)
EXPOSE 8080

CMD ["python", "-m", "server"]

2. requirements.txt
text

mcp-server-sdk
google-cloud-firestore
google-cloud-kms
httpx

3. src/auth.py (The Secure Token Manager)
This is the heart of your security. It handles the Firestore+KMS integration.
python

from google.cloud import firestore, kms
import json

class TokenManager:
    def __init__(self):
        self.db = firestore.AsyncClient()
        self.kms_client = kms.KeyManagementServiceAsyncClient()
        self.key_name = os.environ['KMS_KEY_NAME'] # e.g., "projects/my-project/locations/global/keyRings/my-key-ring/cryptoKeys/my-key"

    async def get_access_token(self, tenant_id: str, user_id: str, provider: str) -> str:
        # 1. Get the encrypted refresh token from Firestore
        doc_ref = self.db.collection("tenants").document(tenant_id).collection("users").document(user_id).collection("providers").document(provider)
        doc = await doc_ref.get()
        
        if not doc.exists:
            raise Exception("No token found")
        
        data = doc.to_dict()
        encrypted_refresh_token = data['encryptedRefreshToken']
        
        # 2. Decrypt the refresh token using KMS
        decrypt_response = await self.kms_client.decrypt(
            request={
                "name": self.key_name,
                "ciphertext": encrypted_refresh_token,
            }
        )
        refresh_token = decrypt_response.plaintext.decode('utf-8')
        
        # 3. Use the refresh token to get a fresh access token
        # (Implement the OAuth2 refresh logic for your provider here)
        access_token = await self._refresh_access_token(provider, refresh_token)
        
        return access_token

4. src/providers/gmail.py (Provider Adapter)
python

import httpx
from auth import TokenManager

class GmailAdapter:
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager

    async def search_emails(self, tenant_id, user_id, query, max_results=10):
        # 1. Get a valid access token
        access_token = await self.token_manager.get_access_token(tenant_id, user_id, "gmail")
        
        # 2. Call the Gmail API
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages?q={query}&maxResults={max_results}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

5. src/server.py (The Main MCP Server)
This ties everything together and exposes the standardized tools.
python

from mcp.server import Server
from mcp.server.stdio import stdio_server
from auth import TokenManager
from providers.gmail import GmailAdapter
import os

# Initialize server and components
server = Server("chimera-hub")
token_manager = TokenManager()
gmail_adapter = GmailAdapter(token_manager)

@server.list_tools()
async def handle_list_tools() -> list:
    return [{
        "name": "email_search",
        "description": "Search user emails across connected providers (Gmail, Outlook).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["gmail", "outlook"]},
                "query": {"type": "string"},
                "max_results": {"type": "number", "maximum": 50}
            },
            "required": ["provider", "query"]
        }
    }]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list:
    if name == "email_search":
        tenant_id = "default_tenant"  # In reality, get this from the MCP context/request
        user_id = "default_user"      # This must be passed from your Laravel app
        
        results = await gmail_adapter.search_emails(
            tenant_id, 
            user_id, 
            arguments["query"], 
            arguments.get("max_results", 10)
        )
        
        # Format the response for the AI
        return [{
            "type": "text",
            "text": f"Found {len(results.get('messages', []))} emails:\n" + 
                    "\n".join([msg['snippet'] for msg in results.get('messages', [])])
        }]
    
    raise Exception(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

Step 3: Deployment & Integration

1. Build & Deploy the Container
bash

# Build the image
docker build -t us-central1-docker.pkg.dev/my-project/chimera-repo/chimera-hub:latest .

# Push to Google Artifact Registry
docker push us-central1-docker.pkg.dev/my-project/chimera-repo/chimera-hub:latest

# Deploy to Cloud Run
gcloud run deploy chimera-hub \
  --image us-central1-docker.pkg.dev/my-project/chimera-repo/chimera-hub:latest \
  --set-env-vars "KMS_KEY_NAME=projects/my-project/locations/global/keyRings/my-key-ring/cryptoKeys/my-key" \
  --service-account chimera-service-account@my-project.iam.gserviceaccount.com \
  --vpc-connector my-vpc-connector \ # Keep traffic private
  --ingress internal # Only allow traffic from your Laravel app

2. Integrate with Laravel
Your Laravel app now acts as the MCP Client. It needs to:

    Receive the user's natural language request.

    Determine the needed MCP tool and arguments.

    Forward the request internally to your Chimera Hub's Cloud Run URL (not via STDIO).

    Get the response and pass it back to the AI.

This is more robust than STDIO for a distributed system and allows your Laravel backend to inject the critical tenant_id and user_id context.
Why This Approach Wins:

    Ultimate Control: You own the entire logic flow.

    Standardization: One gateway to rule all providers, presenting a clean, consistent API to your AI.

    Production Ready: Dockerized, scalable on Cloud Run, and integrates seamlessly with GCP's security primitives (Firestore, KMS, IAM).

    Hybrid Flexibility: The providers/ directory is where you can mix and match. You can start with a third-party API client for speed and later replace it with your own logic without changing the interface.

This requires more upfront investment than Option 1, but it builds a foundation that is infinitely more scalable, secure, and maintainable for a serious application. You are building critical infrastructure, not just a integration.
