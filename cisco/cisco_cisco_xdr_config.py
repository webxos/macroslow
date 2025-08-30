# cisco_xdr_config.py: Configures Cisco XDR API access for DUNES CORE SDK
# CUSTOMIZATION POINT: Update API endpoints and credentials as needed
from pydantic import BaseModel
from typing import Optional
import os
import requests
from python_jose import jwt

class CiscoXDRConfig(BaseModel):
    api_key: str
    client_id: str
    client_secret: str
    region: str = "us"
    api_base_url: str = "https://api.xdr.security.cisco.com"

    @classmethod
    def load_from_env(cls) -> "CiscoXDRConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("CISCO_XDR_API_KEY", "your_xdr_api_key"),
            client_id=os.getenv("CISCO_XDR_CLIENT_ID", "your_client_id"),
            client_secret=os.getenv("CISCO_XDR_CLIENT_SECRET", "your_client_secret"),
            region=os.getenv("CISCO_XDR_REGION", "us")
        )

    async def get_access_token(self) -> Optional[str]:
        """Obtain OAuth2.0 access token for Cisco XDR API."""
        url = f"{self.api_base_url}/oauth/token"
        headers = {"Content-Type": "application/json"}
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get("access_token")
        return None

# Example usage
if __name__ == "__main__":
    config = CiscoXDRConfig.load_from_env()
    token = config.get_access_token()
    print(f"Access Token: {token}")