import click
import yaml
import json
import requests
import logging
from oqs import Signature
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@click.group()
def dunes():
    """DUNES CLI for managing .MAML.ml files with 🐪 branding."""
    pass

@dunes.command()
@click.option('--template', default='src/maml/workflows/maml_ml_template.maml.ml', help='Path to .MAML.ml template')
@click.option('--output', default='src/maml/workflows/my_workflow.maml.ml', help='Output .MAML.ml file')
def create(template, output):
    """
    Create a new .MAML.ml file from a template.
    Path: webxos-vial-mcp/src/services/dunes_cli.py
    """
    try:
        with open(template, 'r') as f:
            template_content = f.read()
        with open(output, 'w') as f:
            f.write(template_content.replace('xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx', str(uuid.uuid4())))
        logger.info(f"DUNES .MAML.ml created at: {output}")
    except Exception as e:
        logger.error(f"DUNES create failed: {str(e)}")
        raise

@dunes.command()
@click.option('--input', required=True, help='Path to .MAML.ml file')
@click.option('--security-mode', default='advanced', type=click.Choice(['advanced', 'lightweight']))
def encrypt(input, security_mode):
    """
    Encrypt a .MAML.ml file with DUNES protocol.
    Path: webxos-vial-mcp/src/services/dunes_cli.py
    """
    try:
        with open(input, 'r') as f:
            data = f.read()
        response = requests.post(
            "http://localhost:8000/api/services/dunes_encrypt",
            json={"data": data, "securityMode": security_mode}
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"DUNES encryption completed: {result['hash']}")
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"DUNES encryption failed: {str(e)}")
        raise

@dunes.command()
@click.option('--input', required=True, help='Path to .MAML.ml file')
@click.option('--oauth-token', required=True, help='OAuth2.0 token')
def sync(input, oauth_token):
    """
    Sync a .MAML.ml file with OAuth2.0 and DUNES validation.
    Path: webxos-vial-mcp/src/services/dunes_cli.py
    """
    try:
        with open(input, 'r') as f:
            data = f.read()
        # Sign with CRYSTALS-Dilithium
        sig = Signature('Dilithium5')
        _, secret_key = sig.keypair()
        signature = sig.sign(data.encode(), secret_key).hex()
        
        response = requests.post(
            "http://localhost:8000/api/services/dunes_oauth_sync",
            json={"mamlData": data, "oauthToken": oauth_token, "dunesHash": hashlib.sha3_512(data.encode()).hexdigest(), "signature": signature},
            headers={"Authorization": f"Bearer {oauth_token}"}
        )
        response.raise_for_status()
        logger.info(f"DUNES sync completed for: {input}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        logger.error(f"DUNES sync failed: {str(e)}")
        raise

if __name__ == "__main__":
    dunes()

# Deployment Instructions
# Path: webxos-vial-mcp/src/services/dunes_cli.py
# Run: pip install click requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python
# Usage: python src/services/dunes_cli.py create --template src/maml/workflows/maml_ml_template.maml.ml --output src/maml/workflows/my_workflow.maml.ml
# Usage: python src/services/dunes_cli.py encrypt --input src/maml/workflows/my_workflow.maml.ml --security-mode advanced
# Usage: python src/services/dunes_cli.py sync --input src/maml/workflows/my_workflow.maml.ml --oauth-token $OAUTH_TOKEN
