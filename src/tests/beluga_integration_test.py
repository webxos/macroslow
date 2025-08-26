```python
import pytest
import requests
import torch
import numpy as np
from oqs import Signature
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib
import json
import yaml

@pytest.fixture
def config():
    with open("config/beluga_mcp_config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def oauth_token():
    return "test_oauth_token"

@pytest.fixture
def wallet_address():
    return "b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e"

@pytest.fixture
def reputation():
    return 2500000000

def test_beluga_maml_processor(config, oauth_token, wallet_address, reputation):
    """Test BELUGA MAML processor integration."""
    maml_data = """
    ---
    maml_version: "1.0.0"
    id: "urn:uuid:test-maml"
    type: "workflow"
    ---
    ## Intent
    Test MAML processing.
    """
    response = requests.post(
        "http://localhost:8000/api/services/beluga_maml_processor",
        json={
            "maml_data": maml_data,
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": wallet_address,
            "reputation": reputation
        }
    )
    assert response.status_code == 200
    assert response.json()['status'] == "success"
    assert "dunes_hash" in response.json()
    assert "signature" in response.json()

def test_beluga_sensor_fusion(config, oauth_token, wallet_address, reputation):
    """Test BELUGA sensor fusion integration."""
    sensor_data = {"sonar": [1.0] * 512, "lidar": [2.0] * 512}
    response = requests.post(
        "http://localhost:8000/api/services/beluga_sensor_fusion",
        json={
            "sonar_data": sensor_data["sonar"],
            "lidar_data": sensor_data["lidar"],
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": wallet_address,
            "reputation": reputation
        }
    )
    assert response.status_code == 200
    assert response.json()['status'] == "success"
    assert "fused_features" in response.json()
    assert "dunes_hash" in response.json()

def test_beluga_adaptive_navigator(config, oauth_token, wallet_address, reputation):
    """Test BELUGA adaptive navigator integration."""
    sensor_data = {"sonar": [1.0] * 512, "lidar": [2.0] * 512}
    response = requests.post(
        "http://localhost:8000/api/services/beluga_adaptive_navigator",
        json={
            "sensor_data": sensor_data,
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": wallet_address,
            "reputation": reputation
        }
    )
    assert response.status_code == 200
    assert response.json()['status'] == "success"
    assert "navigation_path" in response.json()
    assert "dunes_hash" in response.json()

def test_beluga_obs_controller(config, oauth_token, wallet_address, reputation):
    """Test BELUGA OBS controller integration."""
    sensor_data = {"sonar": [1.0] * 512, "lidar": [2.0] * 512}
    response = requests.post(
        "http://localhost:8000/api/services/beluga_obs_controller",
        json={
            "sensor_data": sensor_data,
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": wallet_address,
            "reputation": reputation
        }
    )
    assert response.status_code == 200
    assert response.json()['status'] == "success"
    assert "visualization_data" in response.json()
    assert "dunes_hash" in response.json()

def test_beluga_sustainability_optimizer(config, oauth_token, wallet_address, reputation):
    """Test BELUGA sustainability optimizer integration."""
    sensor_data = {"sonar": [1.0] * 512, "lidar": [2.0] * 512}
    response = requests.post(
        "http://localhost:8000/api/services/beluga_sustainability_optimizer",
        json={
            "sensor_data": sensor_data,
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": wallet_address,
            "reputation": reputation
        }
    )
    assert response.status_code == 200
    assert response.json()['status'] == "success"
    assert "optimization_plan" in response.json()
    assert "dunes_hash" in response.json()

# Deployment Instructions
# Path: webxos-vial-mcp/src/tests/beluga_integration_test.py
# Run: pip install pytest requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python torch numpy pyyaml
# Test: pytest src/tests/beluga_integration_test.py
```
