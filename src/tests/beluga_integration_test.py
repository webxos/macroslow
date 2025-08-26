import pytest
import requests
import numpy as np
from src.services.beluga_dunes_interface import SOLIDAREngine
from src.services.beluga_quantum_validator import QuantumGraphDB
from src.services.beluga_gps_denied_navigator import GPSDeniedNavigator
from src.services.beluga_federated_learning import FederatedLearningServer
from src.services.beluga_threejs_visualizer import ThreeJSVisualizerResponse
import yaml

@pytest.fixture
def config():
    with open("config/beluga_mcp_config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def oauth_token():
    return "test-oauth-token"

def test_react_dashboard_endpoint(config, oauth_token):
    """Test BELUGA React dashboard endpoint integration."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_dashboard",
        json={
            "environment": "space",
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "f2a3b4c5-d6e7-4f8a-90ab-e1c2d3e4f5a6",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "threat_data" in response.json()
    assert "visualization_data" in response.json()
    assert "navigation_data" in response.json()
    assert "optimization_data" in response.json()

def test_desert_workflow(config, oauth_token):
    """Test BELUGA desert workflow execution."""
    with open("src/maml/workflows/beluga_desert_workflow.maml.ml", "r") as f:
        maml_data = f.read()
    response = requests.post(
        "http://localhost:8080/api/mcp/maml_execute",
        json={
            "maml_data": maml_data,
            "sensor_data": {"sonar": [1.0] * 512, "lidar": [2.0] * 512},
            "oauth_token": oauth_token,
            "knowledge_graph": "",
            "security_mode": "advanced",
            "wallet_address": "g3b4c5d6-e7f8-5a9b-a1bc-f2d3e4f5a6b7",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "threejs_content" in response.json()

def test_threejs_visualizer(config, oauth_token):
    """Test BELUGA Three.js visualizer."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_threejs_visualizer",
        json={
            "navigation_path": [[0, 0], [1, 1], [2, 2]],
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "g3b4c5d6-e7f8-5a9b-a1bc-f2d3e4f5a6b7",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "threejs_content" in response.json()
    assert "THREE.Scene" in response.json()["threejs_content"]

def test_client_training(config, oauth_token):
    """Test BELUGA client-side training."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_client_training",
        json={
            "sensor_data": {"sonar": [1.0] * 512, "lidar": [2.0] * 512},
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "g3b4c5d6-e7f8-5a9b-a1bc-f2d3e4f5a6b7",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "local_weights" in response.json()
    assert len(response.json()["local_weights"]) > 0

def test_federated_learning_integration(config, oauth_token):
    """Test BELUGA federated learning integration."""
    trainer = ClientTrainer()
    local_weights = trainer.train([1.0] * 256)
    response = requests.post(
        "http://localhost:8000/api/services/beluga_federated_learning",
        json={
            "sensor_data": {"sonar": [1.0] * 512, "lidar": [2.0] * 512},
            "model_weights": local_weights,
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "g3b4c5d6-e7f8-5a9b-a1bc-f2d3e4f5a6b7",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "aggregated_weights" in response.json()

# Deployment Instructions
# Path: webxos-vial-mcp/src/tests/beluga_integration_test.py
# Run: pip install pytest requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python torch numpy pyyaml psycopg2-binary pgvector flower
# Test: pytest src/tests/beluga_integration_test.py
