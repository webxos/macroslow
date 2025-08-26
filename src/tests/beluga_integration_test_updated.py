import pytest
import requests
import numpy as np
from src.services.beluga_dunes_interface import SOLIDAREngine
from src.services.beluga_quantum_validator import QuantumGraphDB
from src.services.beluga_gps_denied_navigator import GPSDeniedNavigator
from src.services.beluga_federated_learning import FederatedLearningServer
from src.services.beluga_threejs_visualizer import ThreeJSVisualizerResponse
from src.services.beluga_nasa_data_service import NASADataResponse
from src.services.beluga_autoencoder_anomaly import Autoencoder
import yaml

@pytest.fixture
def config():
    with open("config/beluga_mcp_config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def oauth_token():
    return "test-oauth-token"

def test_jungle_workflow(config, oauth_token):
    """Test BELUGA jungle workflow execution."""
    with open("src/maml/workflows/beluga_jungle_workflow.maml.ml", "r") as f:
        maml_data = f.read()
    response = requests.post(
        "http://localhost:8080/api/mcp/maml_execute",
        json={
            "maml_data": maml_data,
            "sensor_data": {
                "sonar": [1.0] * 512,
                "lidar": [2.0] * 512,
                "iot": [3.0] * 512,
                "nasa": {"imagery": [], "metadata": {"timestamp": "2025-08-26T11:17:00-04:00"}}
            },
            "oauth_token": oauth_token,
            "knowledge_graph": "",
            "security_mode": "advanced",
            "wallet_address": "i5d6e7f8-a9b0-7c1d-c3de-h4f5a6b7c8d9",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "nasa_data" in response.json()
    assert "threejs_content" in response.json()

def test_nasa_data_service(config, oauth_token):
    """Test BELUGA NASA data service."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_nasa_data_service",
        json={
            "environment": "jungle",
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "i5d6e7f8-a9b0-7c1d-c3de-h4f5a6b7c8d9",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "nasa_data" in response.json()
    assert "imagery" in response.json()["nasa_data"]

def test_cli_tool_execute_workflow(config, oauth_token):
    """Test BELUGA CLI tool for workflow execution."""
    import subprocess
    result = subprocess.run(
        [
            "python", "src/services/beluga_cli_tool.py", "execute-workflow",
            "--workflow-path", "src/maml/workflows/beluga_jungle_workflow.maml.ml",
            "--oauth-token", oauth_token,
            "--wallet-address", "i5d6e7f8-a9b0-7c1d-c3de-h4f5a6b7c8d9"
        ],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "success" in result.stdout

def test_autoencoder_anomaly(config, oauth_token):
    """Test BELUGA autoencoder anomaly detection."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_autoencoder_anomaly",
        json={
            "sensor_data": {"sonar": [1.0] * 512, "lidar": [2.0] * 512, "iot": [3.0] * 512},
            "nasa_data": {"imagery": [], "metadata": {"timestamp": "2025-08-26T11:17:00-04:00"}},
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "i5d6e7f8-a9b0-7c1d-c3de-h4f5a6b7c8d9",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "anomaly_report" in response.json()
    assert "anomalies_detected" in response.json()["anomaly_report"]

def test_federated_learning_integration(config, oauth_token):
    """Test BELUGA federated learning integration with NASA data."""
    from src.services.beluga_client_training import ClientTrainer
    trainer = ClientTrainer()
    local_weights = trainer.train([1.0] * 256)
    response = requests.post(
        "http://localhost:8000/api/services/beluga_federated_learning",
        json={
            "sensor_data": {"sonar": [1.0] * 512, "lidar": [2.0] * 512, "iot": [3.0] * 512},
            "model_weights": local_weights,
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "i5d6e7f8-a9b0-7c1d-c3de-h4f5a6b7c8d9",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "aggregated_weights" in response.json()

# Deployment Instructions
# Path: webxos-vial-mcp/src/tests/beluga_integration_test_updated.py
# Run: pip install pytest requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python torch numpy pyyaml psycopg2-binary pgvector flower
# Test: pytest src/tests/beluga_integration_test_updated.py
