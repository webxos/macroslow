import pytest
import requests
import redis
import json
import yaml
from src.services.beluga_data_aggregator import DataAggregatorResponse
from src.services.beluga_alert_service import AlertResponse
from src.services.beluga_performance_optimizer import PerformanceOptimizerResponse
from src.services.beluga_nasa_data_service import NASADataResponse
from src.services.beluga_autoencoder_anomaly import Autoencoder

@pytest.fixture
def config():
    with open("config/beluga_mcp_config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def oauth_token():
    return "test-oauth-token"

@pytest.fixture
def redis_client():
    return redis.Redis(host='localhost', port=6379, db=0)

def test_coastal_workflow(config, oauth_token):
    """Test BELUGA coastal workflow execution."""
    with open("src/maml/workflows/beluga_coastal_workflow.maml.ml", "r") as f:
        maml_data = f.read()
    response = requests.post(
        "http://localhost:8080/api/mcp/maml_execute",
        json={
            "maml_data": maml_data,
            "sensor_data": {
                "sonar": [1.0] * 512,
                "lidar": [2.0] * 512,
                "iot": [3.0] * 512,
                "nasa": {"imagery": [], "metadata": {"timestamp": "2025-08-26T12:28:00-04:00"}}
            },
            "oauth_token": oauth_token,
            "knowledge_graph": "",
            "security_mode": "advanced",
            "wallet_address": "k7f8a9b0-c1d2-9e3f-e5f0-j6a7b8c9d0e1",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "nasa_data" in response.json()
    assert "alerts" in response.json()
    assert "performance_plan" in response.json()

def test_data_aggregator(config, oauth_token, redis_client):
    """Test BELUGA data aggregator service."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_data_aggregator",
        json={
            "sensor_data": {"sonar": [1.0] * 512, "lidar": [2.0] * 512, "iot": [3.0] * 512},
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "k7f8a9b0-c1d2-9e3f-e5f0-j6a7b8c9d0e1",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "aggregated_data" in response.json()
    redis_key = f"beluga:aggregated_data:k7f8a9b0-c1d2-9e3f-e5f0-j6a7b8c9d0e1"
    assert redis_client.get(redis_key) is not None

def test_alert_service(config, oauth_token):
    """Test BELUGA alert service."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_alert_service",
        json={
            "alert_type": "coastal_threat",
            "alert_data": {"threat_level": "high"},
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "k7f8a9b0-c1d2-9e3f-e5f0-j6a7b8c9d0e1",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "alerts" in response.json()
    assert response.json()["alerts"][0]["type"] == "coastal_threat"

def test_performance_optimizer(config, oauth_token):
    """Test BELUGA performance optimizer service."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_performance_optimizer",
        json={
            "sensor_data": {"sonar": [1.0] * 512, "lidar": [2.0] * 512, "iot": [3.0] * 512},
            "nasa_data": {"imagery": [], "metadata": {"timestamp": "2025-08-26T12:28:00-04:00"}},
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "k7f8a9b0-c1d2-9e3f-e5f0-j6a7b8c9d0e1",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "performance_plan" in response.json()
    assert "cpu_allocation" in response.json()["performance_plan"]

def test_nasa_data_integration(config, oauth_token):
    """Test BELUGA NASA data integration."""
    response = requests.post(
        "http://localhost:8000/api/services/beluga_nasa_data_service",
        json={
            "environment": "coastal",
            "oauth_token": oauth_token,
            "security_mode": "advanced",
            "wallet_address": "k7f8a9b0-c1d2-9e3f-e5f0-j6a7b8c9d0e1",
            "reputation": 2500000000
        }
    )
    assert response.status_code == 200
    assert "nasa_data" in response.json()
    assert "imagery" in response.json()["nasa_data"]

# Deployment Instructions
# Path: webxos-vial-mcp/src/tests/beluga_unit_test_updated.py
# Run: pip install pytest requests qiskit>=0.45 pycryptodome>=3.18 liboqs-python torch pyyaml redis aiohttp psutil
# Test: pytest src/tests/beluga_unit_test_updated.py
