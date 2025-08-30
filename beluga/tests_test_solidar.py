# test_solidar.py
# Description: Unit tests for BELUGA’s SOLIDAR™ engine.
# Ensures correct fusion of SONAR and LIDAR data.
# Usage: Run with pytest to validate SOLIDAR functionality.

import pytest
import torch
from core.solidar_engine import SOLIDAREngine

@pytest.fixture
def solidar_engine():
    return SOLIDAREngine()

def test_solidar_fusion(solidar_engine):
    """
    Tests SOLIDAR™ fusion process.
    """
    sonar_data = torch.randn(100)
    lidar_data = torch.randn(100)
    result = solidar_engine.process_data(sonar_data.numpy(), lidar_data.numpy())
    assert "fused_graph" in result
    assert isinstance(result["fused_graph"], torch.Tensor)

# Run tests: pytest test_solidar.py