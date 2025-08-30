# test_fusion_extended.py
# Description: Unit tests for BELUGA’s extended SOLIDAR™ engine.
# Validates fusion of SONAR, LIDAR, CAMERA, and IMU data.
# Usage: Run with pytest to verify functionality.

import pytest
import numpy as np
import cv2
from core.solidar_fusion_extended import SOLIDARExtended

@pytest.fixture
def solidar_extended():
    return SOLIDARExtended()

def test_fusion_extended(solidar_extended):
    """
    Tests extended SOLIDAR™ fusion process for all sensor types.
    """
    sonar_data = np.random.rand(100)
    lidar_data = np.random.rand(100)
    camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    imu_data = {"accel": np.array([0.1, 0.2, 0.3]), "gyro": np.zeros(3), "magneto": np.zeros(3)}
    timestamp = 1.0
    result = solidar_extended.process_all_data(sonar_data, lidar_data, camera_frame, imu_data, timestamp)
    assert "fused_graph" in result
    assert "vo_position" in result
    assert "imu_position" in result
    assert isinstance(result["fused_graph"], torch.Tensor)

# Run tests: pytest test_fusion_extended.py