# data_validator.py
# Description: Validates sensor data inputs for BELUGA’s SOLIDAR™ system.
# Ensures data integrity for SONAR, LIDAR, CAMERA, and IMU inputs.
# Usage: Call validate_sensor_data to check data before processing.

from pydantic import BaseModel, validator
from typing import List, Dict
import numpy as np

class SensorDataModel(BaseModel):
    sonar_data: List[float]
    lidar_data: List[float]
    camera_frame: List[List[List[int]]]
    imu_data: Dict[str, List[float]]
    timestamp: float

    @validator("sonar_data", "lidar_data")
    def check_array_length(cls, v):
        if len(v) < 1:
            raise ValueError("Array must not be empty")
        return v

    @validator("camera_frame")
    def check_frame_shape(cls, v):
        arr = np.array(v)
        if arr.shape != (480, 640, 3):
            raise ValueError("Camera frame must be 480x640x3")
        return v

    @validator("imu_data")
    def check_imu_keys(cls, v):
        required_keys = {"accel", "gyro", "magneto"}
        if not all(k in v for k in required_keys):
            raise ValueError("IMU data must include accel, gyro, magneto")
        return v

def validate_sensor_data(data: dict) -> SensorDataModel:
    """
    Validates sensor data inputs for BELUGA processing.
    Input: Dictionary with sensor data.
    Output: Validated SensorDataModel instance.
    """
    return SensorDataModel(**data)

# Example usage:
# data = {
#     "sonar_data": [1.0, 2.0],
#     "lidar_data": [3.0, 4.0],
#     "camera_frame": np.zeros((480, 640, 3)).tolist(),
#     "imu_data": {"accel": [0.1, 0.2, 0.3], "gyro": [0.0, 0.0, 0.0], "magneto": [0.0, 0.0, 0.0]},
#     "timestamp": 1.0
# }
# validated = validate_sensor_data(data)
# print(validated)