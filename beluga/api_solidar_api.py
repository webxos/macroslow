# solidar_api.py
# Description: FastAPI endpoint for SOLIDAR™ data processing.
# Provides REST and WebSocket access for SONAR, LIDAR, CAMERA, and IMU data.
# Usage: Run with uvicorn to expose API endpoints.

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from core.solidar_fusion_extended import SOLIDARExtended
import numpy as np
import cv2

app = FastAPI()

class SensorData(BaseModel):
    sonar_data: list
    lidar_data: list
    camera_frame: list  # Simplified: Base64-encoded frame in production
    imu_data: dict
    timestamp: float

class SOLIDARAPI:
    """
    API for processing SONAR, LIDAR, CAMERA, and IMU data with SOLIDAR™.
    Integrates with CHIMERA 2048 for secure, quantum-parallel processing.
    """
    def __init__(self):
        self.engine = SOLIDARExtended()

    @app.post("/api/v1/solidar/process")
    async def process_data(self, data: SensorData):
        """
        Processes sensor data via REST API.
        Input: SensorData model with SONAR, LIDAR, CAMERA, IMU data.
        Output: Fused 3D model data.
        """
        camera_frame = np.array(data.camera_frame, dtype=np.uint8).reshape(480, 640, 3)  # Simplified
        result = self.engine.process_all_data(
            np.array(data.sonar_data),
            np.array(data.lidar_data),
            camera_frame,
            data.imu_data,
            data.timestamp
        )
        return result

    @app.websocket("/api/v1/solidar/stream")
    async def stream_data(self, websocket: WebSocket):
        """
        Streams fused sensor data via WebSocket for real-time applications.
        """
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            result = self.process_data(SensorData(**data))
            await websocket.send_json(result)

# Run: uvicorn solidar_api:app --host 0.0.0.0 --port 8000
