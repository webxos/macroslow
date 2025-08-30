# video_stream_api.py
# Description: FastAPI endpoint for streaming processed video data from BELUGA.
# Supports real-time video output for AR goggles and other devices.
# Usage: Run with uvicorn to expose video streaming endpoints.

from fastapi import FastAPI, WebSocket
from core.solidar_fusion_extended import SOLIDARExtended
import numpy as np
import cv2

app = FastAPI()

class VideoStreamAPI:
    """
    API for streaming BELUGA’s processed video data.
    Integrates with SOLIDAR™ for real-time 3D model streaming.
    """
    def __init__(self):
        self.solidar = SOLIDARExtended()

    @app.websocket("/api/v1/video/stream")
    async def stream_video(self, websocket: WebSocket):
        """
        Streams processed video data via WebSocket.
        Receives sensor data and sends fused 3D model data.
        """
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            result = self.solidar.process_all_data(
                np.array(data.get("sonar_data", [])),
                np.array(data.get("lidar_data", [])),
                np.array(data.get("camera_frame", [])).reshape(480, 640, 3),
                data.get("imu_data", {'accel': np.zeros(3), 'gyro': np.zeros(3), 'magneto': np.zeros(3)}),
                data.get("timestamp", 0.0)
            )
            await websocket.send_json({"fused_graph": result["fused_graph"].cpu().numpy().tolist()})

# Run: uvicorn video_stream_api:app --host 0.0.0.0 --port 8001
