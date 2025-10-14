# Quantum-Enhanced YOLOv8 Guide: Edge AI for Pothole Detection and Beyond

## Page 4: OBS Studio Integration for Real-Time API Connections

### Broadcasting Detections: From Edge to Dashboard

**OBS Studio** (Open Broadcaster Software) enables low-latency streaming of YOLOv8 pothole detections to live dashboards, ideal for municipal monitoring or insurer alerts. This page integrates YOLOv8 outputs with OBS via WebSockets, piping real-time pothole data (bounding boxes, confidence scores) to remote systems, aligned with the Model Context Protocol (MCP) for structured, verifiable streaming.

#### Step 1: OBS Setup for API Hooks
Install OBS Studio (free, cross-platform) and the **obs-websocket** plugin (v5+) for API control:
- Download: [obsproject.com](https://obsproject.com/forum/resources/obs-websocket-remote-control-obs-studio-from-websockets.466/)
- Setup: In OBS, go to **Tools > WebSocket Server Settings**, enable the server (default port: 4455), and set a secure password.

#### Step 2: YOLOv8 to OBS via WebSocket
Use Python’s `obs-websocket-py` to overlay detection data (e.g., pothole counts, confidence scores) on OBS streams.

```bash
pip install obs-websocket-py websocket-client
```

```python
import obswebsocket
from obswebsocket import requests
from ultralytics import YOLO
import json
import cv2

model = YOLO('best.pt')
client = obswebsocket.obsws("localhost", 4455, "your_password")
client.connect()

def stream_detection(video_source):
    # Create OBS text source for overlays
    client.call(requests.CreateSource(
        sourceName="DetectionOverlay",
        sourceKind="text_gdiplus_v2",
        sceneName="Scene"
    ))
    
    for frame in video_source:  # e.g., from drone cam or file
        results = model(frame)
        
        # MCP Filter: Extract high-confidence detections
        boxes = [(box.xyxy[0].tolist(), box.conf.item()) for box in results[0].boxes if box.conf > 0.7]
        
        # Send to OBS: Text overlay with pothole count and confidences
        overlay_text = json.dumps({
            "potholes": len(boxes),
            "confidences": [conf for _, conf in boxes]
        })
        client.call(requests.SetInputSettings(
            inputName="DetectionOverlay",
            inputSettings={"text": overlay_text}
        ))
        
        # Stream to RTMP (for IoT dashboards)
        client.call(requests.StartStream())  # Configure RTMP in OBS settings

# Example: Stream from webcam
cap = cv2.VideoCapture(0)
stream_detection(cap)
client.disconnect()
```

#### Step 3: API Gateway for Remote Access
Expose YOLOv8 detections via FastAPI WebSocket, integrating with MCP for context-aware streaming:

```python
from fastapi import FastAPI, WebSocket
import asyncio
from ultralytics import YOLO

app = FastAPI()
model = YOLO('best.pt')

async def get_yolo_detections(frame):
    results = model(frame)
    detections = [{"box": box.xyxy[0].tolist(), "conf": box.conf.item()} 
                  for box in results[0].boxes if box.conf > 0.7]
    return {"context": "pothole_detection", "detections": detections}

@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = await get_yolo_detections(frame)
        await websocket.send_json(detections)
        # OBS hooks into this WebSocket for live updates
        await asyncio.sleep(0.033)  # ~30 FPS
    cap.release()
```

**Use Case**: Stream drone footage to an OBS dashboard for real-time pothole alerts; notify insurers via RTMP feeds. **Latency**: <100ms on local networks; test with OBS Browser Source for WebSocket integration.

**MCP Integration**: Log stream metadata (e.g., frame hashes) in SQLite for auditability, aligning with MCP’s verifiable context.

**Pro Tip**: For noisy drone feeds, flag frames for quantum validation (Page 8) in OBS metadata using `client.call(requests.SetSceneItemProperties(...))`.

*(End of Page 4. Page 5 explores IoT deployment on edge devices.)*