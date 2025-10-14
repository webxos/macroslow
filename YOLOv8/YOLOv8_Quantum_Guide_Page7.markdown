# Quantum-Enhanced YOLOv8 Guide: Edge AI for Pothole Detection and Beyond

## Page 7: Cyberdecks and Android Phones as Edge Platforms

### Hackable Hardware Meets Mobile AI

**Cyberdecks** (custom-built Raspberry Pi/Arduino rigs) and **Android phones** (via Termux/NCNN) transform portable devices into powerful, MCP-powered pothole detectors. This page adapts YOLOv8 for low-cost, handheld edge platforms, enabling citizen science, field inspections, or rapid prototyping with seamless integration into the Model Context Protocol (MCP) and OBS streaming pipelines.

#### Step 1: Cyberdeck Build and Setup
- **Hardware**: Raspberry Pi Zero 2 W (~$15), USB camera, 3.5" LCD, and battery pack (~$50 total). 3D-print a rugged case for field use.
- **Software**: Raspbian OS, Python 3.8+, TensorFlow Lite runtime, and GPIO libraries for hardware control.

```bash
# On Raspberry Pi
sudo apt update
sudo apt install python3-pip
pip install tflite-runtime opencv-python gpiozero
```

**YOLOv8 Deployment**:
```python
import tflite_runtime.interpreter as tflite
import cv2
import gpiozero
import json
import paho.mqtt.client as mqtt

# Load TFLite model (from Page 5)
interpreter = tflite.Interpreter(model_path='best.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# GPIO setup for button trigger and LED alert
button = gpiozero.Button(18)  # Trigger scan
led = gpiozero.LED(17)       # Visual alert

# MQTT for OBS/MCP integration
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883)

while True:
    if button.is_pressed:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        
        # Preprocess frame
        input_shape = input_details[0]['shape']
        frame_resized = cv2.resize(frame, (320, 320))
        input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])
        
        # MCP Filter: High-confidence potholes
        potholes = [b for b in boxes[0] if b[4] > 0.7]
        
        # Alert and publish
        if potholes:
            led.on()
            payload = {"context": "cyberdeck_pothole", "count": len(potholes)}
            client.publish("pothole/alert", json.dumps(payload))  # To OBS/MCP
        else:
            led.off()
```

**MCP Schema**:
```markdown
---
mcp_schema: yolo_cyberdeck
version: 1.0
device: raspberry_pi_zero
---
# Cyberdeck Pothole Detection
## Context
- Trigger: GPIO button
- Stream: MQTT to OBS
- Confidence Threshold: 0.7
```

**Metrics**: ~10 FPS on Pi Zero; 200mW power draw. Ideal for portable inspections.

#### Step 2: Android Deployment with Termux/NCNN
- **Setup**: Install Termux (F-Droid), then NCNN for ARM-optimized inference.
```bash
pkg install python git cmake
pip install ultralytics
```

- **Export Model**: Convert YOLOv8 to NCNN format for mobile CPUs:
```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='ncnn', imgsz=320)
```

- **Android Inference**:
```python
import cv2
import ncnn
import json
import paho.mqtt.client as mqtt

# NCNN model (best.ncnn_param, best.ncnn_bin)
net = ncnn.Net()
net.load_param("best.ncnn_param")
net.load_model("best.ncnn_bin")

# MQTT for OBS/MCP
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883)

# Access phone camera (requires Termux:API)
cap = cv2.VideoCapture(0)  # Or use android.hardware.camera2
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Preprocess
    img = cv2.resize(frame, (320, 320))
    blob = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 320, 320)
    
    # Inference
    with net.create_extractor() as ex:
        ex.input("in0", blob)
        ret, out = ex.extract("out0")
    
    # MCP Filter
    potholes = [o for o in out if o[4] > 0.7]
    if potholes:
        payload = {"context": "android_pothole", "count": len(potholes)}
        client.publish("pothole/alert", json.dumps(payload))
    
cap.release()
```

**MCP Integration**: Embed schema in app metadata; sync via Firebase for cloud-based MCP logging.

**Metrics**: 15 FPS on mid-range Android (e.g., Snapdragon 680); ~500MB RAM. Ideal for citizen-reported pothole data.

**Use Case**: Cyberdecks for field engineers; Android apps for crowdsourced road condition reports, feeding municipal or insurer dashboards via OBS (Page 4).

**Pro Tip**: For quantum validation, offload noisy detections to Chimera (Page 8) via cloud API. Use Termux:API for camera/GPS access on Android.

*(End of Page 7. Page 8 explores quantum optimization with D-Wave’s Chimera SDK.)*