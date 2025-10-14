# Quantum-Enhanced YOLOv8 Guide: Edge AI for Pothole Detection and Beyond

## Page 2: YOLOv8 Setup and Kaggle Pothole Dataset Training

### Rapid Prototyping: From Dataset to Detection in Under an Hour

YOLOv8's modular design excels for quick iterations. We'll train on the Kaggle "Pothole Detection Dataset" (~1,000 images, balanced classes) to detect potholes at 30+ FPS on edge hardware.

#### Step 1: Environment Setup
Install Ultralytics via pip (no GPU needed for starters; CUDA optional for speed).

```bash
pip install ultralytics torch torchvision
pip install kaggle  # For dataset download
```

Download the dataset:
```bash
# Auth with Kaggle API key
kaggle datasets download -d atulyakumar98/pothole-detection-dataset
unzip pothole-detection-dataset.zip
```

#### Step 2: Data Preparation with YAML Config
Structure data in YOLO format (images/labels split: train/val/test). Use a `data.yaml` for MCP compatibility (semantic tagging).

```yaml
# data.yaml (MCP-Enhanced)
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 1  # num_classes (pothole only)
names: ['pothole']

# MCP Context: Embed model metadata
mcp_version: 1.0
context: "Pothole detection for rural roads; quantum-optimized thresholds"
```

#### Step 3: Model Training
Train with ~100 lines of code. On a laptop (i7, 16GB RAM), expect 50 epochs in 45-60 minutes.

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # Nano for edge; use 'yolov8s.pt' for accuracy

# Train
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else 'cpu'
)

# Export for edge (ONNX for IoT)
model.export(format='onnx')
```

**Expected Metrics:** mAP@0.5: 0.85+; Inference: 15ms/image. Visualize with `results.plot()` for bounding boxes/confidence.

#### MCP Wrapper: Embed in Model Context
Save as `.maml.md` for protocol compliance:
```markdown
---
mcp_schema: yolo_v8_pothole
version: 1.0
quantum_opt: true  # Flag for Chimera integration
---
# YOLOv8 Pothole Model
## Code Block
```python
# Insert training code here
```
## Validation Schema
- Confidence > 0.7
- IoT Stream: RTMP via OBS
```

**Pro Tip:** For real-time video (e.g., 15s rural clip), run inference:
```python
results = model('video.mp4', save=True)  # Outputs annotated video
```

Test on your laptop—next, we layer in MCP for agentic extensibility.

*(End of Page 2. Page 3 dives into MCP for structured model orchestration.)*