Alchemist User Guide
Overview
The Alchemist is a model training agent within the WebXOS 2025 Vial MCP SDK, designed to orchestrate machine learning workflows using PyTorch and integrate with The Mechanic for resource management.
Getting Started

Clone the Repository:git clone https://github.com/webxos/webxos-vial-mcp.git


Configure Environment:Copy .env.example to .env and update with your credentials.
Run the Application:docker-compose up -d
helm install webxos ./deploy/helm/mcp-stack -f deploy/helm/mcp-stack/alchemist.yaml



Features

Model Training: Train models with customizable epochs via /api/alchemist/train.
Progress Visualization: Monitor training progress with alchemist_viz.js.
State Retrieval: Fetch model states using /api/alchemist/state.

Usage

Training a Model: Send a POST request to /api/alchemist/train with a TrainingRequest JSON body.
Visualization: View training progress on the canvas in index.html.
State Check: Retrieve model state with a GET request to /api/alchemist/state/{model_id}.

Troubleshooting

Training Fails: Ensure GPU resources are available and data paths are valid.
Visualization Issues: Check WebSocket connection to /ws/alchemist.

Contributing
Fork the repo, enhance training algorithms, and submit a PR with detailed descriptions!
