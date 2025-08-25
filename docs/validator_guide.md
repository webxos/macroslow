Validator User Guide
Overview
The Validator is a model validation agent within the WebXOS 2025 Vial MCP SDK, designed to assess model accuracy, distribute rewards via The Chancellor, and integrate with The Alchemist for training data.
Getting Started

Clone the Repository:git clone https://github.com/webxos/webxos-vial-mcp.git


Configure Environment:Copy .env.example to .env and update with your credentials.
Run the Application:docker-compose up -d
helm install webxos ./deploy/helm/mcp-stack -f deploy/helm/mcp-stack/validator.yaml



Features

Model Validation: Validate models via /api/validator/validate.
Reward Distribution: Earn rewards for successful validations.
Visualization: Monitor validation results with validator_viz.js.

Usage

Validating a Model: Send a POST request to /api/validator/validate with a ValidationRequest JSON body.
Visualization: View validation accuracy on the canvas in index.html.
Reward Check: Verify rewards with The Chancellor’s /api/chancellor/get_balance.

Troubleshooting

Validation Fails: Ensure the model exists and data is accessible.
Visualization Issues: Check WebSocket connection to /ws/validator.

Contributing
Fork the repo, enhance validation algorithms, and submit a PR with detailed descriptions!
