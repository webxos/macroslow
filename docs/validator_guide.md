Validator User Guide
Overview
The Validator is a model validation agent within the WebXOS 2025 Vial MCP SDK, assessing model accuracy, distributing rewards via The Chancellor, and integrating with Rube.app for automated notifications.
Getting Started

Clone the Repository:git clone https://github.com/webxos/webxos-vial-mcp.git


Configure Environment:Copy .env.example to .env and update with your Rube token and credentials.
Run the Application:docker-compose up -d
helm install webxos ./deploy/helm/mcp-stack -f deploy/helm/mcp-stack/validator.yaml



Features

Model Validation: Validate models via /api/validator/validate.
Reward Distribution: Earn rewards for successful validations.
Rube Integration: Automatically notify via Slack using Rube tools.
Visualization: Monitor results with validator_viz.js.

Usage

Validating a Model: Send a POST request to /api/validator/validate with a ValidationRequest JSON body.
Visualization: View validation accuracy on the canvas in index.html.
Rube Notifications: Successful validations trigger Slack messages via Rube.
Reward Check: Verify rewards with The Chancellor’s /api/chancellor/get_balance.

Troubleshooting

Validation Fails: Ensure the model exists and data is accessible.
Rube Issues: Check Rube token and network connectivity to https://api.rube.app.
Visualization Issues: Verify WebSocket connection to /ws/validator.

Contributing
Fork the repo, enhance validation or Rube integration, and submit a PR with detailed descriptions!
