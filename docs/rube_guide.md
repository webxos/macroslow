Rube Integration User Guide
Overview
Rube.app integrates over 500 business and productivity apps into the WebXOS 2025 Vial MCP SDK, enhancing agentic workflows for scientific data study and OBS live streams.
Getting Started

Clone the Repository:git clone https://github.com/webxos/webxos-vial-mcp.git


Configure Environment:Copy .env.example to .env and update with your Rube token and app credentials.
Run the Application:docker-compose up -d
helm install webxos ./deploy/helm/mcp-stack -f deploy/helm/mcp-stack/rube.yaml



Features

App Connectivity: Access GitHub, Slack, Notion, and more for scientific tasks.
Automation: Use natural language commands to execute tasks via Rube.
Visualization: Monitor Rube task status with rube_viz.js.
Optimization: Enhance tool execution with rube_optimizer.py.

Usage

Connecting Apps: Authenticate apps in the Rube dashboard once.
Executing Tasks: Use commands like "Create a GitHub issue for this dataset" in Claude.
Visualization: View task status on the canvas in index.html.
Optimization: Leverage rube_optimizer.py for efficient tool use.

Troubleshooting

Connection Issues: Verify Rube token and network to https://api.rube.app.
Task Failures: Check app authentication and Rube tool availability.
Visualization Issues: Ensure WebSocket connection to /ws/rube.

Contributing
Fork the repo, enhance Rube integrations, and submit a PR with detailed descriptions!
