Annot8 User Guide
Overview
Annot8 is a real-time collaborative annotation system integrated into the WebXOS 2025 Vial MCP SDK, enhanced with agentic workflows and analytics.
Getting Started

Clone the Repository:git clone https://github.com/webxos/webxos-vial-mcp.git


Configure Environment:Copy .env.example to .env and update with your credentials.
Run the Application:docker-compose up -d
helm install webxos ./deploy/helm/mcp-stack -f deploy/helm/mcp-stack/annot8-analytics.yaml



Features

Real-Time Annotation: Use the WebSocket endpoint /ws/enhanced/{client_id} to annotate collaboratively.
Analytics: Access annotation stats via /api/annot8/analytics.
Visualization: View annotations on a canvas with annot8_viz.js.

Usage

Annotating: Click on the canvas in index.html to add annotations, visible in real-time.
Analytics: Fetch user annotation counts with a valid token.
Visualization: Enable the canvas to display annotations dynamically.

Troubleshooting

Connection Issues: Ensure WebSocket and API endpoints are active.
Authorization Errors: Verify your JWT token in localStorage.

Contributing
Fork the repo, make changes, and submit a PR with detailed descriptions!
