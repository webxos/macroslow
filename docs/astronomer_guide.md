Astronomer User Guide
Overview
The Astronomer is a space data visualization agent within the WebXOS 2025 Vial MCP SDK, designed to fetch and process astronomical data from NASA APIs, including APOD and GIBS telescope data.
Getting Started

Clone the Repository:git clone https://github.com/webxos/webxos-vial-mcp.git


Configure Environment:Copy .env.example to .env and update with your NASA API key.
Run the Application:docker-compose up -d
helm install webxos ./deploy/helm/mcp-stack -f deploy/helm/mcp-stack/astronomer.yaml



Features

Data Fetching: Retrieve space data via /api/astronomer/fetch_space_data.
Telescope Processing: Process GIBS data with /api/astronomer/process_telescope_data.
Visualization: Monitor data on a canvas with astronomer_viz.js.

Usage

Fetching Data: Send a POST request to /api/astronomer/fetch_space_data with a SpaceDataRequest JSON body.
Visualization: View astronomical data on the canvas in index.html.
Telescope Data: Process specific datasets with a GET request to /api/astronomer/process_telescope_data/{dataset_id}.

Troubleshooting

Data Fetch Fails: Ensure NASA API key is valid and network is active.
Visualization Issues: Check WebSocket connection to /ws/astronomer.

Contributing
Fork the repo, enhance visualization features, and submit a PR with detailed descriptions!
