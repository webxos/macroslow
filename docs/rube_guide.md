## Rube Integration User Guide

# Overview

Rube.app integrates over 500 business and productivity apps into the MCP SDK, enhancing agentic workflows for scientific data study and OBS live streams.

# Features

App Connectivity: Access GitHub, Slack, Notion, and more for scientific tasks.
Automation: Use natural language commands to execute tasks via Rube.
Visualization: Monitor Rube task status with rube_viz.js.
Optimization: Enhance tool execution with rube_optimizer.py.

# Usage

Connecting Apps: Authenticate apps in the Rube dashboard once.
Executing Tasks: Use commands like "Create a GitHub issue for this dataset" in Claude Code for example.
Visualization: View task status on the canvas in index.html.
Optimization: Leverage rube_optimizer.py for efficient tool use.

# Troubleshooting

Connection Issues: Verify Rube token and network to https://api.rube.app.
Task Failures: Check app authentication and Rube tool availability.
Visualization Issues: Ensure WebSocket connection to /ws/rube.
