## MAML Validator User Guide

# Overview

The Validator is a model validation agent within the MAML syntax to validate the code, assessing model accuracy, distributing rewards via a 'chancellor' agent, and integrating with automated notifications.

# Features

MAML Validation: Validate models via /api/validator/validate.
Model Reward Distribution: Models earn rewards for successful validations.
Visualization: Monitor results with validator_viz.js.

# Usage

Validating a Model: Send a POST request to /api/validator/validate with a ValidationRequest JSON body.
Visualization: View validation accuracy on the canvas in index.html.
Optional Rube Notifications: Successful validations trigger Slack messages via Rube.
Reward Check: Verify rewards with The Chancellor’s /api/chancellor/get_balance.

# Troubleshooting

Validation Fails: Ensure the model exists and data is accessible.
Visualization Issues: Verify WebSocket connection to /ws/validator.
