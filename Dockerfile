# Use a official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables for Python and secure secrets handling
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUBE_SERVER_URL="https://api.rube.app" \
    RUBE_TOKEN=""

# Install system dependencies required for your agents and any potential MCP client libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it for security
RUN useradd --create-home --shell /bin/bash agent
USER agent
WORKDIR /home/agent/app

# Copy application code and install Python dependencies
COPY --chown=agent:agent requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY --chown=agent:agent . .

# Expose necessary ports if your MCP client runs a server
EXPOSE 3000

# Command to run your agentic application
CMD ["python", "main_orchestrator.py"]
