# Use a official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables for Python and secure secrets handling
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUBE_SERVER_URL="https://api.rube.app" \
    RUBE_TOKEN="" \
    DATABASE_URL="sqlite:///wallet.db"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd --create-home --shell /bin/bash agent
USER agent
WORKDIR /home/agent/app

# Copy application code
COPY --chown=agent:agent . .

# Expose ports
EXPOSE 8000-8006

# Command to run the application
CMD ["python", "-m", "backend.app.mcp.alchemist_manager"]
