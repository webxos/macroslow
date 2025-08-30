Guide: Building a Real-Time Annotation System with MCP-like Protocols

This guide will help you structure a GitHub repository that others can fork to instantly have a working environment for collaborative data annotation, perfect for data science teams.
1. Project Overview & Philosophy

Project Name: annot8 (or a name of your choice)
Core Concept: A web-based platform where authenticated users can annotate a central document (or data point) in real-time. Annotations are persisted, user-specific, and viewable by all authorized users instantly.
MCP (Model Context Protocol) Analogy: While not a literal MCP server, this system embodies the MCP spirit: it's a standalone tool that provides a structured "context" (the annotated document and its annotations) that could be queried and used by an AI model or other data science tools.
2. Repository Structure

A clear structure is key for a forkable repo.
text

annot8/
├── .github/
│   └── workflows/
│       └── ci-cd.yml                 # GitHub Actions CI/CD Pipeline
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI application core
│   │   ├── auth.py                   # OAuth 2.0 logic
│   │   ├── database.py               # DB connection & models
│   │   ├── models.py                 # Pydantic & SQLAlchemy models
│   │   └── websockets.py             # WebSocket manager
│   ├── scripts/
│   │   └── generate_html.py          # Build-time HTML generator (Jinja2)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── static/
│   │   ├── js/
│   │   │   ├── app.js                # Main JS for real-time logic
│   │   │   └── auth.js               # Handles OAuth login flow
│   │   └── css/
│   │       └── style.css
│   ├── templates/
│   │   └── index.html.j2             # Jinja2 Template
│   └── package.json                  # (Optional: for JS bundling)
├── docker-compose.yml                # For local development
├── .env.example                      # Environment variables template
└── README.md                         # Comprehensive setup guide

3. Technology Stack Deep Dive
Component	Technology	Rationale
CI/CD	GitHub Actions (YAML)	Tightly integrated with GitHub, easy for forks.
Containerization	Docker & Docker Compose	Ensures environment consistency.
Backend (API)	FastAPI (Python)	Modern, fast, built-in support for ASGI, WebSockets, and OpenAPI docs.
Real-Time Comms	WebSockets (via websockets lib)	True real-time bidirectional communication for annotations.
Database	PostgreSQL	Robust, relational, perfect for user data and annotations.
OAuth 2.0	authlib or python-jose	Professional handling of JWT tokens.
HTML Generation	Jinja2	Python-powered templating for build-time flexibility.
Frontend	Vanilla JS + WebSocket API	Lightweight, no framework required, easy to understand for forks.
4. Core Implementation Guide

A. The OAuth 2.0 "Wallet Database" (backend/app/auth.py)

This is the heart of user management and data isolation.

    Setup: Use an OAuth provider like Google or GitHub. Create a new OAuth App in their developer console. Set the redirect URI to http://localhost:8000/auth/callback for local dev and your production URL later.

    Flow: Implement the Authorization Code Grant flow with PKCE for best security.

    Database Models:
    python

    # backend/app/models.py
    from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
    from sqlalchemy.orm import relationship
    from backend.app.database import Base

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True, index=True)
        email = Column(String, unique=True, index=True)
        name = Column(String)
        # Store a unique provider ID (e.g., Google's 'sub')
        provider_id = Column(String, unique=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        # Relationship: One User to Many Annotations
        annotations = relationship("Annotation", back_populates="user")

    class Annotation(Base):
        __tablename__ = "annotations"
        id = Column(Integer, primary_key=True, index=True)
        text = Column(Text)
        x_percent = Column(Float)  # For positioning on the page
        y_percent = Column(Float)
        user_id = Column(Integer, ForeignKey("users.id"))
        created_at = Column(DateTime, default=datetime.utcnow)
        # Relationship: Many Annotations to One User
        user = relationship("User", back_populates="annotations")

    Token Handling: Upon successful OAuth callback, create or fetch the User record. Generate a session token or JWT to keep the user logged in. This token is the key to their personal "wallet" of annotations.

B. Real-Time Backend with FastAPI (backend/app/main.py, websockets.py)
python

# backend/app/main.py (Simplified)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.templating import Jinja2Templates
from .auth import get_current_user
from .websockets import ConnectionManager

app = FastAPI()
templates = Jinja2Templates(directory="../frontend/templates")
manager = ConnectionManager()

# Serve the main page (build-time generated HTML is served here)
@app.get("/")
async def get_index_page(request: Request):
    # You could pass build-time data here if needed
    return templates.TemplateResponse("index.html.j2", {"request": request})

# WebSocket endpoint for real-time annotations
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            # Receive new annotation from a client
            data = await websocket.receive_json()
            # **CRITICAL: Validate user and save to DB here**
            # ... (logic to save annotation to PostgreSQL, linked to user_id)

            # Broadcast the new annotation to ALL connected clients
            await manager.broadcast_json(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # Optional: broadcast a "user left" message

# REST API endpoint to fetch all annotations on page load
@app.get("/api/annotations")
async def get_annotations(user=Depends(get_current_user)):
    # Fetch annotations from DB, perhaps paginated
    annotations = ...  # DB query logic
    return annotations

C. Build-Time Automation (backend/scripts/generate_html.py, .github/workflows/ci-cd.yml)

generate_html.py (Jinja2 Templating):
python

from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import os

env = Environment(loader=FileSystemLoader('../frontend/templates/'))
template = env.get_template('index.html.j2')

# You can fetch data here at build-time from an API or DB for truly static generation
html_output = template.render(
    generated_date=datetime.utcnow().isoformat(),
    page_title="Our Collaborative Data Annotation Hub"
)

with open('../frontend/static/index.html', 'w') as f:
    f.write(html_output)

.github/workflows/ci-cd.yml (GitHub Actions):
yaml

name: CI/CD for annot8

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt

      - name: Run HTML Generation Script
        run: |
          cd backend
          python scripts/generate_html.py
        env:
          # Can pass build-time secrets/variables here for the template
          BUILD_TIMESTAMP: ${{ github.run_number }}

      - name: Build Docker image
        run: docker build -t annot8-backend:latest ./backend

      - name: Run Tests
        run: |
          cd backend
          python -m pytest tests/

  deploy:
    needs: build-and-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production server..."
          # Add your deployment commands here, e.g.:
          # scp docker-compose.prod.yml user@server:/app/
          # ssh user@server "docker compose -f /app/docker-compose.prod.yml pull && docker compose -f /app/docker-compose.prod.yml up -d"

D. Docker Multi-Stage Build (backend/Dockerfile)
dockerfile

# Stage 1: Builder
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /app
ENV PYTHONPATH=/app
ENV PATH="/app/.local/bin:${PATH}"

# Copy from builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/requirements.txt .

# Copy application code and generated HTML
COPY ./app ./app
COPY --from=builder /app/scripts/generate_html.py ./scripts/
# This assumes the CI job ran the script first. Alternatively, run it here:
# RUN python scripts/generate_html.py

# Copy static HTML files generated during the build
COPY ./frontend/static /app/static

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

5. Making it Forkable and Data-Science Ready

    Detailed README.md: Include:

        "Fork this repo" as the first step.

        A graphic architecture diagram.

        Setup instructions: cp .env.example .env, docker-compose up -d.

        How to register OAuth apps and configure .env variables.

        Example usage with curl commands and screenshots.

    .env.example File:
    text

DATABASE_URL=postgresql://user:password@db:5432/annot8
GOOGLE_OAUTH_CLIENT_ID=your_google_client_id
GOOGLE_OAUTH_CLIENT_SECRET=your_google_secret
SECRET_KEY=your_random_secret_key_for_sessions

docker-compose.yml for Development:
yaml

    version: '3.8'
    services:
      db:
        image: postgres:15
        environment:
          POSTGRES_DB: annot8
          POSTGRES_USER: user
          POSTGRES_PASSWORD: password
        volumes:
          - postgres_data:/var/lib/postgresql/data
      backend:
        build: ./backend
        command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
        volumes:
          - ./backend:/app
          - ./frontend/static:/app/static # Live-reload for frontend dev
        ports:
          - "8000:8000"
        environment:
          - DATABASE_URL=postgresql://user:password@db:5432/annot8
        depends_on:
          - db
    volumes:
      postgres_data:

    Data Science Integration:

        Add an API Endpoint: GET /api/annotations/export that returns all annotations as a CSV or JSON file for easy analysis in a Jupyter notebook.

        Include Example Notebooks: Add a notebooks/ directory with a Jupyter notebook showing how to load the exported data and perform basic analysis (e.g., most active annotators, common terms).

I'll help you troubleshoot the connection issues with your Streamable HTTP client and servers. Let me analyze the code you provided.

## Issues Identified:

### 1. **Port Conflicts in main.py**
Your `main.py` is trying to start both servers on the same port (8000):

```python
# Both servers are trying to use port 8000
ev_points_server_thread = threading.Thread(target=ev_points_server.run_server, args=('localhost', 8000))
weather_server_thread = threading.Thread(target=weather_server.run_server, args=('localhost', 8000))
```

### 2. **Incorrect URL Construction in client.py**
The client is using hardcoded paths that don't match your server endpoints:

```python
# Current problematic code
response = requests.get(f"http://{server_address}/ev_points")
response = requests.get(f"http://{server_address}/weather")
```

### 3. **Missing Server Endpoints**
Your servers don't have the endpoints that the client is trying to access.

## Fixed Code:

### main.py (Fixed)
```python
import threading
import ev_points_server
import weather_server

def main():
    # Use different ports for each server
    ev_points_server_thread = threading.Thread(target=ev_points_server.run_server, args=('localhost', 8001))
    weather_server_thread = threading.Thread(target=weather_server.run_server, args=('localhost', 8002))
    
    ev_points_server_thread.start()
    weather_server_thread.start()
    
    print("Servers started on ports 8001 (EV Points) and 8002 (Weather)")
    
    ev_points_server_thread.join()
    weather_server_thread.join()

if __name__ == "__main__":
    main()
```

### client.py (Fixed)
```python
import requests
import json
import time

class StreamableHTTPClient:
    def __init__(self, servers):
        self.servers = servers
    
    def fetch_data(self, server_type):
        results = []
        for server in self.servers:
            if server['type'] == server_type:
                try:
                    # Construct proper URL based on server type
                    if server_type == 'ev_points':
                        url = f"http://{server['address']}:{server['port']}/ev/points"
                    elif server_type == 'weather':
                        url = f"http://{server['address']}:{server['port']}/weather/data"
                    else:
                        continue
                    
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        results.append(response.json())
                    else:
                        print(f"Error from {server['address']}: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to connect to {server['address']}:{server['port']} - {e}")
        
        return results
    
    def stream_data(self, interval=5):
        try:
            while True:
                print("\n--- Fetching Data ---")
                
                # Get EV points data
                ev_data = self.fetch_data('ev_points')
                if ev_data:
                    print("EV Points Data:")
                    for data in ev_data:
                        print(json.dumps(data, indent=2))
                
                # Get weather data
                weather_data = self.fetch_data('weather')
                if weather_data:
                    print("Weather Data:")
                    for data in weather_data:
                        print(json.dumps(data, indent=2))
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nClient stopped by user")

# Example usage
if __name__ == "__main__":
    servers = [
        {'type': 'ev_points', 'address': 'localhost', 'port': 8001},
        {'type': 'weather', 'address': 'localhost', 'port': 8002}
    ]
    
    client = StreamableHTTPClient(servers)
    client.stream_data(interval=3)
```

### ev_points_server.py (Enhanced)
```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random

class EVPointsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/ev/points':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Sample EV points data
            ev_data = {
                "points": [
                    {"id": 1, "location": "Downtown", "available": random.choice([True, False]), "power_kw": 50},
                    {"id": 2, "location": "Shopping Mall", "available": random.choice([True, False]), "power_kw": 22},
                    {"id": 3, "location": "Airport", "available": random.choice([True, False]), "power_kw": 150}
                ],
                "timestamp": "2024-01-15T10:30:00Z"
            }
            
            self.wfile.write(json.dumps(ev_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Endpoint not found')

def run_server(host='localhost', port=8001):
    server = HTTPServer((host, port), EVPointsHandler)
    print(f"EV Points Server running on http://{host}:{port}")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
```

### weather_server.py (Enhanced)
```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random

class WeatherHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/weather/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Sample weather data
            weather_data = {
                "temperature": round(random.uniform(15, 30), 1),
                "humidity": random.randint(40, 90),
                "conditions": random.choice(["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]),
                "wind_speed": round(random.uniform(0, 15), 1),
                "timestamp": "2024-01-15T10:30:00Z"
            }
            
            self.wfile.write(json.dumps(weather_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Endpoint not found')

def run_server(host='localhost', port=8002):
    server = HTTPServer((host, port), WeatherHandler)
    print(f"Weather Server running on http://{host}:{port}")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
```

## How to Run:

1. **Start the servers:**
   ```bash
   python main.py
   ```

2. **Run the client in a separate terminal:**
   ```bash
   python client.py
   ```

## Key Changes Made:

1. **Different ports** for each server (8001 for EV, 8002 for Weather)
2. **Proper endpoint URLs** that match between client and servers
3. **Added CORS headers** to avoid cross-origin issues
4. **Error handling** for connection failures
5. **Proper JSON response formatting**

## Testing:

You can test each server individually using curl:
```bash
# Test EV server
curl http://localhost:8001/ev/points

# Test Weather server  
curl http://localhost:8002/weather/data
```

This should resolve your connection issues. Let me know if you encounter any specific errors!

By following this guide, your GitHub repo will be a powerful, self-contained, and easily forkable template for anyone looking to build a real-time, authenticated collaborative application, perfectly suited for data annotation and MCP-inspired context building.
