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
COPY ./scripts/annot8_generate_html.py ./scripts/
RUN python scripts/annot8_generate_html.py
COPY ./frontend/static /app/static

# Install additional WebXOS dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
CMD ["uvicorn", "app.annot8.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
