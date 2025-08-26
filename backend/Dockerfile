# Stage 1: Builder
FROM nvidia/cuda:11.8.0-base-ubuntu20.04 AS builder
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y python3.11 python3-pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
WORKDIR /app
ENV PYTHONPATH=/app
ENV PATH="/root/.local/bin:${PATH}"

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/requirements.txt .
COPY backend/app ./app
COPY frontend/static ./static

RUN apt-get update && apt-get install -y python3.11 && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
