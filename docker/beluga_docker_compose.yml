version: '3.8'

services:
  beluga_mcp_server:
    image: webxos/beluga-mcp-server:latest
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - WEBXOS_API_TOKEN=${WEBXOS_API_TOKEN}
      - COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
      - COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
      - DB_PASSWORD=${DB_PASSWORD}
      - BLOCKCHAIN_RPC_URL=${BLOCKCHAIN_RPC_URL}
      - BLOCKCHAIN_CONTRACT_ADDRESS=${BLOCKCHAIN_CONTRACT_ADDRESS}
      - BLOCKCHAIN_ACCOUNT=${BLOCKCHAIN_ACCOUNT}
      - BLOCKCHAIN_PRIVATE_KEY=${BLOCKCHAIN_PRIVATE_KEY}
    volumes:
      - ./config:/app/config
      - ./src:/app/src
    depends_on:
      - beluga_postgres
      - beluga_redis
    networks:
      - beluga_network

  beluga_postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=beluga_db
      - POSTGRES_USER=beluga_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init_postgres.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - beluga_network

  beluga_redis:
    image: redis:7
    volumes:
      - redis_data:/data
    networks:
      - beluga_network

  beluga_federated_learning:
    image: webxos/beluga-federated-learning:latest
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - WEBXOS_API_TOKEN=${WEBXOS_API_TOKEN}
      - COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
      - COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
    volumes:
      - ./src/services/beluga_federated_learning.py:/app/beluga_federated_learning.py
    depends_on:
      - beluga_postgres
      - beluga_redis
    networks:
      - beluga_network

  beluga_svg_visualizer:
    image: webxos/beluga-svg-visualizer:latest
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8081:8081"
    environment:
      - WEBXOS_API_TOKEN=${WEBXOS_API_TOKEN}
      - COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
      - COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
    volumes:
      - ./src/services/beluga_svg_visualizer.py:/app/beluga_svg_visualizer.py
    depends_on:
      - beluga_postgres
      - beluga_redis
    networks:
      - beluga_network

  beluga_threejs_visualizer:
    image: webxos/beluga-threejs-visualizer:latest
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8082:8082"
    environment:
      - WEBXOS_API_TOKEN=${WEBXOS_API_TOKEN}
      - COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
      - COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
    volumes:
      - ./src/services/beluga_threejs_visualizer.py:/app/beluga_threejs_visualizer.py
    depends_on:
      - beluga_postgres
      - beluga_redis
    networks:
      - beluga_network

  beluga_client_training:
    image: webxos/beluga-client-training:latest
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8083:8083"
    environment:
      - WEBXOS_API_TOKEN=${WEBXOS_API_TOKEN}
      - COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
      - COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
    volumes:
      - ./src/services/beluga_client_training.py:/app/beluga_client_training.py
    depends_on:
      - beluga_postgres
      - beluga_redis
    networks:
      - beluga_network

  beluga_websocket_server:
    image: webxos/beluga-websocket-server:latest
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8084:8084"
    environment:
      - WEBXOS_API_TOKEN=${WEBXOS_API_TOKEN}
      - COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
      - COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
    volumes:
      - ./src/services/beluga_websocket_server.py:/app/beluga_websocket_server.py
    depends_on:
      - beluga_postgres
      - beluga_redis
    networks:
      - beluga_network

  beluga_anomaly_detector:
    image: webxos/beluga-anomaly-detector:latest
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8085:8085"
    environment:
      - WEBXOS_API_TOKEN=${WEBXOS_API_TOKEN}
      - COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
      - COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
    volumes:
      - ./src/services/beluga_anomaly_detector.py:/app/beluga_anomaly_detector.py
    depends_on:
      - beluga_postgres
      - beluga_redis
    networks:
      - beluga_network

networks:
  beluga_network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:

# Deployment Instructions
# Path: webxos-vial-mcp/docker/beluga_docker_compose.yml
# Run: docker-compose -f docker/beluga_docker_compose.yml up -d
# Prerequisites: Set environment variables in .env file
