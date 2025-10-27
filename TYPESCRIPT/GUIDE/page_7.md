# 🐪 PROJECT DUNES 2048-AES: TypeScript Guide for Quantum-Secure Model Context Protocol (MCP) Server

*TypeScript-Powered Quantum MCP Server with DUNES Minimalist SDK for Legacy and Quantum Integration*

## PAGE 7: Deploying the MCP Server with Docker and Kubernetes

Following the security implementation in Page 6, this seventh page of the **PROJECT DUNES 2048-AES TypeScript Guide** focuses on deploying the **Model Context Protocol (MCP)** server using **Docker** and **Kubernetes** within the **DUNES Minimalist SDK**. This deployment ensures scalability, resilience, and seamless integration with quantum and legacy systems, leveraging **TypeScript**’s robust architecture to orchestrate **MAML (Markdown as Medium Language)** workflows, **MARKUP Agent** processes, quantum logic, and legacy bridges. By containerizing the MCP server and deploying it on a Kubernetes cluster, developers can achieve high availability, load balancing, and efficient resource utilization, all while maintaining 2048-bit AES encryption and CRYSTALS-Dilithium signatures. This page provides detailed instructions, TypeScript-related deployment scripts, and best practices for containerized deployment, with Helm charts for simplified Kubernetes management. Guided by the camel emoji (🐪), let’s deploy the MCP server to navigate the computational frontier.

### Overview of Deployment Strategy

The **DUNES Minimalist SDK** is designed for scalable deployment, using **Docker** to containerize the MCP server and **Kubernetes** to manage clusters for high-performance, quantum-secure applications. The deployment strategy includes:

- **Dockerization**: Package the MCP server, including TypeScript code, Python dependencies (e.g., Qiskit, PyTorch), and NVIDIA CUDA libraries, into a multi-stage Docker image for portability.
- **Kubernetes Orchestration**: Deploy the Docker image on a Kubernetes cluster, using Helm charts to manage configurations, scaling, and monitoring.
- **NVIDIA GPU Support**: Enable GPU acceleration for quantum simulations and AI workloads, leveraging NVIDIA’s CUDA Toolkit and cuQuantum SDK.
- **Security Integration**: Ensure all containers use 2048-AES encryption and CRYSTALS-Dilithium signatures, with secure communication via TLS and OAuth2.0.
- **Monitoring and Scaling**: Integrate Prometheus for real-time monitoring and Kubernetes’ Horizontal Pod Autoscaling (HPA) for dynamic scaling based on workload.
- **Legacy and Quantum Compatibility**: Support connections to legacy systems (REST, SQL) and quantum microservices (CUDA-Q, Dilithium) within the cluster.

This deployment integrates with the components from previous pages: the MAML processor (Page 2), MARKUP Agent (Page 3), quantum layer (Page 4), legacy bridge (Page 5), and security layer (Page 6).

### Updating the Project Structure

To support deployment, update the project structure from Page 6 to include Docker and Kubernetes configuration files:

```
dunes-2048-aes/
├── src/
│   ├── server.ts              # Main Fastify server
│   ├── maml_processor.ts      # MAML parsing and execution
│   ├── markup_agent.ts        # MARKUP Agent logic
│   ├── markup_parser.ts       # Parses .mu syntax
│   ├── markup_receipts.ts     # Digital receipts
│   ├── markup_shutdown.ts     # Shutdown scripts
│   ├── markup_learner.ts      # PyTorch-based error detection
│   ├── markup_visualizer.ts   # Plotly visualization
│   ├── quantum_layer.ts       # Quantum circuit execution
│   ├── quantum_circuits.ts    # Quantum circuit definitions
│   ├── legacy_bridge.ts       # Legacy system integration
│   ├── legacy_rest.ts         # REST API integration
│   ├── legacy_sql.ts          # SQL database integration
│   ├── security.ts            # 2048-AES and CRYSTALS-Dilithium
│   ├── security_dilithium.ts  # CRYSTALS-Dilithium signatures
│   ├── types.ts              # TypeScript interfaces
├── docker/
│   ├── Dockerfile            # Multi-stage Dockerfile
│   ├── quantum_service.py    # Python microservice for CUDA-Q
│   ├── dilithium_service.py  # Python microservice for Dilithium
├── helm/
│   ├── Chart.yaml            # Helm chart metadata
│   ├── values.yaml           # Helm chart configurations
│   ├── templates/
│   │   ├── deployment.yaml   # Kubernetes deployment
│   │   ├── service.yaml      # Kubernetes service
│   │   ├── ingress.yaml      # Kubernetes ingress
├── .env                       # Environment variables
├── tsconfig.json             # TypeScript configuration
├── package.json              # Node.js dependencies
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
```

### Creating the Dockerfile

Create a multi-stage `docker/Dockerfile` to build the MCP server with TypeScript and Python dependencies:

```dockerfile
# Stage 1: Build TypeScript
FROM node:18 AS builder
WORKDIR /app
COPY package.json tsconfig.json ./
COPY src ./src
RUN npm install
RUN npx tsc

# Stage 2: Build Python dependencies
FROM python:3.10-slim AS python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final image with NVIDIA CUDA
FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./
COPY --from=python /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY docker/quantum_service.py docker/dilithium_service.py ./
RUN npm install --production
RUN apt-get update && apt-get install -y python3 && ln -s /usr/bin/python3 /usr/bin/python
COPY .env .
EXPOSE 8000 9000 9001
CMD ["bash", "-c", "uvicorn quantum_service:app --host 0.0.0.0 --port 9000 & uvicorn dilithium_service:app --host 0.0.0.0 --port 9001 & node dist/server.js"]
```

This Dockerfile:
- Builds TypeScript code using Node.js.
- Installs Python dependencies (Qiskit, PyTorch, SQLAlchemy).
- Uses an NVIDIA CUDA base image for GPU support.
- Runs the Fastify server and Python microservices (quantum and Dilithium).

Build the Docker image:

```bash
docker build -f docker/Dockerfile -t dunes-mcp:1.0.0 .
```

### Creating Helm Charts

Create Helm charts in `helm/` for Kubernetes deployment. Start with `helm/Chart.yaml`:

```yaml
apiVersion: v2
name: dunes-mcp
description: Quantum-Secure MCP Server for PROJECT DUNES
version: 1.0.0
appVersion: 1.0.0
```

Configure `helm/values.yaml`:

```yaml
replicaCount: 3
image:
  repository: dunes-mcp
  tag: "1.0.0"
  pullPolicy: IfNotPresent
service:
  type: ClusterIP
  port: 8000
ingress:
  enabled: true
  hosts:
    - host: mcp.webxos.local
      paths:
        - path: /
          pathType: Prefix
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    cpu: 500m
    memory: 1Gi
env:
  MCP_API_HOST: "0.0.0.0"
  MCP_API_PORT: "8000"
  DB_URI: "sqlite:///mcp_logs.db"
  QUANTUM_API_URL: "http://localhost:9000/quantum"
  JWT_SECRET: "your_jwt_secret_here"
```

Create a Kubernetes deployment in `helm/templates/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Chart.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Chart.Name }}
  template:
    metadata:
      labels:
        app: {{ .Chart.Name }}
    spec:
      containers:
        - name: mcp-server
          image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
          ports:
            - containerPort: 8000
            - containerPort: 9000
            - containerPort: 9001
          env:
            {{- range $key, $value := .Values.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
          resources:
            {{ toYaml .Values.resources | nindent 12 }}
```

Create a Kubernetes service in `helm/templates/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ .Chart.Name }}
spec:
  selector:
    app: {{ .Chart.Name }}
  ports:
    - protocol: TCP
      port: {{ .Values.service.port }}
      targetPort: 8000
  type: {{ .Values.service.type }}
```

Create an ingress in `helm/templates/ingress.yaml`:

```yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ .Chart.Name }}
spec:
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ $.Chart.Name }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}
```

### Deploying with Helm

Deploy the MCP server to a Kubernetes cluster:

1. **Install Helm** (if not already installed):

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

2. **Deploy the Helm Chart**:

```bash
helm install dunes-mcp ./helm
```

3. **Verify Deployment**:

```bash
kubectl get pods
kubectl get services
```

Access the MCP server at `http://mcp.webxos.local/maml/execute`.

### Configuring Monitoring

Integrate Prometheus for monitoring by updating `src/monitoring.ts`:

```typescript
import { FastifyInstance } from 'fastify';
import promClient from 'prom-client';

export class Monitoring {
  private requestCounter: promClient.Counter<string>;

  constructor() {
    promClient.collectDefaultMetrics();
    this.requestCounter = new promClient.Counter({
      name: 'mcp_requests_total',
      help: 'Total number of MCP requests',
      labelNames: ['endpoint'],
    });
  }

  registerRoutes(server: FastifyInstance) {
    server.get('/metrics', async (request, reply) => {
      reply.header('Content-Type', promClient.register.contentType);
      return promClient.register.metrics();
    });

    server.addHook('onRequest', (request, reply, done) => {
      this.requestCounter.inc({ endpoint: request.url });
      done();
    });
  }
}
```

Update `src/server.ts` to include monitoring:

```typescript
import { Monitoring } from './monitoring';

const monitoring = new Monitoring();
monitoring.registerRoutes(server);
```

Access metrics at `http://mcp.webxos.local/metrics`.

### Testing Deployment

Test the deployed MCP server:

```bash
curl -X POST -H "Content-Type: text/markdown" --data-binary @test.maml.md http://mcp.webxos.local/maml/execute
```

Verify metrics:

```bash
curl http://mcp.webxos.local/metrics
```

### Next Steps

This page has deployed the MCP server using Docker and Kubernetes, with GPU support and Prometheus monitoring. Subsequent pages will cover:

- **Page 8**: Use cases for healthcare, real estate, and cybersecurity.
- **Page 9**: Advanced monitoring and visualization with Prometheus and Plotly.
- **Page 10**: Advanced features and future enhancements.

**© 2025 WebXOS Research Group. All Rights Reserved. Licensed under MIT with attribution to [webxos.netlify.app](https://webxos.netlify.app).**


This page provides a detailed guide for deploying the MCP server with Docker and Kubernetes, including Helm charts and monitoring setup. Let me know if you’d like to proceed with additional pages or focus on specific aspects!
