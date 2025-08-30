# Cisco XDR Advanced Developer’s Cut Guide: Part 5 - Scalable Deployment Architectures 🚀

Welcome to **Part 5**! 🌟 This part deploys the Cisco XDR-integrated MCP server using **Kubernetes**, **Vercel**, and **CI/CD pipelines** for enterprise scalability.[](https://www.cisco.com/c/en/us/products/collateral/security/xdr/xdr-ds.html)

## 🌟 Overview
- **Goal**: Deploy MCP server with high availability and scalability.
- **Tools**: Kubernetes, Vercel, GitHub Actions, DUNES CORE SDK.
- **Use Cases**: Global SOC operations, multi-cloud threat response.

## 📋 Steps

### 1. Create Kubernetes Deployment
Create `cisco/k8s_deployment.yaml` for Kubernetes.

<xaiArtifact artifact_id="02cdb3c0-4687-4bb7-9c26-dcb887a8714f" artifact_version_id="48b71c86-0e3b-4de2-9c5e-b445e372f81a" title="cisco/k8s_deployment.yaml" contentType="text/yaml">
# k8s_deployment.yaml: Kubernetes deployment for Cisco XDR + DUNES MCP server
# CUSTOMIZATION POINT: Update replicas, resources, and environment variables
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cisco-xdr-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cisco-xdr-mcp
  template:
    metadata:
      labels:
        app: cisco-xdr-mcp
    spec:
      containers:
      - name: cisco-xdr-mcp
        image: cisco-xdr-mcp:latest
        ports:
        - containerPort: 8000
        env:
        - name: DUNES_OEM_NAME
          value: "CiscoXDREnhancedSDK"
        - name: DUNES_DB_URI
          value: "sqlite:///cisco/dunes_logs.db"
        - name: CISCO_XDR_API_KEY
          valueFrom:
            secretKeyRef:
              name: xdr-secrets
              key: api-key
---
apiVersion: v1
kind: Service
metadata:
  name: cisco-xdr-mcp-service
spec:
  selector:
    app: cisco-xdr-mcp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer