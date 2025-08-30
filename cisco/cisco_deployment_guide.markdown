# Cisco XDR + DUNES CORE SDK Deployment Guide 📖

This guide deploys the Cisco XDR-integrated MCP server using Docker or Vercel. 🚀

## Prerequisites
- Docker or Vercel CLI
- Cisco XDR API credentials
- DUNES CORE SDK repository

## Steps
1. **Build Docker Image**:
   ```bash
   docker build -t cisco-xdr-mcp -f cisco/Dockerfile .
   ```
2. **Run Container**:
   ```bash
   docker run -p 8000:8000 --env-file cisco/.env cisco-xdr-mcp
   ```
3. **Vercel Deployment**:
   - Install Vercel CLI: `npm i -g vercel`
   - Deploy: `vercel --prod`
4. **Test API**:
   ```bash
   curl -X POST http://localhost:8000/process_maml -H "Content-Type: application/json" -d '{"content": "---\ntitle: Test\n---\n## Objective\nTest"}'
   ```

## Customization
- Update `cisco_xdr_config.py` for additional telemetry endpoints.
- Add playbooks in `xdr_automation_playbook.py` for custom responses.
- Modify `xdr_workflow_visualizer.py` for specific visualizations.