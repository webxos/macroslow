```markdown
# 🐪 PROJECT DUNES 2048-AES: TypeScript Guide for Quantum-Secure Model Context Protocol (MCP) Server

*TypeScript-Powered Quantum MCP Server with DUNES Minimalist SDK for Legacy and Quantum Integration*

## PAGE 5: Building the Legacy System Bridge for REST and SQL Integration

Following the quantum integration outlined in Page 4, this fifth page of the **PROJECT DUNES 2048-AES TypeScript Guide** focuses on building the **legacy system bridge**, a critical component of the **Model Context Protocol (MCP)** server within the **DUNES Minimalist SDK**. This bridge enables seamless integration with legacy systems—such as REST APIs, SQL databases, and enterprise software—while maintaining quantum-secure workflows with 2048-bit AES encryption. By leveraging **TypeScript**’s type safety and interoperability with JavaScript, developers can connect the MCP server to existing infrastructure, ensuring compatibility with systems like SAP, Oracle, or custom REST endpoints, while enabling quantum-enhanced processing via MAML (Markdown as Medium Language) workflows. This page provides detailed instructions, TypeScript code examples, and best practices for implementing the legacy system bridge, ensuring robust data exchange and transformation. Guided by the camel emoji (🐪), let’s bridge the past and future of computing.

### Overview of the Legacy System Bridge

The **legacy system bridge** is a TypeScript-based module that connects the MCP server to classical systems, enabling it to query and transform data from REST APIs, SQL databases, and other enterprise platforms. This integration is essential for organizations transitioning to quantum-ready architectures while maintaining compatibility with existing infrastructure. Key objectives include:

- **REST API Integration**: Connect to legacy REST endpoints using TypeScript’s `axios` or `fetch` libraries, mapping responses to MAML schemas for quantum-secure processing.
- **SQL Database Integration**: Query legacy databases (e.g., MySQL, PostgreSQL) using TypeORM or a TypeScript-to-Python bridge with SQLAlchemy, transforming results into MAML workflows.
- **Enterprise System Compatibility**: Interface with systems like SAP, Oracle, or custom APIs, ensuring seamless data flow between legacy and quantum environments.
- **Real-Time Data Streams**: Support WebSocket-based streams from legacy IoT devices, feeding data into the MCP server for quantum-enhanced analysis.
- **Security and Validation**: Enforce 2048-AES encryption and CRYSTALS-Dilithium signatures for data transfers, with TypeScript’s type system ensuring robust validation.

The legacy system bridge works in tandem with the MARKUP Agent (Page 3) and quantum layer (Page 4), enabling the MCP server to orchestrate workflows across classical and quantum domains.

### Updating the Project Structure

To incorporate the legacy system bridge, update the project structure from Page 4 to include legacy-specific files:

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
│   ├── database.ts            # TypeORM/SQLAlchemy integration
│   ├── monitoring.ts          # Prometheus metrics
│   ├── types.ts              # TypeScript interfaces
├── Dockerfile                 # Multi-stage Dockerfile
├── helm/                      # Helm charts
├── .env                       # Environment variables
├── tsconfig.json             # TypeScript configuration
├── package.json              # Node.js dependencies
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation дитя

```

### Implementing the Legacy System Bridge

The legacy system bridge, implemented in `src/legacy_bridge.ts`, coordinates REST and SQL integrations. Below is the core implementation:

```typescript
import axios from 'axios';
import { DataSource } from 'typeorm';
import { AppDataSource } from './database';

interface LegacyResponse {
  data: any;
  status: number;
  headers: Record<string, string>;
}

interface LegacyQuery {
  endpoint?: string; // For REST
  sql?: string; // For SQL
  params?: Record<string, any>;
}

export class LegacyBridge {
  private restBaseUrl: string;
  private dataSource: DataSource;

  constructor() {
    this.restBaseUrl = process.env.LEGACY_REST_URL || 'http://legacy-api:8080';
    this.dataSource = AppDataSource;
  }

  async queryRest(endpoint: string, params: Record<string, any> = {}): Promise<LegacyResponse> {
    try {
      const response = await axios.get(`${this.restBaseUrl}/${endpoint}`, { params });
      return {
        data: response.data,
        status: response.status,
        headers: response.headers as Record<string, string>,
      };
    } catch (error) {
      throw new Error(`REST query failed: ${error.message}`);
    }
  }

  async querySql(query: string, params: Record<string, any> = {}): Promise<any[]> {
    try {
      const connection = this.dataSource;
      const result = await connection.query(query, Object.values(params));
      return result;
    } catch (error) {
      throw new Error(`SQL query failed: ${error.message}`);
    }
  }

  async transformToMaml(data: any, metadata: { id: string; type: string }): Promise<string> {
    // Transform legacy data into MAML format
    const mamlContent = `---
maml_version: "0.1.0"
id: "${metadata.id}"
type: "${metadata.type}"
origin: "agent://legacy-bridge"
created_at: "${new Date().toISOString()}"
---
## Data
${JSON.stringify(data, null, 2)}
`;
    return mamlContent;
  }
}
```

### Implementing REST Integration

In `src/legacy_rest.ts`, implement specific REST API interactions:

```typescript
import { LegacyBridge } from './legacy_bridge';

export class LegacyRest {
  private bridge: LegacyBridge;

  constructor(bridge: LegacyBridge) {
    this.bridge = bridge;
  }

  async fetchPatientRecords(patientId: string): Promise<string> {
    const response = await this.bridge.queryRest(`patients/${patientId}`);
    return this.bridge.transformToMaml(response.data, {
      id: `patient_${patientId}_${Date.now()}`,
      type: 'dataset',
    });
  }

  async fetchInventory(stockId: string): Promise<string> {
    const response = await this.bridge.queryRest(`inventory/${stockId}`);
    return this.bridge.transformToMaml(response.data, {
      id: `inventory_${stockId}_${Date.now()}`,
      type: 'dataset',
    });
  }
}
```

### Implementing SQL Integration

In `src/legacy_sql.ts`, implement SQL database queries:

```typescript
import { LegacyBridge } from './legacy_bridge';

export class LegacySql {
  private bridge: LegacyBridge;

  constructor(bridge: LegacyBridge) {
    this.bridge = bridge;
  }

  async fetchPatientRecords(patientId: string): Promise<string> {
    const query = 'SELECT * FROM patients WHERE patient_id = ?';
    const result = await this.bridge.querySql(query, { patientId });
    return this.bridge.transformToMaml(result, {
      id: `patient_${patientId}_${Date.now()}`,
      type: 'dataset',
    });
  }

  async fetchInventory(stockId: string): Promise<string> {
    const query = 'SELECT * FROM inventory WHERE stock_id = ?';
    const result = await this.bridge.querySql(query, { stockId });
    return this.bridge.transformToMaml(result, {
      id: `inventory_${stockId}_${Date.now()}`,
      type: 'dataset',
    });
  }
}
```

### Integrating with the MCP Server

Update `src/server.ts` to include legacy system routes:

```typescript
import { FastifyInstance } from 'fastify';
import { LegacyBridge } from './legacy_bridge';
import { LegacyRest } from './legacy_rest';
import { LegacySql } from './legacy_sql';
import { initializeDatabase } from './database';

const server: FastifyInstance = Fastify({ logger: true });
const legacyBridge = new LegacyBridge();
const legacyRest = new LegacyRest(legacyBridge);
const legacySql = new LegacySql(legacyBridge);

async function startServer() {
  await initializeDatabase();

  server.get<{ Params: { patientId: string } }>('/legacy/patient/:patientId', async (request, reply) => {
    try {
      const mamlContent = await legacyRest.fetchPatientRecords(request.params.patientId);
      return reply.status(200).send({ maml: mamlContent });
    } catch (error) {
      return reply.status(500).send({ error: error.message });
    }
  });

  server.get<{ Params: { stockId: string } }>('/legacy/inventory/:stockId', async (request, reply) => {
    try {
      const mamlContent = await legacySql.fetchInventory(request.params.stockId);
      return reply.status(200).send({ maml: mamlContent });
    } catch (error) {
      return reply.status(500).send({ error: error.message });
    }
  });

  const port = parseInt(process.env.MCP_API_PORT || '8000', 10);
  await server.listen({ port, host: process.env.MCP_API_HOST || '0.0.0.0' });
  console.log(`MCP Server running on port ${port}`);
}

startServer();
```

### Testing Legacy Integration

Test the REST integration with a curl request:

```bash
curl http://localhost:8000/legacy/patient/12345
```

Expected output is a MAML-formatted response containing patient data. For SQL integration:

```bash
curl http://localhost:8000/legacy/inventory/67890
```

Expected output is a MAML-formatted response with inventory data.

### Securing Legacy Data Transfers

Ensure all legacy data transfers are encrypted with 2048-AES. Update `src/security.ts` (to be detailed in Page 6) to include encryption for legacy responses:

```typescript
import * as crypto from 'crypto';

export class Security {
  static encryptData(data: string, key: string): string {
    const cipher = crypto.createCipheriv('aes-512-cbc', Buffer.from(key), Buffer.alloc(16));
    return cipher.update(data, 'utf8', 'hex') + cipher.final('hex');
  }
}
```

Apply encryption in `LegacyBridge`:

```typescript
async transformToMaml(data: any, metadata: { id: string; type: string }): Promise<string> {
  const encryptedData = Security.encryptData(JSON.stringify(data), process.env.JWT_SECRET || 'secret');
  const mamlContent = `---
maml_version: "0.1.0"
id: "${metadata.id}"
type: "${metadata.type}"
origin: "agent://legacy-bridge"
created_at: "${new Date().toISOString()}"
---
## Data
${encryptedData}
`;
  return mamlContent;
}
```

### Next Steps

This page has implemented the legacy system bridge, enabling REST and SQL integration with MAML transformation. Subsequent pages will cover:

- **Page 6**: Configuring 2048-AES security and CRYSTALS-Dilithium signatures.
- **Page 7**: Deploying the MCP server with Docker and Kubernetes.
- **Page 8**: Use cases for healthcare, real estate, and cybersecurity.
- **Page 9**: Monitoring and visualization with Prometheus and Plotly.
- **Page 10**: Advanced features and future enhancements.

**© 2025 WebXOS Research Group. All Rights Reserved. Licensed under MIT with attribution to [webxos.netlify.app](https://webxos.netlify.app).**
```

This page provides a detailed implementation of the legacy system bridge, including TypeScript code for REST and SQL integration, with secure data transformation into MAML format. Let me know if you’d like to proceed with additional pages or focus on specific aspects!
