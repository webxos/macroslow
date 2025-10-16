## Page 7: Server-Side Event Handling with MCP for Scalable MACROSLOW Applications

### Introduction to Server-Side Event Handling in MCP
Server-side event handling is a critical component of the **Model Context Protocol (MCP)**, enabling robust, scalable, and secure communication between clients and servers in **MACROSLOW 2048-AES** applications (DUNES, GLASTONBURY, CHIMERA). This page provides an in-depth exploration of building MCP servers using **TypeScript**, focusing on handling **tool executions**, **notifications**, and **session persistence** with a **PostgreSQL event store**. We address the community’s need for clear session storage examples, as highlighted by the query: *“I can’t find examples of what an event store is supposed to look like :)”*. By integrating with **MAML (Markdown as Medium Language)** and `.markup (.mu)` files, this page demonstrates how to create a scalable MCP server that supports real-time event processing, quantum-resistant security, and multi-node deployments, tailored for MACROSLOW’s decentralized and quantum-ready ecosystem.

Server-side event handling in MCP involves:
- **Tool Registration**: Defining and executing tools (e.g., `greet`, `process_solidar`, `run_quantum_circuit`) with structured inputs/outputs.
- **Notification Streaming**: Sending real-time updates to clients via **Streamable HTTP transport** or legacy **SSE transport**.
- **Session Persistence**: Logging events in a persistent event store for resumability and auditability.
- **Error Detection**: Using `.mu` receipts to validate event integrity, leveraging the **MARKUP Agent**.
- **Scalability**: Supporting multi-node deployments with stateless, persistent storage, or message queue-based architectures.

This page provides detailed TypeScript examples, MAML workflows, and `.mu` receipt generation, ensuring compatibility with DUNES (minimalist prototyping), GLASTONBURY (robotics), and CHIMERA (quantum API gateway).

### Why Server-Side Event Handling Matters
MCP servers act as the backbone of event-driven applications, orchestrating interactions between clients, tools, and external systems. TypeScript’s static typing ensures robust event handling by enforcing strict interfaces for tool arguments, notification payloads, and session data. Key benefits include:
- **Reliability**: Persistent event stores ensure no data loss during server restarts or client reconnections.
- **Scalability**: Multi-node setups (stateless, persistent storage, or message queue-based) handle high-throughput scenarios.
- **Security**: Integration with MAML’s quantum-resistant cryptography (e.g., CRYSTALS-Dilithium) secures event data.
- **Flexibility**: MAML and `.mu` files enable structured workflows and error detection, enhancing server reliability.
- **Community Support**: Clear event store examples address documentation gaps, making server setup accessible to novices.

### MCP Server Architecture
An MCP server with **Streamable HTTP transport** (protocol version 2025-03-26) handles:
- **POST /mcp**: Initializes sessions and processes tool/prompt requests.
- **GET /mcp**: Streams notifications via Server-Sent Events (SSE).
- **DELETE /mcp**: Terminates sessions, cleaning up event store data.
- **Event Store**: Persists session events for resumability, using `mcp-session-id` and `Last-Event-ID`.

For multi-node deployments, MCP supports:
- **Stateless Mode**: No session tracking, ideal for simple API proxies.
- **Persistent Storage Mode**: Stores session data in a database (e.g., PostgreSQL), allowing any node to handle requests.
- **Message Queue Mode**: Routes requests to specific nodes for in-memory state, using pub/sub systems.

### TypeScript Server Implementation
Below is a comprehensive TypeScript implementation of an MCP server using the **StreamableHTTPServerTransport**, integrating with the `PostgresEventStore` from Page 3 and supporting DUNES, GLASTONBURY, and CHIMERA workflows.

```typescript
// src/server/mcpServer.ts
import express from 'express';
import { StreamableHTTPServerTransport } from 'mcp-typescript-sdk';
import { PostgresEventStore } from '../utils/eventStore';
import { v4 as uuidv4 } from 'uuid';
import { CustomMarkupAgent } from '../utils/markupAgent';

interface ToolArgs {
  [key: string]: any;
}

const app = express();
app.use(express.json());

const eventStore = new PostgresEventStore();
const markupAgent = new CustomMarkupAgent();

const transport = new StreamableHTTPServerTransport({
  app,
  path: '/mcp',
  sessionIdGenerator: () => uuidv4(),
  eventStore,
});

/**
 * Registers a generic tool with event logging and .mu receipt generation.
 */
async function registerTool(name: string, handler: (args: ToolArgs) => Promise<any>) {
  transport.registerTool(name, async (args: ToolArgs) => {
    try {
      const result = await handler(args);
      await eventStore.saveEvent(transport.sessionId, 'tool_call', {
        tool: name,
        args,
        result,
      }, { source: 'mcp_server', timestamp: new Date().toISOString() });

      // Generate .mu receipt
      const muReceipt = await markupAgent.generateMu(JSON.stringify(result));
      console.log(`Tool ${name} executed. Receipt:`, muReceipt);

      return result;
    } catch (error) {
      console.error(`Error executing tool ${name}:`, error);
      throw new Error(`Tool execution failed: ${error.message}`);
    }
  });
}

// Register tools for DUNES, GLASTONBURY, and CHIMERA
registerTool('greet', async (args: { name: string }) => {
  return `Hello, ${args.name}!`;
});

registerTool('process_solidar', async (args: { sonar: number[]; lidar: number[] }) => {
  // Simplified SOLIDAR fusion
  const fused = args.sonar.map((s, i) => (s + args.lidar[i]) / 2);
  return fused;
});

registerTool('run_quantum_circuit', async (args: { circuit: any }) => {
  // Placeholder: Execute via Qiskit server-side (requires Python bridge)
  const result = { counts: { '00': 512, '11': 512 } };
  return result;
});

// Handle notifications
transport.on('notification', async (event) => {
  try {
    await eventStore.saveEvent(transport.sessionId, 'notification', event.payload, {
      source: 'mcp_server',
      timestamp: new Date().toISOString(),
    });
    console.log('Server notification:', event.payload);

    // Generate .mu receipt for notification
    const muReceipt = await markupAgent.generateMu(JSON.stringify(event.payload));
    console.log('Notification receipt:', muReceipt);
  } catch (error) {
    console.error('Error handling notification:', error);
  }
});

// Handle session termination
transport.on('session_terminated', async (sessionId: string) => {
  try {
    await eventStore.deleteSessionEvents(sessionId);
    console.log(`Session ${sessionId} terminated, events cleared`);
  } catch (error) {
    console.error(`Error terminating session ${sessionId}:`, error);
  }
});

app.listen(3000, () => console.log('MCP server running on port 3000'));
```

### MAML Workflow for Server Tools
Define a MAML file to structure the `greet`, `process_solidar`, and `run_quantum_circuit` tools:
```markdown
// src/maml/server_tools.maml.md
---
tools:
  - name: greet
    arguments:
      name: string
  - name: process_solidar
    arguments:
      sonar: array
      lidar: array
  - name: run_quantum_circuit
    arguments:
      circuit: object
---
## Context
Defines server-side tools for DUNES, GLASTONBURY, and CHIMERA workflows.

## Input_Schema
```json
{
  "greet": {
    "type": "object",
    "properties": { "name": { "type": "string", "minLength": 1 } },
    "required": ["name"]
  },
  "process_solidar": {
    "type": "object",
    "properties": {
      "sonar": { "type": "array", "items": { "type": "number" } },
      "lidar": { "type": "array", "items": { "type": "number" } }
    },
    "required": ["sonar", "lidar"]
  },
  "run_quantum_circuit": {
    "type": "object",
    "properties": { "circuit": { "type": "object" } },
    "required": ["circuit"]
  }
}
```

## Code_Blocks
```typescript
function greet(name: string): string {
  return `Hello, ${name}!`;
}

function processSolidar(args: { sonar: number[]; lidar: number[] }): number[] {
  return args.sonar.map((s, i) => (s + args.lidar[i]) / 2);
}

function runQuantumCircuit(args: { circuit: any }): any {
  // Placeholder: Qiskit execution
  return { counts: { '00': 512, '11': 512 } };
}
```

## Output_Schema
```json
{
  "greet": { "type": "string", "pattern": "^Hello, .+!$" },
  "process_solidar": { "type": "array", "items": { "type": "number" } },
  "run_quantum_circuit": { "type": "object", "properties": { "counts": { "type": "object" } } }
}
```
```

### Generating .markup (.mu) Receipts
Use the **MARKUP Agent** to generate `.mu` receipts for event validation:
```typescript
// src/utils/markupAgent.ts
import { MarkupAgent } from '@macroslow/dunes';

export class CustomMarkupAgent extends MarkupAgent {
  async generateMu(input: string): Promise<string> {
    try {
      // Reverse the input string for .mu receipt
      const reversed = input.split('').reverse().join('');
      const receipt = `## Receipt\n${reversed}`;
      return receipt;
    } catch (error) {
      console.error('Error generating .mu receipt:', error);
      throw new Error(`Receipt generation failed: ${error.message}`);
    }
  }

  async validateMu(input: string, receipt: string): Promise<boolean> {
    const expected = input.split('').reverse().join('');
    return receipt.includes(expected);
  }
}
```

Example `.mu` receipt for a tool call:
```markdown
// src/mu/tool_receipt.mu
## Receipt
olleH, Alice!
```

### Multi-Node Deployment with Message Queues
For scalability, configure the server for **message queue mode** using a pub/sub system (e.g., Redis):
```typescript
// src/server/mcpServerWithQueue.ts
import express from 'express';
import { StreamableHTTPServerTransport } from 'mcp-typescript-sdk';
import { PostgresEventStore } from '../utils/eventStore';
import { v4 as uuidv4 } from 'uuid';
import { createClient } from 'redis';

const app = express();
app.use(express.json());

const eventStore = new PostgresEventStore();
const redis = createClient({ url: 'redis://localhost:6379' });
await redis.connect();

const transport = new StreamableHTTPServerTransport({
  app,
  path: '/mcp',
  sessionIdGenerator: () => uuidv4(),
  eventStore,
});

// Route requests via Redis pub/sub
transport.on('request', async (request) => {
  const { sessionId, payload } = request;
  const sessionOwner = await redis.get(`session:${sessionId}`);
  if (sessionOwner && sessionOwner !== 'this_node') {
    // Forward to owning node
    await redis.publish(`mcp:requests:${sessionId}`, JSON.stringify(payload));
  } else {
    // Handle locally and register session
    await redis.set(`session:${sessionId}`, 'this_node');
  }
});

// Subscribe to responses
redis.subscribe(`mcp:responses:${uuidv4()}`, async (message) => {
  const response = JSON.parse(message);
  await eventStore.saveEvent(response.sessionId, 'response', response.payload);
});

registerTool('greet', async (args: { name: string }) => {
  return `Hello, ${args.name}!`;
});

app.listen(3000, () => console.log('MCP server with message queue running on port 3000'));
```

### Performance Considerations
- **Event Store Optimization**: Use indexes on `session_id` and `event_id` (Page 3) for fast queries.
- **Notification Batching**: Batch notifications to reduce database writes in high-throughput scenarios.
- **Security**: Encrypt event payloads with CRYSTALS-Dilithium signatures for quantum resistance.
- **TypeScript Interfaces**: Define strict types for tool arguments and payloads:
  ```typescript
  interface ToolPayload {
    tool: string;
    args: Record<string, any>;
    result: any;
  }
  ```

### Use Cases for Server-Side Event Handling
1. **Decentralized Exchanges (DEXs)**: Process trade orders and stream updates, logging events securely.
2. **Robotics Control**: Handle GLASTONBURY’s sensor data and control commands in real-time.
3. **Quantum Workflows**: Execute CHIMERA’s quantum circuits, storing results for analysis.
4. **Auditability**: Use `.mu` receipts to validate events for compliance and debugging.

### Troubleshooting
- **Connection Issues**: Ensure the server is accessible at `http://localhost:3000/mcp`.
- **Event Store Errors**: Verify PostgreSQL connectivity and schema (Page 3).
- **Message Queue Issues**: Check Redis connectivity and session ownership logic.
- **Receipt Mismatches**: Debug `.mu` generation by logging intermediate reversed outputs.

### Why This Matters
Server-side event handling with MCP enables:
- **Scalability**: Multi-node setups support high-throughput applications.
- **Reliability**: Persistent event stores ensure session resumability, addressing community documentation gaps.
- **Security**: Quantum-resistant encryption protects sensitive event data.
- **Flexibility**: MAML and `.mu` files provide structured workflows and error detection.

### Next Steps
- Dive into MAML/.mu workflows for advanced orchestration (Page 8).
- Address community challenges and use cases (Page 9).
- Explore real-world applications with DUNES, GLASTONBURY, and CHIMERA (Page 10).

**This server implementation empowers developers to build scalable, secure MCP applications with robust event handling, leveraging TypeScript and MACROSLOW SDKs for decentralized and quantum-ready systems.**