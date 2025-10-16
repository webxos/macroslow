## Page 2: In-Depth Setup of the MACROSLOW TypeScript Environment for Event Handling

### Introduction to TypeScript and Its Role in MCP
TypeScript, a superset of JavaScript, introduces static typing, interfaces, and advanced tooling to enhance code reliability and scalability. Its type safety is critical for building robust **Model Context Protocol (MCP)** applications, where precise handling of events, sessions, and data structures is essential. In the context of **MACROSLOW SDKs (DUNES, GLASTONBURY, CHIMERA)**, TypeScript ensures that developers can define clear interfaces for tools, prompts, and notifications, reducing runtime errors in distributed systems. For MCP, TypeScript’s type system aligns perfectly with the protocol’s need for structured communication between clients and servers, especially when managing complex event-driven workflows involving **MAML (Markdown as Medium Language)** and `.markup (.mu)` files. This page provides a detailed setup guide, explaining how TypeScript integrates with MCP for event handling, session persistence, and quantum-ready applications.

TypeScript’s benefits for MCP include:
- **Type Safety**: Ensures that session IDs, event payloads, and tool arguments conform to expected formats, critical for secure and reliable event handling.
- **Modularity**: Supports reusable components for DUNES (minimalist SDK), GLASTONBURY (robotics), and CHIMERA (quantum API gateway).
- **Scalability**: Facilitates large-scale deployments with clear interfaces for distributed systems, such as multi-node MCP servers.
- **Tooling**: Offers autocompletion and error detection in IDEs, improving developer productivity for complex workflows like quantum simulations with Qiskit.

### Why TypeScript for Event Handling in MCP?
Event handling in MCP involves managing **sessions** (tracked via `mcp-session-id`), **notifications** (streamed via Streamable HTTP or SSE), and **tool/prompt execution** across distributed nodes. TypeScript’s static types ensure that event payloads are correctly formatted, preventing issues like malformed JSON or missing session IDs. For MACROSLOW, event handling is central to:
- **Session Persistence**: Storing events in an **event store** for resumability, addressing the user’s concern: *"I can’t find examples of what an event store is supposed to look like :)"*.
- **Real-Time Notifications**: Processing streams of data, such as sensor updates in GLASTONBURY or quantum states in CHIMERA.
- **MAML/.markup Workflows**: Orchestrating commands and generating `.mu` receipts for error detection and auditability.

### Prerequisites
To build event-driven MCP applications with MACROSLOW SDKs, ensure the following are installed:

- **Node.js (v18+)**: Provides the runtime for TypeScript and MCP SDKs. Verify installation:
  ```bash
  node --version
  # Expected: >= v18.0.0
  ```
- **TypeScript (v5+)**: Enables compilation and execution of TypeScript code. Install globally:
  ```bash
  npm install -g typescript tsx
  tsc --version
  # Expected: >= v5.0.0
  ```
- **Docker**: Supports containerized deployments for scalability and portability.
  ```bash
  docker --version
  # Ensure Docker is installed
  ```
- **Python and Qiskit**: Required for CHIMERA’s quantum simulations, leveraging Qiskit for bilinear qubit networks.
  ```bash
  pip install qiskit
  python -c "import qiskit; print(qiskit.__version__)"
  # Expected: >= v0.46.0
  ```
- **PostgreSQL**: Used for persistent event storage, critical for session resumability.
  ```bash
  psql --version
  # Ensure PostgreSQL is installed
  ```
- **Dependencies**: Install MACROSLOW SDKs, MCP TypeScript SDK, and supporting libraries:
  ```bash
  npm install @macroslow/dunes @macroslow/glastonbury @macroslow/chimera mcp-typescript-sdk express pg uuid @types/express @types/pg @types/uuid
  ```

### Project Setup
Set up a TypeScript project tailored for MCP event handling:

1. **Initialize the Project**:
   Create a new directory and initialize a Node.js project:
   ```bash
   mkdir macroslow-event-handling
   cd macroslow-event-handling
   npm init -y
   ```

2. **Install Dependencies**:
   Install required packages for TypeScript, MCP, and MACROSLOW SDKs:
   ```bash
   npm install typescript tsx @macroslow/dunes @macroslow/glastonbury @macroslow/chimera mcp-typescript-sdk express pg uuid @types/express @types/pg @types/uuid
   ```

3. **Configure TypeScript**:
   Create a `tsconfig.json` file to enable strict typing, modern JavaScript features, and source map support for debugging:
   ```json
   {
     "compilerOptions": {
       "target": "ES2020",
       "module": "commonjs",
       "strict": true,
       "esModuleInterop": true,
       "moduleResolution": "node",
       "outDir": "./dist",
       "sourceMap": true,
       "resolveJsonModule": true,
       "types": ["node", "express", "pg", "uuid"]
     },
     "include": ["src/**/*"],
     "exclude": ["node_modules"]
   }
   ```

4. **Directory Structure**:
   Organize the project to separate client, server, utilities, and MAML/.mu files:
   ```bash
   mkdir -p src/{client,server,utils,maml,mu}
   touch src/index.ts
   touch src/utils/eventStore.ts
   touch src/client/mcpClient.ts
   touch src/server/simpleServer.ts
   touch src/maml/workflow.maml.md
   touch src/mu/receipt.mu
   ```

### Configuring MCP and MACROSLOW SDKs
The MACROSLOW SDKs integrate with the MCP TypeScript SDK to handle events, sessions, and workflows. Below are key configurations:

- **MCP Client Configuration**:
  Create a reusable MCP client with session persistence using a PostgreSQL event store:
  ```typescript
  // src/client/mcpClient.ts
  import { StreamableHTTPClient } from 'mcp-typescript-sdk';
  import { v4 as uuidv4 } from 'uuid';
  import { PostgresEventStore } from '../utils/eventStore';

  export interface McpClientConfig {
    url: string;
    eventStore?: PostgresEventStore;
  }

  export function createMcpClient(config: McpClientConfig): StreamableHTTPClient {
    return new StreamableHTTPClient({
      url: config.url,
      sessionIdGenerator: () => uuidv4(),
      eventStore: config.eventStore || new PostgresEventStore(),
    });
  }
  ```

- **Event Store Implementation**:
  Address the user’s query about event store examples by implementing a PostgreSQL-based store for session persistence. The event store tracks session events (e.g., tool calls, notifications) with `mcp-session-id` and `Last-Event-ID` for resumability:
  ```typescript
  // src/utils/eventStore.ts
  import { Pool } from 'pg';
  import { v4 as uuidv4 } from 'uuid';

  interface Event {
    sessionId: string;
    eventId: string;
    eventType: string;
    payload: any;
    createdAt: Date;
  }

  export class PostgresEventStore {
    private pool: Pool;

    constructor() {
      this.pool = new Pool({
        user: 'postgres',
        host: 'localhost',
        database: 'mcp_db',
        password: 'your_password', // Replace with secure password
        port: 5432,
      });
    }

    async saveEvent(sessionId: string, eventType: string, payload: any): Promise<void> {
      const eventId = uuidv4();
      try {
        await this.pool.query(
          'INSERT INTO mcp_events (session_id, event_id, event_type, payload) VALUES ($1, $2, $3, $4)',
          [sessionId, eventId, eventType, JSON.stringify(payload)]
        );
      } catch (error) {
        console.error('Error saving event:', error);
        throw error;
      }
    }

    async getEvents(sessionId: string, lastEventId?: string): Promise<Event[]> {
      try {
        const query = lastEventId
          ? 'SELECT * FROM mcp_events WHERE session_id = $1 AND event_id > $2 ORDER BY created_at'
          : 'SELECT * FROM mcp_events WHERE session_id = $1 ORDER BY created_at';
        const values = lastEventId ? [sessionId, lastEventId] : [sessionId];
        const result = await this.pool.query(query, values);
        return result.rows.map(row => ({
          sessionId: row.session_id,
          eventId: row.event_id,
          eventType: row.event_type,
          payload: row.payload,
          createdAt: row.created_at,
        }));
      } catch (error) {
        console.error('Error retrieving events:', error);
        throw error;
      }
    }

    async close(): Promise<void> {
      await this.pool.end();
    }
  }
  ```

- **Database Setup**:
  Create the `mcp_events` table in PostgreSQL to store session events:
  ```bash
  psql -U postgres -c "CREATE DATABASE mcp_db"
  psql -U postgres -d mcp_db
  ```
  ```sql
  CREATE TABLE mcp_events (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    event_id VARCHAR(36) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_session_event UNIQUE (session_id, event_id)
  );
  ```

### MAML and .markup (.mu) Integration
**MAML** and `.markup (.mu)` files are central to MACROSLOW’s event-driven workflows, encoding commands, tools, and receipts for error detection and auditability. Create sample files to integrate with TypeScript:

- **MAML Workflow**:
  Define a tool execution workflow in a MAML file:
  ```markdown
  // src/maml/workflow.maml.md
  ---
  tool: greet
  arguments:
    name: string
  ---
  ## Context
  Executes a greeting tool with a name argument, generating a greeting message.

  ## Input_Schema
  ```json
  {
    "type": "object",
    "properties": {
      "name": { "type": "string" }
    },
    "required": ["name"]
  }
  ```

  ## Code_Blocks
  ```typescript
  function greet(name: string): string {
    return `Hello, ${name}!`;
  }
  ```

  ## Output_Schema
  ```json
  {
    "type": "string"
  }
  ```
  ```

- **.markup (.mu) Receipt**:
  Generate a `.mu` file for error detection and auditability:
  ```markdown
  // src/mu/receipt.mu
  ## Receipt
  olleH, Alice!
  ```

### Docker Configuration for Scalability
Containerize the MCP server for production-grade deployments, ensuring portability and scalability:
```dockerfile
# Dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npx tsc
EXPOSE 3000
ENV POSTGRES_HOST=host.docker.internal
ENV POSTGRES_PASSWORD=your_password
CMD ["node", "dist/index.js"]
```

Build and run the container:
```bash
docker build -t macroslow-event-handling .
docker run -p 3000:3000 -e POSTGRES_HOST=host.docker.internal -e POSTGRES_PASSWORD=your_password macroslow-event-handling
```

### Testing the Setup
Create a simple MCP server to test the environment and event handling:
```typescript
// src/server/simpleServer.ts
import express from 'express';
import { StreamableHTTPServerTransport } from 'mcp-typescript-sdk';
import { PostgresEventStore } from '../utils/eventStore';
import { v4 as uuidv4 } from 'uuid';

const app = express();
app.use(express.json());

const eventStore = new PostgresEventStore();
const transport = new StreamableHTTPServerTransport({
  app,
  path: '/mcp',
  sessionIdGenerator: () => uuidv4(),
  eventStore,
});

transport.registerTool('greet', async (args: { name: string }) => {
  const greeting = `Hello, ${args.name}!`;
  await eventStore.saveEvent(transport.sessionId, 'tool_call', {
    tool: 'greet',
    args,
    result: greeting,
  });
  return greeting;
});

transport.on('notification', async (event) => {
  await eventStore.saveEvent(transport.sessionId, 'notification', event.payload);
  console.log('Notification saved:', event.payload);
});

app.listen(3000, () => console.log('MCP server running on port 3000'));
```

Compile and run:
```bash
npx tsc
npx tsx src/server/simpleServer.ts
```

### Troubleshooting
- **Database Connection Issues**: Ensure PostgreSQL is running and credentials are correct. Use `psql -U postgres -h localhost` to test.
- **TypeScript Errors**: Verify `tsconfig.json` settings and installed `@types/*` packages.
- **Docker Issues**: Confirm Docker is running and the `POSTGRES_HOST` environment variable is set correctly (e.g., `host.docker.internal` for local development).

### Why This Setup Matters
This environment establishes a foundation for event-driven MCP applications, enabling:
- **Session Persistence**: The `PostgresEventStore` ensures events are saved and retrievable, addressing the user’s need for clear event store examples.
- **Scalability**: Docker and TypeScript’s modularity support multi-node deployments (see Page 7).
- **Interoperability**: MAML and `.mu` files integrate with DUNES, GLASTONBURY, and CHIMERA for diverse use cases.
- **Quantum Readiness**: Qiskit integration prepares the setup for CHIMERA’s quantum event handling.

### Next Steps
- Test the server with a client implementation (Page 4).
- Explore GLASTONBURY for robotics event handling (Page 5).
- Dive into CHIMERA for quantum workflows (Page 6).
- Learn advanced server-side event handling and multi-node setups (Page 7).

**This setup empowers developers to build secure, scalable, and quantum-ready MCP applications with MACROSLOW SDKs, leveraging TypeScript’s robustness for event handling.**