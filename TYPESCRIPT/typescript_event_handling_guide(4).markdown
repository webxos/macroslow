## Page 4: DUNES SDK - Mastering Event Handling for Minimalist MCP Applications

### Introduction to DUNES and Event Handling
The **DUNES SDK**, part of the **MACROSLOW 2048-AES** lineup, is a minimalist toolkit designed for rapid prototyping of **Model Context Protocol (MCP)** servers and clients. Tailored for developers seeking lightweight, flexible solutions, DUNES excels in building decentralized applications with event-driven architectures. This page provides an in-depth exploration of event handling with DUNES, leveraging **TypeScript**, **MAML (Markdown as Medium Language)**, and `.markup (.mu)` files to create seamless workflows for tool execution, notifications, and session persistence. We address the community’s need for clear examples, particularly around session storage, by demonstrating how DUNES integrates with a PostgreSQL **event store** and supports **quantum-resistant workflows** for secure, scalable applications.

DUNES is ideal for:
- **Rapid Prototyping**: Quickly building MCP servers for decentralized exchanges (DEXs) or IoT systems.
- **Event-Driven Workflows**: Managing tool calls, notifications, and session events with minimal overhead.
- **MAML Integration**: Using structured Markdown for executable workflows and validation.
- **.markup (.mu) Receipts**: Generating reversed receipts for error detection and auditability.

This page dives into DUNES’s event handling capabilities, providing detailed TypeScript examples, MAML workflows, and `.mu` receipt generation, while ensuring compatibility with MCP’s **Streamable HTTP transport** and persistent storage mode.

### Why DUNES for Event Handling?
DUNES’s minimalist design makes it perfect for developers new to MCP, offering a low-barrier entry to building event-driven systems. Its event handling features include:
- **Session Management**: Tracks client-server interactions using `mcp-session-id` and an event store for resumability.
- **Real-Time Notifications**: Streams updates via Streamable HTTP, supporting use cases like live status updates in decentralized networks.
- **Tool Execution**: Executes predefined tools (e.g., `greet`) with structured inputs defined in MAML files.
- **Error Detection**: Uses `.mu` files to validate event integrity by reversing content (e.g., “Hello” to “olleH”).
- **TypeScript Benefits**: Leverages TypeScript’s static typing to ensure robust event payloads, reducing runtime errors in distributed systems.

By integrating with a persistent event store, DUNES addresses the user’s query: *“I can’t find examples of what an event store is supposed to look like :)”*, providing clear, practical implementations.

### DUNES Event Handling Workflow
DUNES handles events through a combination of MCP’s client-server communication and MAML/.mu workflows:
1. **Client Initialization**: A TypeScript client connects to an MCP server using `StreamableHTTPClient`.
2. **Tool Calls**: Clients invoke tools (e.g., `greet`) defined in MAML files, with results stored in the event store.
3. **Notifications**: Servers stream notifications (e.g., status updates) to clients, logged for resumability.
4. **Session Persistence**: Events are saved in a PostgreSQL event store, enabling reconnection without data loss.
5. **Error Detection**: The **MARKUP Agent** generates `.mu` receipts to validate event integrity.

### TypeScript Client Implementation
Below is a comprehensive TypeScript client using DUNES to handle events, integrating with the `PostgresEventStore` from Page 3.

```typescript
// src/client/dunesClient.ts
import { StreamableHTTPClient } from 'mcp-typescript-sdk';
import { v4 as uuidv4 } from 'uuid';
import { PostgresEventStore } from '../utils/eventStore';
import { MarkupAgent } from '@macroslow/dunes';

interface DunesClientConfig {
  url: string;
  eventStore?: PostgresEventStore;
}

export class DunesClient {
  private client: StreamableHTTPClient;
  private markupAgent: MarkupAgent;

  constructor(config: DunesClientConfig) {
    this.client = new StreamableHTTPClient({
      url: config.url,
      sessionIdGenerator: () => uuidv4(),
      eventStore: config.eventStore || new PostgresEventStore(),
    });
    this.markupAgent = new MarkupAgent();
  }

  /**
   * Connects to the MCP server and sets up notification handling.
   */
  async connect(): Promise<void> {
    try {
      await this.client.connect();
      console.log('Connected to MCP server:', this.client.url);

      // Handle notifications
      this.client.on('notification', async (event) => {
        const payload = event.payload;
        await this.client.eventStore.saveEvent(
          this.client.sessionId,
          'notification',
          payload,
          { source: 'client', timestamp: new Date().toISOString() }
        );
        console.log('Received notification:', payload);

        // Generate .mu receipt for validation
        const muReceipt = await this.markupAgent.generateMu(JSON.stringify(payload));
        console.log('Generated .mu receipt:', muReceipt);
      });
    } catch (error) {
      console.error('Failed to connect:', error);
      throw new Error(`Connection error: ${error.message}`);
    }
  }

  /**
   * Calls a tool and logs the result in the event store.
   */
  async callTool(toolName: string, args: any): Promise<any> {
    try {
      const result = await this.client.callTool(toolName, args);
      await this.client.eventStore.saveEvent(
        this.client.sessionId,
        'tool_call',
        { tool: toolName, args, result },
        { source: 'client', timestamp: new Date().toISOString() }
      );

      // Generate .mu receipt
      const muReceipt = await this.markupAgent.generateMu(result);
      console.log(`Tool ${toolName} executed. Receipt:`, muReceipt);

      return result;
    } catch (error) {
      console.error(`Error calling tool ${toolName}:`, error);
      throw new Error(`Tool execution failed: ${error.message}`);
    }
  }

  /**
   * Disconnects the client and closes the event store.
   */
  async disconnect(): Promise<void> {
    await this.client.disconnect();
    if (this.client.eventStore instanceof PostgresEventStore) {
      await this.client.eventStore.close();
    }
    console.log('Disconnected from MCP server');
  }
}
```

### Example Usage
Run the DUNES client to interact with an MCP server:
```typescript
// src/index.ts
import { DunesClient } from './client/dunesClient';

async function main() {
  const client = new DunesClient({ url: 'http://localhost:3000/mcp' });
  try {
    await client.connect();

    // Call the 'greet' tool
    const response = await client.callTool('greet', { name: 'Alice' });
    console.log('Tool Response:', response);

    // Keep the client running to receive notifications
    await new Promise((resolve) => setTimeout(resolve, 5000));
  } catch (error) {
    console.error('Error:', error);
  } finally {
    await client.disconnect();
  }
}

main().catch(console.error);
```

Compile and run:
```bash
npx tsc
npx tsx src/index.ts
```

### MAML Workflow for Tool Execution
Define a `greet` tool in a MAML file to structure the workflow and validate inputs/outputs:
```markdown
// src/maml/greet.maml.md
---
tool: greet
arguments:
  name: string
---
## Context
Executes a greeting tool that returns a personalized message.

## Input_Schema
```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string", "minLength": 1 }
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
  "type": "string",
  "pattern": "^Hello, .+!$"
}
```
```

### Generating .markup (.mu) Receipts
Use the **MARKUP Agent** to generate a `.mu` file for error detection:
```typescript
// src/utils/markupAgent.ts
import { MarkupAgent } from '@macroslow/dunes';

export class CustomMarkupAgent extends MarkupAgent {
  async generateMu(input: string): Promise<string> {
    try {
      // Reverse the input string for .mu receipt
      const reversed = input.split('').reverse().join('');
      const receipt = `## Receipt\n${reversed}`;
      // Optionally save to file (e.g., src/mu/receipt.mu)
      // await fs.writeFile('src/mu/receipt.mu', receipt);
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

Example `.mu` receipt:
```markdown
// src/mu/receipt.mu
## Receipt
olleH, Alice!
```

### Server-Side Integration
To test the client, ensure an MCP server is running with the `greet` tool:
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
  console.log('Server notification:', event.payload);
});

app.listen(3000, () => console.log('MCP server running on port 3000'));
```

Run the server:
```bash
npx tsc
npx tsx src/server/simpleServer.ts
```

### Error Detection with .mu Receipts
Validate tool execution results using the MARKUP Agent:
```typescript
// src/index.ts (continued)
import { CustomMarkupAgent } from './utils/markupAgent';

async function validateToolResult(client: DunesClient, result: string) {
  const markupAgent = new CustomMarkupAgent();
  const muReceipt = await markupAgent.generateMu(result);
  const isValid = await markupAgent.validateMu(result, muReceipt);
  console.log('Validation result:', isValid ? 'Valid' : 'Invalid');
}
```

### Performance Considerations
- **Event Store Efficiency**: Use indexes on `session_id` and `event_id` (see Page 3) to optimize queries.
- **Notification Handling**: Batch notifications to reduce overhead in high-frequency scenarios.
- **TypeScript Types**: Define strict interfaces for tool arguments and payloads to prevent runtime errors:
  ```typescript
  interface GreetArgs {
    name: string;
  }

  interface GreetResult {
    greeting: string;
  }
  ```

### Use Cases for DUNES Event Handling
1. **Decentralized Exchanges (DEXs)**: Execute trade orders as tools, log events, and validate with `.mu` receipts.
2. **IoT Prototyping**: Stream sensor data as notifications, storing events for analysis.
3. **Workflow Automation**: Use MAML to define automated tasks, with `.mu` files ensuring integrity.
4. **Error Recovery**: Detect and correct errors in event payloads using reversed `.mu` receipts.

### Troubleshooting
- **Connection Issues**: Ensure the MCP server is running and accessible at `http://localhost:3000/mcp`.
- **Event Store Errors**: Verify PostgreSQL connectivity and table schema (Page 3).
- **MAML Validation**: Check that MAML schemas match tool inputs/outputs to avoid runtime errors.
- **Receipt Mismatches**: Debug `.mu` generation by logging intermediate reversed outputs.

### Why This Matters
DUNES’s minimalist approach simplifies event handling for MCP, making it accessible for novices while supporting advanced features like:
- **Session Persistence**: The event store ensures reliable session resumption, addressing community documentation gaps.
- **Error Detection**: `.mu` receipts provide a unique mechanism for validating event integrity.
- **Scalability**: Lightweight design supports rapid iteration and deployment in decentralized systems.
- **Interoperability**: MAML workflows integrate seamlessly with GLASTONBURY and CHIMERA for broader applications.

### Next Steps
- Explore GLASTONBURY for robotics event handling (Page 5).
- Dive into CHIMERA for quantum event workflows (Page 6).
- Learn advanced server configurations for multi-node setups (Page 7).

**DUNES empowers developers to build lightweight, event-driven MCP applications with robust session management and error detection, leveraging TypeScript and MAML for secure, scalable workflows.**