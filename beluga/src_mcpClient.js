// mcpClient.js
// Description: Client-side JavaScript module for MCP server communication.
// Integrates BELUGA with CHIMERA 2048 and other MCP-compliant systems.
// Usage: Import and instantiate MCPClient for API and WebSocket communication.

class MCPClient {
    /**
     * Initializes MCP client for BELUGA.
     * @param {string} baseUrl - MCP server URL (default: http://localhost:8080).
     */
    constructor(baseUrl = "http://localhost:8080") {
        this.baseUrl = baseUrl;
        this.wsUrl = baseUrl.replace("http", "ws") + "/ws";
    }

    /**
     * Sends data to MCP server via REST.
     * @param {string} endpoint - API endpoint.
     * @param {Object} data - Data to send.
     * @returns {Promise<Object>} - Server response.
     */
    async sendMCPMessage(endpoint, data) {
        const response = await fetch(`${this.baseUrl}/${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        return response.json();
    }

    /**
     * Establishes WebSocket connection for real-time data.
     */
    connectWebSocket() {
        const ws = new WebSocket(this.wsUrl);
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            console.log(`Received: ${message.type}`, message.data);
        };
    }
}

// Example usage:
// const client = new MCPClient();
// client.sendMCPMessage("sensor_data", { data: [1, 2, 3] }).then(console.log);
// client.connectWebSocket();