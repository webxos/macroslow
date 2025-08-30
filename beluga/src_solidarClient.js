// solidarClient.js
// Description: Client-side JavaScript module for interacting with BELUGA’s SOLIDAR™ API.
// Facilitates real-time 3D model streaming for AR applications.
// Usage: Import and instantiate SOLIDARClient to connect to BELUGA server.

class SOLIDARClient {
    /**
     * Initializes a client to communicate with BELUGA’s SOLIDAR™ API.
     * @param {string} apiUrl - Base URL of BELUGA server (default: http://localhost:8000).
     */
    constructor(apiUrl = "http://localhost:8000") {
        this.apiUrl = apiUrl;
    }

    /**
     * Fetches fused 3D model data from the SOLIDAR™ API.
     * @param {Object} sensorData - SONAR and LIDAR data to process.
     * @returns {Promise<Object>} - Fused 3D model data.
     */
    async fetchFusedModel(sensorData) {
        const response = await fetch(`${this.apiUrl}/solidar/process`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(sensorData)
        });
        return response.json();
    }
}

// Example usage:
// const client = new SOLIDARClient();
// client.fetchFusedModel({ sonar: [1, 2, 3], lidar: [4, 5, 6] }).then(console.log);