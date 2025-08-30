// monitoringClient.js
// Description: Client-side module for monitoring BELUGA system metrics.
// Connects to Prometheus for real-time GPU and system monitoring.
// Usage: Import and instantiate MonitoringClient to fetch metrics.

class MonitoringClient {
    /**
     * Initializes monitoring client for BELUGA.
     * @param {string} prometheusUrl - Prometheus server URL (default: http://localhost:9090).
     */
    constructor(prometheusUrl = "http://localhost:9090") {
        this.prometheusUrl = prometheusUrl;
    }

    /**
     * Fetches system metrics from Prometheus.
     * @returns {Promise<Object>} - Metrics data.
     */
    async fetchMetrics() {
        const response = await fetch(`${this.prometheusUrl}/api/v1/query?query=up`);
        return response.json();
    }
}

// Example usage:
// const monitor = new MonitoringClient();
// monitor.fetchMetrics().then(console.log);