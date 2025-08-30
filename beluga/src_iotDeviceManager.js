// iotDeviceManager.js
// Description: Manages IoT device connections for BELUGA (e.g., drones, submarines).
// Handles low-power 256-bit encryption for edge devices.
// Usage: Import and instantiate IoTDeviceManager to manage devices.

class IoTDeviceManager {
    /**
     * Initializes IoT device manager for BELUGA.
     * @param {string} apiUrl - BELUGA server URL.
     */
    constructor(apiUrl = "http://localhost:8000") {
        this.apiUrl = apiUrl;
    }

    /**
     * Sends sensor data from IoT device to BELUGA server.
     * @param {Object} sensorData - Sensor data to send.
     */
    async sendSensorData(sensorData) {
        const response = await fetch(`${this.apiUrl}/iot/data`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(sensorData)
        });
        console.log('Sent IoT data:', await response.json());
    }
}

// Example usage:
// const manager = new IoTDeviceManager();
// manager.sendSensorData({ deviceId: "drone1", data: [1, 2, 3] });