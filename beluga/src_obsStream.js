// obsStream.js
// Description: JavaScript module for streaming BELUGA’s 3D models to OBS for AR visualization.
// Connects to OBS WebSocket for real-time video feeds.
// Usage: Import and instantiate OBSStream to stream data.

import OBSWebSocket from 'obs-websocket-js';

class OBSStream {
    /**
     * Initializes OBS WebSocket connection for streaming BELUGA data.
     * @param {string} wsUrl - OBS WebSocket URL (default: ws://localhost:4455).
     * @param {string} password - OBS WebSocket password.
     */
    constructor(wsUrl = "ws://localhost:4455", password = "") {
        this.obs = new OBSWebSocket();
        this.wsUrl = wsUrl;
        this.password = password;
    }

    /**
     * Connects to OBS and streams 3D model data.
     * @param {ArrayBuffer} modelData - Fused 3D model data.
     */
    async streamModel(modelData) {
        await this.obs.connect(this.wsUrl, this.password);
        await this.obs.call('SetInputSettings', {
            inputName: 'BELUGA_AR',
            inputSettings: { buffer: modelData }
        });
        console.log('Streaming to OBS...');
    }
}

// Example usage:
// const stream = new OBSStream();
// stream.streamModel(new ArrayBuffer(128));