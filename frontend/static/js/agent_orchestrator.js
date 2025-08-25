class AgentOrchestrator {
    constructor() {
        this.ws = new WebSocket(`ws://${location.host}/ws/orchestrator`);
        this.agents = {};
        this.setupWebSocket();
    }

    setupWebSocket() {
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.agent) this.handleAgentResponse(data);
        };
    }

    async requestAgentAction(agent, action, payload) {
        const message = { agent, action, payload, token: localStorage.getItem('token') };
        this.ws.send(JSON.stringify(message));
    }

    handleAgentResponse(data) {
        if (data.agent === 'Curator' && data.action === 'fetch_dataset') {
            document.getElementById('datasetDisplay').innerText = JSON.stringify(data.payload);
        }
        // Add handlers for other agents as needed
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const orchestrator = new AgentOrchestrator();
    document.getElementById('fetchDatasetBtn').addEventListener('click', () => {
        orchestrator.requestAgentAction('Curator', 'fetch_dataset', { dataset_id: 'nasa_123' });
    });
});
