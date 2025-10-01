const axios = require('axios');
const fs = require('fs');
const Plotly = require('plotly')({ offline: true }); // Offline mode for CI compatibility

async function test3DVisualization() {
  try {
    // Simulate fetching agent task output (e.g., from agent_sandbox.py)
    const response = await axios.post('http://localhost:8000/mcp/agents/crewai', {
      task: 'Emergency Response Coordination'
    }, {
      headers: { 'Authorization': 'Bearer <token>' } // Replace with valid token
    }).catch(() => ({
      data: { result: 'CrewAI task executed: Emergency Response Coordination' } // Mock response for testing
    }));

    const taskData = response.data;

    // Sample data for 3D visualization (e.g., agent execution metrics)
    const data = [{
      x: [1, 2, 3, 4], // Time points
      y: [10, 15, 13, 17], // Task success scores
      z: [5, 6, 7, 8], // Resource usage
      type: 'scatter3d',
      mode: 'markers',
      marker: { size: 12, color: '#1f77b4', opacity: 0.8 }
    }];

    const layout = {
      title: `3D Visualization of CrewAI Task: ${taskData.result}`,
      scene: {
        xaxis: { title: 'Time (s)' },
        yaxis: { title: 'Success Score' },
        zaxis: { title: 'Resource Usage (MB)' }
      }
    };

    // Save visualization data (offline mode for CI)
    fs.writeFileSync('test-3d-output.json', JSON.stringify({ data, layout }, null, 2));
    console.log('3D visualization data saved to test-3d-output.json');
  } catch (error) {
    console.error('Error testing 3D visualization:', error.message);
    process.exit(1);
  }
}

test3DVisualization();