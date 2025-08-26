class MAMLVisualizer {
    constructor() {
        this.canvas = document.createElement('canvas');
        document.body.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 800;
        this.canvas.height = 400;
    }

    visualizeExecution(result) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = 'blue';
        this.ctx.font = '16px Arial';
        let y = 20;
        for (let [key, value] of Object.entries(result.outputs)) {
            this.ctx.fillText(`${key}: ${value}`, 10, y);
            y += 20;
        }
        this.ctx.fillStyle = 'green';
        this.ctx.fillText(`Status: ${result.status}`, 10, y);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const viz = new MAMLVisualizer();
    // Example usage with WebSocket
    const ws = new WebSocket('ws://localhost:8000/ws/maml');
    ws.onmessage = (event) => {
        const result = JSON.parse(event.data);
        viz.visualizeExecution(result);
    };
});
