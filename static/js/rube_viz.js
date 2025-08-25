class RubeViz {
    constructor() {
        this.canvas = document.getElementById('rubeCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.status = "idle";
        this.setupWebSocket();
    }

    setupWebSocket() {
        this.ws = new WebSocket(`ws://${location.host}/ws/rube`);
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status) this.status = data.status;
            this.render();
        };
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = this.status === "success" ? 'green' : 'red';
        this.ctx.fillRect(0, 0, this.canvas.width * (this.status === "success" ? 1 : 0.5), 20);
        this.ctx.fillStyle = 'white';
        this.ctx.fillText(this.status, this.canvas.width / 2, 15);
    }
}

document.addEventListener('DOMContentLoaded', () => new RubeViz());
