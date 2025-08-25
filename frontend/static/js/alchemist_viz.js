class AlchemistViz {
    constructor() {
        this.canvas = document.getElementById('alchemistCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.progress = 0;
        this.setupWebSocket();
    }

    setupWebSocket() {
        this.ws = new WebSocket(`ws://${location.host}/ws/alchemist`);
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status) this.progress = data.status === "trained" ? 100 : this.progress + 10;
            this.render();
        };
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = 'green';
        this.ctx.fillRect(0, 0, (this.progress / 100) * this.canvas.width, 20);
        this.ctx.fillStyle = 'white';
        this.ctx.fillText(`${this.progress}%`, this.canvas.width / 2, 15);
    }
}

document.addEventListener('DOMContentLoaded', () => new AlchemistViz());
