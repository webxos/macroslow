class AstronomerViz {
    constructor() {
        this.canvas = document.getElementById('astronomerCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.data = [];
        this.setupWebSocket();
    }

    setupWebSocket() {
        this.ws = new WebSocket(`ws://${location.host}/ws/astronomer`);
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.data) this.data = data.data;
            this.render();
        };
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.data.forEach((item, index) => {
            this.ctx.beginPath();
            this.ctx.arc(index * 20, this.canvas.height - (item.value * this.canvas.height / 100), 5, 0, 2 * Math.PI);
            this.ctx.fillStyle = 'blue';
            this.ctx.fill();
            this.ctx.fillText(item.date, index * 20, this.canvas.height - 10);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => new AstronomerViz());
