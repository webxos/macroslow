class ValidatorViz {
    constructor() {
        this.canvas = document.getElementById('validatorCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.results = { accuracy: 0, status: "pending" };
        this.setupWebSocket();
    }

    setupWebSocket() {
        this.ws = new WebSocket(`ws://${location.host}/ws/validator`);
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status) this.results = { accuracy: data.accuracy, status: data.status };
            this.render();
        };
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = this.results.status === "valid" ? 'green' : 'red';
        this.ctx.fillRect(0, 0, (this.results.accuracy * this.canvas.width), 20);
        this.ctx.fillStyle = 'white';
        this.ctx.fillText(`${this.results.accuracy * 100}% (${this.results.status})`, this.canvas.width / 2, 15);
    }
}

document.addEventListener('DOMContentLoaded', () => new ValidatorViz());
